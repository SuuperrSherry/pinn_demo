# train.py
import os
from typing import Tuple, Dict, List
import torch
from torch.amp import autocast, GradScaler

from sampler import UniformSampler, AdaptiveSampler
from config import Config, set_random_seeds
from model import PINN
from losses import compute_loss
from bifurcation import export_branch_csv, SaddleNodeDetector


def _sync_cfg_instance_to_class(cfg_obj):
    """把实例 cfg 的公开属性同步到 Config 类属性，确保 losses 等模块读取一致。"""
    for k in dir(cfg_obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(cfg_obj, k)
        except Exception:
            continue
        if isinstance(v, (int, float, bool, str, tuple, list, dict, torch.device)):
            setattr(Config, k, v)


class PINNTrainer:
    """稳健训练器：AMP(可选)+自适应采样+梯度裁剪+LR调度+CSV导出"""
    def __init__(self, cfg: Config, F):
        _sync_cfg_instance_to_class(cfg)
        set_random_seeds()
        self.cfg = cfg
        self.F = F

        # 设备 & AMP
        self.device = getattr(cfg, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.scaler = GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and bool(getattr(cfg, "AMP", False)))
        )

        # —— 采样器 —— #
        S_max = float(getattr(cfg, "S_MAX", 7.0))
        pts   = int(getattr(cfg, "POINTS_PER_SEGMENT", 80))

        self.use_adaptive        = bool(getattr(cfg, "USE_ADAPTIVE", False))
        self.adapt_warmup_iters  = int(getattr(cfg, "ADAPTIVE_WARMUP_ITERS", 800))
        self.adapt_update_every  = int(getattr(cfg, "ADAPTIVE_UPDATE_EVERY", 200))
        self.batch_points        = int(getattr(cfg, "BATCH_POINTS", pts))

        self.uni_sampler = UniformSampler(S_max, self.device)
        self.adp_sampler = AdaptiveSampler(
            S_max, self.device,
            grid_size=int(getattr(cfg, "ADAPTIVE_GRID_SIZE", 256)),
            score_type=str(getattr(cfg, "ADAPTIVE_SCORE", "res")),  # "res" 或 "sigma"
            mix=float(getattr(cfg, "ADAPTIVE_MIX", 0.5)),
            temperature=float(getattr(cfg, "ADAPTIVE_TEMP", 0.5))
        )

        # 初始：均匀训练网格
        self.s_train = self.uni_sampler.sample_grid(pts)

        # 初值（y0 = [x0..., p0...]）
        y0 = getattr(cfg, "Y0", [2.0, 0.0])
        self.y0 = torch.tensor([y0], dtype=torch.float32, device=self.device)  # [1, nx+np]

        # 模型
        nx = int(getattr(cfg, "NX", 1)); np_ = int(getattr(cfg, "NP", 1))
        hidden = int(getattr(cfg, "HIDDEN_NEURONS", 32))
        layers = int(getattr(cfg, "HIDDEN_LAYERS", 3))
        act = getattr(cfg, "ACT", "tanh")
        self.net = PINN(nx=nx, np_=np_, hidden=hidden, layers=layers, act=act).to(self.device)

        # 优化器 & 调度器
        lr = float(getattr(cfg, "LEARNING_RATE", 1e-3))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        step_size = int(getattr(cfg, "LR_STEP_SIZE", 2000))
        gamma = float(getattr(cfg, "LR_GAMMA", 0.3))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=step_size, gamma=gamma)

        # 分叉探测器
        eps_r = float(getattr(cfg, "EPS_R", 1e-3))
        eps_sigma = float(getattr(cfg, "EPS_SIGMA", 1e-3))
        eps_tau = float(getattr(cfg, "EPS_TAU", 1e-3))
        window = int(getattr(cfg, "DEBOUNCE_WINDOW", 5))
        self.det = SaddleNodeDetector(eps_r=eps_r, eps_sigma=eps_sigma, eps_tau=eps_tau, window=window)

        # 训练控制
        self.print_every = int(getattr(cfg, "PRINT_INTERVAL", 1000))
        self.clip_max_norm = float(getattr(cfg, "CLIP_MAX_NORM", 1.0))

    def _make_s_eval(self) -> torch.Tensor:
        n_eval = int(getattr(self.cfg, "POINTS_PER_SEGMENT", 80))
        return torch.linspace(
            0.0, float(getattr(self.cfg, "S_MAX", Config.S_MAX)),
            steps=n_eval, device=self.device
        ).view(-1, 1)

    def train(self) -> Tuple[str, List[Dict[str, float]]]:
        cfg = self.cfg
        history: List[Dict[str, float]] = []
        iters = int(getattr(cfg, "EPOCHS", Config.STEPS))

        for it in range(iters):
            self.opt.zero_grad(set_to_none=True)

            # —— 自适应采样：warm-up 后按周期刷新分布并重采样 —— #
            if self.use_adaptive and it >= self.adapt_warmup_iters:
                # 周期性更新重要性分布
                if (it - self.adapt_warmup_iters) % self.adapt_update_every == 0:
                    self.adp_sampler.update(self.net, self.F)
                # 重采样训练网格（整网格或 mini-batch）
                self.s_train = self.adp_sampler.sample_grid(self.batch_points)
            # （均匀模式则保持初始 s_train 不变；如需每步均匀重采样可改为：
            # else: self.s_train = self.uni_sampler.sample_grid(self.batch_points)）

            # 计算损失（AMP）
            with autocast("cuda", enabled=self.scaler.is_enabled()):
                total, comps = compute_loss(
                    model=self.net,
                    s=self.s_train,
                    y0=self.y0,
                    physics_fn=self.F,
                )

            # 反传 + 梯度裁剪 + 更新
            if self.scaler.is_enabled():
                self.scaler.scale(total).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.clip_max_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.clip_max_norm)
                self.opt.step()

            self.scheduler.step()

            # 记录
            comps = dict(comps)
            comps["grad_norm"] = float(grad_norm)
            comps["lr"] = float(self.scheduler.get_last_lr()[0])
            # 方便观察采样范围
            comps["s_min"] = float(self.s_train.min().item())
            comps["s_max"] = float(self.s_train.max().item())
            history.append(comps)

            if it % self.print_every == 0 or it == iters - 1:
                print(
                    f"it={it:05d} | total={comps.get('total', 0):.3e} "
                    f"| phys={comps.get('phys', 0):.3e} | arc={comps.get('arc', 0):.3e} "
                    f"| ic={comps.get('ic', 0):.3e} | smooth={comps.get('smooth', 0):.3e} "
                    f"| dir={comps.get('dir', 0):.3e} | grad={comps['grad_norm']:.2e} | lr={comps['lr']:.1e} "
                    f"| s∈[{comps['s_min']:.2f},{comps['s_max']:.2f}]"
                )

        # 导出 CSV 供画图
        s_export = self._make_s_eval()
        out_dir = getattr(cfg, "OUT_DIR", "assets")
        os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
        out_csv = os.path.join(out_dir, "tables", "branch.csv")

        export_branch_csv(
            self.F, self.net, s_export, out_csv,
            nx=int(getattr(cfg, "NX", 1)),
            np_=int(getattr(cfg, "NP", 1)),
            detector=self.det
        )
        print(f"Saved CSV -> {out_csv}")
        return out_csv, history

    @torch.no_grad()
    def evaluate_path(self, s_eval: torch.Tensor):
        self.net.eval()
        x, p = self.net(s_eval.to(self.device))
        return x.squeeze(), p.squeeze()
