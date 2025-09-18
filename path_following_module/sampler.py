# sampler.py
import torch
from typing import Callable, Optional

class UniformSampler:
    def __init__(self, S_max: float, device):
        self.S_max, self.device = float(S_max), device

    @torch.no_grad()
    def sample_grid(self, N: int) -> torch.Tensor:
        """均匀网格（含端点）"""
        return torch.linspace(0.0, self.S_max, steps=int(N), device=self.device).view(-1, 1)

    @torch.no_grad()
    def sample(self, B: int) -> torch.Tensor:
        """均匀随机采样"""
        return (torch.rand(int(B), 1, device=self.device) * self.S_max)


class AdaptiveSampler:
    """
    重要性采样器：
      - 先在一个固定细网格上评估 score(s)（residual 或 sigma_min 的函数）
      - 把 score→概率分布（softmax/温度/稳健化）
      - 混合：pi = mix * uniform + (1-mix) * importance
      - 采样 N 个 s 组成新的训练网格（或随机 mini-batch）
    """
    def __init__(
        self,
        S_max: float,
        device,
        grid_size: int = 256,
        score_type: str = "res",   # "res" 或 "sigma"
        mix: float = 0.5,          # 与均匀分布的混合比例
        temperature: float = 0.5,  # softmax 温度，越小越尖锐
        eps: float = 1e-8
    ):
        self.S_max, self.device = float(S_max), device
        self.grid_size = int(grid_size)
        self.score_type = score_type
        self.mix = float(mix)
        self.temperature = float(temperature)
        self.eps = float(eps)

        # 评估用的细网格
        self.s_fine = torch.linspace(0.0, self.S_max, steps=self.grid_size, device=self.device).view(-1, 1)
        self.prob = torch.full((self.grid_size,), 1.0 / self.grid_size, device=self.device)  # 初始均匀

    @torch.no_grad()
    def _scores(self, model, F: Callable) -> torch.Tensor:
        """
        计算每个 s_fine 的 score；默认用 residual=||F||，
        想用 σ_min 时置 score_type="sigma"（会更贵）。
        """
        x, p = model(self.s_fine)
        if self.score_type == "res":
            r = F(x, p)                                     # [G, nx]
            s = torch.linalg.norm(r, dim=1)                 # [G]
            # 让分布更平滑、可微
            return s

        elif self.score_type == "sigma":
            # 轻量版 σ_min（对 1D/2D 还好；若觉得慢，仍建议用 residual）
            from physics import jac_x
            Jx = jac_x(F, x.detach().requires_grad_(True), p.detach().requires_grad_(True))  # [G,nx,nx]
            S = torch.linalg.svdvals(Jx)                         # [G, min(nx,nx)]
            sig_min = S[..., -1]
            # 想把“越小越难”转成“越大越重要”，用 1/(sig_min+eps)
            return 1.0 / (sig_min + self.eps)

        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

    @torch.no_grad()
    def update(self, model, F: Callable):
        """刷新重要性分布 self.prob"""
        s = self._scores(model, F)                    # [G]
        # 归一化到概率（带温度、避免全 0）
        logits = torch.log(s + self.eps) / max(self.temperature, self.eps)
        p_imp = torch.softmax(logits, dim=0)          # 重要性分布
        p_uni = torch.full_like(p_imp, 1.0 / self.grid_size)
        self.prob = self.mix * p_uni + (1.0 - self.mix) * p_imp
        # 数值稳健
        self.prob = (self.prob + self.eps)
        self.prob = self.prob / self.prob.sum()

    @torch.no_grad()
    def sample_grid(self, N: int) -> torch.Tensor:
        """
        根据当前分布从 s_fine 采样 N 个点作为“训练网格”（可替换原来的均匀 s_train）。
        """
        idx = torch.multinomial(self.prob, num_samples=int(N), replacement=True)
        return self.s_fine.index_select(0, idx).view(-1, 1)

    @torch.no_grad()
    def sample(self, B: int) -> torch.Tensor:
        """随机采样 B 个 s（做 mini-batch 训练时用）"""
        return self.sample_grid(B)
