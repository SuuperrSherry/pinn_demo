import matplotlib.pyplot as plt
import numpy as np
import torch
from physics import physics_equation, get_theoretical_saddle_node


class Visualizer:
    def __init__(self, output_dir: str = "assets"):
        self.output_dir = output_dir

    def plot_comparison(self, results):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for strategy, data in results.items():
            s = np.array(data['s'])
            x = np.array(data['x'])
            p = np.array(data['p'])
            axes[0, 0].plot(s, x, label=f'x(s) - {strategy}', linewidth=2)
        axes[0, 0].set_xlabel('Arc Length s')
        axes[0, 0].set_ylabel('x')
        axes[0, 0].set_title('x(s) across strategies')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        for strategy, data in results.items():
            s = np.array(data['s'])
            p = np.array(data['p'])
            axes[0, 1].plot(s, p, label=f'p(s) - {strategy}', linewidth=2)
        axes[0, 1].set_xlabel('Arc Length s')
        axes[0, 1].set_ylabel('p')
        axes[0, 1].set_title('p(s) across strategies')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # phase space with stability linestyle for learned path
        for strategy, data in results.items():
            x = np.array(data['x'])
            p = np.array(data['p'])
            if 'bifurcation' in data and 'stable_mask' in data['bifurcation']:
                stable_mask = np.array(data['bifurcation']['stable_mask'], dtype=bool)
            else:
                stable_mask = (x < 0)
            seg_start = 0
            for i in range(1, len(x) + 1):
                if i == len(x) or (stable_mask[i] != stable_mask[i-1]):
                    ls = '-' if stable_mask[i-1] else '--'
                    axes[1, 0].plot(x[seg_start:i], p[seg_start:i], linestyle=ls, linewidth=2, label=strategy if seg_start==0 else None)
                    seg_start = i

        # theory overlay: stable solid, unstable dashed
        th = get_theoretical_saddle_node()
        p_th = th['p']
        axes[1, 0].plot(th['x_minus'], p_th, 'k-', alpha=0.7, label='theory stable (x<0)')
        axes[1, 0].plot(th['x_plus'],  p_th, 'k--', alpha=0.7, label='theory unstable (x>0)')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('p')
        axes[1, 0].set_title('Phase Space (x, p)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')

        # residual panel
        for strategy, data in results.items():
            s = np.array(data['s'])
            residual = np.abs(physics_equation(torch.tensor(data['x']), torch.tensor(data['p'])).numpy())
            axes[1, 1].semilogy(s, residual, label=f'{strategy}', linewidth=2)
        axes[1, 1].set_xlabel('Arc Length s')
        axes[1, 1].set_ylabel('|Physics Residual|')
        axes[1, 1].set_title('Physics Residual |F(x,p)|')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_loss_comparison(self, results):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for strategy, data in results.items():
            epochs = range(len(data['loss_history']))
            total_losses = [h['total'] for h in data['loss_history']]
            physics_losses = [h['physics'] for h in data['loss_history']]
            arc_losses = [h['arc'] for h in data['loss_history']]
            ic_losses = [h['ic'] for h in data['loss_history']]

            axes[0, 0].semilogy(epochs, total_losses, label=strategy, linewidth=2)
            axes[0, 1].semilogy(epochs, physics_losses, label=strategy, linewidth=2)
            axes[1, 0].semilogy(epochs, arc_losses, label=strategy, linewidth=2)
            axes[1, 1].semilogy(epochs, ic_losses, label=strategy, linewidth=2)

        titles = ['total loss', 'physics loss', 'arclength loss', 'initial condition loss']
        for ax, title in zip(axes.flat, titles):
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_bifurcation_markers(self, results):
        # p–x diagram with fold candidates marked
        fig, ax = plt.subplots(figsize=(7, 6))
        for strategy, data in results.items():
            x = np.array(data['x'])
            p = np.array(data['p'])
            if 'bifurcation' in data and 'stable_mask' in data['bifurcation']:
                stable_mask = np.array(data['bifurcation']['stable_mask'], dtype=bool)
            else:
                stable_mask = (x < 0)
            seg_start = 0
            for i in range(1, len(x) + 1):
                if i == len(x) or (stable_mask[i] != stable_mask[i-1]):
                    ls = '-' if stable_mask[i-1] else '--'
                    ax.plot(p[seg_start:i], x[seg_start:i], linestyle=ls, linewidth=2, label=strategy if seg_start==0 else None)
                    seg_start = i
            if 'bifurcation' in data and data['bifurcation']['points']:
                pts = data['bifurcation']['points']
                bp = np.array([[d['p'], d['x']] for d in pts])
                ax.scatter(bp[:, 0], bp[:, 1], marker='o', s=120, edgecolors='k', linewidths=1.0, zorder=3)
                for j, (pp, xx) in enumerate(bp):
                    ax.annotate(str(j), (pp, xx), xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel('p (parameter)')
        ax.set_ylabel('x (state / amplitude)')
        ax.set_title('Bifurcation diagram with fold candidates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/bifurcation_markers.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_direction_diagnostics(self, results):
        # placeholder summary panels (we only have aggregate numbers now)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for strategy, data in results.items():
            s = np.array(data['s'])
            diag = data.get('direction_diag', {})
            mp = diag.get('mean_projection')
            mc = diag.get('mean_cosine')
            label1 = f"{strategy}: cos≈{mc:.2f}" if mc is not None else f"{strategy}: cos NA"
            label2 = f"{strategy}: proj≈{mp:.2f}" if mp is not None else f"{strategy}: proj NA"
            axes[0].plot([s[0], s[-1]], [mc or 0, mc or 0], label=label1)
            axes[1].plot([s[0], s[-1]], [mp or 0, mp or 0], label=label2)
        axes[0].set_title('mean cosine to global dir (summary)')
        axes[1].set_title('mean projection to global dir (summary)')
        for ax in axes:
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/direction_diag.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
