import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List
from physics import physics_equation, get_theoretical_ellipse

class Visualizer:
    def __init__(self, output_dir="assets"):
        self.output_dir = output_dir
    
    def plot_comparison(self, results):
        """plot comparison of different strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # x(s) comparison
        for strategy, data in results.items():
            axes[0, 0].plot(data['s'], data['x'], label=f'x(s) - {strategy}', linewidth=2)
        axes[0, 0].set_xlabel('Arc Length s')
        axes[0, 0].set_ylabel('x')
        axes[0, 0].set_title('x(s) different strategies comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # p(s) comparison
        for strategy, data in results.items():
            axes[0, 1].plot(data['s'], data['p'], label=f'p(s) - {strategy}', linewidth=2)
        axes[0, 1].set_xlabel('Arc Length s')
        axes[0, 1].set_ylabel('p')
        axes[0, 1].set_title('p(s) different strategies comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Phase space comparison
        for strategy, data in results.items():
            axes[1, 0].plot(data['x'], data['p'], label=strategy, linewidth=2)
        
        # Add theoretical ellipse
        x_theory, p_theory = get_theoretical_ellipse()
        axes[1, 0].plot(x_theory, p_theory, 'k--', alpha=0.5, label='Theoretical Path')
        axes[1, 0].plot(-x_theory, p_theory, 'k--', alpha=0.5)
        
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('p')
        axes[1, 0].set_title('Phase Space (x, p)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # Physics residual comparison
        for strategy, data in results.items():
            residual = np.abs(physics_equation(
                torch.tensor(data['x']), 
                torch.tensor(data['p'])
            ).numpy())
            axes[1, 1].semilogy(data['s'], residual, label=f'{strategy}', linewidth=2)
        
        axes[1, 1].set_xlabel('Arc Length s')
        axes[1, 1].set_ylabel('|Physics Residual|')
        axes[1, 1].set_title('Physics Residual |F(x,p)|')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_loss_comparison(self, results):
        """Plot loss comparison for different strategies."""
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
        
        titles = ['total loss', 'phys loss', 'arclenth loss', 'initial condition loss']
        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()