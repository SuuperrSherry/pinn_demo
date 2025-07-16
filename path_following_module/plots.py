import numpy as np
import matplotlib.pyplot as plt

def plot_results(path_x, path_p, path_s, start_point, physics_fn_inv):
    p_true = np.linspace(-1.5, 1.5, 300)
    x_true = [physics_fn_inv(p) for p in p_true]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    if len(path_x) < 30:
        print(f" Warning: path_x only has {len(path_x)} points, zoom plot may look empty.")
    
    
    axes[0, 0].plot(x_true, p_true, 'r-', linewidth=2, label='True Path')
    axes[0, 0].plot(path_x, path_p, 'bo-', markersize=4, linewidth=1, label='PINN Path')
    axes[0, 0].plot(start_point[0], start_point[1], 'go', markersize=8, label='Start')
    axes[0, 0].legend(); axes[0, 0].grid(); axes[0, 0].axis('equal')
    axes[0, 0].set_title('Path Following Results')
    
    
    errors = [abs(x - physics_fn_inv(p)) for x, p in zip(path_x, path_p)]
    axes[0, 1].plot(path_s, errors)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(); axes[0, 1].set_title('Physics Constraint Error')
    
    
    axes[1, 0].plot(path_s, path_x, 'b-', label='x(s)')
    axes[1, 0].plot(path_s, path_p, 'r-', label='p(s)')
    axes[1, 0].legend(); axes[1, 0].grid()
    axes[1, 0].set_xlabel('Arc length s'); axes[1, 0].set_title('Path Components vs Arc Length')
    
    n_zoom = min(30, len(path_x))
    axes[1, 1].plot(x_true, p_true, 'r-', linewidth=2, label='True Path')
    axes[1, 1].plot(path_x[:n_zoom], path_p[:n_zoom], 'bo-', markersize=6, 
                    label=f'PINN Path (first {n_zoom} points)')
    axes[1, 1].plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    axes[1, 1].legend(); axes[1, 1].grid()
    axes[1, 1].set_title('Zoom: Starting Region')
    
    plt.tight_layout(); plt.show()
    return errors


def print_statistics(path_x, path_p, path_s, errors):
    print(f"\nTotal points: {len(path_x)}, Arc length covered: {path_s[-1]:.3f}")
    print(f"Mean err: {np.mean(errors):.6f}, Max err: {np.max(errors):.6f}")