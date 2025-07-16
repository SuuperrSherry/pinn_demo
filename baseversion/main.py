from continuation import para_stab
from config import para_stab
from plots import plot_results, print_statistics

def physics_equation(x, p):
    return x - p**2


def main():
    set_random_seeds()
    print("Starting improved PINN path following...")
    print(f"Configuration: {Config.HIDDEN_LAYERS} layers, {Config.HIDDEN_NEURONS} neurons")
    
    # YOUR PHYSICS FUNCTION
    def physics_equation(x, p):
        return x - p**2

    start_x, start_p = 0.25, 0.5
    path_x, path_p, path_s = path_following(
        (start_x, start_p), 
        physics_fn=physics_equation,   # 👈 必须加上这个参数
        total_length=Config.TOTAL_LENGTH, 
        base_step=Config.BASE_STEP_SIZE,
        max_segments=Config.MAX_SEGMENTS
    )
    
    errors = plot_results(path_x, path_p, path_s, (start_x, start_p))
    print_statistics(path_x, path_p, path_s, errors)
