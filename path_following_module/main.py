from config import set_random_seeds, Config
from continuation import path_following
from plots import plot_results, print_statistics

def physics_equation(x, p):
    return x - p**2

def physics_fn_inv(p):
    return p**2

def main():
    set_random_seeds()
    print(f"Config: {Config.HIDDEN_LAYERS} layers, {Config.HIDDEN_NEURONS} neurons")
    start_x, start_p = 0.25, 0.5
    path_x, path_p, path_s = path_following(
        (start_x, start_p), physics_fn=physics_equation,
        total_length=Config.TOTAL_LENGTH, base_step=Config.BASE_STEP_SIZE,
        max_segments=Config.MAX_SEGMENTS
    )
    errors = plot_results(path_x, path_p, path_s, (start_x, start_p), physics_fn_inv)

    print_statistics(path_x, path_p, path_s, errors)

if __name__ == "__main__":
    main()