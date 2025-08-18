from config import Config
from physics import physics_equation
from trainer import PINNTrainer
from comparator import StrategyComparator


def main():
    print("Running on device:", Config.DEVICE)
    strategies = ['l2', 'cosine', 'smooth', 'global_forward', 'forward_consistency']
#, 'smooth', 'global_forward', 'forward_consistency'
    comparator = StrategyComparator(trainer=PINNTrainer())
    results = comparator.compare_strategies(
        strategies=strategies,
        physics_fn=physics_equation,
        start_x=Config.START_X,
        start_p=Config.START_P,
        arc_length=Config.ARC_LENGTH,
        epochs=Config.EPOCHS,
    )
    print("Done. Outputs in assets/")
    return results


if __name__ == "__main__":
    main()
