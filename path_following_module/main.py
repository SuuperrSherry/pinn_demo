from config import Config
from comparator import StrategyComparator
from strategies import get_available_strategies

def main():
    """Main function to run the PINN path following model training and strategy comparison."""
    print("PINN Path Following Module")
    print("="*60)
    
    # print configuration details
    print(f"physics equation: (x²)/4 + p² - 1 = 0")
    print(f"starting point: ({Config.START_X}, {Config.START_P})")
    print(f"arc lenth: {Config.ARC_LENGTH}")
    print(f"epochs: {Config.EPOCHS}")
    print(f"initial direction: {Config.INIT_DIRECTION.numpy()}")
    print(f"strategies: {get_available_strategies()}")
    
    # build the strategy comparator
    comparator = StrategyComparator()
    
    # define strategies to compare
    strategies = [ 'l2', 'forward_consistency', 'cosine', 'smooth', 'global_forward' ]
    
    print(f"\ncomparison of: {strategies}")
    print("="*60)
    
    # compare strategies
    results = comparator.compare_strategies(strategies)
    
    print(f"\ncomparison finished!")
    print("result save as:")
    print("  - assets/strategy_comparison.png")
    print("  - assets/loss_comparison.png")
    print("  - assets/models/ (strategy_name).pt")
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = main()