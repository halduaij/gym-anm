"""Quick test of dataset generation with fewer episodes."""

from generate_final_offline_datasets import *

def quick_test():
    """Quick test with minimal episodes."""
    print("QUICK DATASET GENERATION TEST")
    print("="*50)
    
    env = IEEE33ProperEnvironment()
    
    controllers = [
        SimpleL0_Random(),
        SimpleL5_Optimal()
    ]
    
    for controller in controllers:
        print(f"\nTesting {controller.name}...")
        
        data = collect_episode_data(env, controller, num_episodes=2)
        
        print(f"  Transitions: {len(data['rewards'])}")
        print(f"  Avg return: {np.mean(data['episode_returns']):.4f}")
        print(f"  Reward range: [{np.min(data['rewards']):.4f}, {np.max(data['rewards']):.4f}]")
    
    print("\n✅ Quick test complete!")

if __name__ == "__main__":
    quick_test()