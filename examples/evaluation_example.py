"""
Evaluation example for PPO-CodeT5.
"""

import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo_codet5 import Config, ReinforcementLearningPromptOptimizer
from ppo_codet5.utils import setup_logging


def main():
    """Evaluation example."""
    print("🚀 Starting Reinforcement Learning–Driven Prompt Optimization Evaluation Example")
    
    # Create configuration
    config = Config()
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize trainer
    print("📦 Initializing trainer...")
    trainer = ReinforcementLearningPromptOptimizer(config)
    
    # Load checkpoint if available
    checkpoint_path = "./examples/outputs/final_model.pt"
    if os.path.exists(checkpoint_path):
        print(f"📥 Loading checkpoint from {checkpoint_path}")
        trainer.agent.load(checkpoint_path)
    else:
        print("⚠️ No checkpoint found. Using untrained model for evaluation.")
    
    # Evaluate
    print("🎯 Starting evaluation...")
    results = trainer.evaluate(num_episodes=20)
    
    # Print detailed results
    print("\n📊 Evaluation Results:")
    print(f"Average Reward: {results['avg_reward']:.3f}")
    print(f"Average Success: {results['avg_success']:.3f}")
    print(f"Success Rate: {results['success_rate']:.3f}")
    
    # Print episode-by-episode results
    print("\n📈 Episode Results:")
    for i, (reward, success) in enumerate(zip(results['episode_rewards'], results['episode_successes'])):
        print(f"Episode {i+1:2d}: Reward={reward:6.3f}, Success={success:2d}")
    
    # Save results
    results_path = "./examples/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Evaluation results saved to {results_path}")
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
