"""
Basic training example for PPO-CodeT5.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo_codet5 import Config, ReinforcementLearningPromptOptimizer
from ppo_codet5.utils import setup_logging


def main():
    """Basic training example."""
    print("🚀 Starting Reinforcement Learning–Driven Prompt Optimization Basic Training Example")
    
    # Create configuration
    config = Config()
    
    # Modify configuration for quick training
    config.training.max_episodes = 50
    config.training.max_steps_per_episode = 5
    config.environment.max_samples = 100  # Use only 100 samples for quick training
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize trainer
    print("📦 Initializing trainer...")
    trainer = ReinforcementLearningPromptOptimizer(config)
    
    # Train
    print("🎯 Starting training...")
    results = trainer.train("./examples/outputs")
    
    # Print results
    print("\n📊 Training Results:")
    print(f"Final average reward: {results['episode_rewards'][-10:].mean():.3f}")
    print(f"Final success rate: {results['episode_successes'][-10:].mean():.3f}")
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
