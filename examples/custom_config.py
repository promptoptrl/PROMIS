"""
Custom configuration example for PPO-CodeT5.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo_codet5 import Config, ReinforcementLearningPromptOptimizer
from ppo_codet5.utils import setup_logging


def main():
    """Custom configuration example."""
    print("🚀 Starting Reinforcement Learning–Driven Prompt Optimization Custom Configuration Example")
    
    # Create custom configuration
    config = Config()
    
    # Customize model configuration
    config.model.codet5_model = "Salesforce/codet5p-770m-py"
    config.model.temperature = 0.8
    config.model.top_p = 0.95
    
    # Customize training configuration
    config.training.learning_rate = 1e-4
    config.training.gamma = 0.95
    config.training.clip_epsilon = 0.3
    config.training.max_episodes = 200
    config.training.max_steps_per_episode = 8
    
    # Customize environment configuration
    config.environment.dataset_name = "mbpp"
    config.environment.max_samples = 200
    config.environment.reward_scale = 2.0
    config.environment.success_reward = 2.0
    config.environment.penalty_scale = -0.2
    
    # Customize cache configuration
    config.cache.output_dir = "./custom_outputs"
    config.cache.results_dir = "./custom_results"
    config.cache.checkpoints_dir = "./custom_checkpoints"
    
    # Setup logging
    setup_logging(level="DEBUG", log_file="./custom_training.log")
    
    # Save configuration
    config.save("./examples/custom_config.json")
    print("💾 Configuration saved to custom_config.json")
    
    # Initialize trainer
    print("📦 Initializing trainer with custom configuration...")
    trainer = ReinforcementLearningPromptOptimizer(config)
    
    # Train
    print("🎯 Starting training with custom configuration...")
    results = trainer.train(config.cache.output_dir)
    
    # Print results
    print("\n📊 Custom Training Results:")
    print(f"Final average reward: {results['episode_rewards'][-10:].mean():.3f}")
    print(f"Final success rate: {results['episode_successes'][-10:].mean():.3f}")
    
    print("\n✅ Custom training completed successfully!")


if __name__ == "__main__":
    main()
