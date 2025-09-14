"""
Command Line Interface for PPO-CodeT5.
"""

import argparse
import sys
import os
from typing import Optional

from .main import PPOCodeT5Trainer
from .config import Config
from .utils import setup_logging


def create_config_command(args):
    """Create configuration file."""
    config = Config()
    
    # Override with command line arguments
    if args.model_name:
        config.model.codet5_model = args.model_name
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.episodes:
        config.training.max_episodes = args.episodes
    if args.steps:
        config.training.max_steps_per_episode = args.steps
    
    config.save(args.output)
    print(f"Configuration saved to {args.output}")


def train_command(args):
    """Train command."""
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Setup logging
    setup_logging(
        level=config.log_level,
        log_file=args.log_file
    )
    
    # Initialize trainer
    trainer = PPOCodeT5Trainer(config)
    
    # Train
    results = trainer.train(args.output_dir)
    
    print(f"Training completed. Results saved to {args.output_dir}")
    return results


def evaluate_command(args):
    """Evaluate command."""
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Setup logging
    setup_logging(
        level=config.log_level,
        log_file=args.log_file
    )
    
    # Initialize trainer
    trainer = PPOCodeT5Trainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.agent.load(args.checkpoint)
    
    # Evaluate
    results = trainer.evaluate(args.episodes)
    
    # Save results
    import json
    eval_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed. Results saved to {eval_path}")
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning–Driven Prompt Optimization for LLM Code Generation: A Hybrid Lexical–Semantic Approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default configuration
  ppo-codet5 config --output config.json
  
  # Train with custom configuration
  ppo-codet5 train --config config.json --output-dir ./results
  
  # Evaluate trained model
  ppo-codet5 eval --checkpoint ./results/final_model.pt --episodes 50
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Create configuration file')
    config_parser.add_argument('--output', '-o', required=True, help='Output config file path')
    config_parser.add_argument('--model-name', help='CodeT5+ model name')
    config_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    config_parser.add_argument('--episodes', type=int, help='Number of episodes')
    config_parser.add_argument('--steps', type=int, help='Steps per episode')
    config_parser.set_defaults(func=create_config_command)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', '-c', help='Configuration file path')
    train_parser.add_argument('--output-dir', '-o', default='./outputs', help='Output directory')
    train_parser.add_argument('--log-file', help='Log file path')
    train_parser.set_defaults(func=train_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--config', '-c', help='Configuration file path')
    eval_parser.add_argument('--checkpoint', help='Model checkpoint path')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    eval_parser.add_argument('--output-dir', '-o', default='./outputs', help='Output directory')
    eval_parser.add_argument('--log-file', help='Log file path')
    eval_parser.set_defaults(func=evaluate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
