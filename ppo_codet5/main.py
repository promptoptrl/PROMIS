"""
Main entry point for Reinforcement Learning–Driven Prompt Optimization training and evaluation.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any

from datasets import load_dataset
from .config import Config
from .models import PromptRewriter, EpicGA, CodeT5Model
from .agents import PPOAgent
from .environments import RLPromptEnv
from .utils import setup_logging, get_logger, ensure_dir


class ReinforcementLearningPromptOptimizer:
    """
    Main trainer class for Reinforcement Learning–Driven Prompt Optimization.
    """
    
    def __init__(self, config: Config):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize models
        self._initialize_models()
        
        # Initialize environment and agent
        self._initialize_environment()
        self._initialize_agent()
    
    def _initialize_models(self):
        """Initialize all models."""
        self.logger.info("Initializing models...")
        
        # Initialize models
        self.prompt_rewriter = PromptRewriter(self.config.model)
        self.epic_ga = EpicGA(self.config.model)
        self.codet5_model = CodeT5Model(self.config.model)
        
        self.logger.info("Models initialized successfully")
    
    def _initialize_environment(self):
        """Initialize RL environment."""
        self.logger.info("Loading dataset...")
        
        # Load dataset
        dataset = load_dataset(self.config.environment.dataset_name)
        train_data = dataset[self.config.environment.dataset_split]
        
        if self.config.environment.max_samples:
            train_data = train_data.select(range(self.config.environment.max_samples))
        
        self.logger.info(f"Loaded {len(train_data)} samples from {self.config.environment.dataset_name}")
        
        # Initialize environment
        self.env = RLPromptEnv(
            dataset=train_data,
            prompt_rewriter=self.prompt_rewriter,
            epic_ga=self.epic_ga,
            codet5_model=self.codet5_model,
            config=self.config.environment
        )
    
    def _initialize_agent(self):
        """Initialize PPO agent."""
        self.logger.info("Initializing PPO agent...")
        
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        
        self.agent = PPOAgent(obs_dim, act_dim, self.config.training)
        
        self.logger.info(f"PPO agent initialized: obs_dim={obs_dim}, act_dim={act_dim}")
    
    def train(self, output_dir: str = "./outputs") -> Dict[str, Any]:
        """
        Train the PPO agent.
        
        Args:
            output_dir: Output directory for results
            
        Returns:
            Training results
        """
        self.logger.info("Starting training...")
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Training metrics
        episode_rewards = []
        episode_successes = []
        training_metrics = []
        
        # Training loop
        for episode in range(self.config.training.max_episodes):
            episode_reward, episode_success = self._train_episode()
            
            episode_rewards.append(episode_reward)
            episode_successes.append(episode_success)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_success = np.mean(episode_successes[-10:])
                self.logger.info(
                    f"Episode {episode}: Avg Reward={avg_reward:.3f}, "
                    f"Avg Success={avg_success:.3f}"
                )
            
            # Update agent
            if len(self.agent.states) > 0:
                metrics = self.agent.update()
                training_metrics.append(metrics)
                self.agent.reset_buffer()
            
            # Save checkpoint
            if episode % self.config.training.save_frequency == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_episode_{episode}.pt")
                self.agent.save(checkpoint_path)
                self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(output_dir, "final_model.pt")
        self.agent.save(final_model_path)
        
        # Save training results
        results = {
            "episode_rewards": episode_rewards,
            "episode_successes": episode_successes,
            "training_metrics": training_metrics,
            "config": self.config.to_dict()
        }
        
        results_path = os.path.join(output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Training completed. Results saved to {output_dir}")
        
        return results
    
    def _train_episode(self) -> tuple[float, int]:
        """
        Train one episode.
        
        Returns:
            Tuple of (episode_reward, episode_success)
        """
        observation, _ = self.env.reset()
        episode_reward = 0
        episode_success = 0
        
        for step in range(self.config.training.max_steps_per_episode):
            # Select action
            action, log_prob, value = self.agent.select_action(observation)
            
            # Take step
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Store experience
            self.agent.store(
                observation, action, reward, terminated, log_prob, value
            )
            
            episode_reward += reward
            episode_success += int(info.get("is_passing", False))
            
            observation = next_observation
            
            if terminated or truncated:
                break
        
        return episode_reward, episode_success
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation results
        """
        self.logger.info(f"Starting evaluation for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_successes = []
        detailed_results = []
        
        for episode in range(num_episodes):
            observation, _ = self.env.reset()
            episode_reward = 0
            episode_success = 0
            episode_results = []
            
            for step in range(self.config.training.max_steps_per_episode):
                # Select action (no gradient)
                action, _, _ = self.agent.select_action(observation)
                
                # Take step
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_success += int(info.get("is_passing", False))
                episode_results.append(info)
                
                observation = next_observation
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_successes.append(episode_success)
            detailed_results.append(episode_results)
            
            self.logger.info(
                f"Evaluation Episode {episode + 1}: "
                f"Reward={episode_reward:.3f}, Success={episode_success}"
            )
        
        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        avg_success = np.mean(episode_successes)
        success_rate = avg_success / self.config.training.max_steps_per_episode
        
        results = {
            "episode_rewards": episode_rewards,
            "episode_successes": episode_successes,
            "detailed_results": detailed_results,
            "avg_reward": avg_reward,
            "avg_success": avg_success,
            "success_rate": success_rate
        }
        
        self.logger.info(
            f"Evaluation completed: Avg Reward={avg_reward:.3f}, "
            f"Success Rate={success_rate:.3f}"
        )
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PPO-CodeT5 Training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Mode")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for evaluation")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Setup logging
    setup_logging(
        level=config.log_level,
        log_file=config.log_file
    )
    
    # Initialize trainer
    trainer = ReinforcementLearningPromptOptimizer(config)
    
    if args.mode == "train":
        # Training mode
        results = trainer.train(args.output_dir)
        print(f"Training completed. Results saved to {args.output_dir}")
    
    elif args.mode == "eval":
        # Evaluation mode
        if args.checkpoint:
            trainer.agent.load(args.checkpoint)
        
        results = trainer.evaluate(args.episodes)
        
        # Save evaluation results
        eval_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation completed. Results saved to {eval_path}")


if __name__ == "__main__":
    main()
