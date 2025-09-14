"""
PPO (Proximal Policy Optimization) agent implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from ..config import TrainingConfig


class PPOPolicy(nn.Module):
    """
    PPO Policy network with actor-critic architecture.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        """
        Initialize PPO policy network.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, act_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Tuple of (action_logits, value)
        """
        shared_features = self.shared(obs)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value


class PPOAgent:
    """
    PPO Agent for reinforcement learning.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, config: TrainingConfig):
        """
        Initialize PPO agent.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            config: Training configuration
        """
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Initialize policy network
        self.model = PPOPolicy(obs_dim, act_dim)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Initialize buffer
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value = self.model(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        done: bool, 
        log_prob: float, 
        value: float
    ):
        """
        Store experience in buffer.
        
        Args:
            state: State
            action: Action taken
            reward: Reward received
            done: Whether episode is done
            log_prob: Log probability of action
            value: State value
        """
        self.states.append(torch.as_tensor(state, dtype=torch.float32))
        self.actions.append(torch.tensor(action))
        self.rewards.append(torch.tensor(reward))
        self.dones.append(torch.tensor(done))
        self.log_probs.append(torch.tensor(log_prob))
        self.values.append(torch.tensor(value))
    
    def compute_returns(self) -> List[float]:
        """
        Compute discounted returns.
        
        Returns:
            List of discounted returns
        """
        returns = []
        G = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            G = reward + self.config.gamma * G * (1 - done.item())
            returns.insert(0, G)
        
        return returns
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        old_values = torch.stack(self.values)
        
        # Compute returns and advantages
        returns = self.compute_returns()
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        
        for _ in range(self.config.update_epochs):
            # Forward pass
            action_logits, values = self.model(states)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            
            # Compute losses
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy loss (PPO)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 
                1 - self.config.clip_epsilon, 
                1 + self.config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            loss = (
                policy_loss + 
                self.config.value_coef * value_loss - 
                self.config.entropy_coef * entropy
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            actor_loss += policy_loss.item()
            critic_loss += value_loss.item()
            entropy_loss += entropy.item()
        
        # Average losses
        num_epochs = self.config.update_epochs
        metrics = {
            "total_loss": total_loss / num_epochs,
            "actor_loss": actor_loss / num_epochs,
            "critic_loss": critic_loss / num_epochs,
            "entropy_loss": entropy_loss / num_epochs,
            "buffer_size": len(self.states)
        }
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save model state.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load model state.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information.
        
        Returns:
            Agent information dictionary
        """
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "clip_epsilon": self.config.clip_epsilon,
            "update_epochs": self.config.update_epochs,
            "buffer_size": len(self.states)
        }
