"""
Rollout buffer for storing experience data during PPO training.

This module handles efficient storage and retrieval of trajectories
collected from parallel environments.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RolloutBufferConfig:
    """Configuration for rollout buffer"""
    buffer_size: int = 2048  # Steps per environment before update
    n_envs: int = 8  # Number of parallel environments
    observation_dim: int = 456
    action_dim: int = 2  # Movement (2D)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"


class RolloutBuffer:
    """
    Buffer for storing rollout data from parallel environments.
    
    Stores observations, actions, rewards, values, and log probs
    for calculating PPO losses.
    """
    
    def __init__(self, config: RolloutBufferConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Buffer size accounting for parallel environments
        self.buffer_size = config.buffer_size
        self.n_envs = config.n_envs
        self.total_size = self.buffer_size * self.n_envs
        
        # Initialize buffers
        self.reset()
        
        logger.info(f"RolloutBuffer initialized: {self.total_size} total steps "
                   f"({self.buffer_size} steps × {self.n_envs} envs)")
    
    def reset(self):
        """Reset all buffers"""
        # Observations
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, self.config.observation_dim),
            dtype=np.float32
        )
        
        # Actions (continuous movement + discrete split)
        self.movement_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.config.action_dim),
            dtype=np.float32
        )
        self.split_actions = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32
        )
        
        # Rewards and episode info
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # Advantages and returns (computed later)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # Tracking
        self.pos = 0
        self.full = False
    
    def add(self,
            obs: np.ndarray,
            movement_action: np.ndarray,
            split_action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: np.ndarray,
            log_prob: np.ndarray):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Observations [n_envs, obs_dim]
            movement_action: Movement actions [n_envs, 2]
            split_action: Split actions [n_envs]
            reward: Rewards [n_envs]
            episode_start: Episode start flags [n_envs]
            value: Value estimates [n_envs]
            log_prob: Action log probabilities [n_envs]
        """
        if self.pos >= self.buffer_size:
            raise ValueError("Buffer is full. Call compute_returns_and_advantage() and reset().")
        
        self.observations[self.pos] = obs
        self.movement_actions[self.pos] = movement_action
        self.split_actions[self.pos] = split_action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantage(self, last_values: np.ndarray, dones: np.ndarray):
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_values: Value estimates for last observation [n_envs]
            dones: Done flags for last observation [n_envs]
        """
        # Convert to numpy if needed
        if isinstance(last_values, torch.Tensor):
            last_values = last_values.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
        
        # GAE computation
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            delta = (self.rewards[step] + 
                    self.config.gamma * next_values * next_non_terminal - 
                    self.values[step])
            
            last_gae_lam = (delta + 
                           self.config.gamma * self.config.gae_lambda * 
                           next_non_terminal * last_gae_lam)
            
            self.advantages[step] = last_gae_lam
        
        # Compute returns
        self.returns = self.advantages + self.values
    
    def get_samples(self) -> Dict[str, torch.Tensor]:
        """
        Get all samples from buffer for training.
        
        Returns:
            Dictionary of tensors
        """
        if not self.full and self.pos < self.buffer_size:
            # Only use filled portion of buffer
            buffer_size = self.pos
        else:
            buffer_size = self.buffer_size
        
        # Flatten buffer for sampling
        def flatten(arr):
            shape = arr.shape
            if len(shape) == 3:  # observations, movement_actions
                return arr[:buffer_size].reshape(-1, shape[-1])
            else:  # rewards, values, etc.
                return arr[:buffer_size].reshape(-1)
        
        data = {
            "observations": torch.FloatTensor(flatten(self.observations)).to(self.device),
            "movement_actions": torch.FloatTensor(flatten(self.movement_actions)).to(self.device),
            "split_actions": torch.FloatTensor(flatten(self.split_actions)).to(self.device),
            "values": torch.FloatTensor(flatten(self.values)).to(self.device),
            "log_probs": torch.FloatTensor(flatten(self.log_probs)).to(self.device),
            "advantages": torch.FloatTensor(flatten(self.advantages)).to(self.device),
            "returns": torch.FloatTensor(flatten(self.returns)).to(self.device),
        }
        
        return data
    
    def get_batches(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get samples in batches for training.
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Dictionary of tensors for each batch
        """
        # Get all data first
        data = self.get_samples()
        n_samples = len(data["observations"])
        indices = np.arange(n_samples)
        
        # Yield random batches
        for _ in range(n_samples // batch_size):
            batch_indices = np.random.choice(indices, batch_size, replace=False)
            batch_data = {}
            
            for key, tensor in data.items():
                batch_data[key] = tensor[batch_indices]
            
            yield batch_data
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics for logging"""
        if self.pos == 0:
            return {
                "buffer_size": 0,
                "mean_reward": 0.0,
                "mean_value": 0.0,
                "mean_advantage": 0.0,
                "mean_return": 0.0,
            }
        
        # Only consider filled portion
        rewards = self.rewards[:self.pos]
        values = self.values[:self.pos]
        advantages = self.advantages[:self.pos] if self.full else np.zeros_like(values)
        returns = self.returns[:self.pos] if self.full else np.zeros_like(values)
        
        return {
            "buffer_size": self.pos * self.n_envs,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_value": float(np.mean(values)),
            "mean_advantage": float(np.mean(advantages)),
            "std_advantage": float(np.std(advantages)),
            "mean_return": float(np.mean(returns)),
        }
