"""
PPO (Proximal Policy Optimization) trainer for Blackholio agent.

This is the main training class that orchestrates the entire training process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
import logging
import time
from pathlib import Path

from ..models import BlackholioModel, BlackholioModelConfig
from .rollout_buffer import RolloutBuffer, RolloutBufferConfig
from .parallel_envs import ParallelBlackholioEnv, ParallelEnvConfig
from .checkpoint_manager import CheckpointManager
from .metrics_logger import MetricsLogger
from .self_play_manager import SelfPlayManager, OpponentConfig
from .curriculum_manager import CurriculumManager, StageConfig

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # Environment
    n_envs: int = 8
    env_host: str = "localhost:3000"
    env_database: Optional[str] = None  # Use ConnectionConfig's default
    
    # Training
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Steps per env before update
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # Value function clip range
    normalize_advantage: bool = True
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: Dict[str, int] = None
    adaptive_curriculum: bool = True
    
    # Self-play
    use_self_play: bool = True
    self_play_config: Optional[Dict[str, Any]] = None
    opponent_pool_dir: str = "opponent_pool"
    
    # Logging and checkpointing
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_interval_minutes: float = 20.0
    console_log_interval: int = 10
    
    # Model
    model_config: Optional[BlackholioModelConfig] = None
    
    def __post_init__(self):
        # Default curriculum stages
        if self.curriculum_stages is None:
            self.curriculum_stages = {
                "food_collection": 200_000,
                "survival": 300_000,
                "basic_combat": 300_000,
                "advanced": 200_000,
            }
        
        # Create model config if not provided
        if self.model_config is None:
            self.model_config = BlackholioModelConfig(device=self.device)
        elif isinstance(self.model_config, dict):
            # Convert dict to BlackholioModelConfig
            if 'device' not in self.model_config:
                self.model_config['device'] = self.device
            self.model_config = BlackholioModelConfig(**self.model_config)


class PPOTrainer:
    """
    PPO trainer for Blackholio agent.
    
    Implements the Proximal Policy Optimization algorithm with:
    - Parallel environment execution
    - Generalized Advantage Estimation (GAE)
    - Clipped objective function
    - Value function clipping
    - Entropy bonus for exploration
    - Curriculum learning support
    """
    
    def __init__(self, config: Union[PPOConfig, Dict[str, Any]] = None):
        """
        Initialize PPO trainer.
        
        Args:
            config: Training configuration
        """
        if config is None:
            self.config = PPOConfig()
        elif isinstance(config, dict):
            self.config = PPOConfig(**config)
        else:
            self.config = config
        
        self.device = torch.device(self.config.device)
        logger.info(f"PPO Trainer initialized with device: {self.device}")
        
        # Create model
        self.model = BlackholioModel(self.config.model_config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            eps=1e-5
        )
        
        # Create parallel environments
        env_config = ParallelEnvConfig(
            n_envs=self.config.n_envs,
            host=self.config.env_host,
            database=self.config.env_database
        )
        self.envs = ParallelBlackholioEnv(env_config)
        
        # Create rollout buffer
        buffer_config = RolloutBufferConfig(
            buffer_size=self.config.n_steps,
            n_envs=self.config.n_envs,
            observation_dim=456,  # From ObservationSpace
            device=self.config.device
        )
        self.rollout_buffer = RolloutBuffer(buffer_config)
        
        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            save_interval_minutes=self.config.save_interval_minutes,
            keep_best=True,
            keep_recent=5,
            metrics_to_track=["mean_reward", "episode_reward"]
        )
        
        # Create metrics logger
        self.metrics_logger = MetricsLogger(
            log_dir=self.config.log_dir,
            console_log_interval=self.config.console_log_interval,
            use_tensorboard=True
        )
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.last_obs = None
        self.last_episode_starts = None
        
        # Initialize curriculum manager
        self.curriculum_manager = None
        if self.config.use_curriculum:
            self.curriculum_manager = CurriculumManager(
                adaptive=self.config.adaptive_curriculum
            )
            logger.info("Curriculum learning enabled with adaptive mode: " + 
                       str(self.config.adaptive_curriculum))
        
        # Initialize self-play manager
        self.self_play_manager = None
        if self.config.use_self_play:
            opponent_config = OpponentConfig(**self.config.self_play_config) if self.config.self_play_config else OpponentConfig()
            self.self_play_manager = SelfPlayManager(
                pool_dir=self.config.opponent_pool_dir,
                config=opponent_config
            )
            logger.info("Self-play enabled with opponent pool at: " + 
                       self.config.opponent_pool_dir)
        
        # Log hyperparameters
        self.metrics_logger.log_hyperparameters(self.config.__dict__)
        
    def train(self):
        """
        Main training loop.
        
        Runs the complete PPO training process for the specified
        number of timesteps.
        """
        logger.info("Starting PPO training...")
        logger.info(f"Total timesteps: {self.config.total_timesteps:,}")
        logger.info(f"Parallel environments: {self.config.n_envs}")
        
        try:
            # Start environments
            self.envs.start()
            
            # Reset environments
            self.last_obs, _ = self.envs.reset()
            self.last_episode_starts = np.ones(self.config.n_envs, dtype=bool)
            
            # Main training loop
            while self.global_step < self.config.total_timesteps:
                # Collect rollouts
                self._collect_rollouts()
                
                # Update policy
                update_metrics = self._update_policy()
                
                # Log metrics
                self._log_training_metrics(update_metrics)
                
                # Save checkpoint if needed
                self._save_checkpoint_if_needed()
                
                # Update curriculum if needed
                self._update_curriculum()
            
            logger.info("Training completed!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            self.envs.close()
            self.metrics_logger.close()
            logger.info("Training cleanup completed")
    
    def _collect_rollouts(self):
        """Collect experience by running policy in environments"""
        self.rollout_buffer.reset()
        
        with torch.no_grad():
            for _ in range(self.config.n_steps):
                # Convert observations to tensor
                obs_tensor = torch.FloatTensor(self.last_obs).to(self.device)
                
                # Get actions from policy
                actions, model_output = self.model.get_action(
                    obs_tensor, 
                    deterministic=False
                )
                
                # Convert actions to numpy
                movement = actions["movement"].cpu().numpy()
                split = actions["split"].cpu().numpy()
                
                # Create action array for environments
                env_actions = np.concatenate([movement, split.reshape(-1, 1)], axis=1)
                
                # Step environments
                new_obs, rewards, dones, truncated, infos = self.envs.step(env_actions)
                
                # Compute log prob for buffer
                log_prob, _ = self.model.get_log_prob(
                    obs_tensor,
                    {
                        "movement": actions["movement"],
                        "split": actions["split"]
                    }
                )
                
                # Store in buffer
                self.rollout_buffer.add(
                    obs=self.last_obs,
                    movement_action=movement,
                    split_action=split,
                    reward=rewards,
                    episode_start=self.last_episode_starts,
                    value=model_output.value.cpu().numpy(),
                    log_prob=log_prob.cpu().numpy()
                )
                
                # Update state
                self.last_obs = new_obs
                self.last_episode_starts = dones | truncated
                
                # Update global step
                self.global_step += self.config.n_envs
                
                # Log step metrics
                self._log_step_metrics(rewards, infos)
        
        # Compute returns and advantages
        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(self.last_obs).to(self.device)
            _, last_model_output = self.model.get_action(last_obs_tensor)
            last_values = last_model_output.value.cpu().numpy()
        
        self.rollout_buffer.compute_returns_and_advantage(
            last_values, 
            self.last_episode_starts
        )
    
    def _update_policy(self) -> Dict[str, float]:
        """Update policy using collected rollouts"""
        # Get all data from buffer
        rollout_data = self.rollout_buffer.get_samples()
        
        # Metrics tracking
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        approx_kl_divs = []
        
        for epoch in range(self.config.n_epochs):
            # Create random indices for mini-batches
            n_samples = len(rollout_data["observations"])
            indices = np.random.permutation(n_samples)
            
            for start_idx in range(0, n_samples, self.config.batch_size):
                batch_indices = indices[start_idx:start_idx + self.config.batch_size]
                
                # Get batch data
                batch_obs = rollout_data["observations"][batch_indices]
                batch_movement = rollout_data["movement_actions"][batch_indices]
                batch_split = rollout_data["split_actions"][batch_indices]
                batch_old_log_probs = rollout_data["log_probs"][batch_indices]
                batch_advantages = rollout_data["advantages"][batch_indices]
                batch_returns = rollout_data["returns"][batch_indices]
                batch_old_values = rollout_data["values"][batch_indices]
                
                # Normalize advantages
                if self.config.normalize_advantage:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                
                # Get current log probs and values
                actions = {
                    "movement": batch_movement,
                    "split": batch_split
                }
                log_prob, entropy = self.model.get_log_prob(batch_obs, actions)
                _, model_output = self.model.get_action(batch_obs)
                values = model_output.value
                
                # Calculate ratio
                ratio = torch.exp(log_prob - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.clip_range_vf is None:
                    # No clipping
                    value_loss = F.mse_loss(values, batch_returns)
                else:
                    # Clip value function
                    values_pred_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf
                    )
                    value_losses_1 = F.mse_loss(values, batch_returns)
                    value_losses_2 = F.mse_loss(values_pred_clipped, batch_returns)
                    value_loss = torch.max(value_losses_1, value_losses_2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.vf_coef * value_loss + 
                       self.config.ent_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                with torch.no_grad():
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.config.clip_range).float()).item()
                    clip_fractions.append(clip_fraction)
                    
                    approx_kl = torch.mean((ratio - 1) - torch.log(ratio)).item()
                    approx_kl_divs.append(approx_kl)
        
        # Return average metrics
        return {
            "policy_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "clip_fraction": np.mean(clip_fractions),
            "approx_kl": np.mean(approx_kl_divs),
            "learning_rate": self.config.learning_rate,
        }
    
    def _log_step_metrics(self, rewards: np.ndarray, infos: List[Dict[str, Any]]):
        """Log metrics for each environment step"""
        # Track episode completions
        for i, info in enumerate(infos):
            if "episode" in info:
                self.episode_count += 1
                episode_metrics = {
                    "reward": info["episode"]["r"],
                    "length": info["episode"]["l"]
                }
                self.metrics_logger.log_episode(self.episode_count, episode_metrics)
        
        # Log step metrics
        step_metrics = {
            "reward": np.mean(rewards),
            "mean_reward": np.mean([info.get("episode_reward", 0) for info in infos]),
        }
        
        # Add buffer statistics
        buffer_stats = self.rollout_buffer.get_statistics()
        step_metrics.update(buffer_stats)
        
        self.metrics_logger.log_step(self.global_step, step_metrics)
    
    def _log_training_metrics(self, update_metrics: Dict[str, float]):
        """Log training update metrics"""
        self.metrics_logger.log_training_update(self.global_step, update_metrics)
    
    def _save_checkpoint_if_needed(self):
        """Save checkpoint if needed"""
        # Get current metrics
        metrics = {
            "mean_reward": np.mean(list(self.metrics_logger.metrics.get("mean_reward", [0]))),
            "episode_reward": np.mean(list(self.metrics_logger.episode_metrics.get("reward", [0])[-100:])),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
        }
        
        # Prepare additional state
        additional_state = {
            "config": self.config
        }
        
        # Add curriculum state if enabled
        if self.curriculum_manager:
            curriculum_stats = self.curriculum_manager.get_statistics()
            additional_state["curriculum_stage"] = curriculum_stats["stage_index"]
            additional_state["curriculum_stats"] = curriculum_stats
        
        # Add self-play state if enabled
        if self.self_play_manager:
            self_play_stats = self.self_play_manager.get_statistics()
            additional_state["self_play_stats"] = self_play_stats
            
            # Save to opponent pool if needed
            if self.self_play_manager.should_save_to_pool(self.global_step):
                self.self_play_manager.save_current_model(self.model, self.global_step)
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save(
            step=self.global_step,
            episode=self.episode_count,
            model=self.model,
            optimizer=self.optimizer,
            metrics=metrics,
            additional_state=additional_state
        )
        
        if checkpoint_path:
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _update_curriculum(self):
        """Update curriculum learning stage if needed"""
        if not self.curriculum_manager:
            return
        
        # Prepare metrics for curriculum manager
        # Get metric lists and handle empty cases
        mean_reward_list = self.metrics_logger.metrics.get("mean_reward", [])
        survival_rate_list = self.metrics_logger.metrics.get("survival_rate", [])
        food_collected_list = self.metrics_logger.metrics.get("food_collected", [])
        kill_rate_list = self.metrics_logger.metrics.get("kill_rate", [])
        episode_length_list = self.metrics_logger.episode_metrics.get("length", [])
        
        curriculum_metrics = {
            "mean_reward": np.mean(mean_reward_list[-100:]) if mean_reward_list else 0.0,
            "survival_rate": np.mean(survival_rate_list[-100:]) if survival_rate_list else 0.0,
            "food_collected_per_episode": np.mean(food_collected_list[-100:]) if food_collected_list else 0.0,
            "kill_rate": np.mean(kill_rate_list[-100:]) if kill_rate_list else 0.0,
            "average_lifespan": np.mean(episode_length_list[-100:]) if episode_length_list else 0.0,
        }
        
        # Update curriculum performance
        self.curriculum_manager.update_performance(curriculum_metrics)
        
        # Get current stage info
        stage_info = self.curriculum_manager.get_current_stage()
        stats = self.curriculum_manager.get_statistics()
        
        # Log curriculum statistics
        self.metrics_logger.log_step(self.global_step, {
            "curriculum_stage_idx": stats["stage_index"],
            "curriculum_promotion_progress": stats["promotion_progress"]["overall"],
        })
        
        # Apply stage-specific reward multipliers if available
        reward_multipliers = self.curriculum_manager.get_reward_multipliers()
        if reward_multipliers and hasattr(self.envs, 'set_reward_multipliers'):
            self.envs.set_reward_multipliers(reward_multipliers)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training"""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        if checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.global_step = checkpoint.get("step", 0)
            self.episode_count = checkpoint.get("episode", 0)
            self.current_curriculum_stage = checkpoint.get("curriculum_stage", 0)
            
            logger.info(f"Loaded checkpoint from step {self.global_step}")
            return True
        
        return False
