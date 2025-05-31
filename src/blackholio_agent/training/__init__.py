"""
Training module for Blackholio RL agent.

This module provides PPO training implementation with support for:
- Parallel environment execution
- Experience collection and replay
- Model checkpointing
- Training metrics and monitoring
"""

from .ppo_trainer import PPOTrainer, PPOConfig
from .rollout_buffer import RolloutBuffer, RolloutBufferConfig
from .parallel_envs import ParallelBlackholioEnv, ParallelEnvConfig
from .checkpoint_manager import CheckpointManager
from .metrics_logger import MetricsLogger
from .self_play_manager import SelfPlayManager, OpponentConfig
from .curriculum_manager import CurriculumManager, StageConfig

__all__ = [
    "PPOTrainer",
    "PPOConfig",
    "RolloutBuffer",
    "RolloutBufferConfig",
    "ParallelBlackholioEnv",
    "ParallelEnvConfig",
    "CheckpointManager",
    "MetricsLogger",
    "SelfPlayManager",
    "OpponentConfig",
    "CurriculumManager",
    "StageConfig",
]
