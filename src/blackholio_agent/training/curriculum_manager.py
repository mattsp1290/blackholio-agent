"""
Curriculum learning manager for progressive training.

This module implements adaptive curriculum learning that adjusts
task difficulty based on agent performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Stages of curriculum learning"""
    FOOD_COLLECTION = "food_collection"
    SURVIVAL = "survival"
    BASIC_COMBAT = "basic_combat"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class StageConfig:
    """Configuration for a curriculum stage"""
    name: str
    min_timesteps: int  # Minimum timesteps in this stage
    promotion_criteria: Dict[str, float]  # Metrics required to advance
    demotion_criteria: Optional[Dict[str, float]] = None  # Metrics that trigger demotion
    reward_multipliers: Optional[Dict[str, float]] = None  # Stage-specific reward scaling
    environment_modifiers: Optional[Dict[str, Any]] = None  # Environment difficulty settings


class PerformanceTracker:
    """
    Tracks agent performance metrics for curriculum decisions.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize performance tracker.
        
        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.stage_start_time: float = time.time()
        self.stage_timesteps: int = 0
        
    def update(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.window_size)
            self.metrics[key].append(value)
        self.stage_timesteps += 1
    
    def get_average(self, metric: str, window: Optional[int] = None) -> float:
        """Get average value of a metric"""
        if metric not in self.metrics or not self.metrics[metric]:
            return 0.0
        
        values = list(self.metrics[metric])
        if window is not None:
            values = values[-window:]
        
        return np.mean(values) if values else 0.0
    
    def get_trend(self, metric: str, window: int = 100) -> float:
        """Get trend of a metric (positive = improving)"""
        if metric not in self.metrics or len(self.metrics[metric]) < window:
            return 0.0
        
        values = list(self.metrics[metric])[-window:]
        x = np.arange(len(values))
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, values, rcond=None)[0]
        
        return m
    
    def reset_stage_stats(self):
        """Reset statistics for new stage"""
        self.stage_start_time = time.time()
        self.stage_timesteps = 0
    
    def get_stage_duration(self) -> float:
        """Get time spent in current stage"""
        return time.time() - self.stage_start_time


class CurriculumManager:
    """
    Manages curriculum learning progression.
    
    Adaptively adjusts training difficulty based on agent performance,
    promoting or demoting between stages as appropriate.
    """
    
    def __init__(self, 
                 stages: Optional[List[StageConfig]] = None,
                 adaptive: bool = True,
                 min_stage_duration: float = 300.0):  # 5 minutes minimum per stage
        """
        Initialize curriculum manager.
        
        Args:
            stages: List of curriculum stages
            adaptive: Whether to use adaptive progression
            min_stage_duration: Minimum time in seconds per stage
        """
        self.stages = stages or self._get_default_stages()
        self.adaptive = adaptive
        self.min_stage_duration = min_stage_duration
        
        self.current_stage_idx = 0
        self.performance_tracker = PerformanceTracker()
        
        # Stage history
        self.stage_history: List[Dict[str, Any]] = []
        self.promotion_count = 0
        self.demotion_count = 0
        
        logger.info(f"CurriculumManager initialized with {len(self.stages)} stages")
    
    def _get_default_stages(self) -> List[StageConfig]:
        """Get default curriculum stages"""
        return [
            StageConfig(
                name=CurriculumStage.FOOD_COLLECTION.value,
                min_timesteps=50000,
                promotion_criteria={
                    "mean_reward": 5.0,
                    "survival_rate": 0.8,
                    "food_collected_per_episode": 20.0
                },
                reward_multipliers={
                    "food_reward": 2.0,
                    "survival_reward": 1.0,
                    "mass_reward": 1.0
                },
                environment_modifiers={
                    "food_density": 1.5,
                    "enemy_density": 0.5,
                    "enemy_aggression": 0.3
                }
            ),
            StageConfig(
                name=CurriculumStage.SURVIVAL.value,
                min_timesteps=100000,
                promotion_criteria={
                    "mean_reward": 10.0,
                    "survival_rate": 0.7,
                    "average_lifespan": 120.0
                },
                demotion_criteria={
                    "mean_reward": 3.0,
                    "survival_rate": 0.4
                },
                reward_multipliers={
                    "food_reward": 1.0,
                    "survival_reward": 2.0,
                    "mass_reward": 1.5
                },
                environment_modifiers={
                    "food_density": 1.0,
                    "enemy_density": 1.0,
                    "enemy_aggression": 0.5
                }
            ),
            StageConfig(
                name=CurriculumStage.BASIC_COMBAT.value,
                min_timesteps=150000,
                promotion_criteria={
                    "mean_reward": 20.0,
                    "kill_rate": 0.3,
                    "mass_rank": 0.5  # Top 50% by mass
                },
                demotion_criteria={
                    "mean_reward": 8.0,
                    "survival_rate": 0.5
                },
                reward_multipliers={
                    "food_reward": 0.5,
                    "kill_reward": 3.0,
                    "mass_reward": 2.0
                },
                environment_modifiers={
                    "food_density": 0.8,
                    "enemy_density": 1.2,
                    "enemy_aggression": 0.7
                }
            ),
            StageConfig(
                name=CurriculumStage.ADVANCED.value,
                min_timesteps=200000,
                promotion_criteria={
                    "mean_reward": 35.0,
                    "kill_rate": 0.5,
                    "split_effectiveness": 0.6
                },
                demotion_criteria={
                    "mean_reward": 15.0,
                    "kill_rate": 0.2
                },
                reward_multipliers={
                    "kill_reward": 2.0,
                    "strategy_reward": 3.0,
                    "mass_reward": 2.5
                },
                environment_modifiers={
                    "enemy_skill_level": "intermediate",
                    "enable_viruses": True,
                    "enable_splits": True
                }
            ),
            StageConfig(
                name=CurriculumStage.EXPERT.value,
                min_timesteps=300000,
                promotion_criteria={
                    "mean_reward": 50.0,
                    "win_rate": 0.7,
                    "strategic_play_score": 0.8
                },
                reward_multipliers={
                    "win_reward": 5.0,
                    "strategy_reward": 4.0,
                    "efficiency_reward": 3.0
                },
                environment_modifiers={
                    "enemy_skill_level": "expert",
                    "tournament_mode": True,
                    "full_game_features": True
                }
            )
        ]
    
    def get_current_stage(self) -> StageConfig:
        """Get current curriculum stage"""
        return self.stages[self.current_stage_idx]
    
    def update_performance(self, metrics: Dict[str, float]):
        """
        Update performance metrics and check for stage transitions.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self.performance_tracker.update(metrics)
        
        if self.adaptive:
            self._check_stage_transition()
    
    def _check_stage_transition(self):
        """Check if agent should be promoted or demoted"""
        current_stage = self.get_current_stage()
        stage_duration = self.performance_tracker.get_stage_duration()
        
        # Check minimum duration
        if stage_duration < self.min_stage_duration:
            return
        
        # Check minimum timesteps
        if self.performance_tracker.stage_timesteps < current_stage.min_timesteps:
            return
        
        # Check promotion criteria
        if self._check_promotion_criteria():
            self._promote_stage()
        # Check demotion criteria
        elif self._check_demotion_criteria():
            self._demote_stage()
    
    def _check_promotion_criteria(self) -> bool:
        """Check if promotion criteria are met"""
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Already at highest stage
        
        current_stage = self.get_current_stage()
        
        for metric, threshold in current_stage.promotion_criteria.items():
            avg_value = self.performance_tracker.get_average(metric, window=1000)
            if avg_value < threshold:
                return False
        
        # Also check if performance is stable (not just lucky streak)
        for metric in current_stage.promotion_criteria.keys():
            trend = self.performance_tracker.get_trend(metric, window=500)
            if trend < 0:  # Declining performance
                return False
        
        return True
    
    def _check_demotion_criteria(self) -> bool:
        """Check if demotion criteria are met"""
        if self.current_stage_idx == 0:
            return False  # Can't demote from first stage
        
        current_stage = self.get_current_stage()
        if current_stage.demotion_criteria is None:
            return False
        
        for metric, threshold in current_stage.demotion_criteria.items():
            avg_value = self.performance_tracker.get_average(metric, window=500)
            if avg_value < threshold:
                return True
        
        return False
    
    def _promote_stage(self):
        """Promote to next stage"""
        old_stage = self.get_current_stage()
        self.current_stage_idx += 1
        new_stage = self.get_current_stage()
        
        self.promotion_count += 1
        self.performance_tracker.reset_stage_stats()
        
        # Record transition
        self.stage_history.append({
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
            "transition_type": "promotion",
            "timestamp": time.time(),
            "metrics": {
                k: self.performance_tracker.get_average(k) 
                for k in old_stage.promotion_criteria.keys()
            }
        })
        
        logger.info(f"Promoted from {old_stage.name} to {new_stage.name}")
    
    def _demote_stage(self):
        """Demote to previous stage"""
        old_stage = self.get_current_stage()
        self.current_stage_idx -= 1
        new_stage = self.get_current_stage()
        
        self.demotion_count += 1
        self.performance_tracker.reset_stage_stats()
        
        # Record transition
        self.stage_history.append({
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
            "transition_type": "demotion",
            "timestamp": time.time(),
            "metrics": {
                k: self.performance_tracker.get_average(k) 
                for k in old_stage.demotion_criteria.keys()
                if old_stage.demotion_criteria
            }
        })
        
        logger.warning(f"Demoted from {old_stage.name} to {new_stage.name}")
    
    def force_stage(self, stage_name: str):
        """Force transition to specific stage"""
        for i, stage in enumerate(self.stages):
            if stage.name == stage_name:
                self.current_stage_idx = i
                self.performance_tracker.reset_stage_stats()
                logger.info(f"Forced transition to stage: {stage_name}")
                return
        
        raise ValueError(f"Unknown stage: {stage_name}")
    
    def get_reward_multipliers(self) -> Dict[str, float]:
        """Get current stage's reward multipliers"""
        stage = self.get_current_stage()
        return stage.reward_multipliers or {}
    
    def get_environment_modifiers(self) -> Dict[str, Any]:
        """Get current stage's environment modifiers"""
        stage = self.get_current_stage()
        return stage.environment_modifiers or {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics"""
        current_stage = self.get_current_stage()
        
        return {
            "current_stage": current_stage.name,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_duration": self.performance_tracker.get_stage_duration(),
            "stage_timesteps": self.performance_tracker.stage_timesteps,
            "promotions": self.promotion_count,
            "demotions": self.demotion_count,
            "stage_history": self.stage_history[-10:],  # Last 10 transitions
            "current_metrics": {
                metric: self.performance_tracker.get_average(metric)
                for metric in current_stage.promotion_criteria.keys()
            },
            "promotion_progress": self._get_promotion_progress()
        }
    
    def _get_promotion_progress(self) -> Dict[str, float]:
        """Get progress towards promotion criteria"""
        if self.current_stage_idx >= len(self.stages) - 1:
            return {"complete": 1.0}
        
        current_stage = self.get_current_stage()
        progress = {}
        
        for metric, threshold in current_stage.promotion_criteria.items():
            current_value = self.performance_tracker.get_average(metric)
            progress[metric] = min(1.0, current_value / threshold)
        
        progress["overall"] = np.mean(list(progress.values()))
        return progress
