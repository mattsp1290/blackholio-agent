"""
Reward calculation for Blackholio environment.

This module calculates rewards based on game events and state changes,
supporting both dense and sparse reward schemes for reinforcement learning.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    # Dense reward weights
    mass_gain_weight: float = 1.0          # Reward per unit of mass gained
    mass_loss_weight: float = -2.0         # Penalty per unit of mass lost
    survival_bonus_per_step: float = 0.01  # Small reward for staying alive
    food_collection_weight: float = 0.1    # Reward for collecting food
    
    # Sparse reward weights
    kill_reward: float = 10.0              # Reward for consuming another player
    death_penalty: float = -50.0           # Penalty for dying
    
    # Reward shaping
    distance_to_food_weight: float = -0.001  # Penalty for distance to nearest food
    size_ratio_weight: float = 0.01          # Reward for favorable size ratios
    
    # Episode tracking
    max_episode_steps: int = 10000         # Maximum steps per episode
    
    # Curriculum learning
    use_curriculum: bool = True            # Enable curriculum learning
    curriculum_stages: List[str] = field(default_factory=lambda: [
        "food_collection",    # Stage 1: Focus on collecting food
        "survival",          # Stage 2: Avoid larger players
        "hunting",           # Stage 3: Hunt smaller players
        "advanced"           # Stage 4: Complex strategies
    ])
    current_stage: int = 0


@dataclass
class EpisodeStats:
    """Statistics for a single episode"""
    total_reward: float = 0.0
    steps: int = 0
    mass_gained: float = 0.0
    mass_lost: float = 0.0
    food_collected: int = 0
    kills: int = 0
    deaths: int = 0
    max_mass: float = 0.0
    survival_time: float = 0.0
    start_time: float = field(default_factory=time.time)


class RewardCalculator:
    """
    Calculates rewards based on game events and state changes.
    
    Supports:
    - Dense rewards: Continuous feedback for every step
    - Sparse rewards: Major events like kills/deaths
    - Reward shaping: Guide learning towards good behaviors
    - Curriculum learning: Adjust rewards based on training progress
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.episode_stats = EpisodeStats()
        self.previous_state: Optional[Dict[str, Any]] = None
        
        logger.info(f"Reward calculator initialized with curriculum: {self.config.use_curriculum}")
    
    def reset(self) -> EpisodeStats:
        """
        Reset for a new episode.
        
        Returns:
            Statistics from the previous episode
        """
        old_stats = self.episode_stats
        self.episode_stats = EpisodeStats()
        self.previous_state = None
        return old_stats
    
    def calculate_step_reward(self,
                            current_state: Dict[str, Any],
                            action: np.ndarray,
                            info: Dict[str, Any] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for a single step.
        
        Args:
            current_state: Current game state
            action: Action taken
            info: Additional information (kills, deaths, etc.)
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        if info is None:
            info = {}
        
        reward_components = {}
        
        # Calculate mass change
        if self.previous_state:
            mass_change = self._calculate_mass_change(self.previous_state, current_state)
            if mass_change > 0:
                reward_components['mass_gain'] = mass_change * self.config.mass_gain_weight
                self.episode_stats.mass_gained += mass_change
            else:
                reward_components['mass_loss'] = mass_change * self.config.mass_loss_weight
                self.episode_stats.mass_lost += abs(mass_change)
        
        # Survival bonus
        reward_components['survival'] = self.config.survival_bonus_per_step
        
        # Food collection
        food_collected = info.get('food_collected', 0)
        if food_collected > 0:
            reward_components['food'] = food_collected * self.config.food_collection_weight
            self.episode_stats.food_collected += food_collected
        
        # Kill/death rewards (sparse)
        kills = info.get('kills', 0)
        if kills > 0:
            reward_components['kills'] = kills * self.config.kill_reward
            self.episode_stats.kills += kills
        
        deaths = info.get('deaths', 0)
        if deaths > 0:
            reward_components['deaths'] = deaths * self.config.death_penalty
            self.episode_stats.deaths += deaths
        
        # Reward shaping
        if self.config.distance_to_food_weight != 0:
            min_food_dist = self._get_min_food_distance(current_state)
            if min_food_dist is not None:
                reward_components['food_distance'] = min_food_dist * self.config.distance_to_food_weight
        
        if self.config.size_ratio_weight != 0:
            size_ratio_bonus = self._calculate_size_ratio_bonus(current_state)
            if size_ratio_bonus != 0:
                reward_components['size_ratio'] = size_ratio_bonus * self.config.size_ratio_weight
        
        # Apply curriculum learning adjustments
        if self.config.use_curriculum:
            reward_components = self._apply_curriculum_adjustments(reward_components)
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        # Update episode stats
        self.episode_stats.total_reward += total_reward
        self.episode_stats.steps += 1
        
        current_mass = self._get_total_mass(current_state)
        if current_mass > self.episode_stats.max_mass:
            self.episode_stats.max_mass = current_mass
        
        # Store state for next step
        self.previous_state = current_state
        
        return total_reward, reward_components
    
    def calculate_episode_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate final reward for the episode.
        
        Returns:
            Tuple of (episode_bonus, bonus_components)
        """
        bonus_components = {}
        
        # Survival time bonus
        survival_time = time.time() - self.episode_stats.start_time
        self.episode_stats.survival_time = survival_time
        bonus_components['survival_time'] = survival_time * 0.01
        
        # Max mass bonus
        bonus_components['max_mass'] = self.episode_stats.max_mass * 0.001
        
        # Efficiency bonus (mass gained vs steps)
        if self.episode_stats.steps > 0:
            efficiency = self.episode_stats.mass_gained / self.episode_stats.steps
            bonus_components['efficiency'] = efficiency * 10.0
        
        # K/D ratio bonus
        if self.episode_stats.deaths == 0 and self.episode_stats.kills > 0:
            bonus_components['kd_ratio'] = self.episode_stats.kills * 5.0
        
        total_bonus = sum(bonus_components.values())
        return total_bonus, bonus_components
    
    def advance_curriculum_stage(self) -> bool:
        """
        Advance to the next curriculum stage.
        
        Returns:
            True if advanced to a new stage, False if already at final stage
        """
        if self.config.current_stage < len(self.config.curriculum_stages) - 1:
            self.config.current_stage += 1
            logger.info(f"Advanced to curriculum stage: {self.config.curriculum_stages[self.config.current_stage]}")
            return True
        return False
    
    def set_curriculum_stage(self, stage: Union[int, str]) -> None:
        """Set the curriculum stage"""
        if isinstance(stage, str):
            if stage in self.config.curriculum_stages:
                self.config.current_stage = self.config.curriculum_stages.index(stage)
            else:
                raise ValueError(f"Unknown curriculum stage: {stage}")
        else:
            self.config.current_stage = max(0, min(stage, len(self.config.curriculum_stages) - 1))
        
        logger.info(f"Set curriculum stage to: {self.config.curriculum_stages[self.config.current_stage]}")
    
    def get_current_stage(self) -> str:
        """Get the current curriculum stage name"""
        return self.config.curriculum_stages[self.config.current_stage]
    
    def _calculate_mass_change(self, prev_state: Dict[str, Any], curr_state: Dict[str, Any]) -> float:
        """Calculate change in total mass"""
        prev_mass = self._get_total_mass(prev_state)
        curr_mass = self._get_total_mass(curr_state)
        return curr_mass - prev_mass
    
    def _get_total_mass(self, state: Dict[str, Any]) -> float:
        """Get total mass from state"""
        player_entities = state.get('player_entities', [])
        return sum(e.get('mass', 0) for e in player_entities)
    
    def _get_min_food_distance(self, state: Dict[str, Any]) -> Optional[float]:
        """Get distance to nearest food"""
        player_entities = state.get('player_entities', [])
        food_entities = state.get('food_entities', [])
        
        if not player_entities or not food_entities:
            return None
        
        # Get player center of mass
        total_mass = sum(e.get('mass', 1) for e in player_entities)
        player_x = sum(e.get('x', 0) * e.get('mass', 1) for e in player_entities) / total_mass
        player_y = sum(e.get('y', 0) * e.get('mass', 1) for e in player_entities) / total_mass
        
        # Find minimum distance to food
        min_dist = float('inf')
        for food in food_entities:
            dist = np.sqrt((food.get('x', 0) - player_x)**2 + (food.get('y', 0) - player_y)**2)
            min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else None
    
    def _calculate_size_ratio_bonus(self, state: Dict[str, Any]) -> float:
        """Calculate bonus based on size relative to nearby entities"""
        player_entities = state.get('player_entities', [])
        other_entities = state.get('other_entities', [])
        
        if not player_entities or not other_entities:
            return 0.0
        
        player_mass = self._get_total_mass(state)
        
        # Count entities we're larger than (potential prey)
        prey_count = sum(1 for e in other_entities if e.get('mass', 0) < player_mass * 0.8)
        
        # Count entities we're smaller than (potential threats)
        threat_count = sum(1 for e in other_entities if e.get('mass', 0) > player_mass * 1.2)
        
        # Bonus for having more prey than threats nearby
        return (prey_count - threat_count) * 0.1
    
    def _apply_curriculum_adjustments(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Apply curriculum-specific reward adjustments"""
        stage = self.config.curriculum_stages[self.config.current_stage]
        
        if stage == "food_collection":
            # Emphasize food collection, reduce combat rewards
            rewards = rewards.copy()
            if 'food' in rewards:
                rewards['food'] *= 2.0
            if 'kills' in rewards:
                rewards['kills'] *= 0.1
            if 'food_distance' in rewards:
                rewards['food_distance'] *= 2.0
                
        elif stage == "survival":
            # Emphasize survival, penalize risky behavior
            rewards = rewards.copy()
            if 'survival' in rewards:
                rewards['survival'] *= 2.0
            if 'deaths' in rewards:
                rewards['deaths'] *= 2.0
            if 'size_ratio' in rewards:
                rewards['size_ratio'] *= 1.5
                
        elif stage == "hunting":
            # Encourage hunting smaller players
            rewards = rewards.copy()
            if 'kills' in rewards:
                rewards['kills'] *= 1.5
            if 'mass_gain' in rewards:
                rewards['mass_gain'] *= 1.5
                
        # "advanced" stage uses default weights
        
        return rewards
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get detailed reward information for logging"""
        return {
            'total_reward': self.episode_stats.total_reward,
            'steps': self.episode_stats.steps,
            'mass_gained': self.episode_stats.mass_gained,
            'mass_lost': self.episode_stats.mass_lost,
            'food_collected': self.episode_stats.food_collected,
            'kills': self.episode_stats.kills,
            'deaths': self.episode_stats.deaths,
            'max_mass': self.episode_stats.max_mass,
            'survival_time': self.episode_stats.survival_time,
            'curriculum_stage': self.get_current_stage(),
            'avg_reward_per_step': self.episode_stats.total_reward / max(1, self.episode_stats.steps)
        }
