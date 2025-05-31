"""
Action space for Blackholio environment.

This module handles agent actions and converts them to game commands,
including movement and split decisions with proper throttling.
"""

import numpy as np
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ActionConfig:
    """Configuration for action space"""
    movement_scale: float = 1.0  # Scale factor for movement vectors
    update_rate: float = 20.0    # Target update rate in Hz
    queue_size: int = 10         # Maximum action queue size
    enable_split: bool = True    # Whether split action is enabled
    split_cooldown: float = 1.0  # Cooldown between splits in seconds


class ActionSpace:
    """
    Handles agent actions and converts to game commands.
    
    Action space consists of:
    - Continuous: 2D movement direction vector
    - Discrete: Binary split decision
    
    Features:
    - Throttling to match game's 20Hz update rate
    - Action queuing during network delays
    - Cooldown management for split actions
    """
    
    def __init__(self, config: ActionConfig = None):
        self.config = config or ActionConfig()
        
        # Action dimensions
        self.movement_dim = 2  # x, y direction
        self.discrete_dim = 1 if self.config.enable_split else 0
        self.total_dim = self.movement_dim + self.discrete_dim
        
        # Throttling
        self.update_interval = 1.0 / self.config.update_rate
        self.last_movement_time = 0.0
        self.last_split_time = 0.0
        
        # Action queue for handling network delays
        self.action_queue = deque(maxlen=self.config.queue_size)
        self.processing_actions = False
        
        # Performance tracking
        self.action_count = 0
        self.successful_actions = 0
        self.failed_actions = 0
        
        logger.info(f"Action space initialized with {self.total_dim} dimensions")
    
    @property
    def shape(self) -> Tuple[int]:
        """Get the shape of the action space"""
        return (self.total_dim,)
    
    @property
    def movement_shape(self) -> Tuple[int]:
        """Get the shape of movement action space"""
        return (self.movement_dim,)
    
    @property
    def discrete_shape(self) -> Tuple[int]:
        """Get the shape of discrete action space"""
        return (self.discrete_dim,) if self.config.enable_split else (0,)
    
    def sample(self) -> np.ndarray:
        """
        Sample a random action from the action space.
        Useful for random exploration.
        
        Returns:
            Random action array
        """
        action = np.zeros(self.total_dim, dtype=np.float32)
        
        # Random movement direction (normalized)
        direction = np.random.randn(2)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        action[:2] = direction
        
        # Random split decision (if enabled)
        if self.config.enable_split:
            action[2] = np.random.random() > 0.95  # 5% chance to split
        
        return action
    
    def parse_action(self, action: Union[np.ndarray, Dict[str, Any]]) -> Tuple[np.ndarray, bool]:
        """
        Parse action into movement and split components.
        
        Args:
            action: Either a numpy array or dict with 'movement' and 'split' keys
            
        Returns:
            Tuple of (movement_vector, should_split)
        """
        if isinstance(action, dict):
            movement = np.array(action.get('movement', [0.0, 0.0]), dtype=np.float32)
            should_split = bool(action.get('split', False)) if self.config.enable_split else False
        else:
            action = np.array(action, dtype=np.float32)
            movement = action[:2] if len(action) >= 2 else np.zeros(2)
            should_split = bool(action[2] > 0.5) if len(action) > 2 and self.config.enable_split else False
        
        # Normalize and scale movement
        movement = self._normalize_movement(movement)
        
        return movement, should_split
    
    async def execute_action(self, 
                           connection,
                           action: Union[np.ndarray, Dict[str, Any]],
                           force: bool = False) -> Dict[str, Any]:
        """
        Execute an action on the game.
        
        Args:
            connection: BlackholioConnection instance
            action: Action to execute
            force: If True, bypass throttling
            
        Returns:
            Dictionary with execution results
        """
        movement, should_split = self.parse_action(action)
        
        result = {
            'movement_executed': False,
            'split_executed': False,
            'queued': False,
            'error': None
        }
        
        # Check if we should queue the action
        if not force and self._should_queue_action():
            self.action_queue.append((movement, should_split, time.time()))
            result['queued'] = True
            return result
        
        # Execute movement
        if not force and not self._can_send_movement():
            # Skip movement update due to throttling
            pass
        else:
            try:
                await connection.update_player_input(movement.tolist())
                self.last_movement_time = time.time()
                result['movement_executed'] = True
                self.successful_actions += 1
            except Exception as e:
                logger.error(f"Failed to execute movement: {e}")
                result['error'] = str(e)
                self.failed_actions += 1
        
        # Execute split if requested
        if should_split and self._can_split():
            try:
                await connection.player_split()
                self.last_split_time = time.time()
                result['split_executed'] = True
            except Exception as e:
                logger.error(f"Failed to execute split: {e}")
                result['error'] = str(e)
        
        self.action_count += 1
        return result
    
    async def process_action_queue(self, connection) -> int:
        """
        Process queued actions.
        
        Args:
            connection: BlackholioConnection instance
            
        Returns:
            Number of actions processed
        """
        if self.processing_actions or not self.action_queue:
            return 0
        
        self.processing_actions = True
        processed = 0
        
        try:
            while self.action_queue and self._can_send_movement():
                movement, should_split, timestamp = self.action_queue.popleft()
                
                # Skip old actions
                if time.time() - timestamp > 1.0:
                    continue
                
                await self.execute_action(connection, {
                    'movement': movement,
                    'split': should_split
                }, force=True)
                
                processed += 1
                
                # Small delay to avoid overwhelming the connection
                await asyncio.sleep(0.01)
        
        finally:
            self.processing_actions = False
        
        return processed
    
    def get_action_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for each action component.
        
        Returns:
            Dictionary mapping action names to (min, max) bounds
        """
        bounds = {
            'movement_x': (-1.0, 1.0),
            'movement_y': (-1.0, 1.0)
        }
        
        if self.config.enable_split:
            bounds['split'] = (0.0, 1.0)
        
        return bounds
    
    def get_action_stats(self) -> Dict[str, Any]:
        """
        Get action execution statistics.
        
        Returns:
            Dictionary with action statistics
        """
        total = self.successful_actions + self.failed_actions
        success_rate = self.successful_actions / total if total > 0 else 0.0
        
        return {
            'total_actions': self.action_count,
            'successful_actions': self.successful_actions,
            'failed_actions': self.failed_actions,
            'success_rate': success_rate,
            'queued_actions': len(self.action_queue),
            'current_update_rate': self._get_current_update_rate()
        }
    
    def reset_stats(self) -> None:
        """Reset action statistics"""
        self.action_count = 0
        self.successful_actions = 0
        self.failed_actions = 0
    
    def _normalize_movement(self, movement: np.ndarray) -> np.ndarray:
        """Normalize and scale movement vector"""
        # Ensure movement is 2D
        if movement.shape != (2,):
            movement = movement[:2] if len(movement) >= 2 else np.zeros(2)
        
        # Clip to [-1, 1] range
        movement = np.clip(movement, -1.0, 1.0)
        
        # Apply scale factor
        movement = movement * self.config.movement_scale
        
        return movement
    
    def _can_send_movement(self) -> bool:
        """Check if we can send a movement update"""
        return time.time() - self.last_movement_time >= self.update_interval
    
    def _can_split(self) -> bool:
        """Check if we can perform a split"""
        if not self.config.enable_split:
            return False
        return time.time() - self.last_split_time >= self.config.split_cooldown
    
    def _should_queue_action(self) -> bool:
        """Determine if action should be queued"""
        # Queue if we're already processing actions or throttled
        return self.processing_actions or not self._can_send_movement()
    
    def _get_current_update_rate(self) -> float:
        """Calculate current actual update rate"""
        if self.last_movement_time == 0:
            return 0.0
        
        time_since_last = time.time() - self.last_movement_time
        if time_since_last > 0:
            return min(1.0 / time_since_last, self.config.update_rate)
        return self.config.update_rate


class HybridActionSpace(ActionSpace):
    """
    Extended action space supporting both continuous and discrete action modes.
    
    This class provides additional functionality for handling hybrid action spaces
    where the agent can switch between continuous control and discrete actions.
    """
    
    def __init__(self, config: ActionConfig = None):
        super().__init__(config)
        
        # Additional discrete actions
        self.discrete_actions = {
            0: "none",
            1: "split",
            2: "eject_mass",  # Future feature
            3: "boost"        # Future feature
        }
    
    def parse_discrete_action(self, action_id: int) -> Dict[str, Any]:
        """
        Parse a discrete action ID into action components.
        
        Args:
            action_id: Integer action ID
            
        Returns:
            Dictionary with action components
        """
        action_name = self.discrete_actions.get(action_id, "none")
        
        return {
            'movement': np.zeros(2),  # No movement for discrete actions
            'split': action_name == "split",
            'action_name': action_name
        }
    
    def get_discrete_action_count(self) -> int:
        """Get the number of discrete actions available"""
        return len(self.discrete_actions)
