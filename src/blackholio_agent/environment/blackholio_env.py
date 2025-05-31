"""
Main Blackholio environment for ML training.

This module provides a Gym-like environment interface for training
reinforcement learning agents to play Blackholio.
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import logging
import time
import gymnasium as gym
from gymnasium import spaces

from .connection import BlackholioConnection, ConnectionConfig, GameState
from .mock_connection import MockBlackholioConnection
from .mock_connection_improved import ImprovedMockBlackholioConnection
from .observation_space import ObservationSpace, ObservationConfig
from .action_space import ActionSpace, ActionConfig
from .reward_calculator import RewardCalculator, RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class BlackholioEnvConfig:
    """Configuration for Blackholio environment"""
    # Connection settings
    host: str = "localhost:3000"
    database: Optional[str] = None  # Let ConnectionConfig use its default
    auth_token: Optional[str] = None
    ssl_enabled: bool = False
    
    # Environment settings
    player_name: str = "ML_Agent"
    render_mode: Optional[str] = None  # 'human', 'rgb_array', or None
    max_episode_steps: int = 10000
    step_interval: float = 0.05  # 20Hz update rate
    
    # Component configs
    connection_config: Optional[ConnectionConfig] = None
    observation_config: Optional[ObservationConfig] = None
    action_config: Optional[ActionConfig] = None
    reward_config: Optional[RewardConfig] = None
    
    def __post_init__(self):
        # Create default configs if not provided
        if self.connection_config is None:
            # Only pass database if it's not None
            config_kwargs = {
                'host': self.host,
                'auth_token': self.auth_token,
                'ssl_enabled': self.ssl_enabled
            }
            if self.database is not None:
                config_kwargs['database'] = self.database
            self.connection_config = ConnectionConfig(**config_kwargs)
        
        if self.observation_config is None:
            self.observation_config = ObservationConfig()
        
        if self.action_config is None:
            self.action_config = ActionConfig()
        
        if self.reward_config is None:
            self.reward_config = RewardConfig(max_episode_steps=self.max_episode_steps)


class BlackholioEnv(gym.Env):
    """
    Gym-like environment for Blackholio game.
    
    This environment provides:
    - Standard Gym interface (reset, step, render, close)
    - Async/await support for SpacetimeDB operations
    - Automatic game state synchronization
    - Reward calculation and episode management
    
    Example:
        ```python
        env = BlackholioEnv(config)
        
        # Synchronous usage (runs async internally)
        obs, info = env.reset()
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Async usage
        obs, info = await env.async_reset()
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = await env.async_step(action)
        ```
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 20
    }
    
    def __init__(self, config: Union[BlackholioEnvConfig, Dict[str, Any]] = None):
        """
        Initialize the Blackholio environment.
        
        Args:
            config: Environment configuration
        """
        super().__init__()
        
        # Handle config
        if config is None:
            self.config = BlackholioEnvConfig()
        elif isinstance(config, dict):
            self.config = BlackholioEnvConfig(**config)
        else:
            self.config = config
        
        # Create components
        self.connection: Union[BlackholioConnection, MockBlackholioConnection, ImprovedMockBlackholioConnection] = None
        self._use_mock = False
        self.observation_space_handler = ObservationSpace(self.config.observation_config)
        self.action_space_handler = ActionSpace(self.config.action_config)
        self.reward_calculator = RewardCalculator(self.config.reward_config)
        
        # Define Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.observation_space_handler.shape,
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.action_space_handler.shape,
            dtype=np.float32
        )
        
        # Episode state
        self.current_obs: Optional[np.ndarray] = None
        self.current_game_state: Optional[Dict[str, Any]] = None
        self.episode_steps = 0
        self.episode_start_time = 0.0
        self.done = False
        
        # Async support
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._game_state_updated = asyncio.Event()
        
        # Rendering
        self.render_mode = self.config.render_mode
        
        logger.info("Blackholio environment initialized")
    
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop"""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment (synchronous wrapper).
        
        Args:
            seed: Random seed
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        return self.loop.run_until_complete(self.async_reset(seed, options))
    
    async def async_reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment (async version).
        
        Args:
            seed: Random seed
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        logger.info("Resetting environment")
        
        # Reset episode state
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.done = False
        
        # Reset components
        episode_stats = self.reward_calculator.reset()
        
        # Initialize connection if needed
        if self.connection is None:
            await self._initialize_connection()
        
        # Ensure connected
        await self.connection.ensure_connected()
        
        # Join game with specified name
        await self._join_game()
        
        # Wait for initial game state
        await self._wait_for_spawn()
        
        # Get initial observation
        self.current_obs = self._get_observation()
        
        info = {
            'episode_stats': episode_stats.__dict__ if episode_stats.steps > 0 else None,
            'player_id': self.connection.player_id,
            'player_identity': self.connection.player_identity
        }
        
        logger.info(f"Environment reset complete. Player ID: {self.connection.player_id}")
        
        return self.current_obs, info
    
    def step(self, action: Union[np.ndarray, Dict[str, Any]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment (synchronous wrapper).
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return self.loop.run_until_complete(self.async_step(action))
    
    async def async_step(self, action: Union[np.ndarray, Dict[str, Any]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment (async version).
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        # Execute action
        action_result = await self.action_space_handler.execute_action(
            self.connection, action
        )
        
        # Process any queued actions
        await self.action_space_handler.process_action_queue(self.connection)
        
        # Wait for next game update
        await self._wait_for_update()
        
        # Get new observation
        self.current_obs = self._get_observation()
        
        # Check if we're dead (no player entities)
        player_entities = self.connection.get_player_entities()
        terminated = len(player_entities) == 0
        
        # Check if episode is truncated (max steps reached)
        self.episode_steps += 1
        truncated = self.episode_steps >= self.config.max_episode_steps
        
        # Calculate reward
        info = self._extract_game_events()
        info['action_result'] = action_result
        
        reward, reward_components = self.reward_calculator.calculate_step_reward(
            self.current_game_state, action, info
        )
        
        info['reward_components'] = reward_components
        
        # Handle episode end
        if terminated or truncated:
            self.done = True
            episode_bonus, bonus_components = self.reward_calculator.calculate_episode_reward()
            reward += episode_bonus
            info['episode_bonus'] = episode_bonus
            info['bonus_components'] = bonus_components
            info['episode_stats'] = self.reward_calculator.get_reward_info()
        
        # Add debug info
        info['game_update_rate'] = self.connection.get_update_rate()
        info['action_stats'] = self.action_space_handler.get_action_stats()
        
        return self.current_obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self.render_mode == 'human':
            # Print game state to console
            self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
        
        return None
    
    def close(self) -> None:
        """Close the environment and cleanup resources"""
        logger.info("Closing environment")
        
        # Disconnect from game if connection exists
        if self.connection and self.connection.is_connected:
            self.loop.run_until_complete(self.connection.disconnect())
        
        # Close event loop if we created it
        if self._loop and not self._loop.is_running():
            self._loop.close()
            self._loop = None
    
    async def _join_game(self) -> None:
        """Join the game with specified player name"""
        logger.info(f"Joining game as '{self.config.player_name}'")
        
        # Call EnterGame reducer
        await self.connection.call_reducer("EnterGame", self.config.player_name)
    
    async def _wait_for_spawn(self, timeout: float = 30.0) -> None:
        """Wait for player to spawn in game"""
        logger.info("Waiting for player spawn...")
        
        start_time = time.time()
        while True:
            player_entities = self.connection.get_player_entities()
            if player_entities:
                logger.info(f"Player spawned with {len(player_entities)} entities")
                break
            
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for player spawn")
            
            await asyncio.sleep(0.1)
    
    async def _wait_for_update(self) -> None:
        """Wait for next game state update"""
        self._game_state_updated.clear()
        
        # Wait for update or timeout
        try:
            await asyncio.wait_for(
                self._game_state_updated.wait(),
                timeout=self.config.step_interval * 2
            )
        except asyncio.TimeoutError:
            # Continue even if no update received
            pass
    
    def _on_game_state_update(self, game_state: GameState) -> None:
        """Handle game state update from connection"""
        self.current_game_state = self._convert_game_state(game_state)
        self._game_state_updated.set()
    
    def _convert_game_state(self, game_state: GameState) -> Dict[str, Any]:
        """Convert GameState to dictionary format for reward calculator"""
        player_entities = self.connection.get_player_entities()
        other_entities = self.connection.get_other_entities()
        
        # Extract food entities (entities without circles)
        food_entities = []
        for entity_id, entity in game_state.entities.items():
            if entity_id not in game_state.circles:
                food_entities.append(entity)
        
        return {
            'player_entities': player_entities,
            'other_entities': other_entities,
            'food_entities': food_entities,
            'timestamp': game_state.timestamp
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from game state"""
        if not self.current_game_state:
            return np.zeros(self.observation_space_handler.shape, dtype=np.float32)
        
        return self.observation_space_handler.process_game_state(
            self.current_game_state['player_entities'],
            self.current_game_state['other_entities'],
            self.current_game_state['food_entities']
        )
    
    def _extract_game_events(self) -> Dict[str, Any]:
        """Extract game events for reward calculation"""
        # TODO: Implement proper event tracking
        # This would involve tracking entity changes to detect kills, food collection, etc.
        return {
            'food_collected': 0,  # TODO: Track food collection
            'kills': 0,           # TODO: Track kills
            'deaths': 0           # TODO: Track deaths
        }
    
    def _render_human(self) -> None:
        """Render game state to console"""
        if not self.current_game_state:
            print("No game state available")
            return
        
        player_entities = self.current_game_state['player_entities']
        
        print(f"\n=== Blackholio Environment (Step {self.episode_steps}) ===")
        print(f"Player entities: {len(player_entities)}")
        
        if player_entities:
            total_mass = sum(e.get('mass', 0) for e in player_entities)
            center_x, center_y = self._calculate_center_of_mass(player_entities)
            print(f"Total mass: {total_mass:.1f}")
            print(f"Position: ({center_x:.1f}, {center_y:.1f})")
        
        print(f"Other entities nearby: {len(self.current_game_state['other_entities'])}")
        print(f"Food nearby: {len(self.current_game_state['food_entities'])}")
        print(f"Reward: {self.reward_calculator.episode_stats.total_reward:.2f}")
        print("=" * 40)
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render game state as RGB array"""
        # Create a simple visualization
        width, height = 800, 600
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not self.current_game_state:
            return image
        
        # TODO: Implement proper visualization
        # This would render entities, food, etc. on the image
        
        return image
    
    def _calculate_center_of_mass(self, entities: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate center of mass for entities"""
        if not entities:
            return 0.0, 0.0
        
        total_mass = sum(e.get('mass', 1) for e in entities)
        if total_mass == 0:
            return 0.0, 0.0
        
        x = sum(e.get('x', 0) * e.get('mass', 1) for e in entities) / total_mass
        y = sum(e.get('y', 0) * e.get('mass', 1) for e in entities) / total_mass
        
        return x, y
    
    async def _initialize_connection(self) -> None:
        """Initialize connection with fallback to mock mode"""
        # Check if mock mode is explicitly requested
        if self.config.connection_config.host.startswith("mock://"):
            logger.info("Mock mode requested - using simulated environment")
            self.connection = ImprovedMockBlackholioConnection(self.config.connection_config)
            await self.connection.connect()
            self._use_mock = True
            self.connection.add_game_state_listener(self._on_game_state_update)
            logger.info("Improved mock connection established")
            return
            
        try:
            # Try real connection first
            logger.info("Attempting to connect to SpacetimeDB...")
            self.connection = BlackholioConnection(self.config.connection_config)
            await self.connection.connect()
            self._use_mock = False
            logger.info("Successfully connected to SpacetimeDB")
            
            # Register game state callback
            self.connection.add_game_state_listener(self._on_game_state_update)
            
        except Exception as e:
            logger.warning(f"Failed to connect to SpacetimeDB: {e}")
            logger.info("Falling back to mock connection for training")
            
            # Use improved mock connection
            self.connection = ImprovedMockBlackholioConnection(self.config.connection_config)
            await self.connection.connect()
            self._use_mock = True
            
            # Register game state callback
            self.connection.add_game_state_listener(self._on_game_state_update)
            
            logger.info("Using improved mock connection - training will proceed with simulated environment")
