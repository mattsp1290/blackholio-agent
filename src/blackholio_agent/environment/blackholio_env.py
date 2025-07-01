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
import uuid
import gymnasium as gym
from gymnasium import spaces

from .connection import ConnectionConfig, GameState
from .blackholio_connection_adapter import BlackholioConnectionAdapter as BlackholioConnectionV112
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
    db_identity: Optional[str] = None  # Database identity for v1.1.2
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
            if self.db_identity is not None:
                config_kwargs['db_identity'] = self.db_identity
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
        self.connection: Union[BlackholioConnectionV112, MockBlackholioConnection, ImprovedMockBlackholioConnection] = None
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
        Reset the environment with improved spawn handling.
        
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
            await self._setup_connection_with_diagnostics()
        
        # Ensure connected (skip for mock mode)
        if not self._use_mock:
            if not self.connection.is_connected():
                await self.connection.connect()
        
        # Generate unique player name
        player_name = self._generate_player_name()
        
        # Enhanced spawn handling for v1.1.2 connections
        if not self._use_mock:
            if isinstance(self.connection, BlackholioConnectionV112):
                # Use improved spawn detection with retry logic
                spawn_success = await self._enhanced_spawn_handling(player_name)
                
                if not spawn_success:
                    logger.error(f"❌ Failed to spawn player {player_name}")
                    # Return safe default state
                    return self._get_default_observation(), {}
            else:
                # Legacy connection handling
                if not self._is_player_spawned():
                    logger.info("Player not spawned yet, joining game...")
                    await self._join_game()
                    await self._wait_for_spawn()
        else:
            logger.info("Mock mode - skipping player spawn checks")
        
        # Wait for initial game state
        initial_state = await self._wait_for_initial_state(timeout=3.0)
        
        if not initial_state:
            logger.warning("⚠️  No initial state received, using empty state")
            initial_state = self._get_empty_state()
        
        self.current_game_state = initial_state
        
        # Get initial observation
        self.current_obs = self._get_observation()
        
        info = {
            'episode_stats': episode_stats.__dict__ if episode_stats.steps > 0 else None,
            'player_id': getattr(self.connection, 'player_id', None),
            'player_identity': getattr(self.connection, 'player_identity', None)
        }
        
        logger.info(f"Environment reset complete. Player ID: {getattr(self.connection, 'player_id', 'None')}")
        
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
        
        # WORKAROUND: Don't terminate immediately if no player entities
        # This handles cases where player detection is broken but game is running
        if len(player_entities) == 0 and self.episode_steps < 5:
            # Give a few steps for player detection to work
            terminated = False
            logger.debug(f"No player entities detected at step {self.episode_steps}, continuing...")
        else:
            terminated = len(player_entities) == 0
        
        # Check if episode is truncated (max steps reached)
        self.episode_steps += 1
        truncated = self.episode_steps >= self.config.max_episode_steps
        
        # Calculate reward
        info = self._extract_game_events()
        info['action_result'] = action_result
        
        # Ensure we have a valid game state for reward calculation
        if self.current_game_state is None:
            logger.warning("No game state available for reward calculation, using empty state")
            self.current_game_state = self._get_empty_state()
        
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
        
        # Add debug info + PERFORMANCE METRICS
        info['game_update_rate'] = self.connection.get_update_rate()
        info['action_stats'] = self.action_space_handler.get_action_stats()
        
        # 🚀 ADD UNIFIED CLIENT PERFORMANCE METRICS
        if hasattr(self.connection, 'get_performance_metrics'):
            info['performance_metrics'] = self.connection.get_performance_metrics()
            
            # Log performance gains periodically
            if self.episode_steps % 100 == 0 and self.episode_steps > 0:
                perf = info['performance_metrics']
                if perf['action_batching']['enabled']:
                    efficiency = perf['action_batching']['batching_efficiency']
                    logger.info(f"🚀 Performance: {efficiency:.1f}% batching efficiency, "
                              f"{perf['action_batching']['avg_batch_size']:.1f} avg batch size")
        
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
    
    def _is_player_spawned(self) -> bool:
        """Check if player is already spawned and has entities"""
        if not self.connection:
            return False
        
        # For v1.1.2 connection, check multiple indicators
        if isinstance(self.connection, BlackholioConnectionV112):
            # Check if we have any players and circles/entities
            if (len(self.connection.players) > 0 and 
                (len(self.connection.circles) > 0 or len(self.connection.entities) > 0)):
                logger.debug(f"Player already spawned: {len(self.connection.players)} players, {len(self.connection.circles)} circles, {len(self.connection.entities)} entities")
                return True
            
            # Also check the existing logic for completeness
            local_player = self.connection.get_local_player()
            if local_player:
                entities = self.connection.get_local_player_entities()
                if entities and len(entities) > 0:
                    logger.debug(f"Player {local_player.name} already spawned with {len(entities)} entities")
                    return True
        else:
            # For legacy connection, check player_id
            if hasattr(self.connection, 'player_id') and self.connection.player_id:
                logger.debug(f"Player {self.connection.player_id} already spawned")
                return True
        
        return False
    
    def _generate_player_name(self) -> str:
        """Generate a unique player name for this environment instance."""
        # Extract base name parts - preserve original prefix and env_id
        parts = self.config.player_name.split('_')
        if len(parts) >= 3:
            # Format: ML_Agent_0 -> ML_Agent_0_uuid
            prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"
        else:
            # Fallback
            prefix = "ML_Agent_0"
        
        # Generate new UUID-based suffix
        unique_id = str(uuid.uuid4())[:8]
        new_name = f"{prefix}_{unique_id}"
        
        logger.info(f"Generated player name: {new_name}")
        self.config.player_name = new_name
        return new_name
    
    def _regenerate_player_name(self) -> str:
        """Regenerate player name on duplicate errors."""
        return self._generate_player_name()

    async def _join_game(self) -> None:
        """Join the game with specified player name, regenerating name on duplicate errors"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            logger.info(f"Joining game as '{self.config.player_name}' (attempt {attempt + 1})")
            
            try:
                # Check if we're using v1.1.2 connection and need to initialize game first
                if isinstance(self.connection, BlackholioConnectionV112):
                    logger.info("Using v1.1.2 connection - ensuring game is initialized...")
                    
                    # Log current state before joining
                    logger.info(f"Pre-join state: Players={len(self.connection.players)}, Entities={len(self.connection.entities)}, Circles={len(self.connection.circles)}")
                    
                    # Don't call connect reducer directly - it's a lifecycle reducer
                    # The v1.1.2 connection handles this automatically
                    
                    # Skip init reducer (SpacetimeDB v1.1.2 lifecycle reducers can't be called directly)
                    logger.info("Skipping init reducer - SpacetimeDB calls this automatically")
                    await asyncio.sleep(1.0)  # Give time for automatic initialization
                    
                    # Use the v1.1.2 specific enter_game method with extended timeout for inference
                    spawn_success = await self.connection.enter_game(self.config.player_name, timeout=15.0)
                    if spawn_success:
                        return  # Success, exit the loop
                    else:
                        logger.warning("Player spawn detection failed, but continuing with training")
                        return  # Don't retry spawn detection failures
                else:
                    # Call EnterGame reducer for legacy connection
                    await self.connection.call_reducer("EnterGame", self.config.player_name)
                    return  # Success, exit the loop
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "duplicate" in error_msg and "player" in error_msg and attempt < max_attempts - 1:
                    logger.warning(f"Duplicate player name error on attempt {attempt + 1}: {e}")
                    self._regenerate_player_name()
                    logger.info(f"Retrying with new player name: {self.config.player_name}")
                    continue
                else:
                    # Re-raise if not a duplicate error or max attempts reached
                    logger.error(f"Failed to join game after {attempt + 1} attempts: {e}")
                    raise
    
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
        """Get current observation from game state with improved error handling."""
        if not self.current_game_state:
            logger.warning("No game state available for observation, using empty state")
            return np.zeros(self.observation_space_handler.shape, dtype=np.float32)
        
        try:
            return self.observation_space_handler.process_game_state(
                self.current_game_state.get('player_entities', []),
                self.current_game_state.get('other_entities', []),
                self.current_game_state.get('food_entities', [])
            )
        except Exception as e:
            logger.warning(f"Error processing game state for observation: {e}")
            return np.zeros(self.observation_space_handler.shape, dtype=np.float32)
    
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
    
    async def _setup_connection_with_diagnostics(self) -> None:
        """Setup connection with enhanced diagnostics."""
        try:
            await self._initialize_connection()
            
            # Run diagnostics if using v1.1.2 connection
            if isinstance(self.connection, BlackholioConnectionV112):
                self.connection._diagnose_subscription_issues()
        except Exception as e:
            logger.error(f"Connection setup failed: {e}")
            raise
    
    async def _enhanced_spawn_handling(self, player_name: str) -> bool:
        """Enhanced spawn handling with improved detection."""
        try:
            logger.info(f"🎮 Enhanced spawn handling for '{player_name}'")
            
            # Use the improved spawn detection from the adapter
            if hasattr(self.connection, '_join_game_with_retry'):
                return await self.connection._join_game_with_retry(player_name)
            else:
                # Fallback to original method
                return await self.connection.enter_game(player_name, timeout=10.0)
                
        except Exception as e:
            logger.error(f"Enhanced spawn handling failed: {e}")
            return False
    
    async def _wait_for_initial_state(self, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """Wait for initial game state after spawn."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Try to get game state
            try:
                player_entities = self.connection.get_player_entities()
                other_entities = self.connection.get_other_entities()
                
                if player_entities or other_entities:
                    state = {
                        'player_entities': player_entities,
                        'other_entities': other_entities, 
                        'food_entities': [],  # Will be populated by proper game state
                        'timestamp': time.time()
                    }
                    logger.info(f"✅ Received initial state: {len(player_entities)} player entities, {len(other_entities)} other entities")
                    return state
            except Exception as e:
                logger.debug(f"Error getting initial state: {e}")
            
            await asyncio.sleep(0.1)
        
        logger.warning("⏰ Timeout waiting for initial state")
        return None
    
    def _get_empty_state(self) -> Dict[str, Any]:
        """Get empty game state as fallback."""
        return {
            'player_entities': [],
            'other_entities': [],
            'food_entities': [],
            'timestamp': time.time()
        }
    
    def _get_default_observation(self) -> np.ndarray:
        """Get default observation when spawn fails."""
        return np.zeros(self.observation_space_handler.shape, dtype=np.float32)
    
    async def _initialize_connection(self) -> None:
        """Initialize connection with fallback to mock mode"""
        # Check if mock mode is explicitly requested
        if self.config.connection_config.host.startswith("mock://"):
            logger.info("Mock mode requested - using adapter in PERFORMANCE test mode")
            self.connection = BlackholioConnectionV112(
                host="localhost:3000",  # Won't actually connect in mock mode
                db_identity="test_mock",
                verbose_logging=False
            )
            
            # 🚀 ENABLE PERFORMANCE MODE FOR ML TRAINING
            self.connection.enable_performance_mode(
                batch_size=20,          # Larger batches for ML training
                batch_timeout_ms=5.0,   # 5ms timeout for ultra-fast training
                cache_ttl_ms=0.2        # 0.2ms cache for maximum speed
            )
            
            # Don't actually connect - just set up for testing
            self._use_mock = True
            logger.info("🔥 PERFORMANCE mock adapter connection established")
            return
            
        try:
            # Always use the adapter now - it handles unified client integration
            logger.info("Using BlackholioConnectionAdapter with unified client integration...")
            
            self.connection = BlackholioConnectionV112(
                host=self.config.host,
                db_identity=self.config.db_identity or self.config.database or "blackholio",
                verbose_logging=False
            )
            
            # Try to connect
            connection_success = await self.connection.connect()
            if not connection_success:
                logger.warning("Adapter connection failed, but continuing with simulation mode")
            
            self._use_mock = False
            
            # Register game state callback
            self.connection.add_game_state_listener(self._on_game_state_update)
            
            logger.info("Successfully initialized connection adapter")
            
            # 🚀 ENABLE PERFORMANCE MODE BASED ON IDENTITY
            if self.config.db_identity:
                # Production mode for specific identities
                self.connection.enable_performance_mode(
                    batch_size=25,          # Large batches for production
                    batch_timeout_ms=3.0,   # 3ms timeout for maximum throughput  
                    cache_ttl_ms=0.1        # 0.1ms cache for extreme speed
                )
                logger.info("🔥 PRODUCTION performance mode enabled")
            else:
                # Standard mode for default connections
                self.connection.enable_performance_mode(
                    batch_size=20,          # Standard batch size
                    batch_timeout_ms=5.0,   # 5ms timeout
                    cache_ttl_ms=0.2        # 0.2ms cache
                )
                logger.info("🔥 STANDARD performance mode enabled")
            
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
