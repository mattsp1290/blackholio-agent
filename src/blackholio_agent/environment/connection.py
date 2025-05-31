"""
Blackholio connection management using SpacetimeDB Python SDK.

This module provides a wrapper around the SpacetimeDB client specifically
for the Blackholio game, handling connection lifecycle, subscriptions,
and automatic reconnection for training stability.
"""

import asyncio
import logging
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
import time
from threading import Thread, Lock
import concurrent.futures

# SpacetimeDB imports
import sys
sys.path.append('/Users/punk1290/git/spacetimedb-python-sdk/src')
from spacetimedb_sdk import Identity
from spacetimedb_sdk.modern_client import ModernSpacetimeDBClient as SpacetimeDBClient

# Import data converter
from .data_converter import (
    extract_entity_data,
    extract_circle_data,
    extract_player_data,
    extract_food_data,
    convert_to_dict
)

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for Blackholio connection"""
    host: str
    database: str = "blackholio"  # Module name
    db_identity: Optional[str] = None  # v1.1.2 requires database identity
    auth_token: Optional[str] = None
    ssl_enabled: bool = False  # Default to False for localhost
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    subscription_timeout: float = 30.0
    
    def __post_init__(self):
        """Parse host URL and set SSL enabled based on protocol"""
        if self.host.startswith("wss://") or self.host.startswith("https://"):
            self.ssl_enabled = True
            # Remove protocol prefix - SDK expects just host:port
            if self.host.startswith("wss://"):
                self.host = self.host[6:]
            elif self.host.startswith("https://"):
                self.host = self.host[8:]
        elif self.host.startswith("ws://") or self.host.startswith("http://"):
            self.ssl_enabled = False
            # Remove protocol prefix - SDK expects just host:port
            if self.host.startswith("ws://"):
                self.host = self.host[5:]
            elif self.host.startswith("http://"):
                self.host = self.host[7:]
        else:
            # Default to non-SSL for localhost
            self.ssl_enabled = False
            # host is already in correct format (no protocol prefix)


@dataclass
class GameState:
    """Current game state snapshot"""
    player_id: Optional[int] = None
    player_identity: Optional[str] = None
    entities: Dict[int, Any] = None  # entity_id -> entity data
    players: Dict[int, Any] = None   # player_id -> player data
    circles: Dict[int, Any] = None   # entity_id -> circle data
    food: List[Any] = None           # list of food entities
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = {}
        if self.players is None:
            self.players = {}
        if self.circles is None:
            self.circles = {}
        if self.food is None:
            self.food = []


class BlackholioConnection:
    """
    Manages SpacetimeDB connection for Blackholio game.
    
    This class handles:
    - Connection lifecycle management
    - Table subscriptions (Entity, Player, Circle, Food)
    - Automatic reconnection for training stability
    - Game state synchronization
    - Reducer calls for player actions
    """
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.client: Optional[SpacetimeDBClient] = None
        self.game_state = GameState()
        self._connected = False
        self._subscribed = False
        self._connection_lock = asyncio.Lock()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._update_thread: Optional[Thread] = None
        self._pending_handlers = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._state_lock = Lock()
        # Use v1.1.2 database identity if not provided
        if not self.config.db_identity:
            self.config.db_identity = self.config.database
        
        # Callbacks
        self._on_connected_callbacks: List[Callable] = []
        self._on_disconnected_callbacks: List[Callable[[str], None]] = []
        self._on_game_state_update_callbacks: List[Callable[[GameState], None]] = []
        
        # Performance tracking
        self._last_update_time = 0.0
        self._update_count = 0
        
    @property
    def is_connected(self) -> bool:
        """Check if currently connected to SpacetimeDB"""
        with self._state_lock:
            return self._connected and self.client and self.client.is_connected
    
    @property
    def is_subscribed(self) -> bool:
        """Check if subscribed to game tables"""
        with self._state_lock:
            return self._subscribed
    
    @property
    def player_identity(self) -> Optional[str]:
        """Get current player identity"""
        with self._state_lock:
            return self.game_state.player_identity
    
    @property
    def player_id(self) -> Optional[int]:
        """Get current player ID"""
        with self._state_lock:
            return self.game_state.player_id
    
    async def connect(self) -> None:
        """
        Establish connection to Blackholio game.
        
        Raises:
            ConnectionError: If connection fails after all retries
        """
        async with self._connection_lock:
            if self.is_connected:
                logger.info("Already connected to Blackholio")
                return
            
            logger.info(f"Connecting to Blackholio at {self.config.host}")
            logger.info(f"Database: {self.config.database}")
            
            # Create a minimal autogen module mock
            class MockAutogenModule:
                __path__ = []  # SDK expects this attribute
            
            autogen = MockAutogenModule()
            
            # Parse host URL similar to pygame client
            host = self.config.host
            if host.startswith("ws://"):
                # Convert ws:// to http:// for SDK
                host = "http://" + host[5:]
            elif host.startswith("wss://"):
                # Convert wss:// to https:// for SDK
                host = "https://" + host[6:]
            elif not host.startswith("http://") and not host.startswith("https://"):
                # Add http:// if no protocol specified
                host = "http://" + host
            
            # Initialize the client with v1.1.2 pattern
            try:
                # Get the current event loop
                self._loop = asyncio.get_event_loop()
                
                # Initialize client in a thread-safe way
                def init_client():
                    # Create client instance (protocol is hardcoded in SDK)
                    client = SpacetimeDBClient(autogen)
                    
                    # Connect with v1.1.2 support
                    # Note: The SDK WebSocketClient has been updated to support db_identity
                    client.connect(
                        auth_token=self.config.auth_token,
                        host=self.config.host,  # Use the original host without protocol prefix
                        address_or_name=self.config.database,
                        ssl_enabled=self.config.ssl_enabled,
                        on_connect=self._handle_connect,
                        on_disconnect=self._handle_disconnect,
                        on_identity=self._handle_identity,
                        on_error=self._handle_error
                    )
                    
                    # The WebSocketClient will use db_identity internally if the SDK supports it
                    # For now, we'll use the static init method as fallback
                    return client
                
                self.client = await self._loop.run_in_executor(self._executor, init_client)
                
                # Register table update handlers
                self._register_table_handlers()
                
                # Start update loop in background thread
                def update_loop():
                    while self._connected and self.client:
                        try:
                            self.client.update()
                            time.sleep(0.01)  # 10ms delay between updates
                        except Exception as e:
                            logger.error(f"Error in update loop: {e}")
                            break
                    logger.info("Update loop ended")
                
                self._update_thread = Thread(target=update_loop, daemon=True)
                self._update_thread.start()
                
                # Wait for connection
                await self._wait_for_connection()
                
                # Subscribe to tables
                await self._subscribe_to_tables()
                
                logger.info("Successfully connected and subscribed to Blackholio")
                
            except Exception as e:
                logger.error(f"Failed to connect to Blackholio: {e}")
                await self.disconnect()
                raise ConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Blackholio game"""
        async with self._connection_lock:
            if self._reconnect_task:
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task
                except asyncio.CancelledError:
                    pass
                self._reconnect_task = None
            
            if self.client:
                logger.info("Disconnecting from Blackholio")
                # Run disconnect in executor to avoid blocking
                if self._loop:
                    await self._loop.run_in_executor(self._executor, self.client.disconnect)
                self.client = None
            
            with self._state_lock:
                self._connected = False
                self._subscribed = False
                self.game_state = GameState()
    
    async def ensure_connected(self) -> None:
        """
        Ensure connection is established, reconnecting if necessary.
        
        This method is useful for training stability - it will automatically
        reconnect if the connection was lost.
        """
        if not self.is_connected:
            logger.info("Connection lost, attempting to reconnect...")
            await self.connect()
    
    async def call_reducer(self, reducer_name: str, *args, timeout: float = 5.0) -> Any:
        """
        Call a reducer on the Blackholio game.
        
        Args:
            reducer_name: Name of the reducer to call
            *args: Arguments for the reducer
            timeout: Timeout for the reducer call
            
        Returns:
            Result of the reducer call
            
        Raises:
            ConnectionError: If not connected
            TimeoutError: If reducer call times out
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Blackholio")
        
        try:
            # Use the legacy client reducer call in executor
            if self._loop:
                await self._loop.run_in_executor(
                    self._executor, 
                    self.client._reducer_call, 
                    reducer_name, 
                    *args
                )
            return True
        except Exception as e:
            logger.error(f"Error calling reducer '{reducer_name}': {e}")
            raise
    
    async def update_player_input(self, direction: List[float]) -> None:
        """
        Update player movement input.
        
        Args:
            direction: 2D movement vector [x, y]
        """
        await self.call_reducer("UpdatePlayerInput", direction)
    
    async def player_split(self) -> None:
        """Execute player split action"""
        await self.call_reducer("PlayerSplit")
    
    def add_connection_listener(self, callback: Callable[[], None]) -> None:
        """Add a callback for connection events"""
        self._on_connected_callbacks.append(callback)
    
    def add_disconnection_listener(self, callback: Callable[[str], None]) -> None:
        """Add a callback for disconnection events"""
        self._on_disconnected_callbacks.append(callback)
    
    def add_game_state_listener(self, callback: Callable[[GameState], None]) -> None:
        """Add a callback for game state updates"""
        self._on_game_state_update_callbacks.append(callback)
    
    def get_player_entities(self) -> List[Any]:
        """Get all entities owned by the current player"""
        with self._state_lock:
            if not self.game_state.player_id:
                return []
            
            player_entities = []
            for entity_id, circle in self.game_state.circles.items():
                if circle.get('player_id') == self.game_state.player_id:
                    entity = self.game_state.entities.get(entity_id)
                    if entity:
                        player_entities.append(entity)
            
            return player_entities
    
    def get_other_entities(self) -> List[Any]:
        """Get all entities not owned by the current player"""
        with self._state_lock:
            if not self.game_state.player_id:
                return list(self.game_state.entities.values())
            
            other_entities = []
            for entity_id, entity in self.game_state.entities.items():
                circle = self.game_state.circles.get(entity_id)
                if not circle or circle.get('player_id') != self.game_state.player_id:
                    other_entities.append(entity)
            
            return other_entities
    
    def get_update_rate(self) -> float:
        """Get current update rate in Hz"""
        if self._update_count < 2:
            return 0.0
        
        current_time = time.time()
        time_diff = current_time - self._last_update_time
        if time_diff > 0:
            return 1.0 / time_diff
        return 0.0
    
    # Private methods
    
    async def _wait_for_connection(self, timeout: float = 10.0) -> None:
        """Wait for connection to be established"""
        start_time = asyncio.get_event_loop().time()
        while not self.is_connected:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError("Connection timeout")
            await asyncio.sleep(0.1)
    
    async def _subscribe_to_tables(self) -> None:
        """Subscribe to all necessary game tables"""
        logger.info("Subscribing to game tables...")
        
        queries = [
            "SELECT * FROM Entity",
            "SELECT * FROM Player", 
            "SELECT * FROM Circle",
            "SELECT * FROM Food"
        ]
        
        # Subscribe in executor to avoid blocking
        if self._loop:
            await self._loop.run_in_executor(
                self._executor,
                self.client.subscribe,
                queries
            )
        
        # Give subscription time to process
        await asyncio.sleep(1.0)
        
        # Assume subscription is successful after subscribe call
        with self._state_lock:
            self._subscribed = True
        logger.info("Subscription request sent to game tables")
    
    async def _auto_reconnect(self) -> None:
        """Automatically reconnect on disconnection"""
        if not self.config.auto_reconnect:
            return
        
        attempt = 0
        while attempt < self.config.max_reconnect_attempts:
            attempt += 1
            logger.info(f"Reconnection attempt {attempt}/{self.config.max_reconnect_attempts}")
            
            try:
                await asyncio.sleep(self.config.reconnect_delay * attempt)
                await self.connect()
                logger.info("Successfully reconnected")
                return
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt} failed: {e}")
        
        logger.error("Max reconnection attempts reached, giving up")
    
    def _register_table_handlers(self) -> None:
        """Register table update handlers with SpacetimeDB client"""
        if not self.client:
            return
            
        # Entity table handler
        def entity_handler(op, old_value, new_value, reducer_event):
            with self._state_lock:
                if op == "insert" and new_value is not None:
                    data = extract_entity_data(new_value)
                    entity_id = data.get('entity_id')
                    if entity_id is not None:
                        self.game_state.entities[entity_id] = data
                elif op == "delete" and old_value is not None:
                    data = extract_entity_data(old_value)
                    entity_id = data.get('entity_id')
                    if entity_id is not None:
                        self.game_state.entities.pop(entity_id, None)
                elif op == "update" and new_value is not None:
                    data = extract_entity_data(new_value)
                    entity_id = data.get('entity_id')
                    if entity_id is not None:
                        self.game_state.entities[entity_id] = data
            self._notify_game_state_update()
        
        # Player table handler
        def player_handler(op, old_value, new_value, reducer_event):
            with self._state_lock:
                if op == "insert" and new_value is not None:
                    data = extract_player_data(new_value)
                    player_id = data.get('player_id')
                    if player_id is not None:
                        self.game_state.players[player_id] = data
                        # Check if this is our player
                        if str(data.get('identity')) == self.game_state.player_identity:
                            self.game_state.player_id = player_id
                            logger.info(f"Our player ID: {player_id}")
                elif op == "delete" and old_value is not None:
                    data = extract_player_data(old_value)
                    player_id = data.get('player_id')
                    if player_id is not None:
                        self.game_state.players.pop(player_id, None)
                elif op == "update" and new_value is not None:
                    data = extract_player_data(new_value)
                    player_id = data.get('player_id')
                    if player_id is not None:
                        self.game_state.players[player_id] = data
            self._notify_game_state_update()
        
        # Circle table handler
        def circle_handler(op, old_value, new_value, reducer_event):
            with self._state_lock:
                if op == "insert" and new_value is not None:
                    data = extract_circle_data(new_value)
                    entity_id = data.get('entity_id')
                    if entity_id is not None:
                        self.game_state.circles[entity_id] = data
                elif op == "delete" and old_value is not None:
                    data = extract_circle_data(old_value)
                    entity_id = data.get('entity_id')
                    if entity_id is not None:
                        self.game_state.circles.pop(entity_id, None)
                elif op == "update" and new_value is not None:
                    data = extract_circle_data(new_value)
                    entity_id = data.get('entity_id')
                    if entity_id is not None:
                        self.game_state.circles[entity_id] = data
            self._notify_game_state_update()
        
        # Food table handler
        def food_handler(op, old_value, new_value, reducer_event):
            # For simplicity, rebuild food list on any update
            self._notify_game_state_update()
        
        # Register handlers
        self.client._register_row_update("Entity", entity_handler)
        self.client._register_row_update("Player", player_handler)
        self.client._register_row_update("Circle", circle_handler)
        self.client._register_row_update("Food", food_handler)
        
        logger.debug("Registered all table update handlers")
    
    # SpacetimeDB callbacks
    
    def _handle_connect(self) -> None:
        """Handle connection established"""
        logger.info("Connected to SpacetimeDB")
        with self._state_lock:
            self._connected = True
        
        # Register any pending handlers
        if self._pending_handlers and self.client:
            for table_name, handler in self._pending_handlers:
                self.client._register_row_update(table_name, handler)
                logger.debug(f"Registered pending handler for table: {table_name}")
            self._pending_handlers.clear()
        
        for callback in self._on_connected_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")
    
    def _handle_disconnect(self, reason: str) -> None:
        """Handle disconnection"""
        logger.warning(f"Disconnected from SpacetimeDB: {reason}")
        with self._state_lock:
            self._connected = False
            self._subscribed = False
        
        for callback in self._on_disconnected_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Error in disconnection callback: {e}")
        
        # Schedule reconnection using thread-safe approach
        if self.config.auto_reconnect and not self._reconnect_task and self._loop:
            try:
                # Schedule the coroutine in the event loop
                self._reconnect_task = asyncio.run_coroutine_threadsafe(
                    self._auto_reconnect(), 
                    self._loop
                ).result()
            except Exception as e:
                logger.error(f"Failed to schedule reconnection: {e}")
    
    def _handle_identity(self, token: str, identity: Any, connection_id: Any) -> None:
        """Handle identity assignment"""
        logger.info(f"Received identity: {identity}")
        with self._state_lock:
            self.game_state.player_identity = str(identity)
    
    def _handle_error(self, error: str) -> None:
        """Handle error from SpacetimeDB"""
        logger.error(f"SpacetimeDB error: {error}")
    
    def _handle_subscription_applied(self) -> None:
        """Handle subscription confirmation"""
        logger.info("Subscription applied")
        with self._state_lock:
            self._subscribed = True
    
    def _notify_game_state_update(self) -> None:
        """Notify listeners of game state update"""
        with self._state_lock:
            self.game_state.timestamp = time.time()
            self._update_count += 1
            self._last_update_time = time.time()
        
        for callback in self._on_game_state_update_callbacks:
            try:
                callback(self.game_state)
            except Exception as e:
                logger.error(f"Error in game state callback: {e}")
