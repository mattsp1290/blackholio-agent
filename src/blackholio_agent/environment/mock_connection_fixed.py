"""
Fixed Mock Blackholio Connection for testing and development.
Handles event loops properly across multiple threads.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from threading import Thread, Lock, Event as ThreadingEvent
import weakref
from queue import Queue, Empty
import concurrent.futures

from .connection import BlackholioConnection, ConnectionConfig, GameState
from .mock_connection import Entity, Player, Food, Circle


logger = logging.getLogger(__name__)


class MockEventLoopManager:
    """Manages a single shared event loop for all mock connections."""
    
    _instance = None
    _lock = Lock()
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _thread: Optional[Thread] = None
    _executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._start()
        return cls._instance
    
    def _start(self):
        """Start the event loop in a separate thread."""
        self._loop = asyncio.new_event_loop()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._loop.set_default_executor(self._executor)
        
        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        # Wait for loop to start
        time.sleep(0.1)
        logger.info("Mock event loop manager started")
    
    def run_coroutine(self, coro):
        """Run a coroutine in the managed event loop."""
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("Event loop not running")
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future
    
    def create_task(self, coro):
        """Create a task in the managed event loop."""
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("Event loop not running")
        
        future = asyncio.run_coroutine_threadsafe(
            self._create_task_internal(coro), 
            self._loop
        )
        return future.result()
    
    async def _create_task_internal(self, coro):
        """Internal method to create task."""
        return asyncio.create_task(coro)
    
    def shutdown(self):
        """Shutdown the event loop manager."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            if self._executor:
                self._executor.shutdown(wait=True)
        logger.info("Mock event loop manager shutdown")


@dataclass
class MockGameSession:
    """Represents a mock game session with multiple players."""
    
    session_id: str
    world_size: int = 1000
    entities: Dict[int, Entity] = field(default_factory=dict)
    players: Dict[int, Player] = field(default_factory=dict)
    circles: Dict[int, Circle] = field(default_factory=dict)
    food: Dict[int, Food] = field(default_factory=dict)
    next_entity_id: int = 1
    next_player_id: int = 1
    lock: Lock = field(default_factory=Lock)
    
    def add_player(self, name: str) -> Player:
        """Add a new player to the session."""
        with self.lock:
            player_id = self.next_player_id
            self.next_player_id += 1
            
            player = Player(
                player_id=player_id,
                name=name,
                identity="mock_identity_" + str(player_id)
            )
            self.players[player_id] = player
            
            # Spawn initial entity for player
            entity_id = self.next_entity_id
            self.next_entity_id += 1
            
            # Random spawn position
            x = random.uniform(100, self.world_size - 100)
            y = random.uniform(100, self.world_size - 100)
            
            entity = Entity(
                entity_id=entity_id,
                x=x,
                y=y,
                mass=100
            )
            self.entities[entity_id] = entity
            
            # Create circle for the entity
            circle = Circle(
                entity_id=entity_id,
                player_id=player_id,
                speed=5.0,
                direction_x=0.0,
                direction_y=0.0,
                last_split_time=0
            )
            self.circles[entity_id] = circle
            
            # Spawn some food
            for _ in range(20):
                self._spawn_food()
            
            return player
    
    def _spawn_food(self):
        """Spawn a food entity."""
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        
        x = random.uniform(50, self.world_size - 50)
        y = random.uniform(50, self.world_size - 50)
        
        entity = Entity(
            entity_id=entity_id,
            x=x,
            y=y,
            mass=10
        )
        self.entities[entity_id] = entity
        
        food = Food(entity_id=entity_id)
        self.food[entity_id] = food
    
    def get_game_state(self, player_id: int) -> GameState:
        """Get current game state for a player."""
        with self.lock:
            # Get player's entities
            player_entity_ids = [
                circle.entity_id 
                for circle in self.circles.values() 
                if circle.player_id == player_id
            ]
            
            # Get all entities
            all_entities = list(self.entities.values())
            
            # Get all players
            all_players = list(self.players.values())
            
            # Get all circles
            all_circles = list(self.circles.values())
            
            # Get all food
            all_food = list(self.food.values())
            
            return GameState(
                player_id=player_id,
                player_entity_ids=player_entity_ids,
                entities=all_entities,
                players=all_players,
                circles=all_circles,
                food=all_food,
                config={'world_size': self.world_size}
            )


class MockConnectionPool:
    """Manages a pool of mock game sessions."""
    
    _instance = None
    _lock = Lock()
    sessions: Dict[str, MockGameSession] = {}
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_or_create_session(self, session_id: str = "default") -> MockGameSession:
        """Get or create a game session."""
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = MockGameSession(session_id)
            return self.sessions[session_id]


class FixedMockBlackholioConnection(BlackholioConnection):
    """Fixed mock connection with proper event loop handling."""
    
    def __init__(self, config: ConnectionConfig):
        """Initialize mock connection."""
        super().__init__(config)
        self.connected = False
        self.player: Optional[Player] = None
        self.session: Optional[MockGameSession] = None
        self._simulation_task: Optional[asyncio.Task] = None
        self._stop_event = ThreadingEvent()
        self._state_lock = Lock()
        self._update_queue = Queue(maxsize=100)
        self._loop_manager = MockEventLoopManager.get_instance()
        self._connection_pool = MockConnectionPool.get_instance()
        
        logger.info("Fixed mock Blackholio connection initialized")
    
    async def connect(self) -> bool:
        """Connect to mock environment."""
        try:
            # Get or create session
            session_id = self.config.auth_token or "default"
            self.session = self._connection_pool.get_or_create_session(session_id)
            
            with self._state_lock:
                self.connected = True
            
            # Start simulation in managed event loop
            self._simulation_task = self._loop_manager.create_task(
                self._simulation_loop()
            )
            
            logger.info(f"Connected to mock session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Mock connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from mock environment."""
        self._stop_event.set()
        
        # Cancel simulation task if running
        if self._simulation_task and not self._simulation_task.done():
            self._simulation_task.cancel()
            try:
                # Use the loop manager to wait for cancellation
                await self._loop_manager.run_coroutine(
                    self._wait_for_task_cancellation(self._simulation_task)
                ).result()
            except asyncio.CancelledError:
                pass
        
        with self._state_lock:
            self.connected = False
            self.player = None
        
        logger.info("Disconnected from mock environment")
    
    async def _wait_for_task_cancellation(self, task):
        """Helper to wait for task cancellation."""
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    async def _simulation_loop(self):
        """Run the game simulation."""
        logger.info("Mock simulation started")
        
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)  # 10 FPS simulation
                
                if self.player and self.session:
                    # Process any pending updates
                    self._process_updates()
                    
                    # Simulate game physics
                    self._simulate_physics()
        
        except asyncio.CancelledError:
            logger.info("Simulation loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            logger.info("Mock simulation stopped")
    
    def _process_updates(self):
        """Process queued updates."""
        try:
            while True:
                update = self._update_queue.get_nowait()
                if update['type'] == 'move':
                    self._update_player_movement(
                        update['direction_x'], 
                        update['direction_y']
                    )
                elif update['type'] == 'split':
                    self._handle_split()
                elif update['type'] == 'respawn':
                    self._handle_respawn()
        except Empty:
            pass
    
    def _simulate_physics(self):
        """Simulate basic game physics."""
        with self.session.lock:
            # Move player entities
            for circle in self.session.circles.values():
                if circle.player_id == self.player.player_id:
                    entity = self.session.entities.get(circle.entity_id)
                    if entity:
                        # Update position based on direction
                        entity.x += circle.direction_x * circle.speed * 0.1
                        entity.y += circle.direction_y * circle.speed * 0.1
                        
                        # Keep in bounds
                        entity.x = max(50, min(self.session.world_size - 50, entity.x))
                        entity.y = max(50, min(self.session.world_size - 50, entity.y))
                        
                        # Simulate mass increase
                        if random.random() < 0.05:  # 5% chance per frame
                            entity.mass = min(entity.mass + 1, 1000)
    
    def _update_player_movement(self, direction_x: float, direction_y: float):
        """Update player movement direction."""
        with self.session.lock:
            for circle in self.session.circles.values():
                if circle.player_id == self.player.player_id:
                    circle.direction_x = direction_x
                    circle.direction_y = direction_y
    
    def _handle_split(self):
        """Handle player split action."""
        # TODO: Implement split logic
        logger.debug("Split action received (not implemented)")
    
    def _handle_respawn(self):
        """Handle player respawn."""
        with self.session.lock:
            # Remove old entities
            to_remove = []
            for entity_id, circle in self.session.circles.items():
                if circle.player_id == self.player.player_id:
                    to_remove.append(entity_id)
            
            for entity_id in to_remove:
                self.session.entities.pop(entity_id, None)
                self.session.circles.pop(entity_id, None)
            
            # Create new entity
            entity_id = self.session.next_entity_id
            self.session.next_entity_id += 1
            
            x = random.uniform(100, self.session.world_size - 100)
            y = random.uniform(100, self.session.world_size - 100)
            
            entity = Entity(
                entity_id=entity_id,
                x=x,
                y=y,
                mass=100
            )
            self.session.entities[entity_id] = entity
            
            circle = Circle(
                entity_id=entity_id,
                player_id=self.player.player_id,
                speed=5.0,
                direction_x=0.0,
                direction_y=0.0,
                last_split_time=0
            )
            self.session.circles[entity_id] = circle
    
    async def enter_game(self, player_name: str) -> bool:
        """Enter the game as a player."""
        if not self.connected or not self.session:
            return False
        
        try:
            # Add player to session
            self.player = self.session.add_player(player_name)
            logger.info(f"Player {player_name} entered game with ID {self.player.player_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enter game: {e}")
            return False
    
    async def update_player_input(self, direction_x: float, direction_y: float) -> bool:
        """Update player input."""
        if not self.connected or not self.player:
            return False
        
        try:
            self._update_queue.put_nowait({
                'type': 'move',
                'direction_x': direction_x,
                'direction_y': direction_y
            })
            return True
        except:
            return False
    
    async def player_split(self) -> bool:
        """Split player cells."""
        if not self.connected or not self.player:
            return False
        
        try:
            self._update_queue.put_nowait({'type': 'split'})
            return True
        except:
            return False
    
    async def respawn(self) -> bool:
        """Respawn player."""
        if not self.connected or not self.player:
            return False
        
        try:
            self._update_queue.put_nowait({'type': 'respawn'})
            return True
        except:
            return False
    
    def get_game_state(self) -> Optional[GameState]:
        """Get current game state."""
        if not self.connected or not self.player or not self.session:
            return None
        
        return self.session.get_game_state(self.player.player_id)
    
    def is_connected(self) -> bool:
        """Check if connected."""
        with self._state_lock:
            return self.connected
