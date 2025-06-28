"""
Improved Mock Blackholio Connection for testing and development.
Properly extends BlackholioConnection with thread-safe event loop handling.
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Optional, Any
from threading import Lock
from dataclasses import dataclass

from .connection import ConnectionConfig, GameState

logger = logging.getLogger(__name__)


@dataclass
class MockEntity:
    """Mock entity data structure"""
    entity_id: int
    x: float
    y: float
    mass: float
    
    def to_dict(self):
        return {
            'entity_id': self.entity_id,
            'x': self.x,
            'y': self.y,
            'mass': self.mass
        }


@dataclass 
class MockPlayer:
    """Mock player data structure"""
    player_id: int
    identity: str
    name: str
    
    def to_dict(self):
        return {
            'player_id': self.player_id,
            'identity': self.identity,
            'name': self.name
        }


@dataclass
class MockCircle:
    """Mock circle data structure"""
    entity_id: int
    player_id: int
    direction_x: float
    direction_y: float
    speed: float
    last_split_time: int
    
    def to_dict(self):
        return {
            'entity_id': self.entity_id,
            'player_id': self.player_id,
            'direction_x': self.direction_x,
            'direction_y': self.direction_y,
            'speed': self.speed,
            'last_split_time': self.last_split_time
        }


@dataclass
class MockFood:
    """Mock food data structure"""
    entity_id: int
    
    def to_dict(self):
        return {
            'entity_id': self.entity_id
        }


class ImprovedMockBlackholioConnection:
    """Improved mock connection that implements the BlackholioConnection interface."""
    
    def __init__(self, config: ConnectionConfig):
        """Initialize mock connection."""
        self.config = config
        self._connected = False
        
        # Mock-specific state
        self._mock_world_size = 1000
        self._mock_entities: Dict[int, MockEntity] = {}
        self._mock_players: Dict[int, MockPlayer] = {}
        self._mock_circles: Dict[int, MockCircle] = {}
        self._mock_food: Dict[int, MockFood] = {}
        self._next_entity_id = 1
        self._next_player_id = 1
        self._simulation_task: Optional[asyncio.Task] = None
        self._mock_lock = Lock()
        
        logger.info("Improved mock Blackholio connection initialized")
    
    @property
    def is_connected(self) -> bool:
        """Override to check mock connection state."""
        with self._state_lock:
            return self._connected
    
    async def connect(self) -> None:
        """
        Override connect to provide mock behavior.
        """
        async with self._connection_lock:
            if self.is_connected:
                logger.info("Already connected to mock Blackholio")
                return
            
            logger.info("Connecting to mock Blackholio environment")
            
            # Set connected state
            with self._state_lock:
                self._connected = True
                self.game_state.player_identity = f"mock_identity_{random.randint(1000, 9999)}"
            
            # Spawn initial food
            await self._spawn_initial_world()
            
            # Start simulation
            self._simulation_task = asyncio.create_task(self._simulation_loop())
            
            # Notify connection listeners
            for callback in self._on_connected_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in connection callback: {e}")
            
            # Mark as subscribed since we're mocking
            with self._state_lock:
                self._subscribed = True
            
            logger.info("Successfully connected to mock Blackholio")
    
    async def disconnect(self) -> None:
        """Override disconnect for mock behavior."""
        async with self._connection_lock:
            if self._simulation_task:
                self._simulation_task.cancel()
                try:
                    await self._simulation_task
                except asyncio.CancelledError:
                    pass
                self._simulation_task = None
            
            with self._state_lock:
                self._connected = False
                self._subscribed = False
                self.game_state = GameState()
            
            # Clear mock state
            with self._mock_lock:
                self._mock_entities.clear()
                self._mock_players.clear()
                self._mock_circles.clear()
                self._mock_food.clear()
            
            # Notify disconnection listeners
            for callback in self._on_disconnected_callbacks:
                try:
                    callback("Mock disconnect")
                except Exception as e:
                    logger.error(f"Error in disconnection callback: {e}")
            
            logger.info("Disconnected from mock Blackholio")
    
    async def call_reducer(self, reducer_name: str, *args, timeout: float = 5.0) -> Any:
        """Override reducer calls for mock behavior."""
        # For mock, we just check the local connected state
        if not self._connected:
            raise ConnectionError("Not connected to mock Blackholio")
        
        logger.debug(f"Mock reducer call: {reducer_name} with args: {args}")
        
        # Handle specific reducers
        if reducer_name == "EnterGame":
            return await self._mock_enter_game(args[0] if args else "MockPlayer")
        elif reducer_name == "UpdatePlayerInput":
            return await self._mock_update_input(args[0] if args else [0, 0])
        elif reducer_name == "PlayerSplit":
            return await self._mock_player_split()
        elif reducer_name == "Respawn":
            return await self._mock_respawn()
        
        return True
    
    async def ensure_connected(self) -> None:
        """Ensure connection - mock always succeeds."""
        if not self.is_connected:
            await self.connect()
    
    async def _spawn_initial_world(self):
        """Spawn initial food in the world."""
        with self._mock_lock:
            # Spawn 50 food entities
            for _ in range(50):
                entity_id = self._next_entity_id
                self._next_entity_id += 1
                
                x = random.uniform(50, self._mock_world_size - 50)
                y = random.uniform(50, self._mock_world_size - 50)
                
                entity = MockEntity(
                    entity_id=entity_id,
                    x=x,
                    y=y,
                    mass=10
                )
                self._mock_entities[entity_id] = entity
                self._mock_food[entity_id] = MockFood(entity_id=entity_id)
    
    async def _mock_enter_game(self, player_name: str) -> bool:
        """Mock entering the game."""
        with self._mock_lock:
            # Create player
            player_id = self._next_player_id
            self._next_player_id += 1
            
            player = MockPlayer(
                player_id=player_id,
                identity=self.game_state.player_identity,
                name=player_name
            )
            self._mock_players[player_id] = player
            
            # Update game state
            with self._state_lock:
                self.game_state.player_id = player_id
            
            # Spawn player entity
            entity_id = self._next_entity_id
            self._next_entity_id += 1
            
            x = random.uniform(100, self._mock_world_size - 100)
            y = random.uniform(100, self._mock_world_size - 100)
            
            entity = MockEntity(
                entity_id=entity_id,
                x=x,
                y=y,
                mass=100
            )
            self._mock_entities[entity_id] = entity
            
            # Create circle for player
            circle = MockCircle(
                entity_id=entity_id,
                player_id=player_id,
                direction_x=0.0,
                direction_y=0.0,
                speed=5.0,
                last_split_time=0
            )
            self._mock_circles[entity_id] = circle
        
        await self._update_game_state()
        logger.info(f"Player {player_name} entered mock game with ID {player_id}")
        return True
    
    async def _mock_update_input(self, direction: List[float]) -> bool:
        """Mock updating player input."""
        if not self.game_state.player_id:
            return False
        
        with self._mock_lock:
            # Update direction for all player circles
            for circle in self._mock_circles.values():
                if circle.player_id == self.game_state.player_id:
                    circle.direction_x = direction[0] if len(direction) > 0 else 0
                    circle.direction_y = direction[1] if len(direction) > 1 else 0
        
        return True
    
    async def _mock_player_split(self) -> bool:
        """Mock player split action."""
        # TODO: Implement split logic
        logger.debug("Mock split action (not implemented)")
        return True
    
    async def _mock_respawn(self) -> bool:
        """Mock player respawn."""
        if not self.game_state.player_id:
            return False
        
        with self._mock_lock:
            # Remove old player entities
            to_remove = []
            for entity_id, circle in list(self._mock_circles.items()):
                if circle.player_id == self.game_state.player_id:
                    to_remove.append(entity_id)
            
            for entity_id in to_remove:
                self._mock_entities.pop(entity_id, None)
                self._mock_circles.pop(entity_id, None)
            
            # Create new entity
            entity_id = self._next_entity_id
            self._next_entity_id += 1
            
            x = random.uniform(100, self._mock_world_size - 100)
            y = random.uniform(100, self._mock_world_size - 100)
            
            entity = MockEntity(
                entity_id=entity_id,
                x=x,
                y=y,
                mass=100
            )
            self._mock_entities[entity_id] = entity
            
            circle = MockCircle(
                entity_id=entity_id,
                player_id=self.game_state.player_id,
                direction_x=0.0,
                direction_y=0.0,
                speed=5.0,
                last_split_time=0
            )
            self._mock_circles[entity_id] = circle
        
        await self._update_game_state()
        return True
    
    async def _simulation_loop(self):
        """Run the mock game simulation."""
        logger.info("Mock simulation started")
        
        try:
            while self.is_connected:
                await asyncio.sleep(0.05)  # 20 FPS
                
                # Simulate physics
                await self._simulate_physics()
                
                # Update game state
                await self._update_game_state()
                
                # Occasionally spawn new food
                if random.random() < 0.02:  # 2% chance per frame
                    await self._spawn_food()
        
        except asyncio.CancelledError:
            logger.info("Mock simulation cancelled")
            raise
        except Exception as e:
            logger.error(f"Mock simulation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("Mock simulation stopped")
    
    async def _simulate_physics(self):
        """Simulate basic game physics."""
        with self._mock_lock:
            # Move entities based on their circles
            for circle in list(self._mock_circles.values()):
                entity = self._mock_entities.get(circle.entity_id)
                if entity:
                    # Update position
                    entity.x += circle.direction_x * circle.speed
                    entity.y += circle.direction_y * circle.speed
                    
                    # Keep in bounds
                    entity.x = max(50, min(self._mock_world_size - 50, entity.x))
                    entity.y = max(50, min(self._mock_world_size - 50, entity.y))
                    
                    # Check food collision for player entities
                    if circle.player_id == self.game_state.player_id:
                        eaten_food = []
                        for food_id, food in self._mock_food.items():
                            food_entity = self._mock_entities.get(food_id)
                            if food_entity:
                                dist = ((entity.x - food_entity.x) ** 2 + 
                                       (entity.y - food_entity.y) ** 2) ** 0.5
                                if dist < entity.mass ** 0.5 + 10:  # Simple collision
                                    eaten_food.append(food_id)
                                    entity.mass += food_entity.mass
                        
                        # Remove eaten food
                        for food_id in eaten_food:
                            self._mock_entities.pop(food_id, None)
                            self._mock_food.pop(food_id, None)
    
    async def _spawn_food(self):
        """Spawn a new food entity."""
        with self._mock_lock:
            entity_id = self._next_entity_id
            self._next_entity_id += 1
            
            x = random.uniform(50, self._mock_world_size - 50)
            y = random.uniform(50, self._mock_world_size - 50)
            
            entity = MockEntity(
                entity_id=entity_id,
                x=x,
                y=y,
                mass=10
            )
            self._mock_entities[entity_id] = entity
            self._mock_food[entity_id] = MockFood(entity_id=entity_id)
    
    async def _update_game_state(self):
        """Update the game state and notify listeners."""
        with self._mock_lock:
            # Convert mock objects to dicts for game state
            entities_dict = {e.entity_id: e.to_dict() for e in self._mock_entities.values()}
            players_dict = {p.player_id: p.to_dict() for p in self._mock_players.values()}
            circles_dict = {c.entity_id: c.to_dict() for c in self._mock_circles.values()}
            food_list = [f.to_dict() for f in self._mock_food.values()]
            
            with self._state_lock:
                self.game_state.entities = entities_dict
                self.game_state.players = players_dict
                self.game_state.circles = circles_dict
                self.game_state.food = food_list
                self.game_state.timestamp = time.time()
        
        # Notify listeners
        self._notify_game_state_update()
