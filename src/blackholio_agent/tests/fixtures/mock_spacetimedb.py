"""
Mock SpacetimeDB client for testing without a real server.

This module provides a realistic mock of the SpacetimeDB client that
simulates game state updates, network delays, and connection behavior.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Set
from unittest.mock import AsyncMock, Mock
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from ...environment.connection import GameState


@dataclass
class MockConfig:
    """Configuration for mock behavior."""
    update_rate: float = 20.0  # Hz
    network_delay: float = 0.01  # seconds
    connection_failure_rate: float = 0.0  # 0-1 probability
    simulate_game_physics: bool = True
    arena_size: tuple = (1000, 800)


class MockSpacetimeDBClient:
    """
    Mock SpacetimeDB client that simulates a real game server.
    
    Features:
    - Simulates game state updates at configurable rate
    - Tracks entity movements and collisions
    - Handles player actions (movement, splitting)
    - Simulates network delays and failures
    """
    
    def __init__(self, config: Optional[MockConfig] = None):
        self.config = config or MockConfig()
        
        # Connection state
        self.is_connected = False
        self.identity = f"mock_identity_{int(time.time())}"
        self.token = f"mock_token_{int(time.time())}"
        
        # Game state
        self.game_state = GameState()
        self.player_id = 100
        self.next_entity_id = 1000
        
        # Callbacks
        self.on_connect: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.table_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Update loop
        self._update_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_update = time.time()
        
        # Player input
        self._player_input = {"x": 0.0, "y": 0.0}
        self._pending_split = False
        
        # Mock methods
        self.connect = AsyncMock(side_effect=self._connect)
        self.disconnect = AsyncMock(side_effect=self._disconnect)
        self.subscribe = AsyncMock(side_effect=self._subscribe)
        self.call_reducer = AsyncMock(side_effect=self._call_reducer)
    
    async def _connect(self, url: str, auth_token: Optional[str] = None):
        """Simulate connection to server."""
        # Simulate network delay
        await asyncio.sleep(self.config.network_delay)
        
        # Simulate connection failure
        if np.random.random() < self.config.connection_failure_rate:
            self.is_connected = False
            if self.on_error:
                self.on_error(Exception("Mock connection failed"))
            raise ConnectionError("Mock connection failed")
        
        self.is_connected = True
        
        # Initialize game state
        self._initialize_game_state()
        
        # Start update loop
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        
        # Call connection callback
        if self.on_connect:
            self.on_connect(self.identity, self.token)
        
        return True
    
    async def _disconnect(self):
        """Simulate disconnection from server."""
        self.is_connected = False
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self.on_disconnect:
            self.on_disconnect()
    
    async def _subscribe(self, table: str):
        """Simulate table subscription."""
        await asyncio.sleep(self.config.network_delay)
        
        # Send initial data for subscribed table
        if table == "Entity" and "Entity" in self.table_callbacks:
            for entity in self.game_state.entities.values():
                for callback in self.table_callbacks["Entity"]:
                    callback("insert", entity, None)
        
        elif table == "Circle" and "Circle" in self.table_callbacks:
            for circle in self.game_state.circles.values():
                for callback in self.table_callbacks["Circle"]:
                    callback("insert", circle, None)
        
        elif table == "Player" and "Player" in self.table_callbacks:
            for player in self.game_state.players.values():
                for callback in self.table_callbacks["Player"]:
                    callback("insert", player, None)
    
    async def _call_reducer(self, reducer_name: str, *args):
        """Simulate reducer calls."""
        await asyncio.sleep(self.config.network_delay)
        
        if reducer_name == "SendPlayerInput":
            if len(args) >= 2:
                self._player_input = {"x": args[0], "y": args[1]}
        
        elif reducer_name == "Split":
            self._pending_split = True
    
    def on_table_update(self, table: str, callback: Callable):
        """Register callback for table updates."""
        self.table_callbacks[table].append(callback)
    
    def _initialize_game_state(self):
        """Initialize game state with player entity."""
        # Create player
        self.game_state.players[self.player_id] = {
            "player_id": self.player_id,
            "name": "MockPlayer",
            "score": 10,
            "alive": True
        }
        
        # Create initial player entity
        entity_id = self._get_next_entity_id()
        self.game_state.entities[entity_id] = {
            "entity_id": entity_id,
            "owner_id": self.player_id,
            "x": self.config.arena_size[0] / 2,
            "y": self.config.arena_size[1] / 2,
            "mass": 10.0,
            "radius": 5.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_at": int(time.time() * 1000)
        }
        self.game_state.circles[entity_id] = {"entity_id": entity_id}
        
        # Add some food
        for i in range(20):
            food_id = self._get_next_entity_id()
            self.game_state.entities[food_id] = {
                "entity_id": food_id,
                "owner_id": 0,
                "x": np.random.uniform(50, self.config.arena_size[0] - 50),
                "y": np.random.uniform(50, self.config.arena_size[1] - 50),
                "mass": 1.0,
                "radius": 3.0,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
                "created_at": int(time.time() * 1000) - 10000
            }
        
        # Add a few other players
        for i in range(3):
            other_player_id = 200 + i
            self.game_state.players[other_player_id] = {
                "player_id": other_player_id,
                "name": f"Bot{i}",
                "score": np.random.randint(5, 50),
                "alive": True
            }
            
            entity_id = self._get_next_entity_id()
            self.game_state.entities[entity_id] = {
                "entity_id": entity_id,
                "owner_id": other_player_id,
                "x": np.random.uniform(100, self.config.arena_size[0] - 100),
                "y": np.random.uniform(100, self.config.arena_size[1] - 100),
                "mass": np.random.uniform(8, 40),
                "radius": np.random.uniform(4, 10),
                "velocity_x": np.random.uniform(-2, 2),
                "velocity_y": np.random.uniform(-2, 2),
                "created_at": int(time.time() * 1000) - 5000
            }
            self.game_state.circles[entity_id] = {"entity_id": entity_id}
    
    async def _update_loop(self):
        """Main update loop that simulates game physics."""
        while self._running:
            try:
                current_time = time.time()
                dt = current_time - self._last_update
                self._last_update = current_time
                
                if self.config.simulate_game_physics:
                    self._update_physics(dt)
                
                # Handle split
                if self._pending_split:
                    self._handle_split()
                    self._pending_split = False
                
                # Update timestamp
                self.game_state.timestamp = int(current_time * 1000)
                
                # Sleep to maintain update rate
                sleep_time = max(0, (1.0 / self.config.update_rate) - dt)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
    
    def _update_physics(self, dt: float):
        """Update entity positions and handle collisions."""
        entities_to_remove = set()
        entities_to_update = {}
        
        for entity_id, entity in self.game_state.entities.items():
            if entity_id in entities_to_remove:
                continue
            
            # Update player entity velocity based on input
            if entity["owner_id"] == self.player_id:
                speed = 5.0 / np.sqrt(entity["mass"])  # Speed decreases with mass
                entity["velocity_x"] = self._player_input["x"] * speed
                entity["velocity_y"] = self._player_input["y"] * speed
            
            # Update position
            new_x = entity["x"] + entity["velocity_x"] * dt * 50
            new_y = entity["y"] + entity["velocity_y"] * dt * 50
            
            # Boundary collision
            new_x = np.clip(new_x, entity["radius"], self.config.arena_size[0] - entity["radius"])
            new_y = np.clip(new_y, entity["radius"], self.config.arena_size[1] - entity["radius"])
            
            # Check collisions with other entities
            for other_id, other in self.game_state.entities.items():
                if other_id == entity_id or other_id in entities_to_remove:
                    continue
                
                dx = new_x - other["x"]
                dy = new_y - other["y"]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Collision detection
                if dist < entity["radius"] + other["radius"]:
                    # Can entity eat other?
                    if entity["mass"] > other["mass"] * 1.25:  # 25% larger
                        entity["mass"] += other["mass"] * 0.8  # 80% efficiency
                        entity["radius"] = np.sqrt(entity["mass"]) * 2
                        entities_to_remove.add(other_id)
                        
                        # Notify callbacks
                        self._notify_entity_update("delete", other)
                        if other_id in self.game_state.circles:
                            self._notify_circle_update("delete", self.game_state.circles[other_id])
            
            # Update entity
            entity["x"] = new_x
            entity["y"] = new_y
            entities_to_update[entity_id] = entity
        
        # Remove eaten entities
        for entity_id in entities_to_remove:
            if entity_id in self.game_state.entities:
                del self.game_state.entities[entity_id]
            if entity_id in self.game_state.circles:
                del self.game_state.circles[entity_id]
        
        # Notify updates
        for entity in entities_to_update.values():
            self._notify_entity_update("update", entity)
        
        # Spawn new food occasionally
        if np.random.random() < 0.1 and len(self.game_state.entities) < 100:
            self._spawn_food()
    
    def _handle_split(self):
        """Handle player split action."""
        player_entities = [e for e in self.game_state.entities.values() 
                          if e["owner_id"] == self.player_id and e["mass"] > 20]
        
        for entity in player_entities:
            if entity["mass"] > 20:
                # Create new entity
                new_id = self._get_next_entity_id()
                new_entity = {
                    "entity_id": new_id,
                    "owner_id": self.player_id,
                    "x": entity["x"] + entity["velocity_x"] * 5,
                    "y": entity["y"] + entity["velocity_y"] * 5,
                    "mass": entity["mass"] / 2,
                    "radius": np.sqrt(entity["mass"] / 2) * 2,
                    "velocity_x": entity["velocity_x"] * 2,
                    "velocity_y": entity["velocity_y"] * 2,
                    "created_at": int(time.time() * 1000)
                }
                
                # Update original entity
                entity["mass"] = entity["mass"] / 2
                entity["radius"] = np.sqrt(entity["mass"]) * 2
                
                # Add new entity
                self.game_state.entities[new_id] = new_entity
                self.game_state.circles[new_id] = {"entity_id": new_id}
                
                # Notify callbacks
                self._notify_entity_update("insert", new_entity)
                self._notify_entity_update("update", entity)
                self._notify_circle_update("insert", self.game_state.circles[new_id])
    
    def _spawn_food(self):
        """Spawn a new food entity."""
        food_id = self._get_next_entity_id()
        food = {
            "entity_id": food_id,
            "owner_id": 0,
            "x": np.random.uniform(10, self.config.arena_size[0] - 10),
            "y": np.random.uniform(10, self.config.arena_size[1] - 10),
            "mass": 1.0,
            "radius": 3.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_at": int(time.time() * 1000)
        }
        
        self.game_state.entities[food_id] = food
        self._notify_entity_update("insert", food)
    
    def _get_next_entity_id(self) -> int:
        """Get next available entity ID."""
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        return entity_id
    
    def _notify_entity_update(self, operation: str, entity: Dict[str, Any]):
        """Notify entity table callbacks."""
        for callback in self.table_callbacks.get("Entity", []):
            old_value = self.game_state.entities.get(entity["entity_id"]) if operation == "update" else None
            callback(operation, entity, old_value)
    
    def _notify_circle_update(self, operation: str, circle: Dict[str, Any]):
        """Notify circle table callbacks."""
        for callback in self.table_callbacks.get("Circle", []):
            old_value = self.game_state.circles.get(circle["entity_id"]) if operation == "update" else None
            callback(operation, circle, old_value)
    
    def set_game_scenario(self, scenario):
        """Load a predefined game scenario."""
        self.game_state.entities = scenario.entities.copy()
        self.game_state.circles = scenario.circles.copy()
        self.game_state.players = scenario.players.copy()
        self.game_state.timestamp = int(time.time() * 1000)
        
        # Notify all callbacks about the new state
        for entity in self.game_state.entities.values():
            self._notify_entity_update("insert", entity)
        
        for circle in self.game_state.circles.values():
            self._notify_circle_update("insert", circle)


def create_mock_client(scenario=None, config: Optional[MockConfig] = None) -> MockSpacetimeDBClient:
    """
    Create a mock SpacetimeDB client with optional scenario.
    
    Args:
        scenario: Optional GameScenario to load
        config: Optional MockConfig for behavior customization
        
    Returns:
        Configured mock client
    """
    client = Mockcreate_game_client()
    
    if scenario:
        # Connect first to initialize
        asyncio.create_task(client.connect())
        # Then load scenario
        client.set_game_scenario(scenario)
    
    return client


__all__ = [
    "MockSpacetimeDBClient",
    "MockConfig",
    "create_mock_client"
]
