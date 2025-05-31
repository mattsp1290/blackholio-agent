"""
Mock Blackholio connection for training when SpacetimeDB is unavailable.

This module provides a mock connection that simulates the Blackholio game
environment for training purposes when the actual SpacetimeDB server is
not accessible.
"""

import asyncio
import logging
import random
import time
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
import numpy as np

from .connection import ConnectionConfig, GameState, BlackholioConnection

logger = logging.getLogger(__name__)


class MockBlackholioConnection(BlackholioConnection):
    """
    Mock connection for training without SpacetimeDB.
    
    This class simulates game state updates and player actions to allow
    training to proceed even when the SpacetimeDB server is unavailable.
    """
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._mock_player_id = 1
        self._mock_entity_counter = 1
        self._simulation_task = None
        logger.info("Using mock Blackholio connection for training")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to mock environment."""
        return self._connected
    
    async def connect(self) -> None:
        """Simulate connection to Blackholio game."""
        async with self._connection_lock:
            if self.is_connected:
                logger.info("Already connected to mock Blackholio")
                # If already connected but no player entities, recreate them
                if not self.get_player_entities():
                    logger.info("Recreating player entity after reset")
                    self._create_player_entity()
                return
            
            logger.info("Connecting to mock Blackholio environment")
            
            # Simulate connection delay
            await asyncio.sleep(0.1)
            
            # Set up mock game state
            self._connected = True
            self.game_state.player_identity = "mock_player_identity"
            self.game_state.player_id = self._mock_player_id
            
            # Initialize with some mock entities
            self._initialize_mock_game_state()
            
            # Start simulation loop
            self._simulation_task = asyncio.create_task(self._simulation_loop())
            
            # Notify connection listeners
            for callback in self._on_connected_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in connection callback: {e}")
            
            # Mark as subscribed
            self._subscribed = True
            
            logger.info("Successfully connected to mock Blackholio")
    
    async def disconnect(self) -> None:
        """Disconnect from mock Blackholio game."""
        async with self._connection_lock:
            if self._simulation_task:
                self._simulation_task.cancel()
                try:
                    await self._simulation_task
                except asyncio.CancelledError:
                    pass
                self._simulation_task = None
            
            self._connected = False
            self._subscribed = False
            self.game_state = GameState()
            
            logger.info("Disconnected from mock Blackholio")
    
    async def call_reducer(self, reducer_name: str, *args, timeout: float = 5.0) -> Any:
        """Simulate reducer calls."""
        if not self.is_connected:
            raise ConnectionError("Not connected to mock Blackholio")
        
        # Simulate network delay
        await asyncio.sleep(0.01)
        
        # Handle specific reducers
        if reducer_name == "UpdatePlayerInput":
            # Simulate player movement
            if args and len(args) > 0:
                direction = args[0]
                self._update_player_position(direction)
        
        elif reducer_name == "PlayerSplit":
            # Simulate split action
            self._simulate_player_split()
        
        return True
    
    def _create_player_entity(self):
        """Create or recreate the player entity."""
        # Create player entity
        player_entity_id = self._get_next_entity_id()
        player_entity = {
            'entity_id': player_entity_id,
            'position': {'x': 0.0, 'y': 0.0},
            'velocity': {'x': 0.0, 'y': 0.0},
            'mass': 50.0,
            'radius': 5.0
        }
        self.game_state.entities[player_entity_id] = player_entity
        
        # Create player circle
        self.game_state.circles[player_entity_id] = {
            'entity_id': player_entity_id,
            'player_id': self._mock_player_id,
            'color': {'r': 255, 'g': 0, 'b': 0}
        }
        
        # Create or update player data
        self.game_state.players[self._mock_player_id] = {
            'player_id': self._mock_player_id,
            'identity': self.game_state.player_identity,
            'name': 'MockPlayer',
            'score': self.game_state.players.get(self._mock_player_id, {}).get('score', 0),
            'kills': self.game_state.players.get(self._mock_player_id, {}).get('kills', 0),
            'deaths': self.game_state.players.get(self._mock_player_id, {}).get('deaths', 0)
        }
        
        # Notify state update
        self._notify_game_state_update()
        
    def _initialize_mock_game_state(self):
        """Initialize mock game state with some entities."""
        # Create player entity
        self._create_player_entity()
        
        # Create some AI entities
        for i in range(10):
            entity_id = self._get_next_entity_id()
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            mass = random.uniform(10, 100)
            
            entity = {
                'entity_id': entity_id,
                'position': {'x': x, 'y': y},
                'velocity': {'x': random.uniform(-1, 1), 'y': random.uniform(-1, 1)},
                'mass': mass,
                'radius': np.sqrt(mass)
            }
            self.game_state.entities[entity_id] = entity
            
            # Some are circles (other players)
            if i < 5:
                self.game_state.circles[entity_id] = {
                    'entity_id': entity_id,
                    'player_id': 1000 + i,  # Mock AI player IDs
                    'color': {'r': random.randint(0, 255), 
                             'g': random.randint(0, 255), 
                             'b': random.randint(0, 255)}
                }
        
        # Create some food entities
        for i in range(20):
            x = random.uniform(-200, 200)
            y = random.uniform(-200, 200)
            self.game_state.food.append({
                'position': {'x': x, 'y': y},
                'mass': 1.0
            })
        
        # Notify initial state
        self._notify_game_state_update()
    
    def _get_next_entity_id(self) -> int:
        """Get next available entity ID."""
        entity_id = self._mock_entity_counter
        self._mock_entity_counter += 1
        return entity_id
    
    def _update_player_position(self, direction: List[float]):
        """Update player position based on input."""
        player_entities = self.get_player_entities()
        if not player_entities:
            return
        
        for entity in player_entities:
            # Update velocity based on direction
            if len(direction) >= 2:
                entity['velocity']['x'] = direction[0] * 10.0  # Scale for movement speed
                entity['velocity']['y'] = direction[1] * 10.0
    
    def _simulate_player_split(self):
        """Simulate player split action."""
        player_entities = self.get_player_entities()
        if not player_entities or len(player_entities) >= 8:  # Max 8 splits
            return
        
        # Split the largest entity
        largest = max(player_entities, key=lambda e: e['mass'])
        if largest['mass'] < 20:  # Min mass to split
            return
        
        # Create new entity
        new_id = self._get_next_entity_id()
        new_entity = {
            'entity_id': new_id,
            'position': {
                'x': largest['position']['x'] + random.uniform(-5, 5),
                'y': largest['position']['y'] + random.uniform(-5, 5)
            },
            'velocity': {
                'x': random.uniform(-5, 5),
                'y': random.uniform(-5, 5)
            },
            'mass': largest['mass'] / 2,
            'radius': np.sqrt(largest['mass'] / 2)
        }
        
        # Update original entity mass
        largest['mass'] /= 2
        largest['radius'] = np.sqrt(largest['mass'])
        
        # Add to game state
        self.game_state.entities[new_id] = new_entity
        self.game_state.circles[new_id] = {
            'entity_id': new_id,
            'player_id': self._mock_player_id,
            'color': {'r': 255, 'g': 0, 'b': 0}
        }
    
    async def _simulation_loop(self):
        """Simulate game physics and state updates."""
        try:
            while self._connected:
                # Update entity positions
                for entity in self.game_state.entities.values():
                    entity['position']['x'] += entity['velocity']['x'] * 0.1
                    entity['position']['y'] += entity['velocity']['y'] * 0.1
                    
                    # Apply friction
                    entity['velocity']['x'] *= 0.95
                    entity['velocity']['y'] *= 0.95
                    
                    # Boundary wrap
                    if abs(entity['position']['x']) > 300:
                        entity['position']['x'] = -entity['position']['x']
                    if abs(entity['position']['y']) > 300:
                        entity['position']['y'] = -entity['position']['y']
                
                # Simulate some entity interactions (eating)
                self._simulate_interactions()
                
                # Respawn food occasionally
                if random.random() < 0.1 and len(self.game_state.food) < 50:
                    self.game_state.food.append({
                        'position': {
                            'x': random.uniform(-200, 200),
                            'y': random.uniform(-200, 200)
                        },
                        'mass': 1.0
                    })
                
                # Notify update
                self._notify_game_state_update()
                
                # Simulate at ~30 FPS
                await asyncio.sleep(0.033)
                
        except asyncio.CancelledError:
            logger.info("Simulation loop cancelled")
    
    def _simulate_interactions(self):
        """Simulate entity interactions (eating, collisions)."""
        entities = list(self.game_state.entities.items())
        
        for i, (id1, e1) in enumerate(entities):
            if id1 not in self.game_state.entities:  # Already consumed
                continue
                
            for j, (id2, e2) in enumerate(entities[i+1:], i+1):
                if id2 not in self.game_state.entities:  # Already consumed
                    continue
                
                # Calculate distance
                dx = e1['position']['x'] - e2['position']['x']
                dy = e1['position']['y'] - e2['position']['y']
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Check collision
                if dist < (e1['radius'] + e2['radius']) * 0.8:
                    # Larger eats smaller
                    if e1['mass'] > e2['mass'] * 1.1:
                        e1['mass'] += e2['mass'] * 0.8
                        e1['radius'] = np.sqrt(e1['mass'])
                        del self.game_state.entities[id2]
                        if id2 in self.game_state.circles:
                            del self.game_state.circles[id2]
                    elif e2['mass'] > e1['mass'] * 1.1:
                        e2['mass'] += e1['mass'] * 0.8
                        e2['radius'] = np.sqrt(e2['mass'])
                        del self.game_state.entities[id1]
                        if id1 in self.game_state.circles:
                            del self.game_state.circles[id1]
                        break
        
        # Player eating food
        player_entities = self.get_player_entities()
        if player_entities:
            new_food = []
            for food in self.game_state.food:
                eaten = False
                for entity in player_entities:
                    dx = entity['position']['x'] - food['position']['x']
                    dy = entity['position']['y'] - food['position']['y']
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < entity['radius']:
                        entity['mass'] += food['mass']
                        entity['radius'] = np.sqrt(entity['mass'])
                        eaten = True
                        break
                
                if not eaten:
                    new_food.append(food)
            
            self.game_state.food = new_food
