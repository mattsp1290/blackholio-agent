"""
Pytest configuration and shared fixtures for Blackholio agent tests.

This module provides common test fixtures, mock objects, and utilities
used across the test suite.
"""

import pytest
import numpy as np
import asyncio
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
import tempfile
import shutil

# Import components to test
from ..environment import (
    ObservationSpace, ObservationConfig,
    ActionSpace, ActionConfig,
    RewardCalculator, RewardConfig,
    BlackholioConnection, ConnectionConfig
)
from ..models import BlackholioModel, BlackholioModelConfig


# --- Test Configuration ---

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# --- Mock SpacetimeDB Client ---

@pytest.fixture
def mock_spacetimedb_client():
    """Create a mock SpacetimeDB client."""
    client = AsyncMock()
    
    # Mock connection methods
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock()
    client.is_connected = True
    
    # Mock identity
    client.identity = "test_identity_12345"
    client.token = "test_token"
    
    # Mock table subscriptions
    client.subscribe = AsyncMock()
    
    # Mock reducer calls
    client.call_reducer = AsyncMock()
    
    # Mock callbacks
    client.on_connect = None
    client.on_disconnect = None
    client.on_error = None
    
    return client


# --- Game State Fixtures ---

@pytest.fixture
def sample_player_entity():
    """Create a sample player entity."""
    return {
        "entity_id": 1,
        "owner_id": 100,
        "x": 500.0,
        "y": 300.0,
        "mass": 50.0,
        "radius": 10.0,
        "velocity_x": 2.0,
        "velocity_y": -1.0,
        "created_at": 1000000
    }


@pytest.fixture
def sample_entities():
    """Create sample entity data."""
    return {
        1: {  # Player entity
            "entity_id": 1,
            "owner_id": 100,
            "x": 500.0,
            "y": 300.0,
            "mass": 50.0,
            "radius": 10.0,
            "velocity_x": 2.0,
            "velocity_y": -1.0,
            "created_at": 1000000
        },
        2: {  # Another player entity (same owner)
            "entity_id": 2,
            "owner_id": 100,
            "x": 520.0,
            "y": 310.0,
            "mass": 30.0,
            "radius": 8.0,
            "velocity_x": 1.5,
            "velocity_y": -0.5,
            "created_at": 1000100
        },
        3: {  # Enemy entity
            "entity_id": 3,
            "owner_id": 101,
            "x": 600.0,
            "y": 350.0,
            "mass": 80.0,
            "radius": 15.0,
            "velocity_x": -1.0,
            "velocity_y": 0.5,
            "created_at": 999000
        },
        4: {  # Food entity (no circle)
            "entity_id": 4,
            "owner_id": 0,
            "x": 450.0,
            "y": 280.0,
            "mass": 1.0,
            "radius": 3.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_at": 900000
        }
    }


@pytest.fixture
def sample_circles():
    """Create sample circle data (entities with mass > food threshold)."""
    return {
        1: {"entity_id": 1, "circle_data": "player1_circle"},
        2: {"entity_id": 2, "circle_data": "player1_circle2"},
        3: {"entity_id": 3, "circle_data": "enemy_circle"}
    }


@pytest.fixture
def sample_players():
    """Create sample player data."""
    return {
        100: {
            "player_id": 100,
            "name": "TestPlayer",
            "score": 80,
            "alive": True
        },
        101: {
            "player_id": 101,
            "name": "Enemy",
            "score": 120,
            "alive": True
        }
    }


@pytest.fixture
def game_state(sample_entities, sample_circles, sample_players):
    """Create a complete game state."""
    from ..environment.connection import GameState
    
    state = GameState()
    state.entities = sample_entities
    state.circles = sample_circles
    state.players = sample_players
    state.timestamp = 1234567890
    
    return state


# --- Component Fixtures ---

@pytest.fixture
def observation_space():
    """Create an ObservationSpace instance."""
    config = ObservationConfig()
    return ObservationSpace(config)


@pytest.fixture
def action_space():
    """Create an ActionSpace instance."""
    config = ActionConfig()
    return ActionSpace(config)


@pytest.fixture
def reward_calculator():
    """Create a RewardCalculator instance."""
    config = RewardConfig()
    return RewardCalculator(config)


@pytest.fixture
def mock_connection(mock_spacetimedb_client, game_state):
    """Create a mock BlackholioConnection."""
    connection = Mock(spec=BlackholioConnection)
    
    # Set basic attributes
    connection.client = mock_spacetimedb_client
    connection.is_connected = True
    connection.player_id = 100
    connection.player_identity = "test_identity_12345"
    
    # Mock game state
    connection.game_state = game_state
    
    # Mock methods
    connection.connect = AsyncMock(return_value=True)
    connection.disconnect = AsyncMock()
    connection.ensure_connected = AsyncMock()
    
    connection.get_player_entities = Mock(return_value=[
        sample_entities[1], sample_entities[2]
    ])
    connection.get_other_entities = Mock(return_value=[
        sample_entities[3]
    ])
    connection.get_food_entities = Mock(return_value=[
        sample_entities[4]
    ])
    
    connection.send_input = AsyncMock()
    connection.call_reducer = AsyncMock()
    
    connection.get_update_rate = Mock(return_value=20.0)
    connection.add_game_state_listener = Mock()
    connection.remove_game_state_listener = Mock()
    
    return connection


@pytest.fixture
def sample_observation(observation_space, game_state, mock_connection):
    """Create a sample observation array."""
    mock_connection.game_state = game_state
    player_entities = [game_state.entities[1], game_state.entities[2]]
    other_entities = [game_state.entities[3]]
    food_entities = [game_state.entities[4]]
    
    return observation_space.process_game_state(
        player_entities, other_entities, food_entities
    )


@pytest.fixture
def blackholio_model():
    """Create a BlackholioModel instance."""
    config = BlackholioModelConfig(
        device="cpu",  # Use CPU for tests
        hidden_size=128,  # Smaller for faster tests
        num_layers=2
    )
    return BlackholioModel(config)


# --- Utility Functions ---

def create_random_entities(n_entities: int, owner_id: int = 100) -> List[Dict[str, Any]]:
    """Create random entity data for testing."""
    entities = []
    for i in range(n_entities):
        entity = {
            "entity_id": i + 1,
            "owner_id": owner_id,
            "x": np.random.uniform(0, 1000),
            "y": np.random.uniform(0, 1000),
            "mass": np.random.uniform(10, 100),
            "radius": np.random.uniform(5, 20),
            "velocity_x": np.random.uniform(-5, 5),
            "velocity_y": np.random.uniform(-5, 5),
            "created_at": 1000000 + i
        }
        entities.append(entity)
    return entities


def create_game_scenario(
    n_player_entities: int = 2,
    n_enemy_entities: int = 5,
    n_food: int = 20,
    player_mass: float = 50.0,
    enemy_mass_range: tuple = (20.0, 150.0)
) -> Dict[str, Any]:
    """Create a complete game scenario for testing."""
    entities = {}
    circles = {}
    
    # Player entities
    for i in range(n_player_entities):
        entity_id = i + 1
        entities[entity_id] = {
            "entity_id": entity_id,
            "owner_id": 100,
            "x": 500.0 + i * 20,
            "y": 300.0 + i * 10,
            "mass": player_mass * (0.8 + i * 0.2),
            "radius": np.sqrt(player_mass) * 2,
            "velocity_x": np.random.uniform(-2, 2),
            "velocity_y": np.random.uniform(-2, 2),
            "created_at": 1000000 + i
        }
        circles[entity_id] = {"entity_id": entity_id}
    
    # Enemy entities
    for i in range(n_enemy_entities):
        entity_id = n_player_entities + i + 1
        mass = np.random.uniform(*enemy_mass_range)
        entities[entity_id] = {
            "entity_id": entity_id,
            "owner_id": 101 + i,
            "x": np.random.uniform(100, 900),
            "y": np.random.uniform(100, 700),
            "mass": mass,
            "radius": np.sqrt(mass) * 2,
            "velocity_x": np.random.uniform(-3, 3),
            "velocity_y": np.random.uniform(-3, 3),
            "created_at": 999000 - i * 100
        }
        circles[entity_id] = {"entity_id": entity_id}
    
    # Food entities
    food_start_id = n_player_entities + n_enemy_entities + 1
    for i in range(n_food):
        entity_id = food_start_id + i
        entities[entity_id] = {
            "entity_id": entity_id,
            "owner_id": 0,
            "x": np.random.uniform(0, 1000),
            "y": np.random.uniform(0, 800),
            "mass": 1.0,
            "radius": 3.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_at": 900000 + i
        }
    
    return {
        "entities": entities,
        "circles": circles,
        "players": {
            100: {"player_id": 100, "name": "TestPlayer", "score": 100, "alive": True}
        }
    }


# --- Async Test Helpers ---

async def advance_game_time(connection: Mock, timesteps: int = 10):
    """Simulate game time advancement."""
    for _ in range(timesteps):
        # Simulate game state updates
        if hasattr(connection, '_game_state_listeners'):
            for listener in connection._game_state_listeners:
                listener(connection.game_state)
        await asyncio.sleep(0.01)


# --- Performance Test Helpers ---

@pytest.fixture
def performance_timer():
    """Simple timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.times.append(self.end - self.start)
        
        @property
        def average(self):
            return sum(self.times) / len(self.times) if self.times else 0
        
        @property
        def last(self):
            return self.times[-1] if self.times else 0
    
    return Timer()
