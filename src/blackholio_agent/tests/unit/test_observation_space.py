"""
Unit tests for ObservationSpace component.

Tests feature extraction, normalization, and observation building
from game states.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from ...environment import ObservationSpace, ObservationConfig
from ..fixtures.game_states import (
    EARLY_GAME_SOLO,
    EARLY_GAME_WITH_THREAT,
    MID_GAME_MULTI_ENTITY,
    LATE_GAME_DOMINANT,
    create_custom_scenario
)


class TestObservationSpace:
    """Test suite for ObservationSpace class."""
    
    def test_initialization(self):
        """Test ObservationSpace initialization with default config."""
        config = ObservationConfig()
        obs_space = ObservationSpace(config)
        
        assert obs_space.config == config
        assert obs_space.shape == (456,)  # 6 + 50*5 + 100*2
        assert obs_space.player_state_dim == 6
        assert obs_space.entity_feature_dim == 5
        assert obs_space.food_feature_dim == 2
    
    def test_custom_configuration(self):
        """Test ObservationSpace with custom configuration."""
        config = ObservationConfig(
            max_entities=30,
            max_food=50,
            normalize_positions=False,
            normalize_mass=False
        )
        obs_space = ObservationSpace(config)
        
        assert obs_space.config.max_entities == 30
        assert obs_space.config.max_food == 50
        assert obs_space.shape == (6 + 30*5 + 50*2,)  # 256
    
    def test_empty_game_state(self):
        """Test observation generation with no entities."""
        obs_space = ObservationSpace()
        
        obs = obs_space.process_game_state([], [], [])
        
        assert obs.shape == (456,)
        assert np.all(obs == 0.0)
    
    def test_single_player_entity(self):
        """Test observation with single player entity."""
        obs_space = ObservationSpace()
        
        player_entity = {
            "entity_id": 1,
            "x": 500.0,
            "y": 400.0,
            "mass": 50.0,
            "radius": 10.0,
            "velocity_x": 2.0,
            "velocity_y": -1.0
        }
        
        obs = obs_space.process_game_state([player_entity], [], [])
        
        # Check player state features
        assert obs[0] == 500.0 / 1000.0  # Normalized x
        assert obs[1] == 400.0 / 800.0   # Normalized y
        assert obs[2] == 50.0 / 500.0    # Normalized mass
        assert obs[3] == 1                # Number of circles
        assert obs[4] == 2.0 / 10.0      # Normalized velocity x
        assert obs[5] == -1.0 / 10.0     # Normalized velocity y
        
        # Other features should be zero
        assert np.all(obs[6:] == 0.0)
    
    def test_multiple_player_entities(self):
        """Test observation with multiple player entities."""
        obs_space = ObservationSpace()
        
        player_entities = [
            {"entity_id": 1, "x": 500.0, "y": 400.0, "mass": 30.0, 
             "radius": 8.0, "velocity_x": 2.0, "velocity_y": 1.0},
            {"entity_id": 2, "x": 520.0, "y": 410.0, "mass": 20.0,
             "radius": 6.0, "velocity_x": 1.5, "velocity_y": 0.5}
        ]
        
        obs = obs_space.process_game_state(player_entities, [], [])
        
        # Check aggregated player state
        total_mass = 50.0
        center_x = (500.0 * 30 + 520.0 * 20) / total_mass
        center_y = (400.0 * 30 + 410.0 * 20) / total_mass
        avg_vx = (2.0 * 30 + 1.5 * 20) / total_mass
        avg_vy = (1.0 * 30 + 0.5 * 20) / total_mass
        
        assert np.isclose(obs[0], center_x / 1000.0)
        assert np.isclose(obs[1], center_y / 800.0)
        assert np.isclose(obs[2], total_mass / 500.0)
        assert obs[3] == 2  # Number of circles
        assert np.isclose(obs[4], avg_vx / 10.0)
        assert np.isclose(obs[5], avg_vy / 10.0)
    
    def test_entity_features(self):
        """Test entity feature extraction and sorting."""
        obs_space = ObservationSpace()
        
        player = {"entity_id": 1, "x": 500.0, "y": 400.0, "mass": 50.0,
                  "radius": 10.0, "velocity_x": 0.0, "velocity_y": 0.0}
        
        # Create entities at different distances
        other_entities = [
            {"entity_id": 2, "x": 600.0, "y": 400.0, "mass": 30.0,  # 100 units away
             "radius": 8.0, "velocity_x": -1.0, "velocity_y": 0.0},
            {"entity_id": 3, "x": 550.0, "y": 400.0, "mass": 80.0,  # 50 units away
             "radius": 15.0, "velocity_x": 0.5, "velocity_y": 0.5},
        ]
        
        obs = obs_space.process_game_state([player], other_entities, [])
        
        # Entities should be sorted by distance
        entity_start = 6
        
        # First entity should be the closer one (id=3)
        assert np.isclose(obs[entity_start], (550.0 - 500.0) / 1000.0)  # rel_x
        assert np.isclose(obs[entity_start + 1], 0.0)  # rel_y
        assert np.isclose(obs[entity_start + 2], 80.0 / 500.0)  # mass
        
        # Second entity (id=2)
        assert np.isclose(obs[entity_start + 5], (600.0 - 500.0) / 1000.0)  # rel_x
        assert np.isclose(obs[entity_start + 6], 0.0)  # rel_y
        assert np.isclose(obs[entity_start + 7], 30.0 / 500.0)  # mass
    
    def test_food_features(self):
        """Test food feature extraction."""
        obs_space = ObservationSpace()
        
        player = {"entity_id": 1, "x": 500.0, "y": 400.0, "mass": 50.0,
                  "radius": 10.0, "velocity_x": 0.0, "velocity_y": 0.0}
        
        food_entities = [
            {"entity_id": 4, "x": 520.0, "y": 400.0, "mass": 1.0, "radius": 3.0},
            {"entity_id": 5, "x": 500.0, "y": 420.0, "mass": 1.0, "radius": 3.0},
        ]
        
        obs = obs_space.process_game_state([player], [], food_entities)
        
        # Food features start after player state and entity features
        food_start = 6 + 50 * 5  # 256
        
        # Food should be sorted by distance
        # First food (y=420, 20 units away)
        assert np.isclose(obs[food_start], 0.0)  # rel_x
        assert np.isclose(obs[food_start + 1], 20.0 / 800.0)  # rel_y
        
        # Second food (x=520, 20 units away but computed after)
        assert np.isclose(obs[food_start + 2], 20.0 / 1000.0)  # rel_x
        assert np.isclose(obs[food_start + 3], 0.0)  # rel_y
    
    def test_max_entities_truncation(self):
        """Test that observations are truncated to max entities."""
        config = ObservationConfig(max_entities=5, max_food=5)
        obs_space = ObservationSpace(config)
        
        player = {"entity_id": 1, "x": 500.0, "y": 400.0, "mass": 50.0,
                  "radius": 10.0, "velocity_x": 0.0, "velocity_y": 0.0}
        
        # Create more entities than max
        other_entities = [
            {"entity_id": i, "x": 500.0 + i*10, "y": 400.0, "mass": 20.0,
             "radius": 6.0, "velocity_x": 0.0, "velocity_y": 0.0}
            for i in range(2, 12)  # 10 entities
        ]
        
        food_entities = [
            {"entity_id": i, "x": 500.0, "y": 400.0 + i*10, "mass": 1.0, "radius": 3.0}
            for i in range(12, 22)  # 10 food
        ]
        
        obs = obs_space.process_game_state([player], other_entities, food_entities)
        
        # Check shape matches config
        expected_shape = 6 + 5 * 5 + 5 * 2  # 41
        assert obs.shape == (expected_shape,)
        
        # Check that closest entities are kept
        entity_start = 6
        # First entity should be closest (id=2, 10 units away)
        assert np.isclose(obs[entity_start], 10.0 / 1000.0)
    
    def test_normalization_disabled(self):
        """Test observation generation without normalization."""
        config = ObservationConfig(
            normalize_positions=False,
            normalize_mass=False,
            normalize_velocities=False
        )
        obs_space = ObservationSpace(config)
        
        player = {"entity_id": 1, "x": 500.0, "y": 400.0, "mass": 50.0,
                  "radius": 10.0, "velocity_x": 2.0, "velocity_y": -1.0}
        
        obs = obs_space.process_game_state([player], [], [])
        
        # Values should not be normalized
        assert obs[0] == 500.0  # x position
        assert obs[1] == 400.0  # y position
        assert obs[2] == 50.0   # mass
        assert obs[4] == 2.0    # velocity x
        assert obs[5] == -1.0   # velocity y
    
    def test_game_scenario_observations(self):
        """Test observations from predefined game scenarios."""
        obs_space = ObservationSpace()
        
        # Test early game scenario
        scenario = EARLY_GAME_SOLO
        player_entities = [e for e in scenario.entities.values() if e["owner_id"] == 100]
        other_entities = [e for e in scenario.entities.values() 
                         if e["owner_id"] not in [0, 100]]
        food_entities = [e for e in scenario.entities.values() if e["owner_id"] == 0]
        
        obs = obs_space.process_game_state(player_entities, other_entities, food_entities)
        
        assert obs.shape == (456,)
        assert obs[3] == len(player_entities)  # Number of player circles
        
        # Test that food is detected
        food_start = 6 + 50 * 5
        food_features = obs[food_start:food_start + len(food_entities) * 2]
        assert not np.all(food_features == 0)
    
    def test_relative_positions(self):
        """Test relative position calculations."""
        obs_space = ObservationSpace()
        
        # Player at origin
        player = {"entity_id": 1, "x": 0.0, "y": 0.0, "mass": 50.0,
                  "radius": 10.0, "velocity_x": 0.0, "velocity_y": 0.0}
        
        # Entity at (100, 100)
        other = {"entity_id": 2, "x": 100.0, "y": 100.0, "mass": 30.0,
                 "radius": 8.0, "velocity_x": 0.0, "velocity_y": 0.0}
        
        obs = obs_space.process_game_state([player], [other], [])
        
        # Check relative position
        entity_start = 6
        assert np.isclose(obs[entity_start], 100.0 / 1000.0)      # rel_x
        assert np.isclose(obs[entity_start + 1], 100.0 / 800.0)   # rel_y
    
    def test_observation_determinism(self):
        """Test that observations are deterministic."""
        obs_space = ObservationSpace()
        
        # Create a complex scenario
        scenario = create_custom_scenario(
            n_player_entities=3,
            n_enemies=10,
            n_food=20
        )
        
        player_entities = [e for e in scenario.entities.values() if e["owner_id"] == 100]
        other_entities = [e for e in scenario.entities.values() 
                         if e["owner_id"] not in [0, 100]]
        food_entities = [e for e in scenario.entities.values() if e["owner_id"] == 0]
        
        # Generate observation multiple times
        obs1 = obs_space.process_game_state(player_entities, other_entities, food_entities)
        obs2 = obs_space.process_game_state(player_entities, other_entities, food_entities)
        obs3 = obs_space.process_game_state(player_entities, other_entities, food_entities)
        
        # All observations should be identical
        assert np.array_equal(obs1, obs2)
        assert np.array_equal(obs2, obs3)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        obs_space = ObservationSpace()
        
        # Test with None inputs
        with pytest.raises((TypeError, AttributeError)):
            obs_space.process_game_state(None, None, None)
        
        # Test with empty lists
        obs = obs_space.process_game_state([], [], [])
        assert obs.shape == (456,)
        assert np.all(obs == 0.0)
        
        # Test with very large values
        player = {"entity_id": 1, "x": 10000.0, "y": 10000.0, "mass": 10000.0,
                  "radius": 100.0, "velocity_x": 100.0, "velocity_y": 100.0}
        
        obs = obs_space.process_game_state([player], [], [])
        
        # Normalized values should be capped
        assert obs[0] <= 1.0  # x position
        assert obs[1] <= 1.0  # y position
        
        # Test with negative positions
        player = {"entity_id": 1, "x": -100.0, "y": -100.0, "mass": 50.0,
                  "radius": 10.0, "velocity_x": 0.0, "velocity_y": 0.0}
        
        obs = obs_space.process_game_state([player], [], [])
        assert obs[0] >= 0.0  # Normalized x should be >= 0
        assert obs[1] >= 0.0  # Normalized y should be >= 0


@pytest.mark.parametrize("scenario_name", [
    "early_game_solo",
    "early_game_with_threat",
    "mid_game_multi_entity",
    "late_game_dominant",
    "split_decision",
    "crowded_area"
])
def test_all_scenarios(scenario_name):
    """Test observation generation for all predefined scenarios."""
    from ..fixtures.game_states import get_scenario
    
    obs_space = ObservationSpace()
    scenario = get_scenario(scenario_name)
    
    player_entities = [e for e in scenario.entities.values() if e["owner_id"] == 100]
    other_entities = [e for e in scenario.entities.values() 
                     if e["owner_id"] not in [0, 100]]
    food_entities = [e for e in scenario.entities.values() if e["owner_id"] == 0]
    
    obs = obs_space.process_game_state(player_entities, other_entities, food_entities)
    
    # Basic checks
    assert obs.shape == (456,)
    assert obs.dtype == np.float32
    assert not np.any(np.isnan(obs))
    assert not np.any(np.isinf(obs))
    
    # Check value ranges for normalized features
    if obs_space.config.normalize_positions:
        assert np.all(obs[:2] >= 0.0) and np.all(obs[:2] <= 1.0)
    
    if obs_space.config.normalize_mass:
        assert obs[2] >= 0.0  # Mass should be non-negative
