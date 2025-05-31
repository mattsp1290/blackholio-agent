"""
Pre-defined game state scenarios for testing.

This module provides various game scenarios that represent different
stages and situations in the Blackholio game.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class GameScenario:
    """Container for a complete game scenario."""
    name: str
    description: str
    entities: Dict[int, Dict[str, Any]]
    circles: Dict[int, Dict[str, Any]]
    players: Dict[int, Dict[str, Any]]
    expected_behavior: str


# --- Early Game Scenarios ---

EARLY_GAME_SOLO = GameScenario(
    name="early_game_solo",
    description="Player just spawned, only food nearby",
    entities={
        1: {
            "entity_id": 1,
            "owner_id": 100,
            "x": 500.0,
            "y": 400.0,
            "mass": 10.0,
            "radius": 5.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_at": 1000000
        },
        # Food entities
        **{i: {
            "entity_id": i,
            "owner_id": 0,
            "x": 500.0 + np.cos(i * 0.5) * (50 + i * 10),
            "y": 400.0 + np.sin(i * 0.5) * (50 + i * 10),
            "mass": 1.0,
            "radius": 3.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_at": 900000 + i
        } for i in range(2, 22)}
    },
    circles={
        1: {"entity_id": 1}
    },
    players={
        100: {"player_id": 100, "name": "TestPlayer", "score": 10, "alive": True}
    },
    expected_behavior="Should focus on collecting nearby food"
)


EARLY_GAME_WITH_THREAT = GameScenario(
    name="early_game_with_threat",
    description="Small player with larger enemy nearby",
    entities={
        1: {
            "entity_id": 1,
            "owner_id": 100,
            "x": 500.0,
            "y": 400.0,
            "mass": 15.0,
            "radius": 6.0,
            "velocity_x": 1.0,
            "velocity_y": 0.5,
            "created_at": 1000000
        },
        2: {  # Larger enemy
            "entity_id": 2,
            "owner_id": 101,
            "x": 550.0,
            "y": 420.0,
            "mass": 40.0,
            "radius": 10.0,
            "velocity_x": -2.0,
            "velocity_y": -1.0,
            "created_at": 999000
        },
        # Some food
        **{i: {
            "entity_id": i,
            "owner_id": 0,
            "x": 450.0 + (i-3) * 20,
            "y": 380.0 + (i-3) * 15,
            "mass": 1.0,
            "radius": 3.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_at": 900000 + i
        } for i in range(3, 8)}
    },
    circles={
        1: {"entity_id": 1},
        2: {"entity_id": 2}
    },
    players={
        100: {"player_id": 100, "name": "TestPlayer", "score": 15, "alive": True},
        101: {"player_id": 101, "name": "Enemy", "score": 40, "alive": True}
    },
    expected_behavior="Should flee from larger enemy while collecting safe food"
)


# --- Mid Game Scenarios ---

MID_GAME_MULTI_ENTITY = GameScenario(
    name="mid_game_multi_entity",
    description="Player has split into multiple entities",
    entities={
        1: {
            "entity_id": 1,
            "owner_id": 100,
            "x": 500.0,
            "y": 400.0,
            "mass": 35.0,
            "radius": 9.0,
            "velocity_x": 2.0,
            "velocity_y": 1.0,
            "created_at": 1000000
        },
        2: {
            "entity_id": 2,
            "owner_id": 100,
            "x": 520.0,
            "y": 410.0,
            "mass": 25.0,
            "radius": 7.5,
            "velocity_x": 1.5,
            "velocity_y": 1.2,
            "created_at": 1000100
        },
        # Smaller enemy
        3: {
            "entity_id": 3,
            "owner_id": 101,
            "x": 600.0,
            "y": 450.0,
            "mass": 20.0,
            "radius": 6.5,
            "velocity_x": -1.0,
            "velocity_y": -0.5,
            "created_at": 999500
        },
        # Larger enemy
        4: {
            "entity_id": 4,
            "owner_id": 102,
            "x": 400.0,
            "y": 350.0,
            "mass": 80.0,
            "radius": 15.0,
            "velocity_x": 1.0,
            "velocity_y": 1.0,
            "created_at": 998000
        },
        # Food
        **{i: {
            "entity_id": i,
            "owner_id": 0,
            "x": np.random.uniform(300, 700),
            "y": np.random.uniform(250, 550),
            "mass": 1.0,
            "radius": 3.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_at": 900000 + i
        } for i in range(5, 15)}
    },
    circles={
        1: {"entity_id": 1},
        2: {"entity_id": 2},
        3: {"entity_id": 3},
        4: {"entity_id": 4}
    },
    players={
        100: {"player_id": 100, "name": "TestPlayer", "score": 60, "alive": True},
        101: {"player_id": 101, "name": "SmallEnemy", "score": 20, "alive": True},
        102: {"player_id": 102, "name": "BigEnemy", "score": 80, "alive": True}
    },
    expected_behavior="Should hunt smaller enemy while avoiding larger one"
)


# --- Late Game Scenarios ---

LATE_GAME_DOMINANT = GameScenario(
    name="late_game_dominant",
    description="Player is one of the largest entities",
    entities={
        # Player entities (split into 3)
        1: {
            "entity_id": 1,
            "owner_id": 100,
            "x": 500.0,
            "y": 400.0,
            "mass": 150.0,
            "radius": 20.0,
            "velocity_x": 1.0,
            "velocity_y": 0.5,
            "created_at": 1000000
        },
        2: {
            "entity_id": 2,
            "owner_id": 100,
            "x": 530.0,
            "y": 420.0,
            "mass": 100.0,
            "radius": 16.0,
            "velocity_x": 1.2,
            "velocity_y": 0.7,
            "created_at": 1000100
        },
        3: {
            "entity_id": 3,
            "owner_id": 100,
            "x": 480.0,
            "y": 380.0,
            "mass": 80.0,
            "radius": 14.0,
            "velocity_x": 0.8,
            "velocity_y": 0.3,
            "created_at": 1000200
        },
        # Various enemies
        **{i: {
            "entity_id": i,
            "owner_id": 100 + i,
            "x": np.random.uniform(200, 800),
            "y": np.random.uniform(200, 600),
            "mass": np.random.uniform(10, 60),
            "radius": np.sqrt(np.random.uniform(10, 60)) * 2,
            "velocity_x": np.random.uniform(-2, 2),
            "velocity_y": np.random.uniform(-2, 2),
            "created_at": 990000 + i * 100
        } for i in range(4, 10)}
    },
    circles={i: {"entity_id": i} for i in range(1, 10)},
    players={
        100: {"player_id": 100, "name": "TestPlayer", "score": 330, "alive": True},
        **{100 + i: {
            "player_id": 100 + i,
            "name": f"Enemy{i}",
            "score": np.random.randint(10, 60),
            "alive": True
        } for i in range(4, 10)}
    },
    expected_behavior="Should aggressively hunt smaller players"
)


# --- Special Scenarios ---

SPLIT_DECISION_SCENARIO = GameScenario(
    name="split_decision",
    description="Situation where splitting might be beneficial",
    entities={
        1: {
            "entity_id": 1,
            "owner_id": 100,
            "x": 500.0,
            "y": 400.0,
            "mass": 80.0,
            "radius": 14.0,
            "velocity_x": 3.0,
            "velocity_y": 0.0,
            "created_at": 1000000
        },
        # Target running away
        2: {
            "entity_id": 2,
            "owner_id": 101,
            "x": 580.0,
            "y": 400.0,
            "mass": 30.0,
            "radius": 8.0,
            "velocity_x": 4.0,
            "velocity_y": 0.0,
            "created_at": 999000
        },
        # Threat from behind
        3: {
            "entity_id": 3,
            "owner_id": 102,
            "x": 400.0,
            "y": 400.0,
            "mass": 120.0,
            "radius": 18.0,
            "velocity_x": 2.5,
            "velocity_y": 0.0,
            "created_at": 998000
        }
    },
    circles={
        1: {"entity_id": 1},
        2: {"entity_id": 2},
        3: {"entity_id": 3}
    },
    players={
        100: {"player_id": 100, "name": "TestPlayer", "score": 80, "alive": True},
        101: {"player_id": 101, "name": "Target", "score": 30, "alive": True},
        102: {"player_id": 102, "name": "Threat", "score": 120, "alive": True}
    },
    expected_behavior="Should consider splitting to catch target while escaping threat"
)


CROWDED_AREA_SCENARIO = GameScenario(
    name="crowded_area",
    description="Many entities in a small area",
    entities={
        # Player
        1: {
            "entity_id": 1,
            "owner_id": 100,
            "x": 500.0,
            "y": 400.0,
            "mass": 45.0,
            "radius": 10.0,
            "velocity_x": 0.5,
            "velocity_y": 0.5,
            "created_at": 1000000
        },
        # Mix of entities around
        **{i: {
            "entity_id": i,
            "owner_id": 100 + (i % 5),
            "x": 500.0 + np.cos(i * 0.8) * (30 + i * 5),
            "y": 400.0 + np.sin(i * 0.8) * (30 + i * 5),
            "mass": 10.0 + i * 3 if i % 3 == 0 else 60.0 - i * 2,
            "radius": np.sqrt(10.0 + i * 3 if i % 3 == 0 else 60.0 - i * 2) * 1.5,
            "velocity_x": np.cos(i) * 2,
            "velocity_y": np.sin(i) * 2,
            "created_at": 990000 + i * 100
        } for i in range(2, 15)}
    },
    circles={i: {"entity_id": i} for i in range(1, 15)},
    players={
        100: {"player_id": 100, "name": "TestPlayer", "score": 45, "alive": True},
        **{100 + i: {
            "player_id": 100 + i,
            "name": f"Player{i}",
            "score": 30 + i * 10,
            "alive": True
        } for i in range(1, 5)}
    },
    expected_behavior="Should carefully navigate to eat smaller entities while avoiding larger ones"
)


# --- Helper Functions ---

def get_scenario(name: str) -> GameScenario:
    """Get a scenario by name."""
    scenarios = {
        "early_game_solo": EARLY_GAME_SOLO,
        "early_game_with_threat": EARLY_GAME_WITH_THREAT,
        "mid_game_multi_entity": MID_GAME_MULTI_ENTITY,
        "late_game_dominant": LATE_GAME_DOMINANT,
        "split_decision": SPLIT_DECISION_SCENARIO,
        "crowded_area": CROWDED_AREA_SCENARIO
    }
    return deepcopy(scenarios.get(name))


def create_custom_scenario(
    n_player_entities: int = 1,
    player_mass: float = 50.0,
    n_enemies: int = 5,
    enemy_mass_range: tuple = (10.0, 100.0),
    n_food: int = 20,
    arena_size: tuple = (1000, 800)
) -> GameScenario:
    """Create a custom game scenario."""
    entities = {}
    circles = {}
    entity_id = 1
    
    # Player entities
    for i in range(n_player_entities):
        mass = player_mass / (i + 1)  # Split mass among entities
        entities[entity_id] = {
            "entity_id": entity_id,
            "owner_id": 100,
            "x": arena_size[0] / 2 + i * 20,
            "y": arena_size[1] / 2 + i * 10,
            "mass": mass,
            "radius": np.sqrt(mass) * 2,
            "velocity_x": np.random.uniform(-1, 1),
            "velocity_y": np.random.uniform(-1, 1),
            "created_at": 1000000 + i
        }
        circles[entity_id] = {"entity_id": entity_id}
        entity_id += 1
    
    # Enemy entities
    for i in range(n_enemies):
        mass = np.random.uniform(*enemy_mass_range)
        entities[entity_id] = {
            "entity_id": entity_id,
            "owner_id": 101 + i,
            "x": np.random.uniform(50, arena_size[0] - 50),
            "y": np.random.uniform(50, arena_size[1] - 50),
            "mass": mass,
            "radius": np.sqrt(mass) * 2,
            "velocity_x": np.random.uniform(-2, 2),
            "velocity_y": np.random.uniform(-2, 2),
            "created_at": 990000 + i * 100
        }
        circles[entity_id] = {"entity_id": entity_id}
        entity_id += 1
    
    # Food entities
    for i in range(n_food):
        entities[entity_id] = {
            "entity_id": entity_id,
            "owner_id": 0,
            "x": np.random.uniform(10, arena_size[0] - 10),
            "y": np.random.uniform(10, arena_size[1] - 10),
            "mass": 1.0,
            "radius": 3.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "created_at": 900000 + i
        }
        entity_id += 1
    
    players = {
        100: {"player_id": 100, "name": "TestPlayer", "score": int(player_mass), "alive": True}
    }
    for i in range(n_enemies):
        players[101 + i] = {
            "player_id": 101 + i,
            "name": f"Enemy{i+1}",
            "score": int(np.random.uniform(*enemy_mass_range)),
            "alive": True
        }
    
    return GameScenario(
        name="custom",
        description="Custom generated scenario",
        entities=entities,
        circles=circles,
        players=players,
        expected_behavior="Depends on specific configuration"
    )


# Export all scenarios
# Add aliases for backward compatibility
SPLIT_DECISION = SPLIT_DECISION_SCENARIO
CROWDED_AREA = CROWDED_AREA_SCENARIO

__all__ = [
    "GameScenario",
    "EARLY_GAME_SOLO",
    "EARLY_GAME_WITH_THREAT", 
    "MID_GAME_MULTI_ENTITY",
    "LATE_GAME_DOMINANT",
    "SPLIT_DECISION_SCENARIO",
    "SPLIT_DECISION",  # Alias for backward compatibility
    "CROWDED_AREA_SCENARIO",
    "CROWDED_AREA",  # Alias for backward compatibility
    "get_scenario",
    "create_custom_scenario"
]
