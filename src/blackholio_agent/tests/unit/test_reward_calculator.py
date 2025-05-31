"""
Unit tests for RewardCalculator component.

Tests reward calculation, curriculum learning stages, and episode statistics.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from ...environment import RewardCalculator, RewardConfig
from ..fixtures.game_states import (
    EARLY_GAME_SOLO,
    EARLY_GAME_WITH_THREAT,
    MID_GAME_MULTI_ENTITY,
    LATE_GAME_DOMINANT
)


class TestRewardCalculator:
    """Test suite for RewardCalculator class."""
    
    def test_initialization(self):
        """Test RewardCalculator initialization with default config."""
        config = RewardConfig()
        calculator = RewardCalculator(config)
        
        assert calculator.config == config
        assert calculator.current_stage == 0
        assert calculator.episode_stats.steps == 0
        assert calculator.episode_stats.total_reward == 0.0
    
    def test_custom_configuration(self):
        """Test RewardCalculator with custom configuration."""
        config = RewardConfig(
            mass_gain_weight=2.0,
            survival_weight=0.5,
            kill_weight=20.0,
            death_penalty=-50.0,
            use_curriculum=False
        )
        calculator = RewardCalculator(config)
        
        assert calculator.config.mass_gain_weight == 2.0
        assert calculator.config.survival_weight == 0.5
        assert calculator.config.kill_weight == 20.0
        assert calculator.config.death_penalty == -50.0
        assert not calculator.config.use_curriculum
    
    def test_reset(self):
        """Test resetting episode statistics."""
        calculator = RewardCalculator()
        
        # Add some stats
        calculator.episode_stats.steps = 100
        calculator.episode_stats.total_reward = 50.0
        calculator.episode_stats.total_mass_gained = 40.0
        calculator.episode_stats.kills = 2
        calculator.previous_mass = 80.0
        
        # Reset
        stats = calculator.reset()
        
        # Verify stats were returned before reset
        assert stats.steps == 100
        assert stats.total_reward == 50.0
        
        # Verify reset
        assert calculator.episode_stats.steps == 0
        assert calculator.episode_stats.total_reward == 0.0
        assert calculator.previous_mass == 0.0
    
    def test_mass_gain_reward(self):
        """Test mass gain reward calculation."""
        calculator = RewardCalculator()
        calculator.previous_mass = 50.0
        
        # Create game state with mass gain
        game_state = {
            'player_entities': [
                {'mass': 30.0},
                {'mass': 30.0}
            ],
            'other_entities': [],
            'food_entities': []
        }
        
        reward, components = calculator.calculate_step_reward(game_state, {}, {})
        
        # Mass increased from 50 to 60
        expected_mass_reward = 10.0 * calculator.config.mass_gain_weight
        assert components['mass_gain'] == expected_mass_reward
        assert calculator.previous_mass == 60.0
    
    def test_mass_loss_penalty(self):
        """Test penalty for mass loss."""
        calculator = RewardCalculator()
        calculator.previous_mass = 100.0
        
        # Create game state with mass loss (e.g., after split)
        game_state = {
            'player_entities': [
                {'mass': 40.0},
                {'mass': 35.0}
            ],
            'other_entities': [],
            'food_entities': []
        }
        
        reward, components = calculator.calculate_step_reward(game_state, {}, {})
        
        # Mass decreased from 100 to 75
        expected_mass_reward = -25.0 * calculator.config.mass_gain_weight
        assert components['mass_gain'] == expected_mass_reward
        assert calculator.previous_mass == 75.0
    
    def test_survival_reward(self):
        """Test survival time reward."""
        config = RewardConfig(survival_weight=0.1)
        calculator = RewardCalculator(config)
        
        game_state = {
            'player_entities': [{'mass': 50.0}],
            'other_entities': [],
            'food_entities': []
        }
        
        # Calculate reward for 5 steps
        for _ in range(5):
            reward, components = calculator.calculate_step_reward(game_state, {}, {})
            assert components['survival'] == 0.1
        
        assert calculator.episode_stats.steps == 5
    
    def test_kill_reward(self):
        """Test reward for killing other players."""
        config = RewardConfig(kill_weight=10.0)
        calculator = RewardCalculator(config)
        
        game_state = {
            'player_entities': [{'mass': 80.0}],
            'other_entities': [],
            'food_entities': []
        }
        
        info = {'kills': 2}  # Killed 2 entities
        
        reward, components = calculator.calculate_step_reward(game_state, {}, info)
        
        assert components['kills'] == 20.0  # 2 * 10.0
        assert calculator.episode_stats.kills == 2
    
    def test_death_penalty(self):
        """Test penalty for dying."""
        config = RewardConfig(death_penalty=-100.0)
        calculator = RewardCalculator(config)
        
        # No player entities = dead
        game_state = {
            'player_entities': [],
            'other_entities': [],
            'food_entities': []
        }
        
        reward, components = calculator.calculate_step_reward(game_state, {}, {})
        
        assert components['death'] == -100.0
        assert calculator.episode_stats.deaths == 1
    
    def test_food_collection_reward(self):
        """Test reward for collecting food."""
        config = RewardConfig(food_weight=0.5)
        calculator = RewardCalculator(config)
        
        game_state = {
            'player_entities': [{'mass': 50.0}],
            'other_entities': [],
            'food_entities': []
        }
        
        info = {'food_collected': 5}
        
        reward, components = calculator.calculate_step_reward(game_state, {}, info)
        
        assert components['food'] == 2.5  # 5 * 0.5
        assert calculator.episode_stats.food_collected == 5
    
    def test_split_penalty(self):
        """Test penalty for splitting."""
        config = RewardConfig(split_penalty=-0.5)
        calculator = RewardCalculator(config)
        
        game_state = {
            'player_entities': [{'mass': 30.0}, {'mass': 30.0}],
            'other_entities': [],
            'food_entities': []
        }
        
        action = {'split': True}
        
        reward, components = calculator.calculate_step_reward(game_state, action, {})
        
        assert components['split'] == -0.5
        assert calculator.episode_stats.splits == 1
    
    def test_curriculum_learning_stages(self):
        """Test curriculum learning stage transitions."""
        config = RewardConfig(use_curriculum=True)
        calculator = RewardCalculator(config)
        
        # Stage 1: Food focus
        assert calculator.current_stage == 0
        assert calculator.config.food_weight == 1.0
        assert calculator.config.kill_weight == 5.0
        
        # Advance to stage 2 (survival)
        calculator.episode_stats.total_mass_gained = 60.0
        calculator._update_curriculum_stage()
        
        assert calculator.current_stage == 1
        assert calculator.config.survival_weight == 0.2
        assert calculator.config.death_penalty == -20.0
        
        # Advance to stage 3 (hunting)
        calculator.episode_stats.steps = 1000
        calculator._update_curriculum_stage()
        
        assert calculator.current_stage == 2
        assert calculator.config.kill_weight == 20.0
        assert calculator.config.mass_gain_weight == 2.0
        
        # Advance to stage 4 (advanced)
        calculator.episode_stats.kills = 5
        calculator._update_curriculum_stage()
        
        assert calculator.current_stage == 3
        assert calculator.config.kill_weight == 30.0
        assert calculator.config.split_penalty == -0.2
    
    def test_curriculum_disabled(self):
        """Test that curriculum doesn't change when disabled."""
        config = RewardConfig(use_curriculum=False, food_weight=2.0)
        calculator = RewardCalculator(config)
        
        original_food_weight = calculator.config.food_weight
        
        # Try to advance stage
        calculator.episode_stats.total_mass_gained = 100.0
        calculator._update_curriculum_stage()
        
        # Config should not change
        assert calculator.current_stage == 0
        assert calculator.config.food_weight == original_food_weight
    
    def test_episode_reward(self):
        """Test episode completion reward calculation."""
        config = RewardConfig(
            episode_length_bonus=0.01,
            max_mass_bonus=0.1,
            efficiency_bonus=10.0
        )
        calculator = RewardCalculator(config)
        
        # Setup episode stats
        calculator.episode_stats.steps = 1000
        calculator.episode_stats.max_mass = 200.0
        calculator.episode_stats.total_mass_gained = 190.0
        calculator.episode_stats.food_collected = 150
        
        bonus, components = calculator.calculate_episode_reward()
        
        # Length bonus: 1000 * 0.01 = 10.0
        assert components['length_bonus'] == 10.0
        
        # Max mass bonus: 200 * 0.1 = 20.0
        assert components['max_mass_bonus'] == 20.0
        
        # Efficiency bonus: (190 / 150) * 10 = 12.67
        assert abs(components['efficiency_bonus'] - 12.67) < 0.1
        
        assert bonus == sum(components.values())
    
    def test_reward_info(self):
        """Test getting reward calculation info."""
        calculator = RewardCalculator()
        
        # Add some stats
        calculator.episode_stats.steps = 500
        calculator.episode_stats.total_reward = 150.0
        calculator.episode_stats.kills = 3
        calculator.episode_stats.deaths = 1
        calculator.current_stage = 2
        
        info = calculator.get_reward_info()
        
        assert info['episode_stats']['steps'] == 500
        assert info['episode_stats']['total_reward'] == 150.0
        assert info['episode_stats']['kills'] == 3
        assert info['episode_stats']['deaths'] == 1
        assert info['current_stage'] == 2
        assert 'current_weights' in info
    
    def test_complete_episode_flow(self):
        """Test a complete episode with various rewards."""
        calculator = RewardCalculator()
        
        total_reward = 0.0
        
        # Step 1: Collect food and gain mass
        game_state = {
            'player_entities': [{'mass': 15.0}],  # Started at 10
            'other_entities': [],
            'food_entities': []
        }
        info = {'food_collected': 5}
        
        reward, _ = calculator.calculate_step_reward(game_state, {}, info)
        total_reward += reward
        
        # Step 2: Kill an enemy
        calculator.previous_mass = 25.0  # Gained more mass
        game_state['player_entities'][0]['mass'] = 40.0
        info = {'kills': 1}
        
        reward, _ = calculator.calculate_step_reward(game_state, {}, info)
        total_reward += reward
        
        # Step 3: Split
        game_state['player_entities'] = [
            {'mass': 20.0},
            {'mass': 20.0}
        ]
        action = {'split': True}
        
        reward, _ = calculator.calculate_step_reward(game_state, action, {})
        total_reward += reward
        
        # Verify total reward matches sum
        assert abs(calculator.episode_stats.total_reward - total_reward) < 0.01
    
    def test_edge_cases(self):
        """Test various edge cases."""
        calculator = RewardCalculator()
        
        # Empty game state
        empty_state = {
            'player_entities': [],
            'other_entities': [],
            'food_entities': []
        }
        
        reward, components = calculator.calculate_step_reward(empty_state, {}, {})
        assert components['death'] < 0  # Should get death penalty
        
        # Negative mass (shouldn't happen but test robustness)
        invalid_state = {
            'player_entities': [{'mass': -10.0}],
            'other_entities': [],
            'food_entities': []
        }
        
        reward, components = calculator.calculate_step_reward(invalid_state, {}, {})
        # Should handle gracefully
        assert isinstance(reward, float)
        
        # Very large numbers
        large_state = {
            'player_entities': [{'mass': 1e6}],
            'other_entities': [],
            'food_entities': []
        }
        info = {'kills': 1000, 'food_collected': 10000}
        
        reward, components = calculator.calculate_step_reward(large_state, {}, info)
        # Should not overflow
        assert np.isfinite(reward)
        assert all(np.isfinite(v) for v in components.values())


@pytest.mark.parametrize("stage,expected_weights", [
    (0, {'food_weight': 1.0, 'kill_weight': 5.0}),
    (1, {'survival_weight': 0.2, 'death_penalty': -20.0}),
    (2, {'kill_weight': 20.0, 'mass_gain_weight': 2.0}),
    (3, {'kill_weight': 30.0, 'split_penalty': -0.2}),
])
def test_curriculum_stage_weights(stage, expected_weights):
    """Test that curriculum stages have correct weights."""
    config = RewardConfig(use_curriculum=True)
    calculator = RewardCalculator(config)
    
    # Force to specific stage
    calculator.current_stage = stage
    calculator._apply_curriculum_weights()
    
    for key, expected_value in expected_weights.items():
        assert getattr(calculator.config, key) == expected_value
