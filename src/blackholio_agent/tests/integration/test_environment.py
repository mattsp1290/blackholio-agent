"""
Integration tests for the complete Blackholio environment.

Tests the full environment with mocked SpacetimeDB connection.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from ...environment import BlackholioEnv, BlackholioEnvConfig
from ..fixtures.mock_spacetimedb import MockSpacetimeDBClient, MockConfig
from ..fixtures.game_states import (
    EARLY_GAME_SOLO,
    EARLY_GAME_WITH_THREAT,
    MID_GAME_MULTI_ENTITY,
    get_scenario
)


class TestBlackholioEnvironment:
    """Integration tests for BlackholioEnv."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock SpacetimeDB client."""
        return MockSpacetimeDBClient(MockConfig(
            simulate_game_physics=True,
            update_rate=20.0
        ))
    
    @pytest.mark.asyncio
    async def test_environment_lifecycle(self, mock_client):
        """Test complete environment lifecycle."""
        # Create environment
        config = BlackholioEnvConfig()
        env = BlackholioEnv(config)
        
        # Mock the connection
        env.connection.client = mock_client
        
        # Connect to server
        connected = await env.connect()
        assert connected is True
        assert env.is_connected is True
        
        # Reset environment
        obs, info = env.reset()
        assert obs.shape == (456,)  # Observation space dimension
        assert isinstance(info, dict)
        
        # Take a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = await env.step(action)
            
            assert obs.shape == (456,)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
        
        # Close environment
        await env.close()
        assert env.is_connected is False
    
    @pytest.mark.asyncio
    async def test_environment_with_scenarios(self, mock_client):
        """Test environment with different game scenarios."""
        config = BlackholioEnvConfig()
        env = BlackholioEnv(config)
        env.connection.client = mock_client
        
        await env.connect()
        
        # Test different scenarios
        scenarios = [
            EARLY_GAME_SOLO,
            EARLY_GAME_WITH_THREAT,
            MID_GAME_MULTI_ENTITY
        ]
        
        for scenario in scenarios:
            # Load scenario into mock client
            mock_client.set_game_scenario(scenario)
            
            # Reset and get observation
            obs, info = env.reset()
            
            # Verify observation is valid
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isinf(obs))
            
            # Take a step
            action = {
                "movement": np.array([0.5, 0.0]),
                "split": 0.0
            }
            
            obs, reward, done, truncated, info = await env.step(action)
            
            # Verify step results
            assert obs.shape == (456,)
            assert isinstance(reward, float)
        
        await env.close()
    
    @pytest.mark.asyncio
    async def test_observation_action_consistency(self, mock_client):
        """Test that observations and actions are properly processed."""
        env = BlackholioEnv()
        env.connection.client = mock_client
        
        await env.connect()
        
        # Reset environment
        obs1, _ = env.reset()
        
        # Take deterministic action
        action = {
            "movement": np.array([1.0, 0.0]),  # Move right
            "split": 0.0
        }
        
        # Step multiple times
        observations = [obs1]
        for _ in range(10):
            obs, _, _, _, _ = await env.step(action)
            observations.append(obs)
        
        # Observations should change (player is moving)
        obs_changed = False
        for i in range(len(observations) - 1):
            if not np.array_equal(observations[i], observations[i + 1]):
                obs_changed = True
                break
        
        assert obs_changed, "Observations should change as player moves"
        
        await env.close()
    
    @pytest.mark.asyncio
    async def test_reward_calculation_integration(self, mock_client):
        """Test reward calculation in different scenarios."""
        env = BlackholioEnv()
        env.connection.client = mock_client
        
        await env.connect()
        
        # Scenario 1: Food collection
        # Set up game state with food nearby
        food_scenario = get_scenario("early_game_solo")
        mock_client.set_game_scenario(food_scenario)
        
        obs, _ = env.reset()
        
        # Move towards food
        total_reward = 0
        for _ in range(20):
            # Simple strategy: move right (assuming food is there)
            action = {"movement": np.array([1.0, 0.0]), "split": 0.0}
            obs, reward, done, truncated, info = await env.step(action)
            total_reward += reward
            
            if done:
                break
        
        # Should have positive reward from survival and possibly food
        assert total_reward > 0
        
        await env.close()
    
    @pytest.mark.asyncio
    async def test_episode_termination(self, mock_client):
        """Test episode termination conditions."""
        env = BlackholioEnv()
        env.connection.client = mock_client
        
        await env.connect()
        
        # Test death scenario
        # Manually trigger player death
        obs, _ = env.reset()
        
        # Remove player entities to simulate death
        env.connection.game_state.entities.clear()
        
        # Next step should detect death
        action = {"movement": np.array([0.0, 0.0]), "split": 0.0}
        obs, reward, done, truncated, info = await env.step(action)
        
        assert done is True
        assert reward < 0  # Death penalty
        
        await env.close()
    
    @pytest.mark.asyncio
    async def test_action_throttling_integration(self, mock_client):
        """Test action throttling at environment level."""
        config = BlackholioEnvConfig()
        env = BlackholioEnv(config)
        env.connection.client = mock_client
        
        await env.connect()
        obs, _ = env.reset()
        
        # Rapid action submission
        action = {"movement": np.array([1.0, 0.0]), "split": 0.0}
        
        # Submit actions rapidly
        start_time = asyncio.get_event_loop().time()
        action_times = []
        
        for _ in range(5):
            await env.step(action)
            action_times.append(asyncio.get_event_loop().time())
        
        # Check that actions were throttled
        # At 20Hz, minimum time between actions is 0.05s
        for i in range(1, len(action_times)):
            time_diff = action_times[i] - action_times[i-1]
            # Allow some tolerance for execution time
            assert time_diff >= 0.04  # Slightly less than 0.05 for tolerance
        
        await env.close()
    
    @pytest.mark.asyncio
    async def test_curriculum_learning_integration(self, mock_client):
        """Test curriculum learning stages in environment."""
        config = BlackholioEnvConfig()
        env = BlackholioEnv(config)
        env.connection.client = mock_client
        
        await env.connect()
        
        # Stage 1: Food collection focus
        obs, _ = env.reset()
        assert env.reward_calculator.current_stage == 0
        
        # Simulate progression through stages
        # Add mass to trigger stage change
        env.reward_calculator.episode_stats.total_mass_gained = 60.0
        env.reward_calculator._update_curriculum_stage()
        
        assert env.reward_calculator.current_stage == 1
        
        await env.close()
    
    @pytest.mark.asyncio
    async def test_multi_entity_handling(self, mock_client):
        """Test handling multiple player entities (after split)."""
        env = BlackholioEnv()
        env.connection.client = mock_client
        
        await env.connect()
        
        # Load multi-entity scenario
        scenario = get_scenario("mid_game_multi_entity")
        mock_client.set_game_scenario(scenario)
        
        obs, _ = env.reset()
        
        # Verify observation includes multiple entities
        # Player state should aggregate multiple entities
        player_state = obs[:6]  # First 6 features are player state
        
        # Take split action
        split_action = {"movement": np.array([0.0, 0.0]), "split": 1.0}
        obs, reward, done, truncated, info = await env.step(split_action)
        
        # Should handle split without errors
        assert not done
        
        await env.close()
    
    @pytest.mark.asyncio
    async def test_render_functionality(self, mock_client):
        """Test rendering (if implemented)."""
        env = BlackholioEnv()
        env.connection.client = mock_client
        
        await env.connect()
        obs, _ = env.reset()
        
        # Test render
        try:
            rendered = env.render()
            # If render is implemented, check output
            if rendered is not None:
                assert isinstance(rendered, (np.ndarray, str))
        except NotImplementedError:
            # Render might not be implemented
            pass
        
        await env.close()
    
    @pytest.mark.asyncio
    async def test_environment_recovery(self, mock_client):
        """Test environment recovery from connection loss."""
        config = BlackholioEnvConfig()
        env = BlackholioEnv(config)
        env.connection.client = mock_client
        
        await env.connect()
        obs, _ = env.reset()
        
        # Simulate connection loss
        env.connection.is_connected = False
        mock_client.is_connected = False
        
        # Try to step - should attempt reconnection
        action = {"movement": np.array([0.0, 0.0]), "split": 0.0}
        
        # Mock reconnection success
        mock_client.connect = AsyncMock(return_value=True)
        
        # This should trigger reconnection attempt
        try:
            obs, reward, done, truncated, info = await env.step(action)
            # If auto-reconnect is enabled, this should work
        except ConnectionError:
            # If auto-reconnect is disabled, this is expected
            pass
        
        await env.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_environments(self):
        """Test multiple environments running concurrently."""
        # Create multiple environments with separate mock clients
        envs = []
        for i in range(3):
            env = BlackholioEnv()
            env.connection.client = MockSpacetimeDBClient(MockConfig(
                simulate_game_physics=True
            ))
            envs.append(env)
        
        # Connect all environments
        await asyncio.gather(*[env.connect() for env in envs])
        
        # Reset all environments
        reset_results = await asyncio.gather(*[
            env.reset() for env in envs
        ])
        
        # All should reset successfully
        assert all(isinstance(r[0], np.ndarray) for r in reset_results)
        
        # Take steps concurrently
        actions = [
            {"movement": np.array([1.0, 0.0]), "split": 0.0},
            {"movement": np.array([0.0, 1.0]), "split": 0.0},
            {"movement": np.array([-1.0, 0.0]), "split": 0.0}
        ]
        
        step_tasks = []
        for env, action in zip(envs, actions):
            step_tasks.append(env.step(action))
        
        step_results = await asyncio.gather(*step_tasks)
        
        # All should step successfully
        assert all(isinstance(r[0], np.ndarray) for r in step_results)
        
        # Close all environments
        await asyncio.gather(*[env.close() for env in envs])
    
    @pytest.mark.asyncio
    async def test_environment_metrics(self, mock_client):
        """Test environment metrics collection."""
        env = BlackholioEnv()
        env.connection.client = mock_client
        
        await env.connect()
        obs, _ = env.reset()
        
        # Take several steps
        for i in range(10):
            action = {
                "movement": np.array([np.sin(i * 0.5), np.cos(i * 0.5)]),
                "split": 0.0 if i < 8 else 1.0  # Split on last steps
            }
            await env.step(action)
        
        # Get environment info
        info = env.get_info()
        
        assert "episode_stats" in info
        assert "action_stats" in info
        assert "reward_info" in info
        
        # Check some stats
        assert info["episode_stats"]["steps"] == 10
        assert info["action_stats"]["total_actions"] == 10
        assert info["action_stats"]["total_splits"] >= 2  # Split twice
        
        await env.close()


@pytest.mark.parametrize("render_mode", [None, "human", "rgb_array"])
def test_environment_creation_modes(render_mode):
    """Test environment creation with different render modes."""
    config = BlackholioEnvConfig(render_mode=render_mode)
    env = BlackholioEnv(config)
    
    assert env.config.render_mode == render_mode
    
    # Check Gym compatibility
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(env, 'close')
    assert hasattr(env, 'render')
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'action_space')
