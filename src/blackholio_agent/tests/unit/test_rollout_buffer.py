"""
Unit tests for RolloutBuffer component.

Tests buffer operations, return calculation, and sampling.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from ...training import RolloutBuffer, RolloutBufferConfig


class TestRolloutBuffer:
    """Test suite for RolloutBuffer class."""
    
    def test_initialization(self):
        """Test RolloutBuffer initialization with default config."""
        config = RolloutBufferConfig()
        buffer = RolloutBuffer(config)
        
        assert buffer.config == config
        assert buffer.buffer_size == 2048
        assert buffer.n_envs == 8
        assert buffer.pos == 0
        assert buffer.full is False
    
    def test_custom_configuration(self):
        """Test RolloutBuffer with custom configuration."""
        config = RolloutBufferConfig(
            buffer_size=1024,
            n_envs=4,
            observation_dim=256,
            gamma=0.95,
            gae_lambda=0.9,
            device="cpu"
        )
        buffer = RolloutBuffer(config)
        
        assert buffer.config.buffer_size == 1024
        assert buffer.config.n_envs == 4
        assert buffer.config.observation_dim == 256
        
        # Check buffer shapes
        assert buffer.observations.shape == (1024, 4, 256)
        assert buffer.movement_actions.shape == (1024, 4, 2)
        assert buffer.split_actions.shape == (1024, 4)
    
    def test_add_single_step(self):
        """Test adding a single step to buffer."""
        buffer = RolloutBuffer()
        
        # Create sample data
        obs = np.random.randn(8, 456).astype(np.float32)
        movement = np.random.randn(8, 2).astype(np.float32)
        split = np.random.randint(0, 2, size=(8,))
        reward = np.random.randn(8).astype(np.float32)
        episode_start = np.zeros(8, dtype=bool)
        episode_start[0] = True
        value = np.random.randn(8).astype(np.float32)
        log_prob = np.random.randn(8).astype(np.float32)
        
        # Add to buffer
        buffer.add(obs, movement, split, reward, episode_start, value, log_prob)
        
        assert buffer.pos == 1
        assert not buffer.full
        
        # Check data was stored correctly
        assert np.array_equal(buffer.observations[0], obs)
        assert np.array_equal(buffer.movement_actions[0], movement)
        assert np.array_equal(buffer.split_actions[0], split)
        assert np.array_equal(buffer.rewards[0], reward)
        assert np.array_equal(buffer.episode_starts[0], episode_start)
        assert np.array_equal(buffer.values[0], value)
        assert np.array_equal(buffer.log_probs[0], log_prob)
    
    def test_buffer_overflow(self):
        """Test buffer behavior when full."""
        config = RolloutBufferConfig(buffer_size=10, n_envs=2)
        buffer = RolloutBuffer(config)
        
        # Fill buffer
        for i in range(10):
            buffer.add(
                obs=np.random.randn(2, 456).astype(np.float32),
                movement_action=np.random.randn(2, 2).astype(np.float32),
                split_action=np.random.randint(0, 2, size=(2,)),
                reward=np.random.randn(2).astype(np.float32),
                episode_start=np.zeros(2, dtype=bool),
                value=np.random.randn(2).astype(np.float32),
                log_prob=np.random.randn(2).astype(np.float32)
            )
        
        assert buffer.pos == 0  # Should wrap around
        assert buffer.full is True
        
        # Adding more should overwrite old data
        new_obs = np.ones((2, 456), dtype=np.float32)
        buffer.add(new_obs, np.zeros((2, 2)), np.zeros(2), np.zeros(2), 
                  np.zeros(2, dtype=bool), np.zeros(2), np.zeros(2))
        
        assert np.array_equal(buffer.observations[0], new_obs)
    
    def test_compute_returns_and_advantage(self):
        """Test GAE computation."""
        config = RolloutBufferConfig(buffer_size=5, n_envs=2, gamma=0.99, gae_lambda=0.95)
        buffer = RolloutBuffer(config)
        
        # Add some steps with known rewards
        for i in range(5):
            buffer.add(
                obs=np.random.randn(2, 456).astype(np.float32),
                movement_action=np.random.randn(2, 2).astype(np.float32),
                split_action=np.random.randint(0, 2, size=(2,)),
                reward=np.array([1.0, 2.0], dtype=np.float32),  # Constant rewards
                episode_start=np.array([i == 0, i == 0], dtype=bool),
                value=np.array([0.5, 1.0], dtype=np.float32) * i,  # Increasing values
                log_prob=np.random.randn(2).astype(np.float32)
            )
        
        # Compute returns
        last_values = np.array([2.5, 5.0], dtype=np.float32)
        dones = np.array([False, False], dtype=bool)
        
        buffer.compute_returns_and_advantage(last_values, dones)
        
        # Check that returns and advantages were computed
        assert buffer.returns is not None
        assert buffer.advantages is not None
        assert buffer.returns.shape == (5, 2)
        assert buffer.advantages.shape == (5, 2)
        
        # Returns should be discounted sum of rewards
        # Advantages should be returns - values
        assert np.all(buffer.returns >= 0)  # With positive rewards
    
    def test_get_samples(self):
        """Test getting all samples from buffer."""
        config = RolloutBufferConfig(buffer_size=10, n_envs=2)
        buffer = RolloutBuffer(config)
        
        # Fill buffer partially
        for i in range(5):
            buffer.add(
                obs=np.random.randn(2, 456).astype(np.float32),
                movement_action=np.random.randn(2, 2).astype(np.float32),
                split_action=np.random.randint(0, 2, size=(2,)),
                reward=np.random.randn(2).astype(np.float32),
                episode_start=np.zeros(2, dtype=bool),
                value=np.random.randn(2).astype(np.float32),
                log_prob=np.random.randn(2).astype(np.float32)
            )
        
        # Compute returns
        buffer.compute_returns_and_advantage(np.zeros(2), np.zeros(2, dtype=bool))
        
        # Get samples
        samples = buffer.get_samples()
        
        # Should return flattened data (5 steps * 2 envs = 10 samples)
        assert samples["observations"].shape == (10, 456)
        assert samples["movement_actions"].shape == (10, 2)
        assert samples["split_actions"].shape == (10,)
        assert samples["rewards"].shape == (10,)
        assert samples["advantages"].shape == (10,)
        assert samples["returns"].shape == (10,)
        assert samples["values"].shape == (10,)
        assert samples["log_probs"].shape == (10,)
        
        # Should be torch tensors
        assert isinstance(samples["observations"], torch.Tensor)
    
    def test_reset(self):
        """Test buffer reset."""
        buffer = RolloutBuffer()
        
        # Add some data
        for i in range(5):
            buffer.add(
                obs=np.random.randn(8, 456).astype(np.float32),
                movement_action=np.random.randn(8, 2).astype(np.float32),
                split_action=np.random.randint(0, 2, size=(8,)),
                reward=np.random.randn(8).astype(np.float32),
                episode_start=np.zeros(8, dtype=bool),
                value=np.random.randn(8).astype(np.float32),
                log_prob=np.random.randn(8).astype(np.float32)
            )
        
        assert buffer.pos == 5
        
        # Reset
        buffer.reset()
        
        assert buffer.pos == 0
        assert buffer.full is False
        assert buffer.returns is None
        assert buffer.advantages is None
    
    def test_get_statistics(self):
        """Test buffer statistics."""
        buffer = RolloutBuffer()
        
        # Add data with known values
        for i in range(3):
            buffer.add(
                obs=np.random.randn(8, 456).astype(np.float32),
                movement_action=np.random.randn(8, 2).astype(np.float32),
                split_action=np.random.randint(0, 2, size=(8,)),
                reward=np.ones(8).astype(np.float32) * (i + 1),  # 1, 2, 3
                episode_start=np.zeros(8, dtype=bool),
                value=np.ones(8).astype(np.float32) * 0.5,
                log_prob=np.random.randn(8).astype(np.float32)
            )
        
        stats = buffer.get_statistics()
        
        assert "buffer_size" in stats
        assert "current_position" in stats
        assert "is_full" in stats
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "mean_value" in stats
        
        assert stats["buffer_size"] == 8 * 3  # 3 steps * 8 envs
        assert stats["current_position"] == 3
        assert stats["is_full"] is False
        assert np.isclose(stats["mean_reward"], 2.0)  # Mean of 1, 2, 3
        assert np.isclose(stats["mean_value"], 0.5)
    
    def test_episode_boundaries(self):
        """Test handling of episode boundaries."""
        config = RolloutBufferConfig(buffer_size=10, n_envs=2, gamma=0.99)
        buffer = RolloutBuffer(config)
        
        # Add steps with episode boundary
        for i in range(5):
            episode_start = np.array([i == 0, i == 3], dtype=bool)  # Env 1 resets at step 3
            
            buffer.add(
                obs=np.random.randn(2, 456).astype(np.float32),
                movement_action=np.random.randn(2, 2).astype(np.float32),
                split_action=np.random.randint(0, 2, size=(2,)),
                reward=np.ones(2).astype(np.float32),
                episode_start=episode_start,
                value=np.ones(2).astype(np.float32),
                log_prob=np.random.randn(2).astype(np.float32)
            )
        
        # Compute returns
        buffer.compute_returns_and_advantage(np.zeros(2), np.zeros(2, dtype=bool))
        
        # Advantages should be reset at episode boundaries
        # This is handled in the GAE computation
        assert buffer.advantages is not None
    
    def test_device_handling(self):
        """Test buffer device placement."""
        if torch.cuda.is_available():
            config = RolloutBufferConfig(device="cuda")
            buffer = RolloutBuffer(config)
            
            # Add data
            buffer.add(
                obs=np.random.randn(8, 456).astype(np.float32),
                movement_action=np.random.randn(8, 2).astype(np.float32),
                split_action=np.random.randint(0, 2, size=(8,)),
                reward=np.random.randn(8).astype(np.float32),
                episode_start=np.zeros(8, dtype=bool),
                value=np.random.randn(8).astype(np.float32),
                log_prob=np.random.randn(8).astype(np.float32)
            )
            
            buffer.compute_returns_and_advantage(np.zeros(8), np.zeros(8, dtype=bool))
            samples = buffer.get_samples()
            
            # Samples should be on CUDA
            assert samples["observations"].device.type == "cuda"
    
    def test_edge_cases(self):
        """Test various edge cases."""
        buffer = RolloutBuffer()
        
        # Get samples from empty buffer should raise error
        with pytest.raises(ValueError):
            buffer.get_samples()
        
        # Add data with extreme values
        buffer.add(
            obs=np.random.randn(8, 456).astype(np.float32) * 1000,
            movement_action=np.clip(np.random.randn(8, 2) * 10, -1, 1).astype(np.float32),
            split_action=np.ones(8, dtype=int),
            reward=np.random.randn(8).astype(np.float32) * 100,
            episode_start=np.ones(8, dtype=bool),  # All episodes start
            value=np.random.randn(8).astype(np.float32) * 50,
            log_prob=np.random.randn(8).astype(np.float32) * -10
        )
        
        # Should handle without errors
        buffer.compute_returns_and_advantage(np.zeros(8), np.zeros(8, dtype=bool))
        samples = buffer.get_samples()
        
        # Check no NaN or inf values
        for key, tensor in samples.items():
            assert not torch.any(torch.isnan(tensor))
            assert not torch.any(torch.isinf(tensor))


@pytest.mark.parametrize("gamma,gae_lambda", [
    (0.99, 0.95),
    (0.95, 0.90),
    (1.0, 1.0),  # No discounting
    (0.9, 0.0),   # TD(0)
])
def test_gae_parameters(gamma, gae_lambda):
    """Test GAE computation with different parameters."""
    config = RolloutBufferConfig(
        buffer_size=5, 
        n_envs=1, 
        gamma=gamma, 
        gae_lambda=gae_lambda
    )
    buffer = RolloutBuffer(config)
    
    # Add constant rewards and values for predictable computation
    for i in range(5):
        buffer.add(
            obs=np.zeros((1, 456), dtype=np.float32),
            movement_action=np.zeros((1, 2), dtype=np.float32),
            split_action=np.zeros(1, dtype=int),
            reward=np.ones(1, dtype=np.float32),
            episode_start=np.array([i == 0], dtype=bool),
            value=np.ones(1, dtype=np.float32) * 0.5,
            log_prob=np.zeros(1, dtype=np.float32)
        )
    
    buffer.compute_returns_and_advantage(np.array([0.5]), np.array([False]))
    
    # Check computation completed
    assert buffer.returns is not None
    assert buffer.advantages is not None
    
    # With constant rewards and values, we can verify some properties
    if gamma == 1.0 and gae_lambda == 1.0:
        # No discounting - returns should accumulate
        assert buffer.returns[0, 0] > buffer.returns[4, 0]
