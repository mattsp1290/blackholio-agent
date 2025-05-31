"""
Unit tests for ActionSpace component.

Tests action validation, throttling, queuing, and execution.
"""

import pytest
import numpy as np
import asyncio
import time
from unittest.mock import Mock, AsyncMock, call

from ...environment.action_space import ActionSpace, ActionConfig


class TestActionSpace:
    """Test suite for ActionSpace class."""
    
    def test_initialization(self):
        """Test ActionSpace initialization with default config."""
        config = ActionConfig()
        action_space = ActionSpace(config)
        
        assert action_space.config == config
        assert action_space.shape == (3,)  # 2D movement + split
        assert action_space.last_movement_time == 0
        assert len(action_space.action_queue) == 0
        assert action_space.action_count == 0
    
    def test_custom_configuration(self):
        """Test ActionSpace with custom configuration."""
        config = ActionConfig(
            update_rate=10.0,  # 10 Hz
            queue_size=20,
            movement_scale=2.0,
            enable_split=False
        )
        action_space = ActionSpace(config)
        
        assert action_space.config.update_rate == 10.0
        assert action_space.config.queue_size == 20
        assert action_space.config.movement_scale == 2.0
        assert not action_space.config.enable_split
        assert action_space.shape == (2,)  # No split dimension
    
    def test_parse_action_dict(self):
        """Test parsing action dictionary."""
        action_space = ActionSpace()
        
        action = {
            "movement": np.array([0.5, -0.3]),
            "split": True
        }
        
        movement, split = action_space.parse_action(action)
        
        assert np.allclose(movement, [0.5, -0.3])
        assert split is True
    
    def test_parse_action_array(self):
        """Test parsing action as numpy array."""
        action_space = ActionSpace()
        
        # 3D array: [movement_x, movement_y, split]
        action = np.array([0.7, -0.2, 0.8])
        
        movement, split = action_space.parse_action(action)
        
        assert np.allclose(movement, [0.7, -0.2])
        assert split is True  # 0.8 > 0.5 threshold
        
        # Test with split = False
        action = np.array([0.7, -0.2, 0.2])
        movement, split = action_space.parse_action(action)
        assert split is False  # 0.2 < 0.5 threshold
    
    def test_normalize_movement(self):
        """Test movement normalization (clipping)."""
        action_space = ActionSpace()
        
        # Test clipping of large movement
        action = np.array([2.0, -1.5, 0.0])
        movement, _ = action_space.parse_action(action)
        
        # Should be clipped to [-1, 1]
        assert movement[0] == 1.0
        assert movement[1] == -1.0
    
    def test_movement_scaling(self):
        """Test movement scaling."""
        config = ActionConfig(movement_scale=2.0)
        action_space = ActionSpace(config)
        
        action = np.array([0.5, -0.3, 0.0])
        movement, _ = action_space.parse_action(action)
        
        # Should be scaled
        assert np.allclose(movement, [1.0, -0.6])
    
    @pytest.mark.asyncio
    async def test_execute_action_throttling(self):
        """Test action throttling based on update rate."""
        config = ActionConfig(update_rate=10.0)  # 10 Hz = 0.1s between actions
        action_space = ActionSpace(config)
        
        mock_connection = AsyncMock()
        
        # First action should execute immediately
        action1 = np.array([0.5, 0.0, 0.0])
        result1 = await action_space.execute_action(mock_connection, action1)
        
        assert result1["movement_executed"] is True
        assert result1["queued"] is False
        assert mock_connection.update_player_input.called
        
        # Second action immediately after should be queued
        action2 = np.array([0.0, 0.5, 0.0])
        result2 = await action_space.execute_action(mock_connection, action2)
        
        assert result2["movement_executed"] is False
        assert result2["queued"] is True
        assert len(action_space.action_queue) == 1
    
    @pytest.mark.asyncio
    async def test_execute_split_action(self):
        """Test split action execution."""
        action_space = ActionSpace()
        mock_connection = AsyncMock()
        
        # Wait to ensure we can move
        action_space.last_movement_time = 0
        
        # Action with split
        action = np.array([0.5, 0.3, 0.8])  # split > 0.5
        result = await action_space.execute_action(mock_connection, action)
        
        assert result["movement_executed"] is True
        assert result["split_executed"] is True
        
        # Check both movement and split were called
        assert mock_connection.update_player_input.called
        assert mock_connection.player_split.called
    
    @pytest.mark.asyncio
    async def test_action_queue_processing(self):
        """Test processing of queued actions."""
        config = ActionConfig(update_rate=20.0, queue_size=5)
        action_space = ActionSpace(config)
        mock_connection = AsyncMock()
        
        # Fill the queue
        for i in range(3):
            action_space.action_queue.append((
                np.array([0.1 * i, 0.2 * i]),
                False,
                time.time()
            ))
        
        assert len(action_space.action_queue) == 3
        
        # Process queue
        processed = await action_space.process_action_queue(mock_connection)
        
        # Should process at least one action
        assert processed > 0
        assert len(action_space.action_queue) < 3
    
    @pytest.mark.asyncio
    async def test_queue_overflow(self):
        """Test queue overflow handling."""
        config = ActionConfig(queue_size=2)
        action_space = ActionSpace(config)
        mock_connection = AsyncMock()
        
        # Set last action time to force queuing
        action_space.last_movement_time = time.time()
        
        # Try to queue more than queue_size actions
        for i in range(4):
            action = np.array([0.1 * i, 0.1 * i, 0.0])
            await action_space.execute_action(mock_connection, action)
        
        # Queue should be at max size
        assert len(action_space.action_queue) == 2
    
    def test_get_action_stats(self):
        """Test action statistics tracking."""
        action_space = ActionSpace()
        
        # Initialize some stats
        action_space.action_count = 100
        action_space.successful_actions = 90
        action_space.failed_actions = 10
        action_space.last_movement_time = time.time()
        
        stats = action_space.get_action_stats()
        
        assert stats["total_actions"] == 100
        assert stats["successful_actions"] == 90
        assert stats["failed_actions"] == 10
        assert stats["success_rate"] == 0.9
        assert stats["queued_actions"] == 0
        assert "current_update_rate" in stats
    
    @pytest.mark.asyncio
    async def test_movement_execution(self):
        """Test movement execution with scaling."""
        config = ActionConfig(movement_scale=2.0)
        action_space = ActionSpace(config)
        mock_connection = AsyncMock()
        
        action = np.array([0.5, -0.25, 0.0])
        await action_space.execute_action(mock_connection, action)
        
        # Check that movement was scaled
        mock_connection.update_player_input.assert_called_once()
        call_args = mock_connection.update_player_input.call_args[0]
        assert call_args[0] == [1.0, -0.5]  # Scaled by 2.0
    
    @pytest.mark.asyncio
    async def test_zero_movement(self):
        """Test handling of zero movement."""
        action_space = ActionSpace()
        mock_connection = AsyncMock()
        
        action = np.array([0.0, 0.0, 0.0])
        result = await action_space.execute_action(mock_connection, action)
        
        assert result["movement_executed"] is True
        mock_connection.update_player_input.assert_called_with([0.0, 0.0])
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self):
        """Test handling of connection failures."""
        action_space = ActionSpace()
        mock_connection = AsyncMock()
        
        # Simulate connection failure
        mock_connection.update_player_input.side_effect = ConnectionError("Connection lost")
        
        action = np.array([0.5, 0.0, 0.0])
        result = await action_space.execute_action(mock_connection, action)
        
        assert result["movement_executed"] is False
        assert result["error"] is not None
        assert action_space.failed_actions == 1
    
    @pytest.mark.asyncio
    async def test_split_cooldown(self):
        """Test split cooldown functionality."""
        config = ActionConfig(split_cooldown=1.0)
        action_space = ActionSpace(config)
        mock_connection = AsyncMock()
        
        # First split should work
        action = np.array([0.0, 0.0, 1.0])
        result1 = await action_space.execute_action(mock_connection, action)
        assert result1["split_executed"] is True
        
        # Second split immediately after should not execute
        result2 = await action_space.execute_action(mock_connection, action)
        assert result2["split_executed"] is False
        
        # Wait for cooldown
        await asyncio.sleep(1.1)
        
        # Now split should work again
        result3 = await action_space.execute_action(mock_connection, action)
        assert result3["split_executed"] is True
    
    def test_get_action_bounds(self):
        """Test action bounds retrieval."""
        action_space = ActionSpace()
        bounds = action_space.get_action_bounds()
        
        assert bounds["movement_x"] == (-1.0, 1.0)
        assert bounds["movement_y"] == (-1.0, 1.0)
        assert bounds["split"] == (0.0, 1.0)
        
        # Test without split
        config = ActionConfig(enable_split=False)
        action_space = ActionSpace(config)
        bounds = action_space.get_action_bounds()
        
        assert "split" not in bounds
    
    def test_sample_action(self):
        """Test random action sampling."""
        action_space = ActionSpace()
        
        for _ in range(10):
            action = action_space.sample()
            
            assert action.shape == (3,)
            # Movement should be normalized
            movement_norm = np.linalg.norm(action[:2])
            assert movement_norm <= 1.0 or np.isclose(movement_norm, 1.0)
            # Split should be 0 or 1 (rare)
            assert action[2] in [0, 1] or 0 <= action[2] <= 1
    
    @pytest.mark.asyncio
    async def test_force_execution(self):
        """Test force execution bypasses throttling."""
        config = ActionConfig(update_rate=10.0)
        action_space = ActionSpace(config)
        mock_connection = AsyncMock()
        
        # Execute first action
        await action_space.execute_action(mock_connection, np.array([0.5, 0.0, 0.0]))
        
        # Force second action immediately
        result = await action_space.execute_action(
            mock_connection, 
            {"movement": [0.0, 0.5], "split": False},
            force=True
        )
        
        assert result["movement_executed"] is True
        assert result["queued"] is False
        assert mock_connection.update_player_input.call_count == 2
    
    def test_reset_stats(self):
        """Test statistics reset."""
        action_space = ActionSpace()
        
        # Set some stats
        action_space.action_count = 100
        action_space.successful_actions = 90
        action_space.failed_actions = 10
        
        # Reset
        action_space.reset_stats()
        
        assert action_space.action_count == 0
        assert action_space.successful_actions == 0
        assert action_space.failed_actions == 0
