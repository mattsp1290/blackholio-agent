"""
Unit tests for MetricsLogger component.

Tests metrics tracking, logging, and tensorboard integration.
"""

import pytest
import tempfile
import shutil
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from ...training import MetricsLogger


class TestMetricsLogger:
    """Test suite for MetricsLogger class."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_log_dir):
        """Test MetricsLogger initialization."""
        logger = MetricsLogger(
            log_dir=temp_log_dir,
            console_log_interval=10,
            use_tensorboard=True
        )
        
        assert logger.log_dir == Path(temp_log_dir)
        assert logger.console_log_interval == 10
        assert logger.use_tensorboard is True
        assert logger.step_count == 0
        assert logger.episode_count == 0
    
    def test_directory_creation(self):
        """Test that log directory is created if it doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        non_existent = os.path.join(temp_dir, "new_logs")
        
        logger = MetricsLogger(log_dir=non_existent)
        
        assert os.path.exists(non_existent)
        shutil.rmtree(temp_dir)
    
    def test_log_step(self, temp_log_dir):
        """Test logging step metrics."""
        logger = MetricsLogger(log_dir=temp_log_dir, console_log_interval=2)
        
        # Log some steps
        logger.log_step(1, {"reward": 10.0, "loss": 0.5})
        logger.log_step(2, {"reward": 15.0, "loss": 0.4})
        logger.log_step(3, {"reward": 20.0, "loss": 0.3})
        
        assert logger.step_count == 3
        assert "reward" in logger.metrics
        assert "loss" in logger.metrics
        assert len(logger.metrics["reward"]) == 3
        assert logger.metrics["reward"] == [10.0, 15.0, 20.0]
    
    def test_log_episode(self, temp_log_dir):
        """Test logging episode metrics."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        # Log episodes
        logger.log_episode(1, {"total_reward": 100.0, "length": 500})
        logger.log_episode(2, {"total_reward": 150.0, "length": 600})
        
        assert logger.episode_count == 2
        assert "total_reward" in logger.episode_metrics
        assert logger.episode_metrics["total_reward"] == [100.0, 150.0]
        assert logger.episode_metrics["length"] == [500, 600]
    
    def test_log_training_update(self, temp_log_dir):
        """Test logging training update metrics."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        update_metrics = {
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 0.5,
            "learning_rate": 0.001
        }
        
        logger.log_training_update(1000, update_metrics)
        
        assert "policy_loss" in logger.training_metrics
        assert logger.training_metrics["policy_loss"] == [0.1]
    
    def test_log_hyperparameters(self, temp_log_dir):
        """Test logging hyperparameters."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        hparams = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "gamma": 0.99,
            "algorithm": "PPO"
        }
        
        logger.log_hyperparameters(hparams)
        
        # Check hyperparameters file
        hparams_file = logger.log_dir / "hyperparameters.json"
        assert hparams_file.exists()
        
        with open(hparams_file, 'r') as f:
            saved_hparams = json.load(f)
        
        assert saved_hparams["learning_rate"] == 0.001
        assert saved_hparams["algorithm"] == "PPO"
    
    def test_console_logging(self, temp_log_dir, capsys):
        """Test console output at intervals."""
        logger = MetricsLogger(
            log_dir=temp_log_dir,
            console_log_interval=2,
            use_tensorboard=False
        )
        
        # Log steps
        logger.log_step(1, {"reward": 10.0})
        captured = capsys.readouterr()
        assert captured.out == ""  # No output yet
        
        logger.log_step(2, {"reward": 20.0})
        captured = capsys.readouterr()
        assert "Step 2" in captured.out
        assert "reward" in captured.out
        assert "15.00" in captured.out  # Average of 10 and 20
    
    def test_get_summary(self, temp_log_dir):
        """Test getting summary statistics."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        # Add various metrics
        for i in range(10):
            logger.log_step(i, {"reward": i * 10.0})
            if i % 2 == 0:
                logger.log_episode(i // 2, {"episode_reward": i * 50.0})
        
        summary = logger.get_summary()
        
        assert "step_metrics" in summary
        assert "episode_metrics" in summary
        assert summary["total_steps"] == 10
        assert summary["total_episodes"] == 5
        
        # Check statistics
        assert "reward" in summary["step_metrics"]
        reward_stats = summary["step_metrics"]["reward"]
        assert reward_stats["mean"] == 45.0  # Mean of 0, 10, 20, ..., 90
        assert reward_stats["min"] == 0.0
        assert reward_stats["max"] == 90.0
        assert "std" in reward_stats
    
    def test_save_metrics(self, temp_log_dir):
        """Test saving metrics to file."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        # Add metrics
        for i in range(5):
            logger.log_step(i, {"reward": i * 10.0})
        
        # Save should happen automatically, but we can trigger it
        logger._save_metrics()
        
        # Check metrics file
        metrics_file = logger.log_dir / "metrics.json"
        assert metrics_file.exists()
        
        with open(metrics_file, 'r') as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics["metrics"]["reward"] == [0.0, 10.0, 20.0, 30.0, 40.0]
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_tensorboard_logging(self, mock_writer_class, temp_log_dir):
        """Test TensorBoard logging functionality."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer
        
        logger = MetricsLogger(
            log_dir=temp_log_dir,
            use_tensorboard=True
        )
        
        # Log metrics
        logger.log_step(100, {"reward": 25.0, "loss": 0.1})
        
        # Check tensorboard calls
        mock_writer.add_scalar.assert_any_call("step/reward", 25.0, 100)
        mock_writer.add_scalar.assert_any_call("step/loss", 0.1, 100)
        
        # Log episode
        logger.log_episode(5, {"episode_reward": 200.0})
        mock_writer.add_scalar.assert_any_call("episode/episode_reward", 200.0, 5)
        
        # Close logger
        logger.close()
        mock_writer.close.assert_called_once()
    
    def test_plot_metrics(self, temp_log_dir):
        """Test metrics plotting functionality."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        # Add data for plotting
        for i in range(20):
            logger.log_step(i, {"reward": np.sin(i * 0.1) * 50 + 50})
            logger.log_step(i, {"loss": np.exp(-i * 0.1) * 0.5})
        
        # Plot metrics
        logger.plot_metrics(["reward", "loss"], save_path=temp_log_dir)
        
        # Check plot files exist
        assert (Path(temp_log_dir) / "reward_plot.png").exists()
        assert (Path(temp_log_dir) / "loss_plot.png").exists()
    
    def test_moving_average(self, temp_log_dir):
        """Test moving average calculation."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        # Add noisy data
        np.random.seed(42)
        for i in range(100):
            logger.log_step(i, {"noisy_reward": 50.0 + np.random.randn() * 10})
        
        # Get moving average
        ma = logger.get_moving_average("noisy_reward", window=10)
        
        assert len(ma) == 91  # 100 - 10 + 1
        # Moving average should be smoother (lower std)
        assert np.std(ma) < np.std(logger.metrics["noisy_reward"])
    
    def test_metric_aggregation(self, temp_log_dir):
        """Test aggregating metrics over intervals."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        # Log many steps
        for i in range(100):
            logger.log_step(i, {
                "reward": i % 10,  # Cycles 0-9
                "constant": 5.0
            })
        
        # Aggregate over windows
        aggregated = logger.aggregate_metrics(window_size=10)
        
        assert "reward_mean" in aggregated
        assert "reward_std" in aggregated
        assert "constant_mean" in aggregated
        
        # Each window should have mean 4.5 (mean of 0-9)
        assert all(abs(m - 4.5) < 0.01 for m in aggregated["reward_mean"])
        assert all(m == 5.0 for m in aggregated["constant_mean"])
    
    def test_multiple_metric_types(self, temp_log_dir):
        """Test handling different metric types."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        # Log different types
        logger.log_step(1, {
            "int_metric": 10,
            "float_metric": 10.5,
            "bool_metric": True,
            "numpy_metric": np.float32(15.5)
        })
        
        # All should be stored as floats
        assert logger.metrics["int_metric"] == [10.0]
        assert logger.metrics["float_metric"] == [10.5]
        assert logger.metrics["bool_metric"] == [1.0]
        assert logger.metrics["numpy_metric"] == [15.5]
    
    def test_concurrent_logging(self, temp_log_dir):
        """Test thread-safe logging."""
        import threading
        
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        def log_worker(worker_id, num_logs):
            for i in range(num_logs):
                logger.log_step(worker_id * 1000 + i, {
                    f"worker_{worker_id}_metric": i
                })
        
        # Create multiple threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=log_worker, args=(i, 10))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check all metrics were logged
        for i in range(4):
            metric_name = f"worker_{i}_metric"
            assert metric_name in logger.metrics
            assert len(logger.metrics[metric_name]) == 10
    
    def test_clear_metrics(self, temp_log_dir):
        """Test clearing metrics."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        # Add metrics
        for i in range(10):
            logger.log_step(i, {"reward": i})
        
        assert len(logger.metrics["reward"]) == 10
        
        # Clear metrics
        logger.clear_metrics()
        
        assert len(logger.metrics) == 0
        assert logger.step_count == 0
    
    def test_load_metrics(self, temp_log_dir):
        """Test loading previously saved metrics."""
        # Create and save metrics
        logger1 = MetricsLogger(log_dir=temp_log_dir)
        for i in range(5):
            logger1.log_step(i, {"reward": i * 10})
        logger1._save_metrics()
        logger1.close()
        
        # Create new logger and load
        logger2 = MetricsLogger(log_dir=temp_log_dir)
        logger2.load_metrics()
        
        assert "reward" in logger2.metrics
        assert logger2.metrics["reward"] == [0, 10, 20, 30, 40]
        assert logger2.step_count == 5
    
    def test_error_handling(self, temp_log_dir):
        """Test error handling in logger."""
        logger = MetricsLogger(log_dir=temp_log_dir)
        
        # Invalid metric values
        logger.log_step(1, {"invalid": None})  # Should skip
        assert "invalid" not in logger.metrics
        
        # Invalid metric names
        logger.log_step(2, {"": 10})  # Empty name should skip
        assert "" not in logger.metrics
        
        # Non-numeric values
        logger.log_step(3, {"string_metric": "not a number"})  # Should skip
        assert "string_metric" not in logger.metrics


@pytest.mark.parametrize("interval,num_logs,expected_outputs", [
    (1, 3, 3),  # Output every log
    (5, 10, 2),  # Output at 5 and 10
    (10, 5, 0),  # No output (interval > num_logs)
])
def test_console_log_intervals(temp_log_dir, capsys, interval, num_logs, expected_outputs):
    """Test different console logging intervals."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        console_log_interval=interval,
        use_tensorboard=False
    )
    
    for i in range(1, num_logs + 1):
        logger.log_step(i, {"metric": i})
    
    captured = capsys.readouterr()
    
    # Count "Step" occurrences in output
    step_count = captured.out.count("Step")
    assert step_count == expected_outputs
