"""
Unit tests for CheckpointManager component.

Tests checkpoint saving, loading, and management functionality.
"""

import pytest
import torch
import tempfile
import shutil
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

from ...training import CheckpointManager
from ...models import BlackholioModel, BlackholioModelConfig


class TestCheckpointManager:
    """Test suite for CheckpointManager class."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_checkpoint_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_interval_minutes=30.0,
            keep_best=True,
            keep_recent=5
        )
        
        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
        assert manager.save_interval_minutes == 30.0
        assert manager.keep_best is True
        assert manager.keep_recent == 5
        assert manager.last_save_time is not None
    
    def test_directory_creation(self):
        """Test that checkpoint directory is created if it doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        non_existent = os.path.join(temp_dir, "new_dir")
        
        manager = CheckpointManager(checkpoint_dir=non_existent)
        
        assert os.path.exists(non_existent)
        shutil.rmtree(temp_dir)
    
    def test_save_checkpoint(self, temp_checkpoint_dir):
        """Test saving a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        
        # Create mock model and optimizer
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint_path = manager.save(
            step=1000,
            episode=50,
            model=model,
            optimizer=optimizer,
            metrics={"mean_reward": 25.5, "episode_reward": 30.0}
        )
        
        assert checkpoint_path is not None
        assert os.path.exists(checkpoint_path)
        assert "step_1000" in checkpoint_path
        
        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint["step"] == 1000
        assert checkpoint["episode"] == 50
        assert checkpoint["metrics"]["mean_reward"] == 25.5
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "timestamp" in checkpoint
    
    def test_save_interval(self, temp_checkpoint_dir):
        """Test that save interval is respected."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_interval_minutes=0.1  # 6 seconds for testing
        )
        
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # First save should succeed
        path1 = manager.save(step=100, episode=5, model=model, optimizer=optimizer)
        assert path1 is not None
        
        # Immediate second save should be skipped
        path2 = manager.save(step=200, episode=10, model=model, optimizer=optimizer)
        assert path2 is None
        
        # Wait for interval and save again
        time.sleep(7)  # Wait more than 6 seconds
        path3 = manager.save(step=300, episode=15, model=model, optimizer=optimizer)
        assert path3 is not None
    
    def test_best_checkpoint_tracking(self, temp_checkpoint_dir):
        """Test tracking of best checkpoint."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            keep_best=True,
            metrics_to_track=["episode_reward"]
        )
        
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoints with different rewards
        manager.save(
            step=100, episode=5, model=model, optimizer=optimizer,
            metrics={"episode_reward": 10.0}
        )
        
        manager.last_save_time = 0  # Reset to allow immediate save
        
        manager.save(
            step=200, episode=10, model=model, optimizer=optimizer,
            metrics={"episode_reward": 20.0}  # Better
        )
        
        manager.last_save_time = 0
        
        manager.save(
            step=300, episode=15, model=model, optimizer=optimizer,
            metrics={"episode_reward": 15.0}  # Worse
        )
        
        # Check best checkpoint
        assert manager.best_checkpoint is not None
        assert manager.best_metrics["episode_reward"] == 20.0
        assert "step_200" in manager.best_checkpoint
        
        # Verify best checkpoint symlink exists
        best_link = manager.checkpoint_dir / "best_checkpoint.pt"
        assert best_link.exists()
    
    def test_recent_checkpoint_limit(self, temp_checkpoint_dir):
        """Test that only recent checkpoints are kept."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            keep_recent=3,
            save_interval_minutes=0  # Disable interval for test
        )
        
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save more checkpoints than keep_recent
        paths = []
        for i in range(5):
            path = manager.save(
                step=i * 100,
                episode=i * 5,
                model=model,
                optimizer=optimizer
            )
            paths.append(path)
            time.sleep(0.1)  # Ensure different timestamps
        
        # Only last 3 should exist
        assert not os.path.exists(paths[0])
        assert not os.path.exists(paths[1])
        assert os.path.exists(paths[2])
        assert os.path.exists(paths[3])
        assert os.path.exists(paths[4])
    
    def test_load_checkpoint(self, temp_checkpoint_dir):
        """Test loading a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        
        # Create and save checkpoint
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        
        original_state = model.state_dict()
        
        checkpoint_path = manager.save(
            step=1000,
            episode=50,
            model=model,
            optimizer=optimizer,
            metrics={"test": 123},
            additional_state={"custom": "data"}
        )
        
        # Modify model again
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        
        # Load checkpoint
        loaded = manager.load_checkpoint(checkpoint_path)
        
        assert loaded is not None
        assert loaded["step"] == 1000
        assert loaded["episode"] == 50
        assert loaded["metrics"]["test"] == 123
        assert loaded["additional_state"]["custom"] == "data"
        
        # Restore model and check
        model.load_state_dict(loaded["model_state_dict"])
        restored_state = model.state_dict()
        
        for key in original_state:
            assert torch.allclose(original_state[key], restored_state[key])
    
    def test_load_latest_checkpoint(self, temp_checkpoint_dir):
        """Test loading the latest checkpoint."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_interval_minutes=0
        )
        
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save multiple checkpoints
        for i in range(3):
            manager.save(
                step=i * 100,
                episode=i * 5,
                model=model,
                optimizer=optimizer
            )
            time.sleep(0.1)
        
        # Load latest
        latest = manager.load_latest_checkpoint()
        
        assert latest is not None
        assert latest["step"] == 200  # Last saved
        assert latest["episode"] == 10
    
    def test_list_checkpoints(self, temp_checkpoint_dir):
        """Test listing available checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_interval_minutes=0
        )
        
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Initially empty
        assert len(manager.list_checkpoints()) == 0
        
        # Save checkpoints
        for i in range(3):
            manager.save(
                step=i * 100,
                episode=i * 5,
                model=model,
                optimizer=optimizer
            )
        
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
        
        # Should be sorted by timestamp
        for i in range(len(checkpoints) - 1):
            assert checkpoints[i]["timestamp"] <= checkpoints[i + 1]["timestamp"]
    
    def test_delete_checkpoint(self, temp_checkpoint_dir):
        """Test deleting a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint_path = manager.save(
            step=100,
            episode=5,
            model=model,
            optimizer=optimizer
        )
        
        assert os.path.exists(checkpoint_path)
        
        # Delete it
        manager.delete_checkpoint(checkpoint_path)
        
        assert not os.path.exists(checkpoint_path)
    
    def test_multiple_metrics_tracking(self, temp_checkpoint_dir):
        """Test tracking multiple metrics for best checkpoint."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            keep_best=True,
            metrics_to_track=["episode_reward", "mean_reward"],
            save_interval_minutes=0
        )
        
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoints
        manager.save(
            step=100, episode=5, model=model, optimizer=optimizer,
            metrics={"episode_reward": 10.0, "mean_reward": 8.0}
        )
        
        manager.save(
            step=200, episode=10, model=model, optimizer=optimizer,
            metrics={"episode_reward": 9.0, "mean_reward": 12.0}  # Better mean
        )
        
        # Should track based on first metric (episode_reward)
        assert manager.best_metrics["episode_reward"] == 10.0
        assert "step_100" in manager.best_checkpoint
    
    def test_checkpoint_info_file(self, temp_checkpoint_dir):
        """Test checkpoint info JSON file."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        manager.save(
            step=1000,
            episode=50,
            model=model,
            optimizer=optimizer,
            metrics={"reward": 25.0}
        )
        
        # Check info file
        info_file = manager.checkpoint_dir / "checkpoint_info.json"
        assert info_file.exists()
        
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        assert len(info["checkpoints"]) == 1
        assert info["checkpoints"][0]["step"] == 1000
        assert info["checkpoints"][0]["metrics"]["reward"] == 25.0
        
        if manager.best_checkpoint:
            assert info["best_checkpoint"] is not None
    
    def test_error_handling(self, temp_checkpoint_dir):
        """Test error handling during save/load."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        
        # Test loading non-existent checkpoint
        loaded = manager.load_checkpoint("non_existent.pt")
        assert loaded is None
        
        # Test loading corrupted checkpoint
        bad_file = os.path.join(temp_checkpoint_dir, "bad.pt")
        with open(bad_file, 'w') as f:
            f.write("not a valid checkpoint")
        
        loaded = manager.load_checkpoint(bad_file)
        assert loaded is None
        
        # Test saving with invalid model
        path = manager.save(
            step=100,
            episode=5,
            model=None,  # Invalid
            optimizer=None
        )
        assert path is None
    
    def test_checkpoint_metadata(self, temp_checkpoint_dir):
        """Test checkpoint metadata handling."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        
        model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save with custom metadata
        custom_data = {
            "training_stage": "advanced",
            "hyperparameters": {"lr": 0.001, "batch_size": 64}
        }
        
        checkpoint_path = manager.save(
            step=1000,
            episode=50,
            model=model,
            optimizer=optimizer,
            metrics={"reward": 100.0},
            additional_state=custom_data
        )
        
        # Load and verify
        loaded = manager.load_checkpoint(checkpoint_path)
        assert loaded["additional_state"]["training_stage"] == "advanced"
        assert loaded["additional_state"]["hyperparameters"]["lr"] == 0.001


@pytest.mark.parametrize("keep_recent,num_saves,expected_kept", [
    (3, 5, 3),
    (5, 3, 3),
    (0, 5, 5),  # 0 means keep all
    (1, 10, 1),
])
def test_checkpoint_retention_policy(temp_checkpoint_dir, keep_recent, num_saves, expected_kept):
    """Test different checkpoint retention policies."""
    manager = CheckpointManager(
        checkpoint_dir=temp_checkpoint_dir,
        keep_recent=keep_recent,
        save_interval_minutes=0
    )
    
    model = BlackholioModel(BlackholioModelConfig(hidden_size=64))
    optimizer = torch.optim.Adam(model.parameters())
    
    # Save multiple checkpoints
    for i in range(num_saves):
        manager.save(
            step=i * 100,
            episode=i * 5,
            model=model,
            optimizer=optimizer
        )
        time.sleep(0.01)
    
    # Count existing checkpoints
    checkpoints = list(manager.checkpoint_dir.glob("checkpoint_*.pt"))
    assert len(checkpoints) == expected_kept
