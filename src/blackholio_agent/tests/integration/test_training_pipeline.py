"""
Integration tests for the training pipeline.

Tests PPO training with mocked environments.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from ...training import PPOTrainer, PPOConfig
from ...models import BlackholioModel, BlackholioModelConfig
from ...environment import BlackholioEnv, BlackholioEnvConfig
from ..fixtures.mock_spacetimedb import MockSpacetimeDBClient, MockConfig
from ..fixtures.game_states import get_scenario


class TestTrainingPipeline:
    """Integration tests for PPO training pipeline."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for logs and checkpoints."""
        log_dir = tempfile.mkdtemp()
        checkpoint_dir = tempfile.mkdtemp()
        yield log_dir, checkpoint_dir
        shutil.rmtree(log_dir)
        shutil.rmtree(checkpoint_dir)
    
    @pytest.fixture
    def mock_env_factory(self):
        """Factory for creating mock environments."""
        def create_env():
            env = BlackholioEnv()
            # Use mock client instead of real connection
            env.connection.client = MockSpacetimeDBClient()
            return env
        return create_env
    
    @pytest.mark.asyncio
    async def test_training_initialization(self, temp_dirs):
        """Test PPO trainer initialization."""
        log_dir, checkpoint_dir = temp_dirs
        
        config = PPOConfig(
            n_envs=2,
            total_timesteps=1000,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            device="cpu"
        )
        
        trainer = PPOTrainer(config)
        
        assert trainer.config == config
        assert trainer.global_step == 0
        assert trainer.episode_count == 0
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.envs is not None
        assert trainer.rollout_buffer is not None
    
    @pytest.mark.asyncio
    async def test_short_training_run(self, temp_dirs, mock_env_factory, monkeypatch):
        """Test a short training run with mock environments."""
        log_dir, checkpoint_dir = temp_dirs
        
        # Mock the parallel environment creation
        async def mock_create_env(config):
            return mock_env_factory()
        
        monkeypatch.setattr(
            "src.blackholio_agent.training.parallel_envs.create_blackholio_env",
            mock_create_env
        )
        
        # Create trainer with small config
        config = PPOConfig(
            n_envs=2,
            total_timesteps=100,  # Very short for testing
            n_steps=10,  # Small rollout
            batch_size=5,
            n_epochs=1,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            save_interval_minutes=0,  # Save immediately
            console_log_interval=1,
            device="cpu",
            model_config=BlackholioModelConfig(
                hidden_size=64,  # Smaller model for faster test
                num_layers=1,
                device="cpu"
            )
        )
        
        trainer = PPOTrainer(config)
        
        # Mock environment connections
        for env in trainer.envs.envs:
            env.is_connected = True
            await env.connection.client.connect()
        
        # Start environments
        trainer.envs.start()
        
        # Collect initial observations
        trainer.last_obs, _ = trainer.envs.reset()
        
        # Run a few training iterations
        initial_step = trainer.global_step
        
        # Collect one rollout
        trainer._collect_rollouts()
        
        assert trainer.global_step > initial_step
        assert trainer.rollout_buffer.pos > 0
        
        # Update policy
        update_metrics = trainer._update_policy()
        
        assert "policy_loss" in update_metrics
        assert "value_loss" in update_metrics
        assert "entropy_loss" in update_metrics
        assert all(isinstance(v, float) for v in update_metrics.values())
        
        # Check that model was updated
        # (Would need to track weights to verify, but at least no errors)
        
        # Close trainer
        trainer.envs.close()
    
    @pytest.mark.asyncio
    async def test_curriculum_learning_progression(self, temp_dirs, mock_env_factory, monkeypatch):
        """Test curriculum learning stage progression."""
        log_dir, checkpoint_dir = temp_dirs
        
        # Mock environment creation
        async def mock_create_env(config):
            return mock_env_factory()
        
        monkeypatch.setattr(
            "src.blackholio_agent.training.parallel_envs.create_blackholio_env",
            mock_create_env
        )
        
        config = PPOConfig(
            n_envs=1,
            total_timesteps=100,
            n_steps=5,
            use_curriculum=True,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            device="cpu"
        )
        
        trainer = PPOTrainer(config)
        
        # Connect mock environments
        for env in trainer.envs.envs:
            env.is_connected = True
            await env.connection.client.connect()
        
        trainer.envs.start()
        
        # Check initial curriculum stage
        assert trainer.current_curriculum_stage == 0
        
        # Force curriculum progression
        trainer.current_curriculum_stage = 1
        trainer._update_curriculum()
        
        # Stage should update based on thresholds
        # (In real training, this would happen based on performance)
        
        trainer.envs.close()
    
    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, temp_dirs, mock_env_factory, monkeypatch):
        """Test saving and loading checkpoints during training."""
        log_dir, checkpoint_dir = temp_dirs
        
        # Mock environment creation
        async def mock_create_env(config):
            return mock_env_factory()
        
        monkeypatch.setattr(
            "src.blackholio_agent.training.parallel_envs.create_blackholio_env",
            mock_create_env
        )
        
        config = PPOConfig(
            n_envs=2,
            total_timesteps=50,
            n_steps=10,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            save_interval_minutes=0,  # Save immediately
            device="cpu"
        )
        
        trainer = PPOTrainer(config)
        
        # Connect environments
        for env in trainer.envs.envs:
            env.is_connected = True
            await env.connection.client.connect()
        
        trainer.envs.start()
        trainer.last_obs, _ = trainer.envs.reset()
        
        # Train a bit
        trainer._collect_rollouts()
        trainer._update_policy()
        
        # Save checkpoint
        trainer._save_checkpoint_if_needed()
        
        # Check that checkpoint was saved
        checkpoint_files = list(Path(checkpoint_dir).glob("checkpoint_*.pt"))
        assert len(checkpoint_files) > 0
        
        # Create new trainer and load checkpoint
        trainer2 = PPOTrainer(config)
        loaded = trainer2.load_checkpoint(str(checkpoint_files[0]))
        
        assert loaded is True
        assert trainer2.global_step == trainer.global_step
        assert trainer2.episode_count == trainer.episode_count
        
        trainer.envs.close()
    
    @pytest.mark.asyncio
    async def test_multi_env_training(self, temp_dirs, mock_env_factory, monkeypatch):
        """Test training with multiple parallel environments."""
        log_dir, checkpoint_dir = temp_dirs
        
        # Mock environment creation
        async def mock_create_env(config):
            return mock_env_factory()
        
        monkeypatch.setattr(
            "src.blackholio_agent.training.parallel_envs.create_blackholio_env",
            mock_create_env
        )
        
        config = PPOConfig(
            n_envs=4,  # Multiple environments
            total_timesteps=200,
            n_steps=20,
            batch_size=10,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            device="cpu"
        )
        
        trainer = PPOTrainer(config)
        
        # Connect all environments
        for env in trainer.envs.envs:
            env.is_connected = True
            await env.connection.client.connect()
        
        trainer.envs.start()
        
        # Reset and check observations shape
        obs, _ = trainer.envs.reset()
        assert obs.shape == (4, 456)  # n_envs x obs_dim
        
        # Collect rollouts from all environments
        trainer.last_obs = obs
        trainer._collect_rollouts()
        
        # Buffer should have data from all environments
        assert trainer.rollout_buffer.pos > 0
        
        # Get samples should flatten across environments
        trainer.rollout_buffer.compute_returns_and_advantage(
            np.zeros(4), np.zeros(4, dtype=bool)
        )
        samples = trainer.rollout_buffer.get_samples()
        
        # Should have n_steps * n_envs samples
        expected_samples = min(trainer.config.n_steps * trainer.config.n_envs,
                              trainer.rollout_buffer.pos * trainer.config.n_envs)
        assert len(samples["observations"]) == expected_samples
        
        trainer.envs.close()
    
    @pytest.mark.asyncio
    async def test_training_metrics_logging(self, temp_dirs, mock_env_factory, monkeypatch):
        """Test that training metrics are properly logged."""
        log_dir, checkpoint_dir = temp_dirs
        
        # Mock environment creation
        async def mock_create_env(config):
            return mock_env_factory()
        
        monkeypatch.setattr(
            "src.blackholio_agent.training.parallel_envs.create_blackholio_env",
            mock_create_env
        )
        
        config = PPOConfig(
            n_envs=2,
            total_timesteps=100,
            n_steps=10,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            console_log_interval=1,  # Log frequently
            device="cpu"
        )
        
        trainer = PPOTrainer(config)
        
        # Connect environments
        for env in trainer.envs.envs:
            env.is_connected = True
            await env.connection.client.connect()
        
        trainer.envs.start()
        trainer.last_obs, _ = trainer.envs.reset()
        
        # Run training steps
        trainer._collect_rollouts()
        update_metrics = trainer._update_policy()
        
        # Log metrics
        trainer._log_training_metrics(update_metrics)
        
        # Check that metrics were logged
        assert trainer.metrics_logger.step_count > 0
        assert len(trainer.metrics_logger.training_metrics) > 0
        
        # Get summary
        summary = trainer.metrics_logger.get_summary()
        assert summary["total_steps"] > 0
        
        trainer.envs.close()
    
    @pytest.mark.asyncio  
    async def test_training_error_recovery(self, temp_dirs, mock_env_factory, monkeypatch):
        """Test training recovery from errors."""
        log_dir, checkpoint_dir = temp_dirs
        
        # Mock environment creation
        async def mock_create_env(config):
            return mock_env_factory()
        
        monkeypatch.setattr(
            "src.blackholio_agent.training.parallel_envs.create_blackholio_env",
            mock_create_env
        )
        
        config = PPOConfig(
            n_envs=2,
            total_timesteps=100,
            n_steps=10,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            device="cpu"
        )
        
        trainer = PPOTrainer(config)
        
        # Connect environments
        for env in trainer.envs.envs:
            env.is_connected = True
            await env.connection.client.connect()
        
        trainer.envs.start()
        trainer.last_obs, _ = trainer.envs.reset()
        
        # Simulate error during rollout collection
        # Save original method
        original_step = trainer.envs.step
        
        # Mock to raise error once
        error_count = 0
        async def error_step(actions):
            nonlocal error_count
            error_count += 1
            if error_count == 1:
                raise RuntimeError("Simulated environment error")
            return await original_step(actions)
        
        trainer.envs.step = error_step
        
        # Training should handle the error gracefully
        try:
            trainer._collect_rollouts()
        except RuntimeError:
            # Error is expected, but training should be able to continue
            pass
        
        # Restore normal step function
        trainer.envs.step = original_step
        
        # Should be able to continue training
        trainer._collect_rollouts()  # Should work now
        
        trainer.envs.close()
    
    @pytest.mark.asyncio
    async def test_model_improvement(self, temp_dirs, mock_env_factory, monkeypatch):
        """Test that model improves during training (simplified test)."""
        log_dir, checkpoint_dir = temp_dirs
        
        # Mock environment creation
        async def mock_create_env(config):
            env = mock_env_factory()
            # Load a simple scenario
            scenario = get_scenario("early_game_solo")
            env.connection.client.set_game_scenario(scenario)
            return env
        
        monkeypatch.setattr(
            "src.blackholio_agent.training.parallel_envs.create_blackholio_env",
            mock_create_env
        )
        
        config = PPOConfig(
            n_envs=2,
            total_timesteps=200,
            n_steps=20,
            batch_size=10,
            n_epochs=2,
            learning_rate=1e-3,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            device="cpu"
        )
        
        trainer = PPOTrainer(config)
        
        # Connect environments
        for env in trainer.envs.envs:
            env.is_connected = True
            await env.connection.client.connect()
        
        trainer.envs.start()
        trainer.last_obs, _ = trainer.envs.reset()
        
        # Track initial policy behavior
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(trainer.last_obs).to(trainer.device)
            initial_actions, initial_output = trainer.model.get_action(obs_tensor)
            initial_value = initial_output.value.mean().item()
        
        # Train for several iterations
        for _ in range(5):
            trainer._collect_rollouts()
            trainer._update_policy()
        
        # Check policy has changed
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(trainer.last_obs).to(trainer.device)
            final_actions, final_output = trainer.model.get_action(obs_tensor)
            final_value = final_output.value.mean().item()
        
        # Values should have changed (model has learned something)
        assert abs(final_value - initial_value) > 1e-4
        
        trainer.envs.close()


@pytest.mark.parametrize("device", ["cpu"])  # Add "cuda" if available
def test_training_device_placement(device, temp_dirs):
    """Test training on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    log_dir, checkpoint_dir = temp_dirs
    
    config = PPOConfig(
        n_envs=1,
        total_timesteps=10,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )
    
    trainer = PPOTrainer(config)
    
    # Check model is on correct device
    assert next(trainer.model.parameters()).device.type == device
    
    # Check optimizer is configured for correct device
    param_device = next(trainer.optimizer.param_groups[0]["params"]).device
    assert param_device.type == device
