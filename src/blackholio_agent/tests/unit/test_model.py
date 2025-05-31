"""
Unit tests for BlackholioModel neural network.

Tests model architecture, forward pass, action sampling, and save/load.
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from ...models import BlackholioModel, BlackholioModelConfig, ModelOutput


class TestBlackholioModel:
    """Test suite for BlackholioModel class."""
    
    def test_initialization(self):
        """Test model initialization with default config."""
        config = BlackholioModelConfig()
        model = BlackholioModel(config)
        
        assert model.config == config
        assert isinstance(model, torch.nn.Module)
        
        # Check that all components are initialized
        assert hasattr(model, 'player_encoder')
        assert hasattr(model, 'entity_attention') or hasattr(model, 'entity_encoder')
        assert hasattr(model, 'food_encoder')
        assert hasattr(model, 'movement_mean')
        assert hasattr(model, 'discrete_policy')
        assert hasattr(model, 'value_head')
    
    def test_custom_configuration(self):
        """Test model with custom configuration."""
        config = BlackholioModelConfig(
            hidden_size=128,
            num_layers=2,
            attention_heads=2,
            use_attention=False,
            use_spatial_features=False,
            use_lstm=True,
            device="cpu"
        )
        model = BlackholioModel(config)
        
        assert model.config.hidden_size == 128
        assert model.config.num_layers == 2
        assert not model.config.use_attention
        assert hasattr(model, 'entity_encoder')  # Should use MLP instead of attention
        assert hasattr(model, 'lstm')
    
    def test_forward_pass(self):
        """Test forward pass with sample observations."""
        model = BlackholioModel()
        batch_size = 4
        obs_dim = 456
        
        # Create random observations
        observations = torch.randn(batch_size, obs_dim)
        
        # Forward pass
        output = model(observations)
        
        # Check output types and shapes
        assert isinstance(output, ModelOutput)
        assert output.movement_mean.shape == (batch_size, 2)
        assert output.movement_std.shape == (batch_size, 2)
        assert output.split_logits.shape == (batch_size, 2)
        assert output.value.shape == (batch_size,)
        assert output.hidden_state is None  # No LSTM by default
    
    def test_forward_pass_with_lstm(self):
        """Test forward pass with LSTM enabled."""
        config = BlackholioModelConfig(use_lstm=True, lstm_hidden_size=64)
        model = BlackholioModel(config)
        
        batch_size = 4
        observations = torch.randn(batch_size, 456)
        
        # First forward pass (no hidden state)
        output1 = model(observations)
        assert output1.hidden_state is not None
        assert len(output1.hidden_state) == 2  # (h, c)
        assert output1.hidden_state[0].shape == (1, batch_size, 64)
        
        # Second forward pass with hidden state
        output2 = model(observations, output1.hidden_state)
        assert output2.hidden_state is not None
    
    def test_get_action(self):
        """Test action sampling from model."""
        model = BlackholioModel()
        observations = torch.randn(2, 456)
        
        # Deterministic action
        actions_det, output_det = model.get_action(observations, deterministic=True)
        
        assert 'movement' in actions_det
        assert 'split' in actions_det
        assert actions_det['movement'].shape == (2, 2)
        assert actions_det['split'].shape == (2,)
        
        # Movement should be bounded
        assert torch.all(actions_det['movement'] >= -1.0)
        assert torch.all(actions_det['movement'] <= 1.0)
        
        # Stochastic action
        actions_stoch, output_stoch = model.get_action(observations, deterministic=False)
        
        # Stochastic actions should vary (with high probability)
        actions_stoch2, _ = model.get_action(observations, deterministic=False)
        assert not torch.allclose(actions_stoch['movement'], actions_stoch2['movement'])
    
    def test_get_log_prob(self):
        """Test log probability calculation."""
        model = BlackholioModel()
        observations = torch.randn(3, 456)
        
        # Sample actions
        actions = {
            'movement': torch.randn(3, 2).clamp(-1, 1),
            'split': torch.randint(0, 2, (3, 1)).float()
        }
        
        # Get log probs
        log_prob, entropy = model.get_log_prob(observations, actions)
        
        assert log_prob.shape == (3,)
        assert entropy.shape == (3,)
        
        # Log probs should be negative (or zero)
        assert torch.all(log_prob <= 0)
        
        # Entropy should be positive
        assert torch.all(entropy >= 0)
    
    def test_save_load(self):
        """Test model save and load functionality."""
        config = BlackholioModelConfig(hidden_size=128, num_layers=2)
        model = BlackholioModel(config)
        
        # Get initial output for comparison
        observations = torch.randn(2, 456)
        with torch.no_grad():
            initial_output = model(observations)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model.save(tmp.name)
            
            # Load model
            loaded_model = BlackholioModel.load(tmp.name)
            
            # Check config is preserved
            assert loaded_model.config.hidden_size == 128
            assert loaded_model.config.num_layers == 2
            
            # Check outputs match
            with torch.no_grad():
                loaded_output = loaded_model(observations)
            
            assert torch.allclose(initial_output.movement_mean, loaded_output.movement_mean)
            assert torch.allclose(initial_output.split_logits, loaded_output.split_logits)
            assert torch.allclose(initial_output.value, loaded_output.value)
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = BlackholioModel()
        param_count = model._count_parameters()
        
        # Should have > 1M parameters as mentioned in docs
        assert param_count > 1_000_000
        
        # Smaller model should have fewer parameters
        small_config = BlackholioModelConfig(hidden_size=64, num_layers=1)
        small_model = BlackholioModel(small_config)
        small_param_count = small_model._count_parameters()
        
        assert small_param_count < param_count
    
    def test_device_handling(self):
        """Test model device handling."""
        # CPU model
        cpu_model = BlackholioModel(BlackholioModelConfig(device="cpu"))
        assert next(cpu_model.parameters()).device.type == "cpu"
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            cuda_model = BlackholioModel(BlackholioModelConfig(device="cuda"))
            assert next(cuda_model.parameters()).device.type == "cuda"
    
    def test_attention_mechanism(self):
        """Test attention mechanism for entity processing."""
        config = BlackholioModelConfig(use_attention=True)
        model = BlackholioModel(config)
        
        # Create observations with varying entity features
        batch_size = 2
        observations = torch.zeros(batch_size, 456)
        
        # Set player state
        observations[:, :6] = torch.randn(batch_size, 6)
        
        # Set some entity features (should attend to these)
        entity_start = 6
        observations[:, entity_start:entity_start+10] = torch.randn(batch_size, 10) * 2
        
        output = model(observations)
        
        # Model should process entities through attention
        assert hasattr(model, 'entity_attention')
        assert output.movement_mean is not None
    
    def test_spatial_features(self):
        """Test spatial feature extraction."""
        config = BlackholioModelConfig(use_spatial_features=True)
        model = BlackholioModel(config)
        
        observations = torch.randn(2, 456)
        output = model(observations)
        
        # Should have spatial extractor
        assert hasattr(model, 'spatial_extractor')
        assert output.movement_mean is not None
    
    def test_activation_functions(self):
        """Test different activation functions."""
        for activation in ["relu", "gelu", "tanh"]:
            config = BlackholioModelConfig(activation=activation)
            model = BlackholioModel(config)
            
            # Should initialize without errors
            observations = torch.randn(1, 456)
            output = model(observations)
            assert output.movement_mean is not None
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = BlackholioModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        observations = torch.randn(4, 456)
        actions = {
            'movement': torch.randn(4, 2).clamp(-1, 1),
            'split': torch.randint(0, 2, (4,)).float()
        }
        
        # Forward pass
        log_prob, entropy = model.get_log_prob(observations, actions)
        _, output = model.get_action(observations)
        
        # Create dummy loss
        loss = -log_prob.mean() + output.value.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.all(param.grad == 0)
    
    def test_output_ranges(self):
        """Test that outputs are in expected ranges."""
        model = BlackholioModel()
        observations = torch.randn(10, 456)
        
        with torch.no_grad():
            output = model(observations)
        
        # Movement should be bounded by tanh
        assert torch.all(output.movement_mean >= -1.0)
        assert torch.all(output.movement_mean <= 1.0)
        
        # Movement std should be positive
        assert torch.all(output.movement_std > 0)
        assert torch.all(output.movement_std <= 1.0)  # Clamped in model
        
        # Split logits can be any value
        # Value can be any value
    
    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        model = BlackholioModel()
        
        # Create observations where one sample is zeros
        observations = torch.randn(3, 456)
        observations[1] = 0
        
        with torch.no_grad():
            output = model(observations)
        
        # Outputs should be different for different inputs
        assert not torch.allclose(output.movement_mean[0], output.movement_mean[1])
        assert not torch.allclose(output.value[0], output.value[1])
    
    def test_edge_cases(self):
        """Test various edge cases."""
        model = BlackholioModel()
        
        # Single sample
        obs_single = torch.randn(1, 456)
        output_single = model(obs_single)
        assert output_single.movement_mean.shape == (1, 2)
        
        # Large batch
        obs_large = torch.randn(128, 456)
        output_large = model(obs_large)
        assert output_large.movement_mean.shape == (128, 2)
        
        # All zeros input
        obs_zeros = torch.zeros(2, 456)
        output_zeros = model(obs_zeros)
        assert not torch.any(torch.isnan(output_zeros.movement_mean))
        
        # Very large values
        obs_large_vals = torch.randn(2, 456) * 1000
        output_large_vals = model(obs_large_vals)
        assert not torch.any(torch.isnan(output_large_vals.movement_mean))
        assert not torch.any(torch.isinf(output_large_vals.movement_mean))


@pytest.mark.parametrize("use_attention,use_spatial,use_lstm", [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True),
])
def test_model_variants(use_attention, use_spatial, use_lstm):
    """Test different model architecture variants."""
    config = BlackholioModelConfig(
        use_attention=use_attention,
        use_spatial_features=use_spatial,
        use_lstm=use_lstm,
        hidden_size=64  # Smaller for faster tests
    )
    model = BlackholioModel(config)
    
    observations = torch.randn(2, 456)
    
    # Should work without errors
    output = model(observations)
    actions, _ = model.get_action(observations)
    
    assert output.movement_mean.shape == (2, 2)
    assert actions['movement'].shape == (2, 2)
    
    if use_lstm:
        assert output.hidden_state is not None
    else:
        assert output.hidden_state is None
