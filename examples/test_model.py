"""
Test script for the BlackholioModel neural network.

This script demonstrates how to use the model and verifies
that it can process observations and produce valid outputs.
"""

import torch
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blackholio_agent.models import BlackholioModel, BlackholioModelConfig
from src.blackholio_agent.environment import ObservationSpace

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_model_forward():
    """Test model forward pass"""
    print("\n=== Testing Model Forward Pass ===")
    
    # Create model with default config
    config = BlackholioModelConfig()
    model = BlackholioModel(config)
    
    # Create dummy observation
    batch_size = 4
    obs = torch.randn(batch_size, config.observation_dim)
    
    # Forward pass
    output = model(obs)
    
    print(f"Input shape: {obs.shape}")
    print(f"Movement mean shape: {output.movement_mean.shape}")
    print(f"Movement std shape: {output.movement_std.shape}")
    print(f"Split logits shape: {output.split_logits.shape}")
    print(f"Value shape: {output.value.shape}")
    
    # Verify shapes
    assert output.movement_mean.shape == (batch_size, 2)
    assert output.movement_std.shape == (batch_size, 2)
    assert output.split_logits.shape == (batch_size, 2)
    assert output.value.shape == (batch_size,)
    
    print("✓ Forward pass test passed!")


def test_model_with_real_observation():
    """Test model with observation from the environment"""
    print("\n=== Testing Model with Real Observation ===")
    
    # Create observation space
    obs_space = ObservationSpace()
    
    # Create dummy game state
    player_entities = [
        {'x': 100, 'y': 100, 'mass': 50, 'vx': 5, 'vy': -3},
        {'x': 105, 'y': 102, 'mass': 30, 'vx': 4, 'vy': -2}
    ]
    
    other_entities = [
        {'x': 150, 'y': 120, 'mass': 80, 'vx': -2, 'vy': 1},
        {'x': 80, 'y': 90, 'mass': 20, 'vx': 3, 'vy': 3},
        {'x': 200, 'y': 200, 'mass': 100, 'vx': 0, 'vy': 0}
    ]
    
    food_entities = [
        {'x': 110, 'y': 110},
        {'x': 90, 'y': 95},
        {'x': 120, 'y': 105}
    ]
    
    # Process game state
    obs_array = obs_space.process_game_state(player_entities, other_entities, food_entities)
    
    # Convert to tensor and add batch dimension
    obs_tensor = torch.from_numpy(obs_array).unsqueeze(0)
    
    # Create model
    model = BlackholioModel()
    
    # Forward pass
    output = model(obs_tensor)
    
    print(f"Observation shape: {obs_tensor.shape}")
    print(f"Movement mean: {output.movement_mean.squeeze().detach().numpy()}")
    print(f"Movement std: {output.movement_std.squeeze().detach().numpy()}")
    print(f"Split logits: {output.split_logits.squeeze().detach().numpy()}")
    print(f"Value estimate: {output.value.item()}")
    
    print("✓ Real observation test passed!")


def test_model_action_sampling():
    """Test action sampling from the model"""
    print("\n=== Testing Action Sampling ===")
    
    # Create model
    model = BlackholioModel()
    
    # Create observation
    obs = torch.randn(1, model.config.observation_dim)
    
    # Get stochastic action
    actions_stochastic, output_stochastic = model.get_action(obs, deterministic=False)
    
    print("Stochastic action:")
    print(f"  Movement: {actions_stochastic['movement'].squeeze().detach().numpy()}")
    print(f"  Split: {actions_stochastic['split'].item()}")
    
    # Get deterministic action
    actions_deterministic, output_deterministic = model.get_action(obs, deterministic=True)
    
    print("\nDeterministic action:")
    print(f"  Movement: {actions_deterministic['movement'].squeeze().detach().numpy()}")
    print(f"  Split: {actions_deterministic['split'].item()}")
    
    # Verify action bounds
    assert torch.all(actions_stochastic['movement'] >= -1.0)
    assert torch.all(actions_stochastic['movement'] <= 1.0)
    assert actions_stochastic['split'].item() in [0.0, 1.0]
    
    print("✓ Action sampling test passed!")


def test_model_log_prob():
    """Test log probability calculation"""
    print("\n=== Testing Log Probability Calculation ===")
    
    # Create model
    model = BlackholioModel()
    
    # Create observation and actions
    batch_size = 8
    obs = torch.randn(batch_size, model.config.observation_dim)
    
    actions = {
        'movement': torch.randn(batch_size, 2).clamp(-1, 1),
        'split': torch.randint(0, 2, (batch_size, 1)).float()
    }
    
    # Calculate log probabilities
    log_prob, entropy = model.get_log_prob(obs, actions)
    
    print(f"Log probability shape: {log_prob.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Mean log prob: {log_prob.mean().item():.4f}")
    print(f"Mean entropy: {entropy.mean().item():.4f}")
    
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert torch.all(torch.isfinite(log_prob))
    assert torch.all(entropy >= 0)
    
    print("✓ Log probability test passed!")


def test_model_save_load():
    """Test model saving and loading"""
    print("\n=== Testing Model Save/Load ===")
    
    # Create model with custom config
    config = BlackholioModelConfig(
        hidden_size=128,
        num_layers=2,
        use_lstm=True,
        dropout=0.2
    )
    model = BlackholioModel(config)
    
    # Put model in eval mode for deterministic behavior
    model.eval()
    
    # Create observation and get output
    obs = torch.randn(2, config.observation_dim)
    with torch.no_grad():
        output_original = model(obs)
    
    # Save model
    save_path = "test_model.pth"
    model.save(save_path)
    
    # Load model
    loaded_model = BlackholioModel.load(save_path)
    loaded_model.eval()
    
    # Get output from loaded model
    with torch.no_grad():
        output_loaded = loaded_model(obs)
    
    # Compare outputs
    assert torch.allclose(output_original.movement_mean, output_loaded.movement_mean)
    assert torch.allclose(output_original.value, output_loaded.value)
    
    print("✓ Save/load test passed!")
    
    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)


def test_model_with_lstm():
    """Test model with LSTM enabled"""
    print("\n=== Testing Model with LSTM ===")
    
    # Create model with LSTM
    config = BlackholioModelConfig(use_lstm=True)
    model = BlackholioModel(config)
    
    # Create sequence of observations
    seq_len = 5
    batch_size = 2
    hidden_state = None
    
    for t in range(seq_len):
        obs = torch.randn(batch_size, config.observation_dim)
        output = model(obs, hidden_state)
        hidden_state = output.hidden_state
        
        print(f"Step {t+1}: Value = {output.value.mean().item():.4f}")
    
    assert hidden_state is not None
    assert len(hidden_state) == 2  # (h, c) for LSTM
    
    print("✓ LSTM test passed!")


def test_model_without_attention():
    """Test model without attention mechanism"""
    print("\n=== Testing Model without Attention ===")
    
    # Create model without attention
    config = BlackholioModelConfig(use_attention=False)
    model = BlackholioModel(config)
    
    # Forward pass
    obs = torch.randn(4, config.observation_dim)
    output = model(obs)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Output value mean: {output.value.mean().item():.4f}")
    
    print("✓ No-attention test passed!")


def main():
    """Run all tests"""
    print("Testing Blackholio Neural Network Model")
    print("=" * 50)
    
    # Run tests
    test_model_forward()
    test_model_with_real_observation()
    test_model_action_sampling()
    test_model_log_prob()
    test_model_save_load()
    test_model_with_lstm()
    test_model_without_attention()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("\nThe BlackholioModel is ready for training!")


if __name__ == "__main__":
    main()
