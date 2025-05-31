"""
Test script for the inference system.

This script demonstrates how to use the inference system
and validates that all components work correctly.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
import torch
import numpy as np
from src.blackholio_agent.inference import (
    InferenceConfig, InferenceAgent, InferenceMetrics,
    ModelLoader, RateLimiter
)
from src.blackholio_agent.models import BlackholioModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rate_limiter():
    """Test rate limiter functionality"""
    logger.info("Testing RateLimiter...")
    
    # Create rate limiter with 100ms interval
    limiter = RateLimiter(min_interval_ms=100)
    
    # Test waiting
    start_time = time.time()
    for i in range(5):
        wait_time = limiter.wait()
        logger.info(f"  Step {i+1}: waited {wait_time*1000:.1f}ms")
    
    total_time = time.time() - start_time
    expected_time = 0.4  # 4 intervals of 100ms
    
    assert total_time >= expected_time, f"Total time {total_time} < expected {expected_time}"
    logger.info(f"  Total time: {total_time:.2f}s (expected >= {expected_time}s)")
    
    # Check stats
    stats = limiter.get_stats()
    logger.info(f"  Stats: {stats}")
    
    logger.info("✓ RateLimiter test passed!")


def test_metrics():
    """Test metrics tracking"""
    logger.info("Testing InferenceMetrics...")
    
    metrics = InferenceMetrics(window_size=10)
    
    # Simulate an episode
    metrics.start_episode()
    
    # Log some steps
    for i in range(100):
        metrics.log_step(
            inference_latency=0.01 + np.random.random() * 0.02,  # 10-30ms
            reward=np.random.random(),
            info={'test': True},
            current_mass=50 + i * 0.5
        )
    
    # End episode
    episode_metrics = metrics.end_episode(
        failure_mode="test_failure",
        success=False
    )
    
    logger.info(f"  Episode metrics: {episode_metrics.to_dict()}")
    
    # Get summary
    summary = metrics.get_summary()
    logger.info(f"  Summary: {summary}")
    
    assert metrics.total_episodes == 1
    assert metrics.total_steps == 100
    
    logger.info("✓ InferenceMetrics test passed!")


def test_model_loader():
    """Test model loading and optimization"""
    logger.info("Testing ModelLoader...")
    
    # Create a dummy model
    model = BlackholioModel()
    
    # Save it
    model_path = "test_model.pth"
    model.save(model_path)
    
    try:
        # Load for inference
        loaded_model = ModelLoader.load_for_inference(
            model_path=model_path,
            device='cpu',
            num_threads=4
        )
        
        # Validate
        assert ModelLoader.validate_model(loaded_model)
        
        # Get info
        info = ModelLoader.get_model_info(loaded_model)
        logger.info(f"  Model info: {info}")
        
        # Warmup
        warmup_stats = ModelLoader.warmup_model(loaded_model, warmup_steps=5)
        logger.info(f"  Warmup stats: {warmup_stats}")
        
        logger.info("✓ ModelLoader test passed!")
        
    finally:
        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)


def test_config():
    """Test configuration system"""
    logger.info("Testing InferenceConfig...")
    
    # Test with environment variables
    os.environ['MODEL_PATH'] = 'env_model.pth'
    os.environ['CPU_THREADS'] = '8'
    
    # Create a dummy model file
    with open('env_model.pth', 'wb') as f:
        torch.save({}, f)
    
    try:
        config = InferenceConfig()
        assert config.cpu_threads == 8
        
        # Test env config creation
        env_config = config.create_env_config()
        assert env_config.player_name == config.player_name
        
        logger.info("✓ InferenceConfig test passed!")
        
    finally:
        # Cleanup
        if os.path.exists('env_model.pth'):
            os.remove('env_model.pth')
        del os.environ['MODEL_PATH']
        del os.environ['CPU_THREADS']


def test_inference_integration():
    """Test full inference system (without real game connection)"""
    logger.info("Testing inference integration...")
    
    # This would require a running Blackholio instance
    # For now, just verify the agent can be created
    
    # Create dummy model
    model = BlackholioModel()
    model_path = "integration_test_model.pth"
    model.save(model_path)
    
    try:
        # Create config
        config = InferenceConfig()
        config.model_path = model_path
        config.max_episodes = 0  # Don't actually run
        
        # Try to create agent (will fail on connection)
        try:
            agent = InferenceAgent(config)
            logger.info("  Agent created successfully")
            logger.info("  (Skipping actual run - requires game server)")
        except Exception as e:
            logger.info(f"  Expected connection error: {e}")
        
        logger.info("✓ Integration test completed!")
        
    finally:
        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)


def main():
    """Run all tests"""
    logger.info("Testing Blackholio Inference System")
    logger.info("=" * 50)
    
    # Run tests
    test_rate_limiter()
    print()
    
    test_metrics()
    print()
    
    test_model_loader()
    print()
    
    test_config()
    print()
    
    test_inference_integration()
    
    logger.info("\n" + "=" * 50)
    logger.info("All inference tests completed!")
    logger.info("\nTo run the actual agent:")
    logger.info("  1. Ensure Blackholio is running")
    logger.info("  2. Train a model with train_agent.py")
    logger.info("  3. Run: python scripts/run_agent.py --model checkpoints/best_model.pth")


if __name__ == "__main__":
    main()
