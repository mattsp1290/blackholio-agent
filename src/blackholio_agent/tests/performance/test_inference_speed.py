"""
Performance tests for inference speed.

Tests that the agent can process observations and generate actions
fast enough for real-time gameplay (20Hz).
"""

import pytest
import torch
import numpy as np
import time
import asyncio
from blackholio_client.models.game_statistics import PlayerStatistics, SessionStatistics

from ...models import BlackholioModel, BlackholioModelConfig
from ...environment import (
    ObservationSpace, ObservationConfig,
    ActionSpace, ActionConfig,
    BlackholioEnv
)
from ..fixtures.mock_spacetimedb import MockSpacetimeDBClient, MockConfig
from ..fixtures.game_states import get_scenario


class TestInferenceSpeed:
    """Performance benchmarks for inference speed."""
    
    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        config = BlackholioModelConfig(device="cpu")
        return BlackholioModel(config)
    
    @pytest.fixture
    def observation_space(self):
        """Create observation space."""
        return ObservationSpace()
    
    def test_model_forward_pass_speed(self, model, benchmark):
        """Test forward pass speed of the model."""
        # Create batch of observations
        batch_size = 16
        obs = torch.randn(batch_size, 456)
        
        # Benchmark forward pass
        def forward_pass():
            with torch.no_grad():
                return model(obs)
        
        result = benchmark(forward_pass)
        
        # Check that we can do inference fast enough for 20Hz
        # Need < 50ms per batch for real-time operation
        assert benchmark.stats["mean"] < 0.05  # 50ms
        print(f"Forward pass time: {benchmark.stats['mean']*1000:.2f}ms")
    
    def test_action_sampling_speed(self, model, benchmark):
        """Test action sampling speed."""
        batch_size = 8
        obs = torch.randn(batch_size, 456)
        
        def sample_action():
            with torch.no_grad():
                return model.get_action(obs, deterministic=True)
        
        result = benchmark(sample_action)
        
        # Should be fast enough for real-time
        assert benchmark.stats["mean"] < 0.05
        print(f"Action sampling time: {benchmark.stats['mean']*1000:.2f}ms")
    
    def test_observation_processing_speed(self, observation_space, benchmark):
        """Test observation space processing speed."""
        # Create sample game entities
        player_entities = [
            {"entity_id": i, "owner_id": 100, "x": 500+i*10, "y": 400+i*10,
             "mass": 50.0, "radius": 10.0, "velocity_x": 1.0, "velocity_y": 0.0}
            for i in range(3)
        ]
        
        other_entities = [
            {"entity_id": i+10, "owner_id": 101+i, "x": 600+i*20, "y": 450+i*20,
             "mass": 30.0+i*10, "radius": 8.0, "velocity_x": -1.0, "velocity_y": 0.5}
            for i in range(20)
        ]
        
        food_entities = [
            {"entity_id": i+100, "owner_id": 0, "x": 400+i*5, "y": 300+i*5,
             "mass": 1.0, "radius": 3.0}
            for i in range(50)
        ]
        
        def process_observation():
            return observation_space.process_game_state(
                player_entities, other_entities, food_entities
            )
        
        result = benchmark(process_observation)
        
        # Should process observations quickly
        assert benchmark.stats["mean"] < 0.01  # 10ms
        print(f"Observation processing time: {benchmark.stats['mean']*1000:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_end_to_end_inference_speed(self, performance_timer):
        """Test complete inference pipeline speed."""
        # Create environment with mock connection
        env = BlackholioEnv()
        env.connection.client = MockSpacetimeDBClient()
        
        await env.connect()
        
        # Load a complex scenario
        scenario = get_scenario("crowded_area")
        env.connection.client.set_game_scenario(scenario)
        
        obs, _ = env.reset()
        
        # Time multiple inference steps
        n_steps = 100
        times = []
        
        for _ in range(n_steps):
            start = time.perf_counter()
            
            # Get observation
            obs = env._get_observation()
            
            # Model inference (if we had the model in env)
            # For now, just simulate with a forward pass
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                # Simulate model inference
                action = env.action_space.sample()
            
            # Execute action
            await env.action_space.execute_action(env.connection, action)
            
            end = time.perf_counter()
            times.append(end - start)
        
        await env.close()
        
        # Calculate statistics
        avg_time = mean(times)
        std_time = stdev(times) if len(times) > 1 else 0
        max_time = max(times)
        
        print(f"End-to-end inference - Avg: {avg_time*1000:.2f}ms, "
              f"Std: {std_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
        
        # Average should be under 50ms for 20Hz operation
        assert avg_time < 0.05
        # Max should not be too high (allow some outliers)
        assert max_time < 0.1
    
    def test_batch_inference_scaling(self, model):
        """Test how inference scales with batch size."""
        batch_sizes = [1, 4, 8, 16, 32]
        times = []
        
        for batch_size in batch_sizes:
            obs = torch.randn(batch_size, 456)
            
            # Time inference
            start = time.perf_counter()
            for _ in range(100):
                with torch.no_grad():
                    model(obs)
            end = time.perf_counter()
            
            avg_time = (end - start) / 100
            times.append(avg_time)
            
            print(f"Batch size {batch_size}: {avg_time*1000:.2f}ms")
        
        # Check that batch processing is efficient
        # Time should not scale linearly with batch size
        time_ratio = times[-1] / times[0]  # 32 vs 1
        batch_ratio = batch_sizes[-1] / batch_sizes[0]
        
        # Should have some benefit from batching
        assert time_ratio < batch_ratio * 0.5  # At least 2x speedup
    
    def test_model_size_impact(self):
        """Test impact of model size on inference speed."""
        model_configs = [
            ("Small", BlackholioModelConfig(hidden_size=64, num_layers=1)),
            ("Medium", BlackholioModelConfig(hidden_size=128, num_layers=2)),
            ("Large", BlackholioModelConfig(hidden_size=256, num_layers=3)),
        ]
        
        obs = torch.randn(8, 456)
        
        for name, config in model_configs:
            model = BlackholioModel(config)
            model.eval()
            
            # Time inference
            start = time.perf_counter()
            for _ in range(100):
                with torch.no_grad():
                    model(obs)
            end = time.perf_counter()
            
            avg_time = (end - start) / 100
            param_count = model._count_parameters()
            
            print(f"{name} model ({param_count:,} params): {avg_time*1000:.2f}ms")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference_speed(self):
        """Test inference speed on GPU."""
        device = "cuda"
        config = BlackholioModelConfig(device=device)
        model = BlackholioModel(config)
        
        batch_size = 16
        obs = torch.randn(batch_size, 456).to(device)
        
        # Warm up GPU
        for _ in range(10):
            with torch.no_grad():
                model(obs)
        
        # Time inference
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(100):
            with torch.no_grad():
                model(obs)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time = (end - start) / 100
        print(f"GPU inference time: {avg_time*1000:.2f}ms")
        
        # GPU should be significantly faster
        assert avg_time < 0.01  # 10ms
    
    def test_parallel_environment_inference(self):
        """Test inference speed with multiple parallel environments."""
        n_envs = 8
        obs_space = ObservationSpace()
        
        # Simulate parallel observations
        all_observations = []
        for _ in range(n_envs):
            player_entities = [{"entity_id": 1, "owner_id": 100, "x": 500, "y": 400,
                               "mass": 50.0, "radius": 10.0, "velocity_x": 1.0, "velocity_y": 0.0}]
            other_entities = [{"entity_id": i+2, "owner_id": 101, "x": 600+i*10, "y": 450+i*10,
                              "mass": 30.0, "radius": 8.0, "velocity_x": -1.0, "velocity_y": 0.5}
                             for i in range(10)]
            food_entities = []
            
            obs = obs_space.process_game_state(player_entities, other_entities, food_entities)
            all_observations.append(obs)
        
        # Stack observations
        batch_obs = np.stack(all_observations)
        
        # Test model inference on batch
        model = BlackholioModel()
        obs_tensor = torch.FloatTensor(batch_obs)
        
        start = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                actions, _ = model.get_action(obs_tensor)
        end = time.perf_counter()
        
        avg_time = (end - start) / 100
        per_env_time = avg_time / n_envs
        
        print(f"Parallel inference ({n_envs} envs): {avg_time*1000:.2f}ms total, "
              f"{per_env_time*1000:.2f}ms per env")
        
        # Should handle multiple environments efficiently
        assert avg_time < 0.05  # 50ms for all environments


@pytest.mark.benchmark(group="inference")
def test_critical_path_latency(benchmark):
    """Test the critical path latency for real-time operation."""
    # Setup
    obs_space = ObservationSpace()
    model = BlackholioModel(BlackholioModelConfig(device="cpu"))
    action_space = ActionSpace()
    
    # Create sample game state
    player_entities = [{"entity_id": 1, "owner_id": 100, "x": 500, "y": 400,
                       "mass": 50.0, "radius": 10.0, "velocity_x": 1.0, "velocity_y": 0.0}]
    other_entities = []
    food_entities = []
    
    def critical_path():
        # 1. Process observation
        obs = obs_space.process_game_state(player_entities, other_entities, food_entities)
        
        # 2. Model inference
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            actions, _ = model.get_action(obs_tensor, deterministic=True)
        
        # 3. Process action
        movement, split = action_space._process_action({
            "movement": actions["movement"][0].numpy(),
            "split": actions["split"][0].item()
        })
        
        return movement, split
    
    result = benchmark(critical_path)
    
    print(f"Critical path latency: {benchmark.stats['mean']*1000:.2f}ms")
    
    # Must be under 50ms for 20Hz operation
    assert benchmark.stats["mean"] < 0.05
