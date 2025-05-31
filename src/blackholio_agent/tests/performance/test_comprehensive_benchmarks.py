"""
Comprehensive performance benchmarks for Blackholio RL agent.

Tests performance characteristics including:
- Inference latency under various conditions
- Memory usage patterns
- Parallel environment scaling
- Training throughput
- Model loading times
"""

import pytest
import time
import torch
import numpy as np
import psutil
import gc
from typing import Dict, List, Tuple
import json
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from ...models import BlackholioModel
from ...environment import BlackholioEnv, ObservationSpace
from ...training import ParallelBlackholioEnv, RolloutBuffer, RolloutBufferConfig
from ..fixtures.mock_spacetimedb import MockSpacetimeDBClient


class PerformanceBenchmarks:
    """Comprehensive performance benchmarking suite"""
    
    @staticmethod
    def measure_inference_latency(model: BlackholioModel, 
                                 batch_sizes: List[int] = [1, 4, 8, 16, 32],
                                 num_iterations: int = 100) -> Dict[int, Dict[str, float]]:
        """Measure inference latency for different batch sizes"""
        
        model.eval()
        device = next(model.parameters()).device
        obs_dim = 456
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create dummy observations
            obs = torch.randn(batch_size, obs_dim, device=device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(obs)
            
            # Measure
            latencies = []
            torch.cuda.synchronize() if device.type == 'cuda' else None
            
            for _ in range(num_iterations):
                start = time.perf_counter()
                
                with torch.no_grad():
                    _ = model(obs)
                    
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)  # Convert to ms
            
            results[batch_size] = {
                "mean_latency_ms": np.mean(latencies),
                "std_latency_ms": np.std(latencies),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "throughput_fps": 1000 / np.mean(latencies) * batch_size
            }
            
        return results
    
    @staticmethod
    def measure_memory_usage(model: BlackholioModel,
                           batch_sizes: List[int] = [1, 8, 32, 128]) -> Dict[int, Dict[str, float]]:
        """Measure memory usage for different batch sizes"""
        
        device = next(model.parameters()).device
        obs_dim = 456
        results = {}
        
        for batch_size in batch_sizes:
            # Clear cache
            gc.collect()
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            # Measure baseline
            process = psutil.Process()
            baseline_mem = process.memory_info().rss / 1024 / 1024  # MB
            
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                baseline_gpu = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Create data and run inference
            obs = torch.randn(batch_size, obs_dim, device=device)
            
            with torch.no_grad():
                output = model(obs)
                
            # Measure peak
            peak_mem = process.memory_info().rss / 1024 / 1024
            
            results[batch_size] = {
                "cpu_memory_mb": peak_mem - baseline_mem,
                "total_memory_mb": peak_mem
            }
            
            if device.type == 'cuda':
                peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024
                results[batch_size]["gpu_memory_mb"] = peak_gpu - baseline_gpu
                results[batch_size]["gpu_peak_mb"] = peak_gpu
        
        return results
    
    @staticmethod
    def benchmark_parallel_environments(num_envs_list: List[int] = [1, 2, 4, 8, 16],
                                      steps_per_env: int = 100) -> Dict[int, Dict[str, float]]:
        """Benchmark parallel environment performance"""
        
        results = {}
        
        for num_envs in num_envs_list:
            # Create mock environments
            mock_clients = [MockSpacetimeDBClient() for _ in range(num_envs)]
            
            # Create parallel env wrapper
            env = ParallelBlackholioEnv(
                env_fns=[lambda c=client: BlackholioEnv(c) for client in mock_clients],
                start_method='spawn' if num_envs > 1 else None
            )
            
            # Warmup
            env.reset()
            for _ in range(10):
                actions = [{"movement": np.random.randn(2), "split": False} 
                          for _ in range(num_envs)]
                env.step(actions)
            
            # Benchmark
            start = time.perf_counter()
            
            obs = env.reset()
            for _ in range(steps_per_env):
                actions = [{"movement": np.random.randn(2), "split": False} 
                          for _ in range(num_envs)]
                obs, rewards, dones, infos = env.step(actions)
            
            end = time.perf_counter()
            total_time = end - start
            total_steps = steps_per_env * num_envs
            
            results[num_envs] = {
                "total_time_s": total_time,
                "steps_per_second": total_steps / total_time,
                "time_per_step_ms": (total_time / steps_per_env) * 1000,
                "speedup": results[1]["steps_per_second"] / (total_steps / total_time) 
                          if 1 in results else 1.0,
                "efficiency": (results[1]["steps_per_second"] / (total_steps / total_time)) / num_envs
                             if 1 in results else 1.0
            }
            
            env.close()
            
        return results
    
    @staticmethod
    def benchmark_training_throughput(model: BlackholioModel,
                                    buffer_sizes: List[int] = [256, 1024, 4096],
                                    batch_size: int = 64,
                                    num_epochs: int = 4) -> Dict[int, Dict[str, float]]:
        """Benchmark training throughput for different buffer sizes"""
        
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        results = {}
        
        for buffer_size in buffer_sizes:
            # Create buffer
            config = RolloutBufferConfig(
                buffer_size=buffer_size,
                n_envs=1,
                observation_dim=456,
                device=str(device)
            )
            buffer = RolloutBuffer(config)
            
            # Fill buffer with dummy data
            for i in range(buffer_size):
                buffer.add(
                    obs=np.random.randn(1, 456),
                    movement_action=np.random.randn(1, 2),
                    split_action=np.random.randint(0, 2, (1,)),
                    reward=np.random.randn(1),
                    episode_start=np.array([i == 0]),
                    value=np.random.randn(1),
                    log_prob=np.random.randn(1)
                )
            
            buffer.compute_returns_and_advantage(
                last_values=np.zeros(1),
                dones=np.ones(1)
            )
            
            # Benchmark training
            start = time.perf_counter()
            
            for epoch in range(num_epochs):
                for batch in buffer.get_samples(batch_size):
                    # Forward pass
                    obs = batch["observations"]
                    actions = torch.cat([
                        batch["movement_actions"],
                        batch["split_actions"].unsqueeze(-1)
                    ], dim=-1)
                    
                    output = model(obs)
                    
                    # Dummy loss
                    loss = output.value.mean()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            end = time.perf_counter()
            total_time = end - start
            total_samples = buffer_size * num_epochs
            
            results[buffer_size] = {
                "total_time_s": total_time,
                "samples_per_second": total_samples / total_time,
                "time_per_epoch_s": total_time / num_epochs,
                "time_per_batch_ms": (total_time / (num_epochs * (buffer_size // batch_size))) * 1000
            }
            
        return results
    
    @staticmethod
    def benchmark_model_operations(model: BlackholioModel) -> Dict[str, float]:
        """Benchmark various model operations"""
        
        device = next(model.parameters()).device
        results = {}
        
        # Model loading time
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save
            start = time.perf_counter()
            torch.save(model.state_dict(), tmp.name)
            save_time = time.perf_counter() - start
            
            # Load
            start = time.perf_counter()
            new_model = BlackholioModel()
            new_model.load_state_dict(torch.load(tmp.name))
            load_time = time.perf_counter() - start
        
        results["save_time_ms"] = save_time * 1000
        results["load_time_ms"] = load_time * 1000
        
        # Model initialization time
        start = time.perf_counter()
        _ = BlackholioModel()
        init_time = time.perf_counter() - start
        results["init_time_ms"] = init_time * 1000
        
        # Parameter count
        results["total_parameters"] = sum(p.numel() for p in model.parameters())
        results["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results["model_size_mb"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        return results


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarks"""
    
    @pytest.fixture
    def model(self):
        """Create model for benchmarking"""
        return BlackholioModel()
    
    @pytest.fixture
    def benchmarks(self):
        """Create benchmark suite"""
        return PerformanceBenchmarks()
    
    @pytest.mark.benchmark
    def test_inference_latency_benchmark(self, model, benchmarks):
        """Benchmark inference latency"""
        results = benchmarks.measure_inference_latency(
            model, 
            batch_sizes=[1, 8, 32],
            num_iterations=50
        )
        
        # Check that latency is reasonable
        for batch_size, metrics in results.items():
            assert metrics["mean_latency_ms"] < 100  # Should be under 100ms
            assert metrics["throughput_fps"] > 10    # At least 10 FPS
            
            print(f"\nBatch size {batch_size}:")
            print(f"  Mean latency: {metrics['mean_latency_ms']:.2f}ms")
            print(f"  P95 latency: {metrics['p95_latency_ms']:.2f}ms")
            print(f"  Throughput: {metrics['throughput_fps']:.1f} FPS")
    
    @pytest.mark.benchmark
    def test_memory_usage_benchmark(self, model, benchmarks):
        """Benchmark memory usage"""
        results = benchmarks.measure_memory_usage(
            model,
            batch_sizes=[1, 32, 128]
        )
        
        for batch_size, metrics in results.items():
            print(f"\nBatch size {batch_size}:")
            print(f"  CPU memory: {metrics['cpu_memory_mb']:.1f}MB")
            if "gpu_memory_mb" in metrics:
                print(f"  GPU memory: {metrics['gpu_memory_mb']:.1f}MB")
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_parallel_env_benchmark(self, benchmarks):
        """Benchmark parallel environment scaling"""
        results = benchmarks.benchmark_parallel_environments(
            num_envs_list=[1, 2, 4, 8],
            steps_per_env=50
        )
        
        for num_envs, metrics in results.items():
            print(f"\nNum environments: {num_envs}")
            print(f"  Steps/second: {metrics['steps_per_second']:.1f}")
            print(f"  Speedup: {metrics['speedup']:.2f}x")
            print(f"  Efficiency: {metrics['efficiency']:.1%}")
    
    @pytest.mark.benchmark
    def test_training_throughput_benchmark(self, model, benchmarks):
        """Benchmark training throughput"""
        results = benchmarks.benchmark_training_throughput(
            model,
            buffer_sizes=[256, 1024],
            batch_size=32,
            num_epochs=2
        )
        
        for buffer_size, metrics in results.items():
            print(f"\nBuffer size: {buffer_size}")
            print(f"  Samples/second: {metrics['samples_per_second']:.1f}")
            print(f"  Time per epoch: {metrics['time_per_epoch_s']:.2f}s")
    
    @pytest.mark.benchmark
    def test_model_operations_benchmark(self, model, benchmarks):
        """Benchmark model operations"""
        results = benchmarks.benchmark_model_operations(model)
        
        print("\nModel operations:")
        print(f"  Save time: {results['save_time_ms']:.1f}ms")
        print(f"  Load time: {results['load_time_ms']:.1f}ms")
        print(f"  Init time: {results['init_time_ms']:.1f}ms")
        print(f"  Model size: {results['model_size_mb']:.1f}MB")
        print(f"  Parameters: {results['total_parameters']:,}")


def generate_performance_report(output_dir: str = "benchmark_results"):
    """Generate comprehensive performance report with visualizations"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model = BlackholioModel()
    benchmarks = PerformanceBenchmarks()
    
    # Run all benchmarks
    print("Running inference latency benchmarks...")
    inference_results = benchmarks.measure_inference_latency(model)
    
    print("Running memory usage benchmarks...")
    memory_results = benchmarks.measure_memory_usage(model)
    
    print("Running parallel environment benchmarks...")
    parallel_results = benchmarks.benchmark_parallel_environments()
    
    print("Running training throughput benchmarks...")
    training_results = benchmarks.benchmark_training_throughput(model)
    
    print("Running model operation benchmarks...")
    operation_results = benchmarks.benchmark_model_operations(model)
    
    # Save results
    all_results = {
        "inference_latency": inference_results,
        "memory_usage": memory_results,
        "parallel_environments": parallel_results,
        "training_throughput": training_results,
        "model_operations": operation_results
    }
    
    with open(output_path / "benchmark_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    # Inference latency plot
    plt.figure(figsize=(10, 6))
    batch_sizes = list(inference_results.keys())
    mean_latencies = [inference_results[bs]["mean_latency_ms"] for bs in batch_sizes]
    p95_latencies = [inference_results[bs]["p95_latency_ms"] for bs in batch_sizes]
    
    plt.plot(batch_sizes, mean_latencies, 'b-o', label='Mean', linewidth=2)
    plt.plot(batch_sizes, p95_latencies, 'r--o', label='P95', linewidth=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.title('Inference Latency vs Batch Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "inference_latency.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Parallel scaling plot
    plt.figure(figsize=(10, 6))
    num_envs = list(parallel_results.keys())
    speedups = [parallel_results[n]["speedup"] for n in num_envs]
    ideal_speedup = num_envs
    
    plt.plot(num_envs, speedups, 'b-o', label='Actual', linewidth=2)
    plt.plot(num_envs, ideal_speedup, 'g--', label='Ideal', linewidth=2)
    plt.xlabel('Number of Environments')
    plt.ylabel('Speedup')
    plt.title('Parallel Environment Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "parallel_scaling.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPerformance report saved to {output_path}")
    return all_results
