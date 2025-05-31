# Blackholio Inference System Documentation

## Overview

The Blackholio Inference System is a production-ready solution for running trained ML agents in the Blackholio game. It provides real-time inference at 20Hz, performance monitoring, and Docker deployment support.

## Architecture

### Core Components

1. **InferenceAgent** (`src/blackholio_agent/inference/agent.py`)
   - Main agent class that orchestrates the inference loop
   - Handles model loading, environment interaction, and metrics collection
   - Implements graceful shutdown and error recovery

2. **ModelLoader** (`src/blackholio_agent/inference/model_loader.py`)
   - CPU-optimized model loading and validation
   - Model warmup for consistent performance
   - Optional torch.compile optimization for PyTorch 2.0+

3. **RateLimiter** (`src/blackholio_agent/inference/rate_limiter.py`)
   - Enforces minimum time between actions (default 50ms)
   - Prevents API spam while maintaining consistent update rate
   - Includes adaptive rate limiting option

4. **InferenceMetrics** (`src/blackholio_agent/inference/metrics.py`)
   - Lightweight metrics tracking without detailed recordings
   - Tracks survival time, mass, failure modes
   - Provides summary statistics and JSON export

5. **InferenceConfig** (`src/blackholio_agent/inference/config.py`)
   - Configuration management with environment variable support
   - Docker-ready with sensible defaults
   - Validates settings at initialization

## Usage

### Command Line

```bash
# Basic usage with model path
python scripts/run_agent.py --model checkpoints/best_model.pth

# With custom settings
python scripts/run_agent.py \
    --model model.pth \
    --host localhost:3000 \
    --name MyAgent \
    --cpu-threads 4 \
    --rate-limit 50 \
    --verbose

# Limited episodes for testing
python scripts/run_agent.py --model model.pth --max-episodes 10
```

### Environment Variables

```bash
# Set via environment variables (useful for Docker)
export MODEL_PATH=checkpoints/best_model.pth
export SPACETIMEDB_HOST=localhost:3000
export AGENT_NAME=ML_Agent_1
export CPU_THREADS=4
export RATE_LIMIT_MS=50

python scripts/run_agent.py
```

### Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t blackholio-agent .

# Run single agent
docker run -v ./checkpoints:/models:ro \
    -e MODEL_PATH=/models/best_model.pth \
    -e SPACETIMEDB_HOST=host.docker.internal:3000 \
    blackholio-agent

# Or use docker-compose
docker-compose up

# Scale multiple agents
docker-compose up --scale blackholio-agent=5
```

## Performance Optimizations

### CPU Optimizations
- Configurable thread count via `torch.set_num_threads()`
- Model compilation with torch.compile (if available)
- Pre-allocated tensors to minimize memory allocation
- Gradient computation disabled for inference

### Rate Limiting
- Minimum 50ms between actions to prevent API spam
- Consistent 20Hz update rate
- Adaptive rate limiting available for dynamic adjustment

### Monitoring
- Real-time FPS calculation
- Inference latency tracking
- Memory-efficient metrics without full episode recording
- Periodic metrics export to JSON

## Configuration Options

| Parameter | CLI Flag | Environment Variable | Default | Description |
|-----------|----------|---------------------|---------|-------------|
| Model Path | `--model` | `MODEL_PATH` | `checkpoints/best_model.pth` | Path to trained model |
| Host | `--host` | `SPACETIMEDB_HOST` | `localhost:3000` | SpacetimeDB host |
| Database | `--database` | `SPACETIMEDB_DB` | `blackholio` | Database name |
| Auth Token | `--auth-token` | `SPACETIMEDB_TOKEN` | None | Authentication token |
| Agent Name | `--name` | `AGENT_NAME` | `ML_Agent_<random>` | Player name in game |
| CPU Threads | `--cpu-threads` | `CPU_THREADS` | 4 | Number of CPU threads |
| Rate Limit | `--rate-limit` | `RATE_LIMIT_MS` | 50 | Min milliseconds between actions |
| Max Episodes | `--max-episodes` | `MAX_EPISODES` | 0 (unlimited) | Episode limit |
| Log Interval | `--log-interval` | `LOG_INTERVAL` | 100 | Episodes between summaries |
| Verbose | `--verbose` | `VERBOSE` | false | Enable debug logging |

## Metrics and Monitoring

### Episode Metrics
- Survival time
- Maximum mass achieved
- Total reward
- Failure mode detection
- Inference latency statistics

### Aggregate Metrics
- Success rate (based on configurable thresholds)
- Average survival time
- Common failure modes
- Performance statistics (FPS, latency)

### Failure Mode Detection
- `eaten_by_larger`: Consumed by bigger player
- `starvation`: No mass gain for extended period
- `out_of_bounds`: Left the game area
- `time_limit`: Reached maximum episode steps
- `unknown`: Other failure reasons

## Testing

Run the inference system tests:

```bash
# Test all inference components
python examples/test_inference.py

# This tests:
# - Rate limiter functionality
# - Metrics tracking
# - Model loading and optimization
# - Configuration system
# - Integration (without game server)
```

## Docker Resource Management

The docker-compose.yml includes resource limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 2G
    reservations:
      cpus: '1.0'
      memory: 1G
```

Adjust these based on your system capabilities and model size.

## Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure model path is correct
   - Check volume mounts in Docker
   - Verify file permissions

2. **Connection failed**
   - Verify Blackholio server is running
   - Check host/port configuration
   - For Docker: use `host.docker.internal` on Mac/Windows

3. **High latency**
   - Reduce CPU threads if system is overloaded
   - Check for background processes
   - Monitor with `--verbose` flag

4. **Memory issues**
   - Adjust Docker memory limits
   - Use smaller batch sizes
   - Monitor with system tools

### Debug Mode

Enable verbose logging for detailed information:

```bash
python scripts/run_agent.py --model model.pth --verbose
```

This provides:
- Step-by-step action logging
- Detailed performance metrics
- Connection status updates
- Error traces

## Next Steps

1. Train a model using the training pipeline (task 3)
2. Deploy multiple agents using Docker scaling
3. Analyze metrics to improve model performance
4. Implement custom failure mode detection
5. Add visualization tools for agent behavior

## Related Documentation

- [Training Pipeline](TRAINING_PIPELINE.md) - How to train models
- [Environment Documentation](../src/blackholio_agent/environment/README.md) - Environment details
- [Model Architecture](../src/blackholio_agent/models/README.md) - Neural network design
