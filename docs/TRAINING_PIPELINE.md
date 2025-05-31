# Blackholio Agent Training Pipeline

This document describes the training pipeline implementation for the Blackholio reinforcement learning agent.

## Overview

The training pipeline implements Proximal Policy Optimization (PPO) with the following features:
- Parallel environment execution for sample efficiency
- Curriculum learning for progressive skill development
- Comprehensive logging and visualization
- Checkpoint management for model persistence
- Performance optimizations for real-time gameplay

## Architecture

### Core Components

1. **PPOTrainer** (`src/blackholio_agent/training/ppo_trainer.py`)
   - Main training orchestrator
   - Implements PPO algorithm with clipped objective
   - Manages training loop and policy updates
   - Integrates self-play and curriculum learning

2. **RolloutBuffer** (`src/blackholio_agent/training/rollout_buffer.py`)
   - Stores experience trajectories
   - Computes advantages using GAE
   - Provides batched samples for training

3. **ParallelEnvs** (`src/blackholio_agent/training/parallel_envs.py`)
   - Manages multiple environment instances
   - Enables vectorized step execution
   - Handles asynchronous environment operations

4. **CheckpointManager** (`src/blackholio_agent/training/checkpoint_manager.py`)
   - Saves model checkpoints periodically
   - Tracks best performing models
   - Enables training resumption

5. **MetricsLogger** (`src/blackholio_agent/training/metrics_logger.py`)
   - Logs training metrics to TensorBoard
   - Generates plots and statistics
   - Tracks episode and training metrics

6. **SelfPlayManager** (`src/blackholio_agent/training/self_play_manager.py`)
   - Manages pool of historical opponents
   - Coordinates agent matchmaking
   - Tracks win rates and performance

7. **CurriculumManager** (`src/blackholio_agent/training/curriculum_manager.py`)
   - Adaptive difficulty progression
   - Performance-based stage transitions
   - Reward shaping per stage

## Configuration

Training is configured through YAML files:

```yaml
# Example configuration
total_timesteps: 10_000_000
n_envs: 8
n_steps: 2048
batch_size: 64
n_epochs: 10
learning_rate: 3.0e-4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
```

See `scripts/configs/` for pre-configured training setups.

## Usage

### Basic Training

```bash
# Start training with default configuration
python scripts/train_agent.py

# Use a specific configuration
python scripts/train_agent.py --config scripts/configs/full_train.yaml

# Resume from checkpoint
python scripts/train_agent.py --resume checkpoints/best_checkpoint.pt
```

### Quick Test Training

```bash
# Run a quick training test (1 minute)
python scripts/train_agent.py --config scripts/configs/quick_train.yaml
```

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir logs/
```

Key metrics tracked:
- Episode rewards and lengths
- Policy loss, value loss, and entropy
- Learning rate schedule
- Curriculum stage progression

## Curriculum Learning

The training pipeline implements adaptive curriculum learning with 5 stages:

1. **Food Collection** (Stage 0)
   - Focus on basic movement and food collection
   - High food rewards, low combat penalties
   - Promotion criteria: mean_reward > 5.0, survival_rate > 0.8

2. **Survival** (Stage 1)
   - Emphasis on staying alive
   - Increased survival rewards and death penalties
   - Promotion criteria: mean_reward > 10.0, average_lifespan > 120s
   - Demotion if: mean_reward < 3.0, survival_rate < 0.4

3. **Basic Combat** (Stage 2)
   - Encourage combat with smaller entities
   - Higher kill rewards, mass gain bonuses
   - Promotion criteria: mean_reward > 20.0, kill_rate > 0.3
   - Demotion if: mean_reward < 8.0, survival_rate < 0.5

4. **Advanced** (Stage 3)
   - Full gameplay with split mechanics
   - Balanced rewards for complex strategies
   - Promotion criteria: mean_reward > 35.0, split_effectiveness > 0.6
   - Demotion if: mean_reward < 15.0, kill_rate < 0.2

5. **Expert** (Stage 4)
   - Tournament-level gameplay
   - Focus on win rate and strategic play
   - No demotion, represents mastery level

### Adaptive Curriculum Features

- **Performance-Based Progression**: Stages advance/demote based on rolling performance metrics
- **Reward Shaping**: Each stage has custom reward multipliers
- **Environment Modifiers**: Difficulty settings adjusted per stage (enemy density, aggression, etc.)
- **Minimum Stage Duration**: 5 minutes minimum before transitions to prevent oscillation

## Self-Play Training

The training pipeline supports self-play for competitive skill development:

### Self-Play Features

1. **Opponent Pool**
   - Maintains pool of historical model checkpoints
   - Saves models every 10,000 steps
   - Keeps best 10 models based on performance

2. **Opponent Selection Strategies**
   - **Uniform**: Random selection from pool
   - **Weighted**: Prioritizes less-played opponents
   - **Latest**: Always plays against most recent model
   - **Self**: 50% chance to play against current self

3. **Performance Tracking**
   - Win rates tracked per model
   - Head-to-head statistics maintained
   - Used to determine pool membership

### Configuration

```yaml
# Enable self-play
use_self_play: true
self_play_config:
  pool_size: 10
  save_interval: 10000
  selection_strategy: "uniform"
  self_play_prob: 0.5
  track_win_rates: true
opponent_pool_dir: "opponent_pool"

# Enable adaptive curriculum
use_curriculum: true
adaptive_curriculum: true
```

## Testing

The training pipeline includes comprehensive tests:

### Unit Tests
```bash
# Run unit tests
python scripts/run_tests.py unit

# Specific component tests
pytest src/blackholio_agent/tests/unit/test_rollout_buffer.py
pytest src/blackholio_agent/tests/unit/test_checkpoint_manager.py
```

### Integration Tests
```bash
# Run integration tests
python scripts/run_tests.py integration

# Test full training pipeline
pytest src/blackholio_agent/tests/integration/test_training_pipeline.py
```

### Performance Benchmarks
```bash
# Run performance tests
python scripts/run_tests.py performance --benchmark

# Test inference speed
pytest src/blackholio_agent/tests/performance/test_inference_speed.py
```

## Performance Considerations

1. **Batch Processing**: The pipeline processes multiple environments in parallel for efficiency
2. **GPU Acceleration**: Training supports CUDA for faster model updates
3. **Optimized Rollout Collection**: Experience collection is vectorized across environments
4. **Efficient Advantage Computation**: GAE is computed in batches

## Best Practices

1. **Hyperparameter Tuning**
   - Start with provided configurations
   - Adjust learning rate based on training stability
   - Increase batch size for more stable updates

2. **Environment Scaling**
   - Use 8-16 parallel environments for good sample efficiency
   - Adjust based on available CPU cores

3. **Checkpoint Strategy**
   - Save checkpoints every 30 minutes
   - Keep best 5 recent checkpoints
   - Track performance metrics for best model selection

4. **Debugging**
   - Use smaller configurations for debugging
   - Enable verbose logging with `--verbose`
   - Check TensorBoard for training anomalies

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size or number of environments
   - Use gradient accumulation for large batches

2. **Slow Training**
   - Enable GPU with `--device cuda`
   - Reduce model size for faster iterations
   - Check for environment bottlenecks

3. **Unstable Training**
   - Reduce learning rate
   - Increase batch size
   - Check reward scaling

### Performance Profiling

```python
# Profile training performance
python scripts/train_agent.py --profile
```

## Future Improvements

1. **Distributed Training**: Support for multi-GPU and multi-node training
2. **Advanced Algorithms**: Implementation of SAC, A3C, or IMPALA
3. **Hyperparameter Optimization**: Automated tuning with Optuna
4. **Online Learning**: Continuous learning from live gameplay

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [GAE Paper](https://arxiv.org/abs/1506.02438)
- [Curriculum Learning](https://arxiv.org/abs/2003.04960)
