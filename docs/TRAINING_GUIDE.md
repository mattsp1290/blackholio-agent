# Blackholio Agent Training Guide

This guide provides comprehensive information for training high-performance Blackholio agents, including hyperparameter tuning, compute requirements, and troubleshooting.

## Table of Contents

1. [Quick Start Training](#quick-start-training)
2. [Hyperparameter Tuning](#hyperparameter-tuning)
3. [Compute Requirements](#compute-requirements)
4. [Training Strategies](#training-strategies)
5. [Monitoring Training](#monitoring-training)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Techniques](#advanced-techniques)

## Quick Start Training

### Basic Training Command

```bash
python scripts/train_agent.py \
    --experiment_name "my_first_agent" \
    --total_timesteps 1000000 \
    --learning_rate 3e-4 \
    --batch_size 256 \
    --num_envs 4
```

### Configuration File Approach

Create a training config file:

```yaml
# configs/my_config.yaml
experiment_name: "my_first_agent"
total_timesteps: 1000000
learning_rate: 3e-4
batch_size: 256
num_envs: 4
save_interval: 50000
eval_interval: 25000

# Environment settings
environment:
  max_episode_steps: 10000
  curriculum_stage: 0  # Start with food collection
  
# Model architecture
model:
  hidden_dim: 256
  num_attention_heads: 8
  use_lstm: false
```

Then run:
```bash
python scripts/train_agent.py --config configs/my_config.yaml
```

## Hyperparameter Tuning

### Core PPO Hyperparameters

#### Learning Rate
- **Default**: `3e-4`
- **Range**: `1e-5` to `1e-3`
- **Tuning**: Start high for exploration, decay over time
- **Signs of poor tuning**:
  - Too high: Training unstable, loss oscillates
  - Too low: Training too slow, gets stuck in local minima

```python
# Learning rate schedule example
learning_rate = 3e-4
lr_schedule = "linear"  # Decay linearly to 0
```

#### Batch Size
- **Default**: `256`
- **Range**: `64` to `1024`
- **Memory constraint**: Larger batches need more GPU memory
- **Stability**: Larger batches = more stable gradients

#### PPO Clip Range
- **Default**: `0.2`
- **Range**: `0.1` to `0.3`
- **Effect**: Controls how much policy can change per update
- **Tuning**: Decrease for more conservative updates

#### Value Function Coefficient
- **Default**: `0.5`
- **Range**: `0.1` to `1.0`
- **Effect**: Balances policy vs value learning
- **Signs**: High value loss → increase coefficient

#### Entropy Coefficient
- **Default**: `0.01`
- **Range**: `0.001` to `0.1`
- **Effect**: Encourages exploration
- **Schedule**: Start high, decay over time

### Model Architecture Hyperparameters

#### Hidden Dimensions
- **Default**: `256`
- **Options**: `128, 256, 512, 1024`
- **Trade-off**: Larger = more capacity but slower training

#### Attention Heads
- **Default**: `8`
- **Range**: `4` to `16`
- **Effect**: More heads = better entity relationship modeling

#### LSTM Usage
- **Default**: `False`
- **When to use**: If agent needs memory of past states
- **Cost**: Significantly slower training

### Recommended Hyperparameter Sets

#### For Fast Experimentation
```yaml
learning_rate: 1e-3
batch_size: 128
num_envs: 8
hidden_dim: 128
total_timesteps: 500000
```

#### For High Performance
```yaml
learning_rate: 3e-4
batch_size: 512
num_envs: 16
hidden_dim: 512
num_attention_heads: 12
total_timesteps: 5000000
```

#### For Limited Compute
```yaml
learning_rate: 3e-4
batch_size: 64
num_envs: 2
hidden_dim: 128
total_timesteps: 1000000
```

## Compute Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **GPU**: Optional but recommended
- **Training time**: 6-12 hours for basic agent

### Recommended Setup
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GTX 1660 or better (6GB+ VRAM)
- **Training time**: 2-4 hours for competitive agent

### High-Performance Setup
- **CPU**: 16+ cores
- **RAM**: 32GB+
- **GPU**: NVIDIA RTX 3080 or better (10GB+ VRAM)
- **Training time**: 1-2 hours for state-of-the-art agent

### Scaling Guidelines

#### Parallel Environments
```python
# Memory usage scales linearly with num_envs
num_envs = min(cpu_cores, available_ram_gb // 2)

# Examples:
# 8 cores, 16GB RAM → num_envs = 8
# 4 cores, 8GB RAM → num_envs = 4
```

#### Batch Size Scaling
```python
# GPU memory limited
if gpu_memory_gb >= 8:
    batch_size = 512
elif gpu_memory_gb >= 6:
    batch_size = 256
else:
    batch_size = 128
```

### Training Time Estimates

| Setup | Timesteps | Wall Time | Performance Level |
|-------|-----------|-----------|------------------|
| Minimal | 500K | 6-8 hours | Basic food collection |
| Standard | 1M | 3-4 hours | Competent play |
| High-perf | 2M | 2-3 hours | Advanced strategies |
| Expert | 5M+ | 4-6 hours | Human-competitive |

## Training Strategies

### Curriculum Learning

The agent supports 4-stage curriculum learning:

#### Stage 0: Food Collection (0-250K timesteps)
```yaml
curriculum_stage: 0
reward_config:
  food_collection_weight: 1.0
  mass_gain_weight: 0.5
  survival_bonus_per_step: 0.01
```

#### Stage 1: Survival (250K-500K timesteps)
```yaml
curriculum_stage: 1
reward_config:
  survival_bonus_per_step: 0.02
  death_penalty: -25.0
  size_ratio_weight: 0.1
```

#### Stage 2: Hunting (500K-1M timesteps)
```yaml
curriculum_stage: 2
reward_config:
  kill_reward: 10.0
  mass_gain_weight: 1.0
  aggressive_bonus: 0.05
```

#### Stage 3: Advanced Strategies (1M+ timesteps)
```yaml
curriculum_stage: 3
reward_config:
  split_strategy_reward: 5.0
  territory_control_bonus: 0.1
  efficiency_bonus: 0.02
```

### Self-Play Training

Enable self-play for competitive improvement:

```bash
python scripts/train_agent.py \
    --use_self_play \
    --opponent_pool_size 5 \
    --self_play_prob 0.7 \
    --update_opponent_interval 25000
```

Benefits:
- Agents learn to counter their own strategies
- Prevents overfitting to static opponents
- Leads to more robust strategies

### Multi-Agent Training

Train coordinated teams:

```bash
python scripts/train_agent.py \
    --multi_agent \
    --team_size 2 \
    --communication_enabled \
    --shared_reward_weight 0.3
```

## Monitoring Training

### TensorBoard Integration

Start TensorBoard to monitor training:

```bash
tensorboard --logdir logs/
```

Key metrics to watch:

#### Training Metrics
- **Episode Reward**: Should increase over time
- **Policy Loss**: Should decrease and stabilize
- **Value Loss**: Should decrease steadily
- **Entropy**: Should start high and gradually decrease

#### Environment Metrics
- **Episode Length**: Should increase (agents survive longer)
- **Max Mass Achieved**: Should increase over time
- **Kill/Death Ratio**: Should improve with curriculum

#### Performance Metrics
- **FPS**: Should remain stable (watch for bottlenecks)
- **Inference Time**: Should be < 50ms for real-time play

### Checkpoint Management

Automatic checkpoint saving:

```python
save_interval = 50000  # Save every 50K timesteps
eval_interval = 25000  # Evaluate every 25K timesteps
```

Best model selection based on evaluation performance.

### Early Stopping

Implement early stopping to prevent overfitting:

```python
patience = 10  # Stop if no improvement for 10 evaluations
min_delta = 0.01  # Minimum improvement threshold
```

## Troubleshooting

### Common Training Issues

#### Training Not Converging
**Symptoms**: Reward stays flat, high variance
**Solutions**:
- Reduce learning rate by 2-3x
- Increase batch size
- Add entropy regularization
- Check reward function design

#### Policy Collapse
**Symptoms**: Agent stops exploring, gets stuck
**Solutions**:
- Increase entropy coefficient
- Reduce batch size
- Add curriculum learning
- Reset to earlier checkpoint

#### Memory Issues
**Symptoms**: OOM errors, slow training
**Solutions**:
- Reduce `num_envs`
- Reduce `batch_size`
- Reduce model `hidden_dim`
- Use gradient checkpointing

#### Unstable Training
**Symptoms**: Loss oscillates wildly
**Solutions**:
- Reduce learning rate
- Reduce PPO clip range
- Increase batch size
- Add gradient clipping

### Environment-Specific Issues

#### Connection Problems
```python
# Add connection retry logic
connection_retries = 5
retry_delay = 2.0  # seconds
```

#### Reward Sparsity
```python
# Add dense reward shaping
reward_config = RewardConfig(
    distance_to_food_weight=-0.001,  # Encourage food seeking
    size_ratio_weight=0.01,          # Reward size advantages
    exploration_bonus=0.005          # Encourage map exploration
)
```

#### Action Space Issues
```python
# Ensure action bounds are correct
action_config = ActionConfig(
    movement_scale=1.0,      # Full movement range
    enable_split=True,       # Allow splitting
    split_cooldown=1.0       # Prevent spam splitting
)
```

## Performance Optimization

### Training Speed Optimization

#### Vectorized Environments
```python
# Use optimized vectorized environments
from blackholio_agent.training import ParallelEnvironments

envs = ParallelEnvironments(
    num_envs=16,
    async_update=True,  # Non-blocking updates
    shared_memory=True  # Faster observation transfer
)
```

#### Model Optimization
```python
# Use mixed precision training
use_mixed_precision = True

# Optimize model for inference
torch.jit.script(model)  # JIT compilation
model.half()             # FP16 inference
```

#### Data Loading
```python
# Optimize data pipeline
num_workers = 4          # Parallel data loading
pin_memory = True        # Faster GPU transfer
prefetch_factor = 2      # Pipeline batches
```

### Memory Optimization

#### Gradient Accumulation
```python
# For large effective batch sizes with limited memory
gradient_accumulation_steps = 4
effective_batch_size = batch_size * gradient_accumulation_steps
```

#### Model Pruning
```python
# Remove unnecessary model components for inference
prune_attention_heads = True
quantize_weights = True
```

### Distributed Training

For very large scale training:

```bash
# Multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train_agent.py \
    --distributed
```

## Advanced Techniques

### Hyperparameter Optimization with Optuna

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    
    # Train model with these hyperparameters
    final_reward = train_model(lr, batch_size, hidden_dim)
    return final_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### Population-Based Training

```python
# Train multiple agents with different hyperparameters
population_size = 8
mutation_rate = 0.1
crossover_rate = 0.3

# Periodically replace worst performers with mutated best performers
```

### Custom Reward Functions

```python
class AdvancedRewardCalculator(RewardCalculator):
    def calculate_reward(self, obs, action, next_obs, info):
        # Base reward
        reward = super().calculate_reward(obs, action, next_obs, info)
        
        # Add custom components
        reward += self._territorial_bonus(obs, next_obs)
        reward += self._efficiency_bonus(action, info)
        reward += self._strategic_bonus(obs, action, info)
        
        return reward
```

### Model Architecture Search

```python
# Search over different architectures
architectures = [
    {'hidden_dim': 256, 'num_heads': 8, 'num_layers': 3},
    {'hidden_dim': 512, 'num_heads': 12, 'num_layers': 4},
    {'hidden_dim': 128, 'num_heads': 4, 'num_layers': 2, 'use_lstm': True},
]

for arch in architectures:
    model = create_model(**arch)
    performance = evaluate_model(model)
    # Track best architecture
```

## Best Practices Summary

1. **Start Simple**: Begin with default hyperparameters
2. **Monitor Closely**: Use TensorBoard for real-time monitoring
3. **Iterate Quickly**: Use smaller timestep counts for initial experiments
4. **Save Everything**: Keep checkpoints and logs for analysis
5. **Test Thoroughly**: Evaluate on diverse scenarios
6. **Document Changes**: Track what works and what doesn't
7. **Be Patient**: Good agents take time to train
8. **Use Curriculum**: Gradual difficulty increase works better
9. **Validate Performance**: Test against known baselines
10. **Plan for Scale**: Design for eventual larger experiments

## Next Steps

After training a basic agent:

1. **Analyze Behavior**: Use behavior analysis notebooks
2. **Compare Models**: Run model comparison scripts
3. **Optimize Performance**: Profile and optimize inference
4. **Deploy**: Set up production inference system
5. **Iterate**: Use insights to train better agents

For more advanced topics, see:
- [API Documentation](API.md)
- [SpacetimeDB ML Integration Guide](SPACETIMEDB_ML_GUIDE.md)
- [Example Notebooks](../examples/notebooks/)
