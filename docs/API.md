# Blackholio Agent API Documentation

Complete reference documentation for all classes, methods, and configuration options in the Blackholio Agent framework.

## Table of Contents

1. [Environment](#environment)
2. [Models](#models)
3. [Training](#training)
4. [Inference](#inference)
5. [Multi-Agent](#multi-agent)
6. [Configuration](#configuration)
7. [Utilities](#utilities)

## Environment

### BlackholioEnv

Main environment class providing Gym-like interface to Blackholio game.

```python
from blackholio_agent import BlackholioEnv, BlackholioEnvConfig

class BlackholioEnv:
    def __init__(self, config: BlackholioEnvConfig)
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]
    def close(self) -> None
    def render(self, mode: str = "human") -> Optional[np.ndarray]
    
    # Async versions
    async def async_reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]
    async def async_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]
```

#### Parameters
- `config`: Configuration object with connection and environment settings

#### Properties
- `observation_space`: Gym space describing observation format
- `action_space`: Gym space describing action format
- `reward_range`: Tuple of (min_reward, max_reward)

#### Methods

##### `reset(seed=None)`
Reset environment to initial state.

**Returns**: `(observation, info)`
- `observation`: Current game state as numpy array
- `info`: Dictionary with metadata (player_id, episode_count, etc.)

##### `step(action)`
Execute one environment step.

**Parameters**:
- `action`: numpy array [move_x, move_y, split] where move values are in [-1,1] and split is binary

**Returns**: `(observation, reward, terminated, truncated, info)`
- `observation`: Next game state
- `reward`: Reward for this step
- `terminated`: True if episode ended naturally (death)
- `truncated`: True if episode hit time limit
- `info`: Metadata dictionary

##### `close()`
Clean up environment resources.

### BlackholioConnection

Manages SpacetimeDB connection and game interactions.

```python
from blackholio_agent.environment import BlackholioConnection

class BlackholioConnection:
    def __init__(self, host: str, database: str, auth_token: Optional[str] = None)
    async def connect(self) -> None
    async def disconnect(self) -> None
    async def spawn_player(self, name: str) -> int
    async def move_player(self, player_id: int, direction: Tuple[float, float]) -> None
    async def split_player(self, player_id: int) -> None
    def get_game_state(self) -> Dict[str, Any]
    def is_player_alive(self, player_id: int) -> bool
```

#### Methods

##### `connect()`
Establish connection to SpacetimeDB server.

##### `spawn_player(name)`
Create a new player in the game.

**Returns**: Player ID for the spawned player

##### `move_player(player_id, direction)`
Send movement command for player.

**Parameters**:
- `player_id`: ID of player to move
- `direction`: Tuple of (x, y) movement vector

##### `split_player(player_id)`
Execute split action for player.

##### `get_game_state()`
Get current state of all game entities.

**Returns**: Dictionary with entities, players, circles, food

### ObservationSpace

Converts game state to neural network input format.

```python
from blackholio_agent.environment import ObservationSpace, ObservationConfig

class ObservationSpace:
    def __init__(self, config: ObservationConfig)
    def process_game_state(self, game_state: Dict, player_id: int) -> np.ndarray
    def get_space(self) -> gym.Space
    def get_observation_info(self) -> Dict[str, Any]
```

#### Methods

##### `process_game_state(game_state, player_id)`
Convert raw game state to observation array.

**Parameters**:
- `game_state`: Raw game state from connection
- `player_id`: ID of the observing player

**Returns**: Flattened numpy array of shape (obs_dim,)

### ActionSpace

Handles action processing and execution.

```python
from blackholio_agent.environment import ActionSpace, ActionConfig

class ActionSpace:
    def __init__(self, config: ActionConfig)
    def process_action(self, action: np.ndarray) -> Dict[str, Any]
    def get_space(self) -> gym.Space
    def sample(self) -> np.ndarray
```

#### Methods

##### `process_action(action)`
Convert neural network output to game commands.

**Parameters**:
- `action`: Array [move_x, move_y, split_decision]

**Returns**: Dictionary with movement and split commands

### RewardCalculator

Computes rewards from game state transitions.

```python
from blackholio_agent.environment import RewardCalculator, RewardConfig

class RewardCalculator:
    def __init__(self, config: RewardConfig)
    def calculate_reward(self, prev_obs: np.ndarray, action: np.ndarray, 
                        obs: np.ndarray, info: Dict) -> float
    def reset_episode(self) -> None
    def update_curriculum_stage(self, stage: int) -> None
```

#### Methods

##### `calculate_reward(prev_obs, action, obs, info)`
Calculate reward for a state transition.

**Returns**: Float reward value

##### `update_curriculum_stage(stage)`
Change curriculum learning stage (0-3).

## Models

### BlackholioModel

Neural network model for the Blackholio agent.

```python
from blackholio_agent.models import BlackholioModel, ModelConfig

class BlackholioModel(nn.Module):
    def __init__(self, config: ModelConfig)
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    def get_action_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    def get_value(self, obs: torch.Tensor) -> torch.Tensor
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

#### Methods

##### `forward(observations)`
Forward pass through the network.

**Returns**: `(action_logits, values)`
- `action_logits`: Action probability logits
- `values`: State value estimates

##### `get_action_and_value(obs)`
Sample actions and compute values for inference.

**Returns**: `(actions, log_probs, values)`

##### `save(path)` / `load(path)`
Save/load model weights and configuration.

### ModelConfig

Configuration for model architecture.

```python
@dataclass
class ModelConfig:
    observation_dim: int = 456
    action_dim: int = 3
    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_attention_layers: int = 3
    use_lstm: bool = False
    lstm_hidden_dim: int = 256
    activation: str = "relu"
    dropout_rate: float = 0.1
```

## Training

### PPOTrainer

Proximal Policy Optimization trainer.

```python
from blackholio_agent.training import PPOTrainer, PPOConfig

class PPOTrainer:
    def __init__(self, model: BlackholioModel, config: PPOConfig)
    def train_step(self, rollouts: RolloutBuffer) -> Dict[str, float]
    def save_checkpoint(self, path: str) -> None
    def load_checkpoint(self, path: str) -> None
```

#### Methods

##### `train_step(rollouts)`
Perform one training update step.

**Returns**: Dictionary with training metrics (losses, etc.)

### RolloutBuffer

Stores experience for training.

```python
from blackholio_agent.training import RolloutBuffer

class RolloutBuffer:
    def __init__(self, buffer_size: int, observation_space: gym.Space, action_space: gym.Space)
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            done: bool, value: float, log_prob: float) -> None
    def get(self) -> Dict[str, torch.Tensor]
    def clear(self) -> None
    def compute_returns_and_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None
```

### CheckpointManager

Handles model checkpointing and best model tracking.

```python
from blackholio_agent.training import CheckpointManager

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5)
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                       step: int, metrics: Dict[str, float]) -> str
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]
    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]
    def get_best_metric(self) -> float
```

### MetricsLogger

Logs training metrics to TensorBoard.

```python
from blackholio_agent.training import MetricsLogger

class MetricsLogger:
    def __init__(self, log_dir: str)
    def log_scalar(self, name: str, value: float, step: int) -> None
    def log_scalars(self, scalars: Dict[str, float], step: int) -> None
    def log_histogram(self, name: str, values: torch.Tensor, step: int) -> None
    def close(self) -> None
```

### SelfPlayManager

Manages self-play training with opponent pool.

```python
from blackholio_agent.training import SelfPlayManager

class SelfPlayManager:
    def __init__(self, opponent_pool_size: int = 5, update_interval: int = 10000)
    def should_update_pool(self, step: int) -> bool
    def add_opponent(self, model_path: str, rating: float) -> None
    def sample_opponent(self) -> str
    def update_ratings(self, results: List[Dict]) -> None
```

### CurriculumManager

Manages curriculum learning progression.

```python
from blackholio_agent.training import CurriculumManager

class CurriculumManager:
    def __init__(self, stages: List[str], thresholds: List[float])
    def update(self, performance_metric: float) -> bool
    def get_current_stage(self) -> int
    def get_stage_config(self) -> Dict[str, Any]
```

## Inference

### InferenceAgent

Production agent for running trained models.

```python
from blackholio_agent.inference import InferenceAgent, InferenceConfig

class InferenceAgent:
    def __init__(self, model_path: str, config: InferenceConfig)
    async def run(self) -> None
    async def step(self) -> None
    def stop(self) -> None
    def get_metrics(self) -> Dict[str, float]
```

#### Methods

##### `run()`
Start the main inference loop.

##### `step()`
Execute one inference step.

##### `get_metrics()`
Get performance metrics (FPS, latency, etc.).

### ModelLoader

Loads and optimizes models for inference.

```python
from blackholio_agent.inference import ModelLoader

class ModelLoader:
    def __init__(self, device: str = "cpu")
    def load_model(self, model_path: str) -> BlackholioModel
    def optimize_for_inference(self, model: BlackholioModel) -> BlackholioModel
    def validate_model(self, model: BlackholioModel) -> bool
```

### RateLimiter

Controls inference timing to match game rate.

```python
from blackholio_agent.inference import RateLimiter

class RateLimiter:
    def __init__(self, target_fps: float = 20.0)
    async def wait(self) -> None
    def get_current_fps(self) -> float
    def reset(self) -> None
```

### InferenceMetrics

Tracks inference performance metrics.

```python
from blackholio_agent.inference import InferenceMetrics

class InferenceMetrics:
    def __init__(self)
    def record_inference_time(self, time_ms: float) -> None
    def record_step_time(self, time_ms: float) -> None
    def get_stats(self) -> Dict[str, float]
    def reset(self) -> None
```

## Multi-Agent

### MultiAgentBlackholioEnv

Environment supporting coordinated multi-agent teams.

```python
from blackholio_agent.multi_agent import MultiAgentBlackholioEnv

class MultiAgentBlackholioEnv:
    def __init__(self, config: MultiAgentConfig, team_size: int = 2)
    def reset(self) -> Dict[str, np.ndarray]
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], 
                                                            Dict[str, float], 
                                                            Dict[str, bool], 
                                                            Dict[str, bool], 
                                                            Dict[str, Dict]]
```

### AgentCommunication

Handles message passing between agents.

```python
from blackholio_agent.multi_agent import AgentCommunication

class AgentCommunication:
    def __init__(self, team_size: int, message_dim: int = 32)
    def send_message(self, sender_id: str, message: np.ndarray, priority: float = 1.0) -> None
    def receive_messages(self, receiver_id: str) -> List[np.ndarray]
    def broadcast(self, sender_id: str, message: np.ndarray) -> None
    def clear_messages(self) -> None
```

### TeamObservationSpace

Extended observation space including teammate information.

```python
from blackholio_agent.multi_agent import TeamObservationSpace

class TeamObservationSpace:
    def __init__(self, base_config: ObservationConfig, team_size: int)
    def process_team_state(self, game_state: Dict, team_player_ids: List[int], 
                          observer_id: int) -> np.ndarray
```

### CoordinationActionSpace

Action space with coordination and communication actions.

```python
from blackholio_agent.multi_agent import CoordinationActionSpace

class CoordinationActionSpace:
    def __init__(self, base_config: ActionConfig, communication_dim: int = 32)
    def process_coordination_action(self, action: np.ndarray) -> Dict[str, Any]
```

### TeamRewardCalculator

Reward calculator supporting team objectives.

```python
from blackholio_agent.multi_agent import TeamRewardCalculator

class TeamRewardCalculator:
    def __init__(self, base_config: RewardConfig, cooperation_weight: float = 0.3)
    def calculate_team_reward(self, team_obs: Dict[str, np.ndarray], 
                             team_actions: Dict[str, np.ndarray],
                             team_next_obs: Dict[str, np.ndarray],
                             team_info: Dict[str, Dict]) -> Dict[str, float]
```

## Configuration

### BlackholioEnvConfig

Main environment configuration.

```python
@dataclass
class BlackholioEnvConfig:
    # Connection settings
    host: str = "localhost:3000"
    database: str = "blackholio"
    auth_token: Optional[str] = None
    ssl_enabled: bool = False
    
    # Environment settings
    player_name: str = "ML_Agent"
    render_mode: Optional[str] = None  # "human", "rgb_array", or None
    max_episode_steps: int = 10000
    step_interval: float = 0.05  # 20Hz
    
    # Component configurations
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)
    action_config: ActionConfig = field(default_factory=ActionConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
```

### ObservationConfig

Observation space configuration.

```python
@dataclass
class ObservationConfig:
    max_entities_tracked: int = 50
    max_food_tracked: int = 100
    observation_radius: float = 500.0
    include_food: bool = True
    include_velocities: bool = True
    arena_width: float = 2000.0
    arena_height: float = 2000.0
    max_mass: float = 10000.0
    max_velocity: float = 100.0
    max_circles: int = 16
```

### ActionConfig

Action space configuration.

```python
@dataclass
class ActionConfig:
    movement_scale: float = 1.0
    update_rate: float = 20.0  # Hz
    queue_size: int = 10
    enable_split: bool = True
    split_cooldown: float = 1.0  # seconds
```

### RewardConfig

Reward calculation configuration.

```python
@dataclass
class RewardConfig:
    # Dense reward weights
    mass_gain_weight: float = 1.0
    mass_loss_weight: float = -2.0
    survival_bonus_per_step: float = 0.01
    food_collection_weight: float = 0.1
    
    # Sparse reward weights
    kill_reward: float = 10.0
    death_penalty: float = -50.0
    
    # Reward shaping
    distance_to_food_weight: float = -0.001
    size_ratio_weight: float = 0.01
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[str] = field(default_factory=lambda: ["food_collection", "survival", "hunting", "advanced"])
    current_stage: int = 0
```

### PPOConfig

PPO training configuration.

```python
@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 4
    clip_range: float = 0.2
    value_function_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
```

### InferenceConfig

Inference system configuration.

```python
@dataclass
class InferenceConfig:
    target_fps: float = 20.0
    max_inference_time_ms: float = 40.0
    device: str = "cpu"
    warmup_steps: int = 10
    metrics_window_size: int = 100
```

## Utilities

### Common Utility Functions

```python
from blackholio_agent.utils import (
    normalize_observation,
    denormalize_action,
    calculate_distance,
    get_relative_position,
    setup_logging,
    save_config,
    load_config
)

def normalize_observation(obs: np.ndarray, obs_config: ObservationConfig) -> np.ndarray
def denormalize_action(action: np.ndarray, action_config: ActionConfig) -> np.ndarray
def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float
def get_relative_position(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> Tuple[float, float]
def setup_logging(log_level: str = "INFO") -> None
def save_config(config: Any, path: str) -> None
def load_config(path: str, config_class: Type) -> Any
```

## Error Handling

### Custom Exceptions

```python
class BlackholioConnectionError(Exception):
    """Raised when connection to SpacetimeDB fails"""
    pass

class InvalidActionError(Exception):
    """Raised when action is invalid for current state"""
    pass

class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass

class TrainingError(Exception):
    """Raised when training encounters an error"""
    pass
```

## Type Hints

### Common Types

```python
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch

# Observation type
Observation = np.ndarray

# Action type  
Action = np.ndarray

# Game state type
GameState = Dict[str, Any]

# Step return type
StepReturn = Tuple[Observation, float, bool, bool, Dict[str, Any]]

# Reset return type
ResetReturn = Tuple[Observation, Dict[str, Any]]
```

## Usage Examples

### Basic Environment Usage

```python
from blackholio_agent import BlackholioEnv, BlackholioEnvConfig

config = BlackholioEnvConfig(
    host="localhost:3000",
    player_name="TestAgent"
)

env = BlackholioEnv(config)
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Training Example

```python
from blackholio_agent import BlackholioModel, PPOTrainer
from blackholio_agent.training import RolloutBuffer, PPOConfig

model = BlackholioModel(config)
trainer = PPOTrainer(model, PPOConfig())
buffer = RolloutBuffer(2048, env.observation_space, env.action_space)

for step in range(100000):
    # Collect experience
    obs, info = env.reset()
    for _ in range(2048):
        action, log_prob, value = model.get_action_and_value(torch.tensor(obs))
        next_obs, reward, terminated, truncated, info = env.step(action.numpy())
        buffer.add(obs, action.numpy(), reward, terminated, value.item(), log_prob.item())
        obs = next_obs if not (terminated or truncated) else env.reset()[0]
    
    # Train
    buffer.compute_returns_and_advantages(torch.zeros(1), 0.99, 0.95)
    metrics = trainer.train_step(buffer)
    buffer.clear()
```

### Inference Example

```python
from blackholio_agent.inference import InferenceAgent, InferenceConfig

config = InferenceConfig(target_fps=20.0)
agent = InferenceAgent("best_model.pth", config)

# Run agent
await agent.run()
```

This API documentation provides complete reference for all public interfaces in the Blackholio Agent framework. For implementation examples and tutorials, see the [Training Guide](TRAINING_GUIDE.md) and [example notebooks](../examples/notebooks/).
