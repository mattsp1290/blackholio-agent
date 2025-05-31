"""
Main neural network model for Blackholio RL agent.

This module implements the policy and value networks used by the
reinforcement learning agent to play Blackholio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BlackholioModelConfig:
    """Configuration for BlackholioModel"""
    # Input dimensions
    observation_dim: int = 456  # From ObservationSpace
    player_state_dim: int = 6
    entity_feature_dim: int = 5
    max_entities: int = 50
    food_feature_dim: int = 2
    max_food: int = 100
    
    # Architecture parameters
    hidden_size: int = 256
    num_layers: int = 3
    attention_heads: int = 4
    attention_hidden: int = 128
    dropout: float = 0.1
    
    # Output dimensions
    movement_dim: int = 2
    discrete_actions: int = 2  # none, split
    
    # Feature processing
    use_attention: bool = True
    use_spatial_features: bool = True
    use_lstm: bool = False
    lstm_hidden_size: int = 128
    
    # Activation function
    activation: str = "relu"  # "relu", "gelu", "tanh"
    
    # Normalization
    use_layer_norm: bool = True
    
    # Device
    device: str = "cpu"


@dataclass
class ModelOutput:
    """Output from the BlackholioModel"""
    movement_mean: torch.Tensor  # Mean of movement distribution
    movement_std: torch.Tensor   # Std of movement distribution
    split_logits: torch.Tensor   # Logits for split action
    value: torch.Tensor          # State value estimate
    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # LSTM hidden state


class EntityAttention(nn.Module):
    """
    Multi-head attention module for processing entity relationships.
    """
    def __init__(self, config: BlackholioModelConfig):
        super().__init__()
        self.config = config
        
        # Project entity features to attention dimension
        self.entity_projection = nn.Linear(
            config.entity_feature_dim, 
            config.attention_hidden
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.attention_hidden,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            config.attention_hidden,
            config.hidden_size
        )
        
        # Query projection for player features
        self.query_projection = nn.Linear(
            config.hidden_size,
            config.attention_hidden
        )
        
        # Layer norm
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, entity_features: torch.Tensor, player_features: torch.Tensor) -> torch.Tensor:
        """
        Process entity features with attention mechanism.
        
        Args:
            entity_features: [batch, max_entities * entity_dim] flattened entity features
            player_features: [batch, hidden_size] processed player features
            
        Returns:
            [batch, hidden_size] attended entity features
        """
        batch_size = entity_features.shape[0]
        
        # Reshape entity features
        entity_features = entity_features.view(
            batch_size, self.config.max_entities, self.config.entity_feature_dim
        )
        
        # Project entities
        entities_projected = self.entity_projection(entity_features)
        
        # Project player features to attention dimension
        query = self.query_projection(player_features).unsqueeze(1)  # [batch, 1, attention_hidden]
        
        # Apply attention
        attended, _ = self.attention(query, entities_projected, entities_projected)
        attended = attended.squeeze(1)  # [batch, attention_hidden]
        
        # Output projection
        output = self.output_projection(attended)
        
        # Layer norm
        if self.config.use_layer_norm:
            output = self.layer_norm(output)
        
        return output


class SpatialFeatureExtractor(nn.Module):
    """
    CNN-based spatial feature extractor for entity and food positions.
    """
    def __init__(self, config: BlackholioModelConfig):
        super().__init__()
        self.config = config
        
        # Spatial grid parameters
        self.grid_size = 32  # 32x32 grid
        self.grid_channels = 3  # entities, food, player
        
        # CNN layers
        self.conv1 = nn.Conv2d(self.grid_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and output
        self.pool = nn.MaxPool2d(2)
        self.flatten_size = 128 * (self.grid_size // 8) ** 2
        self.output = nn.Linear(self.flatten_size, config.hidden_size)
        
        # Activation
        self.activation = self._get_activation(config.activation)
    
    def _get_activation(self, name: str):
        """Get activation function by name"""
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU()
    
    def _create_spatial_grid(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Create spatial grid from observations.
        
        This is a simplified implementation - in practice, you might want
        to use more sophisticated spatial encoding.
        """
        batch_size = observations.shape[0]
        grid = torch.zeros(batch_size, self.grid_channels, self.grid_size, self.grid_size)
        
        # For now, return empty grid - implement actual spatial encoding if needed
        return grid.to(observations.device)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from observations"""
        # Create spatial grid
        grid = self._create_spatial_grid(observations)
        
        # Apply CNN
        x = self.activation(self.conv1(grid))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.pool(x)
        
        # Flatten and project
        x = x.view(x.size(0), -1)
        x = self.output(x)
        
        return x


class BlackholioModel(nn.Module):
    """
    Main neural network model for Blackholio RL agent.
    
    Architecture:
    1. Feature extraction for different observation components
    2. Attention mechanism for entity relationships
    3. Feature fusion
    4. Separate policy and value heads
    
    Supports both continuous movement and discrete split actions.
    """
    
    def __init__(self, config: BlackholioModelConfig = None):
        super().__init__()
        self.config = config or BlackholioModelConfig()
        
        # Move model to specified device
        self.device = torch.device(self.config.device)
        
        # Get activation function
        self.activation = self._get_activation(self.config.activation)
        
        # Feature extraction indices
        self.player_slice = slice(0, self.config.player_state_dim)
        self.entity_slice = slice(
            self.config.player_state_dim,
            self.config.player_state_dim + self.config.max_entities * self.config.entity_feature_dim
        )
        self.food_slice = slice(
            self.config.player_state_dim + self.config.max_entities * self.config.entity_feature_dim,
            self.config.observation_dim
        )
        
        # Player state processing
        self.player_encoder = nn.Sequential(
            nn.Linear(self.config.player_state_dim, self.config.hidden_size),
            self.activation,
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
        # Entity attention (if enabled)
        if self.config.use_attention:
            self.entity_attention = EntityAttention(self.config)
        else:
            # Simple MLP for entity features
            self.entity_encoder = nn.Sequential(
                nn.Linear(
                    self.config.max_entities * self.config.entity_feature_dim,
                    self.config.hidden_size
                ),
                self.activation,
                nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )
        
        # Food processing
        self.food_encoder = nn.Sequential(
            nn.Linear(
                self.config.max_food * self.config.food_feature_dim,
                self.config.hidden_size // 2
            ),
            self.activation,
            nn.Linear(self.config.hidden_size // 2, self.config.hidden_size // 2)
        )
        
        # Spatial features (if enabled)
        if self.config.use_spatial_features:
            self.spatial_extractor = SpatialFeatureExtractor(self.config)
            fusion_input_size = self.config.hidden_size * 3 + self.config.hidden_size // 2
        else:
            fusion_input_size = self.config.hidden_size * 2 + self.config.hidden_size // 2
        
        # Feature fusion
        self.fusion_layers = nn.ModuleList()
        current_size = fusion_input_size
        
        for i in range(self.config.num_layers):
            layer = nn.Sequential(
                nn.Linear(current_size, self.config.hidden_size),
                self.activation
            )
            if self.config.use_layer_norm:
                layer.add_module("norm", nn.LayerNorm(self.config.hidden_size))
            if self.config.dropout > 0:
                layer.add_module("dropout", nn.Dropout(self.config.dropout))
            
            self.fusion_layers.append(layer)
            current_size = self.config.hidden_size
        
        # LSTM layer (if enabled)
        if self.config.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.config.hidden_size,
                hidden_size=self.config.lstm_hidden_size,
                batch_first=True
            )
            policy_input_size = self.config.lstm_hidden_size
            value_input_size = self.config.lstm_hidden_size
        else:
            policy_input_size = self.config.hidden_size
            value_input_size = self.config.hidden_size
        
        # Policy head for continuous movement
        self.movement_mean = nn.Sequential(
            nn.Linear(policy_input_size, self.config.hidden_size // 2),
            self.activation,
            nn.Linear(self.config.hidden_size // 2, self.config.movement_dim)
        )
        
        # Learned log std for movement
        self.movement_log_std = nn.Parameter(
            torch.zeros(1, self.config.movement_dim)
        )
        
        # Policy head for discrete actions (split)
        self.discrete_policy = nn.Sequential(
            nn.Linear(policy_input_size, self.config.hidden_size // 2),
            self.activation,
            nn.Linear(self.config.hidden_size // 2, self.config.discrete_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(value_input_size, self.config.hidden_size // 2),
            self.activation,
            nn.Linear(self.config.hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"BlackholioModel initialized with {self._count_parameters()} parameters")
    
    def _get_activation(self, name: str):
        """Get activation function by name"""
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, 
                observations: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> ModelOutput:
        """
        Forward pass through the model.
        
        Args:
            observations: [batch, observation_dim] tensor of observations
            hidden_state: Optional LSTM hidden state
            
        Returns:
            ModelOutput with policy and value predictions
        """
        batch_size = observations.shape[0]
        
        # Extract observation components
        player_features = observations[:, self.player_slice]
        entity_features = observations[:, self.entity_slice]
        food_features = observations[:, self.food_slice]
        
        # Process player state
        player_encoded = self.player_encoder(player_features)
        
        # Process entity features
        if self.config.use_attention:
            entity_encoded = self.entity_attention(entity_features, player_encoded)
        else:
            entity_encoded = self.entity_encoder(entity_features)
        
        # Process food features
        food_encoded = self.food_encoder(food_features)
        
        # Combine features
        features = [player_encoded, entity_encoded, food_encoded]
        
        # Add spatial features if enabled
        if self.config.use_spatial_features:
            spatial_features = self.spatial_extractor(observations)
            features.append(spatial_features)
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=-1)
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            combined_features = layer(combined_features)
        
        # Apply LSTM if enabled
        if self.config.use_lstm:
            # Reshape for LSTM (add time dimension)
            lstm_input = combined_features.unsqueeze(1)
            
            if hidden_state is None:
                lstm_output, new_hidden = self.lstm(lstm_input)
            else:
                lstm_output, new_hidden = self.lstm(lstm_input, hidden_state)
            
            # Remove time dimension
            features_final = lstm_output.squeeze(1)
        else:
            features_final = combined_features
            new_hidden = None
        
        # Compute policy outputs
        movement_mean = self.movement_mean(features_final)
        movement_mean = torch.tanh(movement_mean)  # Bound to [-1, 1]
        
        movement_std = torch.exp(self.movement_log_std).expand_as(movement_mean)
        movement_std = torch.clamp(movement_std, min=0.01, max=1.0)
        
        split_logits = self.discrete_policy(features_final)
        
        # Compute value
        value = self.value_head(features_final)
        
        return ModelOutput(
            movement_mean=movement_mean,
            movement_std=movement_std,
            split_logits=split_logits,
            value=value.squeeze(-1),
            hidden_state=new_hidden
        )
    
    def get_action(self, 
                   observations: torch.Tensor,
                   deterministic: bool = False,
                   hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[Dict[str, torch.Tensor], ModelOutput]:
        """
        Get action from observations.
        
        Args:
            observations: Observation tensor
            deterministic: If True, return deterministic action
            hidden_state: Optional LSTM hidden state
            
        Returns:
            Tuple of (actions_dict, model_output)
        """
        model_output = self.forward(observations, hidden_state)
        
        # Sample or get deterministic movement
        if deterministic:
            movement = model_output.movement_mean
        else:
            movement_dist = torch.distributions.Normal(
                model_output.movement_mean,
                model_output.movement_std
            )
            movement = movement_dist.sample()
            # Clamp movement to valid range
            movement = torch.clamp(movement, -1.0, 1.0)
        
        # Sample or get deterministic split action
        if deterministic:
            split = (model_output.split_logits.argmax(dim=-1) == 1).float()
        else:
            split_probs = F.softmax(model_output.split_logits, dim=-1)
            split_dist = torch.distributions.Categorical(split_probs)
            split = (split_dist.sample() == 1).float()
        
        actions = {
            'movement': movement,
            'split': split
        }
        
        return actions, model_output
    
    def get_log_prob(self,
                     observations: torch.Tensor,
                     actions: Dict[str, torch.Tensor],
                     hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probabilities of actions given observations.
        
        Args:
            observations: Observation tensor
            actions: Dictionary with 'movement' and 'split' tensors
            hidden_state: Optional LSTM hidden state
            
        Returns:
            Tuple of (log_prob, entropy)
        """
        model_output = self.forward(observations, hidden_state)
        
        # Movement log prob
        movement_dist = torch.distributions.Normal(
            model_output.movement_mean,
            model_output.movement_std
        )
        movement_log_prob = movement_dist.log_prob(actions['movement']).sum(dim=-1)
        movement_entropy = movement_dist.entropy().sum(dim=-1)
        
        # Split log prob
        split_probs = F.softmax(model_output.split_logits, dim=-1)
        split_dist = torch.distributions.Categorical(split_probs)
        split_action = actions['split'].long().squeeze(-1)
        split_log_prob = split_dist.log_prob(split_action)
        split_entropy = split_dist.entropy()
        
        # Combine log probs and entropies
        total_log_prob = movement_log_prob + split_log_prob
        total_entropy = movement_entropy + split_entropy
        
        return total_log_prob, total_entropy
    
    def save(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'BlackholioModel':
        """Load model from file"""
        # Load with weights_only=False since we need to load the config object
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        config = checkpoint['config']
        if device is not None:
            config.device = device
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(model.device)
        
        logger.info(f"Model loaded from {path}")
        return model
