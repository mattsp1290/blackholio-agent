"""
Observation space for Blackholio environment.

This module converts raw game state into normalized numpy arrays
suitable for neural network input.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ObservationConfig:
    """Configuration for observation space"""
    max_entities_tracked: int = 50  # Maximum number of other entities to track
    max_food_tracked: int = 100     # Maximum number of food items to track
    observation_radius: float = 500.0  # Radius around player to observe
    include_food: bool = True       # Whether to include food in observations
    include_velocities: bool = True  # Whether to include velocity information
    arena_width: float = 2000.0     # Arena width for normalization
    arena_height: float = 2000.0    # Arena height for normalization
    max_mass: float = 10000.0       # Maximum mass for normalization
    max_velocity: float = 100.0     # Maximum velocity for normalization
    max_circles: int = 16           # Maximum number of circles per player


class ObservationSpace:
    """
    Converts game state to ML-friendly numpy arrays.
    
    The observation includes:
    - Player state: mass, position, velocity, number of circles
    - Nearby entities: relative positions, masses, velocities
    - Food positions: relative positions within visible range
    - Arena information: boundaries and safe zones
    
    All values are normalized to appropriate ranges for neural network input.
    """
    
    def __init__(self, config: ObservationConfig = None):
        self.config = config or ObservationConfig()
        
        # Calculate observation dimensions
        self.player_state_dim = 6  # mass, x, y, vx, vy, num_circles
        self.entity_feature_dim = 5  # rel_x, rel_y, mass, vx, vy
        self.food_feature_dim = 2  # rel_x, rel_y
        
        self.entities_dim = self.config.max_entities_tracked * self.entity_feature_dim
        self.food_dim = self.config.max_food_tracked * self.food_feature_dim if self.config.include_food else 0
        
        self.total_dim = self.player_state_dim + self.entities_dim + self.food_dim
        
        logger.info(f"Observation space initialized with total dimension: {self.total_dim}")
    
    @property
    def shape(self) -> Tuple[int]:
        """Get the shape of the observation space"""
        return (self.total_dim,)
    
    def process_game_state(self, 
                          player_entities: List[Dict[str, Any]], 
                          other_entities: List[Dict[str, Any]],
                          food_entities: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Process game state into observation array.
        
        Args:
            player_entities: List of entities owned by the player
            other_entities: List of other entities in the game
            food_entities: List of food entities (optional)
            
        Returns:
            Normalized observation array
        """
        if not player_entities:
            # Return zero observation if player has no entities (dead)
            return np.zeros(self.total_dim, dtype=np.float32)
        
        # Calculate player state
        player_state = self._calculate_player_state(player_entities)
        
        # Get player center of mass for relative calculations
        player_x, player_y = self._calculate_center_of_mass(player_entities)
        
        # Process nearby entities
        entity_features = self._process_nearby_entities(
            other_entities, player_x, player_y
        )
        
        # Process food if included
        food_features = np.zeros(self.food_dim, dtype=np.float32)
        if self.config.include_food and food_entities:
            food_features = self._process_food(
                food_entities, player_x, player_y
            )
        
        # Combine all features
        observation = np.concatenate([
            player_state,
            entity_features,
            food_features
        ])
        
        return observation.astype(np.float32)
    
    def _calculate_player_state(self, player_entities: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate aggregated player state"""
        total_mass = sum(e.get('mass', 0) for e in player_entities)
        num_circles = len(player_entities)
        
        # Calculate center of mass and average velocity
        center_x, center_y = self._calculate_center_of_mass(player_entities)
        avg_vx, avg_vy = self._calculate_average_velocity(player_entities)
        
        # Normalize values
        norm_mass = np.clip(total_mass / self.config.max_mass, 0, 1)
        norm_x = (center_x / self.config.arena_width) * 2 - 1  # [-1, 1]
        norm_y = (center_y / self.config.arena_height) * 2 - 1  # [-1, 1]
        norm_vx = np.clip(avg_vx / self.config.max_velocity, -1, 1)
        norm_vy = np.clip(avg_vy / self.config.max_velocity, -1, 1)
        norm_circles = num_circles / self.config.max_circles
        
        return np.array([
            norm_mass, norm_x, norm_y, norm_vx, norm_vy, norm_circles
        ], dtype=np.float32)
    
    def _calculate_center_of_mass(self, entities: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate center of mass for a group of entities"""
        if not entities:
            return 0.0, 0.0
        
        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        for entity in entities:
            mass = entity.get('mass', 1.0)
            x = entity.get('x', 0.0)
            y = entity.get('y', 0.0)
            
            total_mass += mass
            weighted_x += x * mass
            weighted_y += y * mass
        
        if total_mass > 0:
            return weighted_x / total_mass, weighted_y / total_mass
        else:
            # Fallback to simple average
            avg_x = sum(e.get('x', 0.0) for e in entities) / len(entities)
            avg_y = sum(e.get('y', 0.0) for e in entities) / len(entities)
            return avg_x, avg_y
    
    def _calculate_average_velocity(self, entities: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate average velocity for a group of entities"""
        if not entities:
            return 0.0, 0.0
        
        if not self.config.include_velocities:
            return 0.0, 0.0
        
        # Weight velocities by mass
        total_mass = sum(e.get('mass', 1.0) for e in entities)
        if total_mass > 0:
            weighted_vx = sum(e.get('vx', 0.0) * e.get('mass', 1.0) for e in entities)
            weighted_vy = sum(e.get('vy', 0.0) * e.get('mass', 1.0) for e in entities)
            return weighted_vx / total_mass, weighted_vy / total_mass
        else:
            avg_vx = sum(e.get('vx', 0.0) for e in entities) / len(entities)
            avg_vy = sum(e.get('vy', 0.0) for e in entities) / len(entities)
            return avg_vx, avg_vy
    
    def _process_nearby_entities(self, 
                                entities: List[Dict[str, Any]], 
                                player_x: float, 
                                player_y: float) -> np.ndarray:
        """Process nearby entities into feature array"""
        features = np.zeros(self.entities_dim, dtype=np.float32)
        
        if not entities:
            return features
        
        # Calculate distances and sort by proximity
        entity_distances = []
        for entity in entities:
            x = entity.get('x', 0.0)
            y = entity.get('y', 0.0)
            dist = np.sqrt((x - player_x)**2 + (y - player_y)**2)
            
            if dist <= self.config.observation_radius:
                entity_distances.append((dist, entity))
        
        # Sort by distance (closest first)
        entity_distances.sort(key=lambda x: x[0])
        
        # Process up to max_entities_tracked
        for i, (dist, entity) in enumerate(entity_distances[:self.config.max_entities_tracked]):
            # Calculate relative position
            rel_x = entity.get('x', 0.0) - player_x
            rel_y = entity.get('y', 0.0) - player_y
            
            # Normalize relative position
            norm_rel_x = np.clip(rel_x / self.config.observation_radius, -1, 1)
            norm_rel_y = np.clip(rel_y / self.config.observation_radius, -1, 1)
            
            # Normalize mass
            mass = entity.get('mass', 0.0)
            norm_mass = np.clip(mass / self.config.max_mass, 0, 1)
            
            # Normalize velocities
            vx = entity.get('vx', 0.0) if self.config.include_velocities else 0.0
            vy = entity.get('vy', 0.0) if self.config.include_velocities else 0.0
            norm_vx = np.clip(vx / self.config.max_velocity, -1, 1)
            norm_vy = np.clip(vy / self.config.max_velocity, -1, 1)
            
            # Fill feature array
            start_idx = i * self.entity_feature_dim
            features[start_idx:start_idx + self.entity_feature_dim] = [
                norm_rel_x, norm_rel_y, norm_mass, norm_vx, norm_vy
            ]
        
        return features
    
    def _process_food(self,
                     food_entities: List[Dict[str, Any]],
                     player_x: float,
                     player_y: float) -> np.ndarray:
        """Process food entities into feature array"""
        features = np.zeros(self.food_dim, dtype=np.float32)
        
        if not food_entities:
            return features
        
        # Calculate distances and sort by proximity
        food_distances = []
        for food in food_entities:
            x = food.get('x', 0.0)
            y = food.get('y', 0.0)
            dist = np.sqrt((x - player_x)**2 + (y - player_y)**2)
            
            if dist <= self.config.observation_radius:
                food_distances.append((dist, food))
        
        # Sort by distance (closest first)
        food_distances.sort(key=lambda x: x[0])
        
        # Process up to max_food_tracked
        for i, (dist, food) in enumerate(food_distances[:self.config.max_food_tracked]):
            # Calculate relative position
            rel_x = food.get('x', 0.0) - player_x
            rel_y = food.get('y', 0.0) - player_y
            
            # Normalize relative position
            norm_rel_x = np.clip(rel_x / self.config.observation_radius, -1, 1)
            norm_rel_y = np.clip(rel_y / self.config.observation_radius, -1, 1)
            
            # Fill feature array
            start_idx = i * self.food_feature_dim
            features[start_idx:start_idx + self.food_feature_dim] = [
                norm_rel_x, norm_rel_y
            ]
        
        return features
    
    def get_feature_indices(self) -> Dict[str, Tuple[int, int]]:
        """
        Get the start and end indices for each feature group in the observation.
        
        Returns:
            Dictionary mapping feature names to (start, end) index tuples
        """
        indices = {}
        current = 0
        
        # Player state
        indices['player_state'] = (current, current + self.player_state_dim)
        current += self.player_state_dim
        
        # Entity features
        indices['entities'] = (current, current + self.entities_dim)
        current += self.entities_dim
        
        # Food features
        if self.config.include_food:
            indices['food'] = (current, current + self.food_dim)
            current += self.food_dim
        
        return indices
    
    def decode_observation(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Decode an observation array back into human-readable format.
        Useful for debugging and visualization.
        
        Args:
            observation: The observation array to decode
            
        Returns:
            Dictionary with decoded observation components
        """
        indices = self.get_feature_indices()
        decoded = {}
        
        # Decode player state
        player_start, player_end = indices['player_state']
        player_state = observation[player_start:player_end]
        decoded['player'] = {
            'mass': player_state[0] * self.config.max_mass,
            'x': (player_state[1] + 1) * self.config.arena_width / 2,
            'y': (player_state[2] + 1) * self.config.arena_height / 2,
            'vx': player_state[3] * self.config.max_velocity,
            'vy': player_state[4] * self.config.max_velocity,
            'num_circles': int(player_state[5] * self.config.max_circles)
        }
        
        # Decode entities
        entities_start, entities_end = indices['entities']
        entities_features = observation[entities_start:entities_end]
        decoded['entities'] = []
        
        for i in range(self.config.max_entities_tracked):
            start_idx = i * self.entity_feature_dim
            end_idx = start_idx + self.entity_feature_dim
            
            if start_idx >= len(entities_features):
                break
                
            entity_feat = entities_features[start_idx:end_idx]
            
            # Skip if entity is empty (all zeros)
            if np.allclose(entity_feat, 0):
                continue
                
            decoded['entities'].append({
                'rel_x': entity_feat[0] * self.config.observation_radius,
                'rel_y': entity_feat[1] * self.config.observation_radius,
                'mass': entity_feat[2] * self.config.max_mass,
                'vx': entity_feat[3] * self.config.max_velocity,
                'vy': entity_feat[4] * self.config.max_velocity
            })
        
        # Decode food if included
        if self.config.include_food and 'food' in indices:
            food_start, food_end = indices['food']
            food_features = observation[food_start:food_end]
            decoded['food'] = []
            
            for i in range(self.config.max_food_tracked):
                start_idx = i * self.food_feature_dim
                end_idx = start_idx + self.food_feature_dim
                
                if start_idx >= len(food_features):
                    break
                    
                food_feat = food_features[start_idx:end_idx]
                
                # Skip if food is empty (all zeros)
                if np.allclose(food_feat, 0):
                    continue
                    
                decoded['food'].append({
                    'rel_x': food_feat[0] * self.config.observation_radius,
                    'rel_y': food_feat[1] * self.config.observation_radius
                })
        
        return decoded
