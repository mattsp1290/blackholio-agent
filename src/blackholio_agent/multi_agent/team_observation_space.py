"""
Team observation space for multi-agent coordination.

This module extends the single-agent observation space to include
information about teammates and team communication.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from ..environment.observation_space import ObservationSpace, ObservationConfig
from .agent_communication import CommunicationMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass
class TeamObservationConfig:
    """Configuration for team observation space"""
    # Base observation config
    base_config: ObservationConfig = None
    
    # Team size limits
    max_teammates: int = 7  # Maximum teammates (total team size = 8)
    
    # Teammate information
    teammate_feature_dim: int = 8  # per teammate: x, y, vx, vy, mass, circles, health, intention
    include_teammate_entities: bool = True
    teammate_entity_limit: int = 20  # entities per teammate
    
    # Communication features
    include_communication: bool = True
    max_recent_messages: int = 10
    message_feature_dim: int = 12  # sender_id_hash, type, urgency, age, content_features...
    
    # Team coordination features
    include_team_stats: bool = True
    team_stats_dim: int = 8  # team_mass, team_spread, team_center, formation_type, etc.
    
    # Relative positioning
    use_relative_positions: bool = True
    normalize_positions: bool = True
    
    def __post_init__(self):
        if self.base_config is None:
            self.base_config = ObservationConfig()


class TeamObservationSpace:
    """
    Extended observation space for multi-agent teams.
    
    Combines individual agent observations with:
    - Teammate positions, velocities, and states
    - Recent team communication messages
    - Team-level statistics and formations
    - Coordination signals and intentions
    """
    
    def __init__(self, config: TeamObservationConfig = None):
        """
        Initialize team observation space.
        
        Args:
            config: Team observation configuration
        """
        self.config = config or TeamObservationConfig()
        
        # Create base observation space
        self.base_obs_space = ObservationSpace(self.config.base_config)
        
        # Calculate dimensions
        self._calculate_dimensions()
        
        logger.info(f"TeamObservationSpace initialized with {self.shape[0]} features")
        logger.info(f"Base observation: {self.base_obs_space.shape[0]} features")
        logger.info(f"Team extensions: {self.shape[0] - self.base_obs_space.shape[0]} features")
    
    def _calculate_dimensions(self):
        """Calculate total observation space dimensions"""
        base_dim = self.base_obs_space.shape[0]
        
        # Teammate features
        teammate_dim = 0
        if self.config.max_teammates > 0:
            teammate_dim = self.config.max_teammates * self.config.teammate_feature_dim
            
            if self.config.include_teammate_entities:
                teammate_entity_dim = (self.config.max_teammates * 
                                     self.config.teammate_entity_limit * 
                                     5)  # x, y, vx, vy, mass per entity
                teammate_dim += teammate_entity_dim
        
        # Communication features
        comm_dim = 0
        if self.config.include_communication:
            comm_dim = self.config.max_recent_messages * self.config.message_feature_dim
        
        # Team statistics
        team_stats_dim = 0
        if self.config.include_team_stats:
            team_stats_dim = self.config.team_stats_dim
        
        total_dim = base_dim + teammate_dim + comm_dim + team_stats_dim
        self.shape = (total_dim,)
        
        # Store component dimensions for debugging
        self.component_dims = {
            "base": base_dim,
            "teammates": teammate_dim,
            "communication": comm_dim,
            "team_stats": team_stats_dim,
            "total": total_dim
        }
    
    def process_team_state(self,
                          player_entities: List[Dict[str, Any]],
                          other_entities: List[Dict[str, Any]],
                          food_entities: List[Dict[str, Any]],
                          teammates: List[Dict[str, Any]],
                          recent_messages: List[CommunicationMessage],
                          team_stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Process team game state into observation vector.
        
        Args:
            player_entities: This agent's entities
            other_entities: Non-team entities in the area
            food_entities: Food entities in the area
            teammates: List of teammate states
            recent_messages: Recent team communication
            team_stats: Pre-calculated team statistics
            
        Returns:
            Team observation vector
        """
        # Get base observation
        base_obs = self.base_obs_space.process_game_state(
            player_entities, other_entities, food_entities
        )
        
        components = [base_obs]
        
        # Add teammate information
        if self.config.max_teammates > 0:
            teammate_obs = self._process_teammates(player_entities, teammates)
            components.append(teammate_obs)
        
        # Add communication information
        if self.config.include_communication:
            comm_obs = self._process_communication(recent_messages)
            components.append(comm_obs)
        
        # Add team statistics
        if self.config.include_team_stats:
            team_obs = self._process_team_stats(player_entities, teammates, team_stats)
            components.append(team_obs)
        
        # Concatenate all components
        full_observation = np.concatenate(components)
        
        # Ensure correct shape
        if len(full_observation) != self.shape[0]:
            logger.warning(f"Observation shape mismatch: {len(full_observation)} != {self.shape[0]}")
            # Pad or truncate to correct size
            if len(full_observation) < self.shape[0]:
                padding = np.zeros(self.shape[0] - len(full_observation))
                full_observation = np.concatenate([full_observation, padding])
            else:
                full_observation = full_observation[:self.shape[0]]
        
        return full_observation.astype(np.float32)
    
    def _process_teammates(self, 
                          player_entities: List[Dict[str, Any]], 
                          teammates: List[Dict[str, Any]]) -> np.ndarray:
        """Process teammate information"""
        teammate_features = []
        
        # Get player center for relative positioning
        player_center = self._get_entity_center(player_entities)
        
        for i in range(self.config.max_teammates):
            if i < len(teammates):
                teammate = teammates[i]
                features = self._extract_teammate_features(teammate, player_center)
            else:
                # Pad with zeros for missing teammates
                features = np.zeros(self.config.teammate_feature_dim)
            
            teammate_features.extend(features)
        
        # Add teammate entity information if enabled
        if self.config.include_teammate_entities:
            for i in range(self.config.max_teammates):
                if i < len(teammates):
                    teammate = teammates[i]
                    entity_features = self._extract_teammate_entities(teammate, player_center)
                else:
                    # Pad with zeros
                    entity_features = np.zeros(self.config.teammate_entity_limit * 5)
                
                teammate_features.extend(entity_features)
        
        return np.array(teammate_features, dtype=np.float32)
    
    def _extract_teammate_features(self, 
                                  teammate: Dict[str, Any], 
                                  player_center: Tuple[float, float]) -> np.ndarray:
        """Extract features for a single teammate"""
        features = []
        
        # Get teammate center position
        teammate_entities = teammate.get('entities', [])
        teammate_center = self._get_entity_center(teammate_entities)
        
        if self.config.use_relative_positions:
            # Relative position to player
            rel_x = teammate_center[0] - player_center[0]
            rel_y = teammate_center[1] - player_center[1]
        else:
            rel_x, rel_y = teammate_center
        
        # Normalize positions if requested
        if self.config.normalize_positions:
            rel_x = np.tanh(rel_x / 1000.0)  # Normalize to [-1, 1]
            rel_y = np.tanh(rel_y / 1000.0)
        
        features.extend([rel_x, rel_y])
        
        # Teammate velocity (estimated from recent positions)
        vx = teammate.get('velocity_x', 0.0)
        vy = teammate.get('velocity_y', 0.0)
        features.extend([vx, vy])
        
        # Teammate mass and entity count
        total_mass = sum(e.get('mass', 1.0) for e in teammate_entities)
        entity_count = len(teammate_entities)
        features.extend([total_mass, entity_count])
        
        # Teammate health/status
        health = teammate.get('health', 1.0)  # 0-1 scale
        features.append(health)
        
        # Teammate intention (if available from communication)
        intention = teammate.get('current_intention', 0.0)  # Encoded intention
        features.append(intention)
        
        return np.array(features[:self.config.teammate_feature_dim], dtype=np.float32)
    
    def _extract_teammate_entities(self, 
                                  teammate: Dict[str, Any], 
                                  player_center: Tuple[float, float]) -> np.ndarray:
        """Extract entity information for a teammate"""
        entities = teammate.get('entities', [])
        entity_features = []
        
        for i in range(self.config.teammate_entity_limit):
            if i < len(entities):
                entity = entities[i]
                
                # Entity position (relative to player)
                if self.config.use_relative_positions:
                    rel_x = entity.get('x', 0.0) - player_center[0]
                    rel_y = entity.get('y', 0.0) - player_center[1]
                else:
                    rel_x = entity.get('x', 0.0)
                    rel_y = entity.get('y', 0.0)
                
                if self.config.normalize_positions:
                    rel_x = np.tanh(rel_x / 1000.0)
                    rel_y = np.tanh(rel_y / 1000.0)
                
                # Entity velocity and mass
                vx = entity.get('velocity_x', 0.0)
                vy = entity.get('velocity_y', 0.0)
                mass = entity.get('mass', 1.0)
                
                entity_features.extend([rel_x, rel_y, vx, vy, mass])
            else:
                # Pad with zeros
                entity_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(entity_features, dtype=np.float32)
    
    def _process_communication(self, messages: List[CommunicationMessage]) -> np.ndarray:
        """Process recent communication messages"""
        comm_features = []
        current_time = np.time.time() if hasattr(np.time, 'time') else 0.0
        
        # Sort messages by priority and recency
        sorted_messages = sorted(messages, 
                               key=lambda m: (m.priority, -m.timestamp))
        
        for i in range(self.config.max_recent_messages):
            if i < len(sorted_messages):
                msg = sorted_messages[i]
                features = self._extract_message_features(msg, current_time)
            else:
                # Pad with zeros
                features = np.zeros(self.config.message_feature_dim)
            
            comm_features.extend(features)
        
        return np.array(comm_features, dtype=np.float32)
    
    def _extract_message_features(self, 
                                 message: CommunicationMessage, 
                                 current_time: float) -> np.ndarray:
        """Extract features from a communication message"""
        features = []
        
        # Sender ID hash (simple hash to fixed range)
        sender_hash = hash(message.sender_id) % 100 / 100.0
        features.append(sender_hash)
        
        # Message type (one-hot encoded to single value)
        msg_type_map = {
            MessageType.POSITION_UPDATE: 0.1,
            MessageType.MOVEMENT_INTENTION: 0.2,
            MessageType.TARGET_DESIGNATION: 0.3,
            MessageType.SPLIT_COORDINATION: 0.4,
            MessageType.DANGER_ALERT: 0.5,
            MessageType.OPPORTUNITY_SIGNAL: 0.6,
            MessageType.FOOD_LOCATION: 0.7,
            MessageType.ATTACK_COORDINATION: 0.8,
            MessageType.RETREAT_SIGNAL: 0.9,
        }
        msg_type_val = msg_type_map.get(message.message_type, 0.0)
        features.append(msg_type_val)
        
        # Message priority and age
        priority_norm = message.priority / 5.0  # Normalize to [0, 1]
        age = min(current_time - message.timestamp, 10.0) / 10.0  # Normalize age
        features.extend([priority_norm, age])
        
        # Extract key content features
        content = message.content
        
        # Position information (if available)
        x = content.get('x', 0.0) / 1000.0  # Normalize position
        y = content.get('y', 0.0) / 1000.0
        features.extend([x, y])
        
        # Threat/opportunity level
        threat_level = content.get('threat_level', 0.0) / 5.0
        mass_value = min(content.get('mass', 0.0), 1000.0) / 1000.0
        features.extend([threat_level, mass_value])
        
        # Action type (for coordination messages)
        action_map = {
            'request': 0.3,
            'confirm': 0.6,
            'deny': 0.1,
            'complete': 0.9
        }
        action_val = action_map.get(content.get('action', ''), 0.0)
        features.append(action_val)
        
        # Target information
        target_x = content.get('target_x', 0.0) / 1000.0
        target_y = content.get('target_y', 0.0) / 1000.0
        features.extend([target_x, target_y])
        
        # Pad to correct size
        while len(features) < self.config.message_feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.config.message_feature_dim], dtype=np.float32)
    
    def _process_team_stats(self, 
                           player_entities: List[Dict[str, Any]],
                           teammates: List[Dict[str, Any]],
                           team_stats: Optional[Dict[str, Any]]) -> np.ndarray:
        """Process team-level statistics"""
        if team_stats:
            # Use pre-calculated stats
            features = self._extract_team_stats_features(team_stats)
        else:
            # Calculate stats on the fly
            features = self._calculate_team_stats(player_entities, teammates)
        
        # Ensure correct size
        while len(features) < self.config.team_stats_dim:
            features.append(0.0)
        
        return np.array(features[:self.config.team_stats_dim], dtype=np.float32)
    
    def _extract_team_stats_features(self, team_stats: Dict[str, Any]) -> List[float]:
        """Extract features from pre-calculated team statistics"""
        features = []
        
        # Team mass and size
        total_mass = team_stats.get('total_mass', 0.0) / 1000.0  # Normalize
        team_size = team_stats.get('active_members', 0.0) / 8.0  # Max team size = 8
        features.extend([total_mass, team_size])
        
        # Team formation spread
        spread = team_stats.get('formation_spread', 0.0) / 1000.0
        compactness = team_stats.get('compactness', 0.0)
        features.extend([spread, compactness])
        
        # Team center (relative to map center)
        center_x = team_stats.get('center_x', 0.0) / 1000.0
        center_y = team_stats.get('center_y', 0.0) / 1000.0
        features.extend([center_x, center_y])
        
        # Team coordination metrics
        coordination_score = team_stats.get('coordination_score', 0.0)
        communication_activity = team_stats.get('comm_activity', 0.0)
        features.extend([coordination_score, communication_activity])
        
        return features
    
    def _calculate_team_stats(self, 
                             player_entities: List[Dict[str, Any]],
                             teammates: List[Dict[str, Any]]) -> List[float]:
        """Calculate team statistics on the fly"""
        features = []
        
        # Collect all team entities
        all_entities = list(player_entities)
        for teammate in teammates:
            all_entities.extend(teammate.get('entities', []))
        
        if not all_entities:
            return [0.0] * self.config.team_stats_dim
        
        # Total team mass
        total_mass = sum(e.get('mass', 1.0) for e in all_entities)
        features.append(total_mass / 1000.0)  # Normalize
        
        # Team size (active members)
        active_members = 1 + len([t for t in teammates if t.get('entities')])
        features.append(active_members / 8.0)
        
        # Team spread and center
        positions = [(e.get('x', 0.0), e.get('y', 0.0)) for e in all_entities]
        center_x = np.mean([p[0] for p in positions])
        center_y = np.mean([p[1] for p in positions])
        
        # Calculate spread as std deviation of distances from center
        distances = [np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in positions]
        spread = np.std(distances) if len(distances) > 1 else 0.0
        compactness = 1.0 / (1.0 + spread / 100.0)  # Inverse of spread
        
        features.extend([spread / 1000.0, compactness])
        features.extend([center_x / 1000.0, center_y / 1000.0])
        
        # Placeholder coordination metrics
        coordination_score = min(1.0, active_members / 4.0)  # Simple metric
        comm_activity = 0.5  # Would be calculated from recent messages
        features.extend([coordination_score, comm_activity])
        
        return features
    
    def _get_entity_center(self, entities: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate center of mass for entities"""
        if not entities:
            return (0.0, 0.0)
        
        total_mass = sum(e.get('mass', 1.0) for e in entities)
        if total_mass == 0:
            return (0.0, 0.0)
        
        center_x = sum(e.get('x', 0.0) * e.get('mass', 1.0) for e in entities) / total_mass
        center_y = sum(e.get('y', 0.0) * e.get('mass', 1.0) for e in entities) / total_mass
        
        return (center_x, center_y)
    
    def get_component_breakdown(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """Break down observation into components for debugging"""
        components = {}
        start_idx = 0
        
        # Base observation
        end_idx = start_idx + self.component_dims["base"]
        components["base"] = observation[start_idx:end_idx]
        start_idx = end_idx
        
        # Teammates
        if self.component_dims["teammates"] > 0:
            end_idx = start_idx + self.component_dims["teammates"]
            components["teammates"] = observation[start_idx:end_idx]
            start_idx = end_idx
        
        # Communication
        if self.component_dims["communication"] > 0:
            end_idx = start_idx + self.component_dims["communication"]
            components["communication"] = observation[start_idx:end_idx]
            start_idx = end_idx
        
        # Team stats
        if self.component_dims["team_stats"] > 0:
            end_idx = start_idx + self.component_dims["team_stats"]
            components["team_stats"] = observation[start_idx:end_idx]
        
        return components
