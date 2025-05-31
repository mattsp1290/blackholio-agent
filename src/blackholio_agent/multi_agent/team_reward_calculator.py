"""
Team reward calculator for multi-agent coordination.

This module provides reward functions that incentivize team coordination,
cooperation, and strategic play in multi-agent Blackholio scenarios.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import time

from ..environment.reward_calculator import RewardCalculator, RewardConfig, EpisodeStats
from .agent_communication import CommunicationMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass
class TeamRewardConfig:
    """Configuration for team reward calculator"""
    # Base reward config
    base_config: RewardConfig = None
    
    # Team coordination rewards
    coordination_weight: float = 0.3
    communication_bonus: float = 0.1
    formation_bonus: float = 0.2
    shared_objective_weight: float = 0.4
    
    # Team survival rewards
    team_survival_bonus: float = 1.0
    member_death_penalty: float = -0.5
    team_wipe_penalty: float = -2.0
    
    # Cooperation incentives
    mass_sharing_bonus: float = 0.3
    coordinated_split_bonus: float = 0.5
    mutual_protection_bonus: float = 0.4
    
    # Strategic rewards
    territory_control_bonus: float = 0.2
    enemy_elimination_bonus: float = 1.0
    strategic_positioning_bonus: float = 0.15
    
    # Team efficiency metrics
    efficiency_weight: float = 0.25
    redundancy_penalty: float = -0.1
    synergy_bonus: float = 0.3
    
    # Individual vs team balance
    individual_weight: float = 0.6  # Weight for individual rewards
    team_weight: float = 0.4        # Weight for team rewards
    
    # Dynamic reward adjustment
    adaptive_rewards: bool = True
    performance_scaling: bool = True
    difficulty_adjustment: bool = True
    
    def __post_init__(self):
        if self.base_config is None:
            self.base_config = RewardConfig()


@dataclass
class TeamEpisodeStats:
    """Statistics for team episode performance"""
    # Team composition
    team_size: int = 0
    active_members: int = 0
    member_deaths: int = 0
    
    # Team performance
    total_team_mass: float = 0.0
    max_team_mass: float = 0.0
    team_kills: int = 0
    team_food_collected: int = 0
    
    # Coordination metrics
    coordination_events: int = 0
    successful_coordinations: int = 0
    communication_messages: int = 0
    formation_changes: int = 0
    
    # Survival metrics
    team_survival_time: float = 0.0
    longest_member_survival: float = 0.0
    shortest_member_survival: float = 0.0
    
    # Strategic metrics
    territory_controlled: float = 0.0
    strategic_objectives_completed: int = 0
    enemy_teams_eliminated: int = 0
    
    # Efficiency metrics
    coordination_efficiency: float = 0.0
    resource_efficiency: float = 0.0
    spatial_efficiency: float = 0.0
    
    # Reward breakdown
    individual_rewards: float = 0.0
    team_rewards: float = 0.0
    coordination_rewards: float = 0.0
    total_rewards: float = 0.0
    
    # Timestamps
    episode_start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    
    def update_survival_time(self):
        """Update survival time metrics"""
        current_time = time.time()
        self.team_survival_time = current_time - self.episode_start_time
        self.last_update_time = current_time


class TeamRewardCalculator:
    """
    Team-based reward calculator for multi-agent coordination.
    
    Provides rewards that encourage:
    - Team coordination and communication
    - Cooperative strategies
    - Shared objectives and mutual protection
    - Efficient resource utilization
    - Strategic team play
    """
    
    def __init__(self, 
                 config: TeamRewardConfig = None,
                 team_id: str = "team_01"):
        """
        Initialize team reward calculator.
        
        Args:
            config: Team reward configuration
            team_id: Unique identifier for this team
        """
        self.config = config or TeamRewardConfig()
        self.team_id = team_id
        
        # Create base reward calculator for individual components
        self.base_calculator = RewardCalculator(self.config.base_config)
        
        # Team statistics
        self.team_stats = TeamEpisodeStats()
        self.member_stats: Dict[str, EpisodeStats] = {}
        
        # Coordination tracking
        self.recent_communications: List[CommunicationMessage] = []
        self.coordination_history: List[Dict[str, Any]] = []
        self.formation_history: List[str] = []
        
        # Performance tracking
        self.performance_history: List[float] = []
        self.adaptive_multiplier = 1.0
        
        logger.info(f"TeamRewardCalculator initialized for team {team_id}")
    
    def reset(self, team_size: int = 1) -> TeamEpisodeStats:
        """
        Reset team statistics for new episode.
        
        Args:
            team_size: Initial team size
            
        Returns:
            Previous episode statistics
        """
        # Store previous stats
        prev_stats = self.team_stats
        
        # Reset team statistics
        self.team_stats = TeamEpisodeStats()
        self.team_stats.team_size = team_size
        self.team_stats.active_members = team_size
        
        # Reset member statistics
        self.member_stats.clear()
        
        # Reset coordination tracking
        self.recent_communications.clear()
        self.coordination_history.clear()
        self.formation_history.clear()
        
        # Update performance history
        if prev_stats.total_rewards != 0:
            self.performance_history.append(prev_stats.total_rewards)
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)
        
        # Update adaptive multiplier
        if self.config.adaptive_rewards:
            self._update_adaptive_multiplier()
        
        logger.debug(f"Team {self.team_id} reset for new episode")
        return prev_stats
    
    def calculate_team_step_reward(self,
                                  team_states: List[Dict[str, Any]],
                                  team_actions: List[Dict[str, Any]],
                                  team_infos: List[Dict[str, Any]],
                                  communications: List[CommunicationMessage]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Calculate step rewards for all team members.
        
        Args:
            team_states: List of game states for each team member
            team_actions: List of actions taken by each team member
            team_infos: List of info dicts for each team member
            communications: Recent team communication messages
            
        Returns:
            Tuple of (individual_rewards, team_reward_breakdown)
        """
        # Update team statistics
        self._update_team_stats(team_states, team_actions, team_infos, communications)
        
        # Calculate individual rewards using base calculator
        individual_rewards = []
        individual_components = []
        
        for i, (state, action, info) in enumerate(zip(team_states, team_actions, team_infos)):
            member_id = f"member_{i}"
            
            # Initialize member stats if needed
            if member_id not in self.member_stats:
                self.member_stats[member_id] = self.base_calculator.reset()
            
            # Calculate base individual reward
            base_reward, base_components = self.base_calculator.calculate_step_reward(
                state, action, info
            )
            individual_rewards.append(base_reward)
            individual_components.append(base_components)
        
        # Calculate team-based reward components
        team_components = self._calculate_team_reward_components(
            team_states, team_actions, team_infos, communications
        )
        
        # Combine individual and team rewards
        final_rewards = []
        for i, individual_reward in enumerate(individual_rewards):
            # Weighted combination of individual and team rewards
            team_reward_share = sum(team_components.values()) / len(individual_rewards)
            
            final_reward = (
                self.config.individual_weight * individual_reward +
                self.config.team_weight * team_reward_share
            )
            
            # Apply adaptive multiplier
            final_reward *= self.adaptive_multiplier
            
            final_rewards.append(final_reward)
        
        # Update reward tracking
        total_team_reward = sum(final_rewards)
        self.team_stats.individual_rewards += sum(individual_rewards)
        self.team_stats.team_rewards += sum(team_components.values())
        self.team_stats.total_rewards += total_team_reward
        
        # Prepare detailed breakdown
        reward_breakdown = {
            "individual_rewards": individual_rewards,
            "individual_components": individual_components,
            "team_components": team_components,
            "final_rewards": final_rewards,
            "total_team_reward": total_team_reward,
            "adaptive_multiplier": self.adaptive_multiplier,
        }
        
        return final_rewards, reward_breakdown
    
    def _update_team_stats(self,
                          team_states: List[Dict[str, Any]],
                          team_actions: List[Dict[str, Any]],
                          team_infos: List[Dict[str, Any]],
                          communications: List[CommunicationMessage]):
        """Update team statistics based on current step"""
        self.team_stats.update_survival_time()
        
        # Update active members
        active_count = sum(1 for state in team_states if state.get('player_entities'))
        self.team_stats.active_members = active_count
        
        # Track member deaths
        if active_count < self.team_stats.team_size:
            self.team_stats.member_deaths = self.team_stats.team_size - active_count
        
        # Update team mass
        total_mass = 0
        for state in team_states:
            member_mass = sum(e.get('mass', 0) for e in state.get('player_entities', []))
            total_mass += member_mass
        
        self.team_stats.total_team_mass = total_mass
        self.team_stats.max_team_mass = max(self.team_stats.max_team_mass, total_mass)
        
        # Track communications
        self.recent_communications.extend(communications)
        self.team_stats.communication_messages += len(communications)
        
        # Clean old communications (keep only recent ones)
        current_time = time.time()
        self.recent_communications = [
            msg for msg in self.recent_communications
            if current_time - msg.timestamp < 10.0  # Keep last 10 seconds
        ]
        
        # Track coordination events
        coordination_events = sum(1 for msg in communications 
                                if msg.message_type in [
                                    MessageType.SPLIT_COORDINATION,
                                    MessageType.ATTACK_COORDINATION,
                                    MessageType.TARGET_DESIGNATION
                                ])
        self.team_stats.coordination_events += coordination_events
    
    def _calculate_team_reward_components(self,
                                        team_states: List[Dict[str, Any]],
                                        team_actions: List[Dict[str, Any]],
                                        team_infos: List[Dict[str, Any]],
                                        communications: List[CommunicationMessage]) -> Dict[str, float]:
        """Calculate team-specific reward components"""
        components = {}
        
        # Coordination rewards
        coord_reward = self._calculate_coordination_reward(communications, team_states)
        components["coordination"] = coord_reward * self.config.coordination_weight
        
        # Communication rewards
        comm_reward = self._calculate_communication_reward(communications)
        components["communication"] = comm_reward * self.config.communication_bonus
        
        # Formation rewards
        formation_reward = self._calculate_formation_reward(team_states)
        components["formation"] = formation_reward * self.config.formation_bonus
        
        # Survival rewards
        survival_reward = self._calculate_survival_reward()
        components["survival"] = survival_reward
        
        # Cooperation rewards
        coop_reward = self._calculate_cooperation_reward(team_states, team_actions, communications)
        components["cooperation"] = coop_reward
        
        # Strategic rewards
        strategic_reward = self._calculate_strategic_reward(team_states, team_infos)
        components["strategic"] = strategic_reward
        
        # Efficiency rewards
        efficiency_reward = self._calculate_efficiency_reward(team_states, team_actions)
        components["efficiency"] = efficiency_reward * self.config.efficiency_weight
        
        # Store coordination rewards separately
        self.team_stats.coordination_rewards += sum(components.values())
        
        return components
    
    def _calculate_coordination_reward(self,
                                     communications: List[CommunicationMessage],
                                     team_states: List[Dict[str, Any]]) -> float:
        """Calculate reward for team coordination"""
        reward = 0.0
        
        # Reward for coordination messages
        coord_messages = [msg for msg in communications if msg.message_type in [
            MessageType.SPLIT_COORDINATION,
            MessageType.ATTACK_COORDINATION,
            MessageType.TARGET_DESIGNATION
        ]]
        
        # Base reward for coordination attempts
        reward += len(coord_messages) * 0.1
        
        # Bonus for successful coordination (simplified heuristic)
        # Check if team members are close together after coordination messages
        if coord_messages and len(team_states) > 1:
            positions = []
            for state in team_states:
                for entity in state.get('player_entities', []):
                    positions.append((entity.get('x', 0), entity.get('y', 0)))
            
            if len(positions) > 1:
                # Calculate team spread
                center_x = np.mean([pos[0] for pos in positions])
                center_y = np.mean([pos[1] for pos in positions])
                spread = np.mean([
                    np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) 
                    for pos in positions
                ])
                
                # Reward tighter formations after coordination
                if spread < 200:  # Close formation
                    reward += 0.3
                    self.team_stats.successful_coordinations += 1
        
        return reward
    
    def _calculate_communication_reward(self, communications: List[CommunicationMessage]) -> float:
        """Calculate reward for effective communication"""
        reward = 0.0
        
        # Basic communication reward
        reward += len(communications) * 0.05
        
        # Bonus for high-priority messages
        high_priority_msgs = [msg for msg in communications if msg.priority >= 4]
        reward += len(high_priority_msgs) * 0.1
        
        # Bonus for diverse message types
        message_types = set(msg.message_type for msg in communications)
        if len(message_types) > 2:
            reward += 0.2
        
        # Penalty for spam (too many messages)
        if len(communications) > 5:
            reward -= (len(communications) - 5) * 0.05
        
        return max(0, reward)
    
    def _calculate_formation_reward(self, team_states: List[Dict[str, Any]]) -> float:
        """Calculate reward for maintaining good formations"""
        if len(team_states) < 2:
            return 0.0
        
        # Collect all team entity positions
        positions = []
        for state in team_states:
            for entity in state.get('player_entities', []):
                positions.append((entity.get('x', 0), entity.get('y', 0)))
        
        if len(positions) < 2:
            return 0.0
        
        # Calculate formation metrics
        center_x = np.mean([pos[0] for pos in positions])
        center_y = np.mean([pos[1] for pos in positions])
        
        distances = [
            np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) 
            for pos in positions
        ]
        
        spread = np.std(distances)
        avg_distance = np.mean(distances)
        
        # Reward based on formation type
        reward = 0.0
        
        # Compact formation bonus
        if spread < 100 and avg_distance < 150:
            reward += 0.3
        
        # Spread formation bonus (for map control)
        elif spread > 300 and len(positions) >= 3:
            reward += 0.2
        
        # Balanced formation bonus
        elif 100 <= spread <= 300:
            reward += 0.1
        
        return reward
    
    def _calculate_survival_reward(self) -> float:
        """Calculate team survival rewards and penalties"""
        reward = 0.0
        
        # Team survival bonus
        if self.team_stats.active_members > 0:
            survival_time_bonus = min(self.team_stats.team_survival_time / 100.0, 1.0)
            reward += survival_time_bonus * self.config.team_survival_bonus
        
        # Member death penalty
        if self.team_stats.member_deaths > 0:
            reward += self.team_stats.member_deaths * self.config.member_death_penalty
        
        # Team wipe penalty
        if self.team_stats.active_members == 0:
            reward += self.config.team_wipe_penalty
        
        return reward
    
    def _calculate_cooperation_reward(self,
                                    team_states: List[Dict[str, Any]],
                                    team_actions: List[Dict[str, Any]],
                                    communications: List[CommunicationMessage]) -> float:
        """Calculate rewards for cooperative behaviors"""
        reward = 0.0
        
        # Mass sharing detection (simplified)
        # Look for coordinated split actions
        split_actions = sum(1 for action in team_actions 
                          if action.get('split', 0) > 0.5)
        
        if split_actions > 1:
            # Multiple simultaneous splits suggest coordination
            reward += split_actions * self.config.coordinated_split_bonus
        
        # Mutual protection bonus
        # Check if team members are helping each other
        danger_alerts = [msg for msg in communications 
                        if msg.message_type == MessageType.DANGER_ALERT]
        
        if danger_alerts:
            # Bonus for warning teammates about danger
            reward += len(danger_alerts) * self.config.mutual_protection_bonus
        
        # Food sharing notifications
        food_messages = [msg for msg in communications 
                        if msg.message_type == MessageType.FOOD_LOCATION]
        
        if food_messages:
            reward += len(food_messages) * self.config.mass_sharing_bonus
        
        return reward
    
    def _calculate_strategic_reward(self,
                                  team_states: List[Dict[str, Any]],
                                  team_infos: List[Dict[str, Any]]) -> float:
        """Calculate strategic play rewards"""
        reward = 0.0
        
        # Territory control (simplified)
        # Reward for spreading across different areas
        if len(team_states) > 1:
            positions = []
            for state in team_states:
                for entity in state.get('player_entities', []):
                    positions.append((entity.get('x', 0), entity.get('y', 0)))
            
            if len(positions) >= 2:
                # Calculate coverage area
                min_x, max_x = min(pos[0] for pos in positions), max(pos[0] for pos in positions)
                min_y, max_y = min(pos[1] for pos in positions), max(pos[1] for pos in positions)
                coverage_area = (max_x - min_x) * (max_y - min_y)
                
                # Reward for good map coverage
                normalized_coverage = min(coverage_area / 1000000, 1.0)
                reward += normalized_coverage * self.config.territory_control_bonus
        
        # Enemy elimination bonus
        total_kills = sum(info.get('kills', 0) for info in team_infos)
        if total_kills > self.team_stats.team_kills:
            new_kills = total_kills - self.team_stats.team_kills
            reward += new_kills * self.config.enemy_elimination_bonus
            self.team_stats.team_kills = total_kills
        
        # Strategic positioning (near center, avoiding edges)
        center_bonus = 0.0
        for state in team_states:
            for entity in state.get('player_entities', []):
                x, y = entity.get('x', 0), entity.get('y', 0)
                # Assume map center is around (0, 0)
                distance_from_center = np.sqrt(x**2 + y**2)
                if distance_from_center < 500:  # Close to center
                    center_bonus += 0.1
        
        reward += center_bonus * self.config.strategic_positioning_bonus
        
        return reward
    
    def _calculate_efficiency_reward(self,
                                   team_states: List[Dict[str, Any]],
                                   team_actions: List[Dict[str, Any]]) -> float:
        """Calculate team efficiency rewards"""
        reward = 0.0
        
        # Resource efficiency - mass per team member
        if self.team_stats.active_members > 0:
            avg_mass_per_member = self.team_stats.total_team_mass / self.team_stats.active_members
            efficiency_score = min(avg_mass_per_member / 100.0, 1.0)
            reward += efficiency_score * self.config.synergy_bonus
        
        # Redundancy penalty - penalize if all members doing the same thing
        if len(team_actions) > 1:
            # Check for action diversity
            movement_vectors = []
            for action in team_actions:
                if 'movement' in action:
                    movement_vectors.append(action['movement'])
            
            if len(movement_vectors) > 1:
                # Calculate action diversity
                similarities = []
                for i in range(len(movement_vectors)):
                    for j in range(i + 1, len(movement_vectors)):
                        similarity = np.dot(movement_vectors[i], movement_vectors[j])
                        similarities.append(similarity)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    if avg_similarity > 0.8:  # Too similar actions
                        reward += self.config.redundancy_penalty
        
        return reward
    
    def _update_adaptive_multiplier(self):
        """Update adaptive reward multiplier based on performance"""
        if not self.config.adaptive_rewards or len(self.performance_history) < 5:
            return
        
        # Calculate recent performance trend
        recent_performance = self.performance_history[-5:]
        avg_performance = np.mean(recent_performance)
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Adjust multiplier based on performance
        if avg_performance < -10:  # Poor performance
            self.adaptive_multiplier = min(self.adaptive_multiplier * 1.1, 2.0)
        elif avg_performance > 20:  # Good performance
            self.adaptive_multiplier = max(self.adaptive_multiplier * 0.95, 0.5)
        
        # Adjust based on trend
        if trend < -1:  # Declining performance
            self.adaptive_multiplier = min(self.adaptive_multiplier * 1.05, 2.0)
        elif trend > 1:  # Improving performance
            self.adaptive_multiplier = max(self.adaptive_multiplier * 0.98, 0.5)
        
        logger.debug(f"Adaptive multiplier updated to {self.adaptive_multiplier:.3f}")
    
    def calculate_episode_reward(self) -> Tuple[float, Dict[str, float]]:
        """Calculate end-of-episode team bonus"""
        bonus_components = {}
        
        # Team survival bonus
        if self.team_stats.active_members > 0:
            survival_bonus = (self.team_stats.team_survival_time / 600.0) * 2.0  # 10 minutes max
            bonus_components["survival_bonus"] = survival_bonus
        
        # Coordination effectiveness bonus
        if self.team_stats.coordination_events > 0:
            coord_effectiveness = (self.team_stats.successful_coordinations / 
                                 max(self.team_stats.coordination_events, 1))
            bonus_components["coordination_effectiveness"] = coord_effectiveness * 1.0
        
        # Communication efficiency bonus
        if self.team_stats.communication_messages > 0:
            comm_efficiency = min(self.team_stats.communication_messages / 50.0, 1.0)
            bonus_components["communication_efficiency"] = comm_efficiency * 0.5
        
        # Team performance bonus
        mass_bonus = min(self.team_stats.max_team_mass / 1000.0, 2.0)
        bonus_components["mass_performance"] = mass_bonus
        
        total_bonus = sum(bonus_components.values())
        return total_bonus, bonus_components
    
    def get_team_statistics(self) -> Dict[str, Any]:
        """Get comprehensive team statistics"""
        return {
            "team_stats": self.team_stats.__dict__,
            "member_count": len(self.member_stats),
            "active_members": self.team_stats.active_members,
            "performance_history": self.performance_history[-10:],  # Last 10 episodes
            "adaptive_multiplier": self.adaptive_multiplier,
            "recent_communications": len(self.recent_communications),
            "coordination_success_rate": (
                self.team_stats.successful_coordinations / 
                max(self.team_stats.coordination_events, 1)
            ) if self.team_stats.coordination_events > 0 else 0.0,
        }
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get detailed reward breakdown"""
        total_rewards = self.team_stats.total_rewards
        if total_rewards == 0:
            return {}
        
        return {
            "individual_percentage": (self.team_stats.individual_rewards / total_rewards) * 100,
            "team_percentage": (self.team_stats.team_rewards / total_rewards) * 100,
            "coordination_percentage": (self.team_stats.coordination_rewards / total_rewards) * 100,
            "total_rewards": total_rewards,
        }
