"""
Multi-agent environment for coordinated Blackholio agents.

This module provides the main environment interface for training
and running multiple coordinated Blackholio agents.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
import gymnasium as gym
from gymnasium import spaces

from ..environment.blackholio_env import BlackholioEnv, BlackholioEnvConfig
from .agent_communication import AgentCommunication, CommunicationProtocol, CommunicationMessage
from .team_observation_space import TeamObservationSpace, TeamObservationConfig
from .coordination_action_space import CoordinationActionSpace, CoordinationActionConfig
from .team_reward_calculator import TeamRewardCalculator, TeamRewardConfig

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentEnvConfig:
    """Configuration for multi-agent environment"""
    # Team settings
    team_size: int = 4
    team_name: str = "TeamML"
    
    # Base environment configs
    base_env_configs: List[BlackholioEnvConfig] = None
    
    # Multi-agent component configs
    team_obs_config: TeamObservationConfig = None
    coord_action_config: CoordinationActionConfig = None
    team_reward_config: TeamRewardConfig = None
    communication_protocol: CommunicationProtocol = None
    
    # Coordination settings
    enable_communication: bool = True
    enable_coordination: bool = True
    shared_rewards: bool = True
    
    # Environment settings
    max_episode_steps: int = 10000
    step_interval: float = 0.05  # 20Hz
    
    # Server distribution
    distribute_across_servers: bool = False
    server_configs: List[Dict[str, str]] = None
    
    def __post_init__(self):
        # Create base environment configs for each agent
        if self.base_env_configs is None:
            self.base_env_configs = []
            for i in range(self.team_size):
                config = BlackholioEnvConfig()
                config.player_name = f"{self.team_name}_Agent_{i+1}"
                self.base_env_configs.append(config)
        
        # Create component configs if not provided
        if self.team_obs_config is None:
            self.team_obs_config = TeamObservationConfig()
            self.team_obs_config.max_teammates = self.team_size - 1
        
        if self.coord_action_config is None:
            self.coord_action_config = CoordinationActionConfig()
            self.coord_action_config.enable_communication = self.enable_communication
            self.coord_action_config.enable_coordination_signals = self.enable_coordination
        
        if self.team_reward_config is None:
            self.team_reward_config = TeamRewardConfig()
        
        if self.communication_protocol is None:
            self.communication_protocol = CommunicationProtocol()


class MultiAgentBlackholioEnv:
    """
    Multi-agent environment for coordinated Blackholio teams.
    
    Provides:
    - Multiple coordinated agents in the same game
    - Team communication and coordination
    - Shared observations including teammate information
    - Team-based reward functions
    - Coordination action spaces
    
    Example:
        ```python
        config = MultiAgentEnvConfig(team_size=4)
        env = MultiAgentBlackholioEnv(config)
        
        observations = env.reset()
        while not done:
            actions = {f"agent_{i}": agent_policies[i].predict(obs) 
                      for i, obs in enumerate(observations)}
            observations, rewards, dones, infos = env.step(actions)
        ```
    """
    
    def __init__(self, config: Union[MultiAgentEnvConfig, Dict[str, Any]] = None):
        """
        Initialize multi-agent environment.
        
        Args:
            config: Multi-agent environment configuration
        """
        # Handle config
        if config is None:
            self.config = MultiAgentEnvConfig()
        elif isinstance(config, dict):
            self.config = MultiAgentEnvConfig(**config)
        else:
            self.config = config
        
        # Create individual environments for each agent
        self.envs: List[BlackholioEnv] = []
        for i, base_config in enumerate(self.config.base_env_configs):
            env = BlackholioEnv(base_config)
            self.envs.append(env)
        
        # Create team observation space
        self.team_obs_space = TeamObservationSpace(self.config.team_obs_config)
        
        # Create coordination action space
        self.coord_action_space = CoordinationActionSpace(self.config.coord_action_config)
        
        # Create team reward calculator
        self.team_reward_calculator = TeamRewardCalculator(
            self.config.team_reward_config,
            team_id=self.config.team_name
        )
        
        # Create communication systems for each agent
        self.agent_comms: List[AgentCommunication] = []
        for i in range(self.config.team_size):
            agent_id = f"{self.config.team_name}_agent_{i}"
            comm = AgentCommunication(agent_id, self.config.communication_protocol)
            self.agent_comms.append(comm)
        
        # Connect all agents for communication
        self._setup_communication_network()
        
        # Define Gym spaces (for compatibility)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.team_obs_space.shape,
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.coord_action_space.shape,
            dtype=np.float32
        )
        
        # Episode state
        self.current_observations: List[np.ndarray] = []
        self.episode_steps = 0
        self.done = False
        
        # Team state tracking
        self.team_states: List[Dict[str, Any]] = []
        self.teammate_info: List[Dict[str, Any]] = []
        
        logger.info(f"MultiAgentBlackholioEnv initialized with {self.config.team_size} agents")
        logger.info(f"Team name: {self.config.team_name}")
        logger.info(f"Communication enabled: {self.config.enable_communication}")
        logger.info(f"Coordination enabled: {self.config.enable_coordination}")
    
    def _setup_communication_network(self):
        """Setup communication network between all agents"""
        # Register all agents with each other for communication
        for i, comm_a in enumerate(self.agent_comms):
            for j, comm_b in enumerate(self.agent_comms):
                if i != j:
                    comm_a.register_agent(comm_b)
        
        logger.debug(f"Communication network established for {len(self.agent_comms)} agents")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Reset the multi-agent environment.
        
        Args:
            seed: Random seed
            options: Additional reset options
            
        Returns:
            Tuple of (observations_dict, infos_dict)
        """
        if seed is not None:
            np.random.seed(seed)
        
        logger.info("Resetting multi-agent environment")
        
        # Reset episode state
        self.episode_steps = 0
        self.done = False
        
        # Reset team reward calculator
        self.team_reward_calculator.reset(team_size=self.config.team_size)
        
        # Reset all individual environments
        individual_observations = []
        individual_infos = []
        
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed + i if seed else None, options=options)
            individual_observations.append(obs)
            individual_infos.append(info)
        
        # Update team state information
        asyncio.run(self._update_team_states())
        
        # Generate team observations
        team_observations = self._generate_team_observations(individual_observations)
        
        # Prepare return values
        observations_dict = {f"agent_{i}": obs for i, obs in enumerate(team_observations)}
        infos_dict = {f"agent_{i}": info for i, info in enumerate(individual_infos)}
        
        # Add team-level info
        for i in range(self.config.team_size):
            infos_dict[f"agent_{i}"]["team_size"] = self.config.team_size
            infos_dict[f"agent_{i}"]["team_name"] = self.config.team_name
            infos_dict[f"agent_{i}"]["agent_id"] = f"agent_{i}"
        
        self.current_observations = team_observations
        
        logger.info("Multi-agent environment reset complete")
        return observations_dict, infos_dict
    
    async def async_reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Async version of reset"""
        return self.reset(seed, options)
    
    def step(self, actions: Dict[str, Union[np.ndarray, Dict[str, Any]]]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, Any]]]:
        """
        Execute actions for all agents.
        
        Args:
            actions: Dictionary mapping agent_id to action
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        # Convert actions to list format
        action_list = []
        for i in range(self.config.team_size):
            agent_key = f"agent_{i}"
            if agent_key in actions:
                action_list.append(actions[agent_key])
            else:
                # Default action if not provided
                action_list.append(self.coord_action_space.sample_random_action())
        
        return asyncio.run(self.async_step(action_list))
    
    async def async_step(self, actions: List[Union[np.ndarray, Dict[str, Any]]]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, Any]]]:
        """
        Async version of step.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        # Execute coordination actions for each agent
        coordination_results = []
        for i, action in enumerate(actions):
            result = await self.coord_action_space.execute_coordination_action(
                connection=self.envs[i].connection,
                action=action,
                agent_communication=self.agent_comms[i],
                teammates=self.teammate_info
            )
            coordination_results.append(result)
        
        # Update agent positions for communication range calculation
        await self._update_agent_positions()
        
        # Process communication messages
        all_communications = []
        for comm in self.agent_comms:
            messages = await comm.receive_messages()
            all_communications.extend(messages)
        
        # Execute base environment steps
        individual_results = []
        for i, env in enumerate(self.envs):
            base_action = coordination_results[i]["base"]["action"] if "base" in coordination_results[i] else actions[i]
            result = await env.async_step(base_action)
            individual_results.append(result)
        
        # Extract individual components
        individual_observations = [result[0] for result in individual_results]
        individual_rewards = [result[1] for result in individual_results]
        individual_terminated = [result[2] for result in individual_results]
        individual_truncated = [result[3] for result in individual_results]
        individual_infos = [result[4] for result in individual_results]
        
        # Update team states
        await self._update_team_states()
        
        # Calculate team rewards
        team_rewards, reward_breakdown = self.team_reward_calculator.calculate_team_step_reward(
            team_states=self.team_states,
            team_actions=[coordination_results[i] for i in range(self.config.team_size)],
            team_infos=individual_infos,
            communications=all_communications
        )
        
        # Generate team observations
        team_observations = self._generate_team_observations(individual_observations)
        
        # Check episode termination
        self.episode_steps += 1
        episode_terminated = any(individual_terminated)
        episode_truncated = (self.episode_steps >= self.config.max_episode_steps) or any(individual_truncated)
        self.done = episode_terminated or episode_truncated
        
        # Prepare return values
        observations_dict = {f"agent_{i}": obs for i, obs in enumerate(team_observations)}
        rewards_dict = {f"agent_{i}": reward for i, reward in enumerate(team_rewards)}
        terminated_dict = {f"agent_{i}": episode_terminated for i in range(self.config.team_size)}
        truncated_dict = {f"agent_{i}": episode_truncated for i in range(self.config.team_size)}
        
        # Enhanced info dictionaries
        infos_dict = {}
        for i in range(self.config.team_size):
            infos_dict[f"agent_{i}"] = {
                **individual_infos[i],
                "coordination_result": coordination_results[i],
                "individual_reward": individual_rewards[i],
                "team_reward": team_rewards[i],
                "reward_breakdown": reward_breakdown,
                "team_communications": len(all_communications),
                "agent_comm_stats": self.agent_comms[i].get_statistics(),
                "team_stats": self.team_reward_calculator.get_team_statistics(),
            }
        
        # Handle episode end
        if self.done:
            episode_bonus, bonus_components = self.team_reward_calculator.calculate_episode_reward()
            for i in range(self.config.team_size):
                rewards_dict[f"agent_{i}"] += episode_bonus / self.config.team_size
                infos_dict[f"agent_{i}"]["episode_bonus"] = episode_bonus / self.config.team_size
                infos_dict[f"agent_{i}"]["bonus_components"] = bonus_components
        
        self.current_observations = team_observations
        
        return observations_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict
    
    async def _update_team_states(self):
        """Update team state information for all agents"""
        self.team_states = []
        self.teammate_info = []
        
        for i, env in enumerate(self.envs):
            # Get current game state for each agent
            player_entities = env.connection.get_player_entities()
            other_entities = env.connection.get_other_entities()
            food_entities = env.connection.get_food_entities() if hasattr(env.connection, 'get_food_entities') else []
            
            state = {
                'player_entities': player_entities,
                'other_entities': other_entities,
                'food_entities': food_entities,
                'agent_id': f"agent_{i}",
                'timestamp': time.time()
            }
            self.team_states.append(state)
            
            # Create teammate info (for other agents)
            teammate_data = {
                'agent_id': f"agent_{i}",
                'entities': player_entities,
                'velocity_x': 0.0,  # Would need to track this
                'velocity_y': 0.0,
                'health': 1.0 if player_entities else 0.0,
                'current_intention': 0.0,  # Could be derived from recent actions
            }
            self.teammate_info.append(teammate_data)
    
    async def _update_agent_positions(self):
        """Update agent positions for communication range calculations"""
        for i, (env, comm) in enumerate(zip(self.envs, self.agent_comms)):
            player_entities = env.connection.get_player_entities()
            if player_entities:
                # Calculate center of mass for agent
                total_mass = sum(e.get('mass', 1.0) for e in player_entities)
                if total_mass > 0:
                    center_x = sum(e.get('x', 0.0) * e.get('mass', 1.0) for e in player_entities) / total_mass
                    center_y = sum(e.get('y', 0.0) * e.get('mass', 1.0) for e in player_entities) / total_mass
                    comm.update_position(center_x, center_y)
    
    def _generate_team_observations(self, individual_observations: List[np.ndarray]) -> List[np.ndarray]:
        """Generate team observations for each agent"""
        team_observations = []
        
        for i in range(self.config.team_size):
            # Get individual observation
            individual_obs = individual_observations[i]
            
            # Extract game state for this agent
            state = self.team_states[i] if i < len(self.team_states) else {
                'player_entities': [],
                'other_entities': [],
                'food_entities': []
            }
            
            # Get teammates (all other agents)
            teammates = [self.teammate_info[j] for j in range(self.config.team_size) if j != i]
            
            # Get recent communications for this agent
            recent_messages = []
            if self.config.enable_communication:
                # Get messages from the agent's communication system
                # Note: This would need to be called in an async context
                # For now, we'll use an empty list as a placeholder
                recent_messages = []
            
            # Generate team observation using team observation space
            # For now, we'll use a simplified version that extends the individual observation
            # In a full implementation, this would use the team_obs_space.process_team_state method
            
            # Create extended observation
            base_obs_size = len(individual_obs)
            team_obs_size = self.team_obs_space.shape[0]
            
            team_obs = np.zeros(team_obs_size, dtype=np.float32)
            team_obs[:base_obs_size] = individual_obs
            
            # Add teammate information (simplified)
            teammate_start_idx = base_obs_size
            for j, teammate in enumerate(teammates):
                if j >= self.config.team_obs_config.max_teammates:
                    break
                
                # Add basic teammate features
                teammate_idx = teammate_start_idx + j * self.config.team_obs_config.teammate_feature_dim
                if teammate_idx + 8 <= team_obs_size:  # Ensure we don't exceed bounds
                    # Position (relative to this agent)
                    teammate_entities = teammate.get('entities', [])
                    if teammate_entities and state['player_entities']:
                        # Calculate relative position
                        my_center = self._calculate_center_of_mass(state['player_entities'])
                        teammate_center = self._calculate_center_of_mass(teammate_entities)
                        
                        rel_x = teammate_center[0] - my_center[0]
                        rel_y = teammate_center[1] - my_center[1]
                        
                        # Normalize
                        rel_x = np.tanh(rel_x / 1000.0)
                        rel_y = np.tanh(rel_y / 1000.0)
                        
                        team_obs[teammate_idx:teammate_idx + 2] = [rel_x, rel_y]
                        
                        # Velocity, mass, etc.
                        teammate_mass = sum(e.get('mass', 0) for e in teammate_entities)
                        team_obs[teammate_idx + 4] = min(teammate_mass / 100.0, 10.0)  # Normalized mass
                        team_obs[teammate_idx + 5] = len(teammate_entities)  # Entity count
                        team_obs[teammate_idx + 6] = teammate.get('health', 1.0)
            
            team_observations.append(team_obs)
        
        return team_observations
    
    def _calculate_center_of_mass(self, entities: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate center of mass for entities"""
        if not entities:
            return (0.0, 0.0)
        
        total_mass = sum(e.get('mass', 1.0) for e in entities)
        if total_mass == 0:
            return (0.0, 0.0)
        
        center_x = sum(e.get('x', 0.0) * e.get('mass', 1.0) for e in entities) / total_mass
        center_y = sum(e.get('y', 0.0) * e.get('mass', 1.0) for e in entities) / total_mass
        
        return (center_x, center_y)
    
    def close(self):
        """Close all environments and cleanup resources"""
        logger.info("Closing multi-agent environment")
        
        # Close all individual environments
        for env in self.envs:
            env.close()
        
        # Clear communication systems
        self.agent_comms.clear()
        
        logger.info("Multi-agent environment closed")
    
    def render(self, mode: str = 'human') -> Optional[List[np.ndarray]]:
        """
        Render the multi-agent environment.
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
            
        Returns:
            List of rendered frames if mode is 'rgb_array'
        """
        if mode == 'human':
            print(f"\n=== Multi-Agent Blackholio Environment (Step {self.episode_steps}) ===")
            print(f"Team: {self.config.team_name}")
            print(f"Active agents: {sum(1 for state in self.team_states if state.get('player_entities'))}")
            
            # Show team statistics
            team_stats = self.team_reward_calculator.get_team_statistics()
            print(f"Team mass: {team_stats['team_stats']['total_team_mass']:.1f}")
            print(f"Coordination events: {team_stats['team_stats']['coordination_events']}")
            print(f"Communications: {team_stats['recent_communications']}")
            print(f"Survival time: {team_stats['team_stats']['team_survival_time']:.1f}s")
            
            # Show individual agent info
            for i, env in enumerate(self.envs):
                player_entities = env.connection.get_player_entities()
                if player_entities:
                    total_mass = sum(e.get('mass', 0) for e in player_entities)
                    print(f"  Agent {i}: {len(player_entities)} entities, {total_mass:.1f} mass")
                else:
                    print(f"  Agent {i}: Dead")
            
            print("=" * 60)
            
        elif mode == 'rgb_array':
            # Render each environment and return frames
            frames = []
            for env in self.envs:
                frame = env.render(mode='rgb_array')
                frames.append(frame)
            return frames
        
        return None
    
    def get_team_statistics(self) -> Dict[str, Any]:
        """Get comprehensive team statistics"""
        team_stats = self.team_reward_calculator.get_team_statistics()
        
        # Add communication statistics
        comm_stats = {}
        for i, comm in enumerate(self.agent_comms):
            comm_stats[f"agent_{i}"] = comm.get_statistics()
        
        # Add coordination statistics
        coord_stats = {
            "action_space_dims": self.coord_action_space.get_statistics(),
            "observation_space_dims": {
                "total": self.team_obs_space.shape[0],
                "components": self.team_obs_space.component_dims
            }
        }
        
        return {
            "team_stats": team_stats,
            "communication_stats": comm_stats,
            "coordination_stats": coord_stats,
            "episode_steps": self.episode_steps,
            "done": self.done,
        }
    
    def sample_actions(self) -> Dict[str, np.ndarray]:
        """Sample random actions for all agents"""
        actions = {}
        for i in range(self.config.team_size):
            actions[f"agent_{i}"] = self.coord_action_space.sample_random_action()
        return actions
    
    def get_action_masks(self) -> Dict[str, np.ndarray]:
        """Get action masks for all agents"""
        masks = {}
        for i in range(self.config.team_size):
            current_state = self.team_states[i] if i < len(self.team_states) else {}
            teammates = [self.teammate_info[j] for j in range(self.config.team_size) if j != i]
            mask = self.coord_action_space.get_action_mask(current_state, teammates)
            masks[f"agent_{i}"] = mask
        return masks
