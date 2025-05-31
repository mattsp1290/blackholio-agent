"""
Multi-agent coordination demo for Blackholio.

This example demonstrates the advanced multi-agent features
including team communication, coordination, and shared rewards.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any

from src.blackholio_agent.multi_agent import (
    MultiAgentBlackholioEnv,
    MultiAgentEnvConfig,
    TeamObservationConfig,
    CoordinationActionConfig,
    TeamRewardConfig,
    CommunicationProtocol
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTeamPolicy:
    """
    Simple policy for demonstration purposes.
    
    This policy demonstrates basic coordination behaviors:
    - Random movement with some coordination
    - Occasional communication
    - Formation awareness
    """
    
    def __init__(self, agent_id: str, team_size: int):
        self.agent_id = agent_id
        self.team_size = team_size
        self.step_count = 0
        
        # Extract agent index from ID
        self.agent_index = int(agent_id.split('_')[-1]) if '_' in agent_id else 0
        
    def predict(self, observation: np.ndarray, team_info: Dict[str, Any] = None) -> np.ndarray:
        """
        Predict action based on observation and team information.
        
        Args:
            observation: Team observation including teammates
            team_info: Additional team information
            
        Returns:
            Action vector for coordination action space
        """
        self.step_count += 1
        
        # Base movement (random with some bias)
        movement_x = np.random.uniform(-0.8, 0.8)
        movement_y = np.random.uniform(-0.8, 0.8)
        
        # Split occasionally
        split_action = 1.0 if np.random.random() < 0.02 else 0.0
        
        # Communication actions (sparse)
        comm_actions = np.zeros(8)  # Default comm_action_dim
        
        # Occasionally send position updates
        if self.step_count % 50 == self.agent_index * 10:  # Staggered communication
            comm_actions[0] = 1.0  # Position update
            comm_actions[3] = movement_x * 0.5  # Target position
            comm_actions[4] = movement_y * 0.5
            comm_actions[2] = 0.8  # High urgency
        
        # Coordination signals (very sparse)
        coord_signals = np.zeros(6)
        
        # Formation requests occasionally
        if self.step_count % 200 == 0 and self.agent_index == 0:  # Leader agent
            coord_signals[1] = 0.9  # Formation request
            coord_signals[2] = 0.7  # Priority
        
        # Formation control (sparse)
        formation_actions = np.zeros(8)  # 6 formation types + 2 params
        
        # Strategy commands (sparse)
        strategy_actions = np.zeros(7)  # 6 strategy types + 1 param
        
        # Combine all action components
        action = np.concatenate([
            [movement_x, movement_y, split_action],  # Base actions
            comm_actions,                            # Communication
            coord_signals,                           # Coordination
            formation_actions,                       # Formation
            strategy_actions                         # Strategy
        ])
        
        return action


async def run_multi_agent_demo():
    """Run a demonstration of multi-agent coordination"""
    logger.info("🚀 Starting Multi-Agent Coordination Demo")
    
    # Configuration
    team_size = 3
    max_steps = 500
    
    # Create multi-agent environment configuration
    config = MultiAgentEnvConfig(
        team_size=team_size,
        team_name="DemoTeam",
        enable_communication=True,
        enable_coordination=True,
        max_episode_steps=max_steps
    )
    
    # Configure communication protocol
    config.communication_protocol = CommunicationProtocol(
        max_messages_per_second=5.0,
        max_communication_range=800.0,
        range_affects_reliability=True
    )
    
    # Configure team observation space
    config.team_obs_config = TeamObservationConfig(
        max_teammates=team_size - 1,
        include_communication=True,
        include_team_stats=True,
        max_recent_messages=5
    )
    
    # Configure coordination actions
    config.coord_action_config = CoordinationActionConfig(
        enable_communication=True,
        enable_coordination_signals=True,
        enable_formation_control=True,
        enable_strategic_commands=True
    )
    
    # Configure team rewards
    config.team_reward_config = TeamRewardConfig(
        coordination_weight=0.4,
        communication_bonus=0.2,
        individual_weight=0.5,
        team_weight=0.5,
        adaptive_rewards=True
    )
    
    # Create environment
    env = MultiAgentBlackholioEnv(config)
    
    # Create simple policies for each agent
    policies = {}
    for i in range(team_size):
        agent_id = f"agent_{i}"
        policies[agent_id] = SimpleTeamPolicy(agent_id, team_size)
    
    try:
        logger.info(f"🎮 Initializing team of {team_size} agents")
        
        # Reset environment
        observations, infos = env.reset(seed=42)
        
        logger.info("📊 Environment Statistics:")
        env_stats = env.get_team_statistics()
        logger.info(f"  - Observation space: {env_stats['coordination_stats']['observation_space_dims']['total']} dimensions")
        logger.info(f"  - Action space: {env_stats['coordination_stats']['action_space_dims']['total_dimensions']} dimensions")
        logger.info(f"  - Communication enabled: {config.enable_communication}")
        logger.info(f"  - Coordination enabled: {config.enable_coordination}")
        
        # Run episode
        step = 0
        total_rewards = {f"agent_{i}": 0.0 for i in range(team_size)}
        
        logger.info("🏃 Starting episode...")
        
        while step < max_steps:
            # Get actions from policies
            actions = {}
            for agent_id, obs in observations.items():
                team_info = infos[agent_id] if agent_id in infos else {}
                actions[agent_id] = policies[agent_id].predict(obs, team_info)
            
            # Execute step
            observations, rewards, terminated, truncated, infos = env.step(actions)
            
            # Update total rewards
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
            
            # Log progress periodically
            if step % 100 == 0:
                logger.info(f"  Step {step}/{max_steps}")
                
                # Show team statistics
                team_stats = env.get_team_statistics()
                active_agents = sum(1 for state in env.team_states if state.get('player_entities'))
                total_communications = sum(
                    stats['messages_sent'] 
                    for stats in team_stats['communication_stats'].values()
                )
                
                logger.info(f"    Active agents: {active_agents}/{team_size}")
                logger.info(f"    Team communications: {total_communications}")
                logger.info(f"    Average reward: {np.mean(list(total_rewards.values())):.2f}")
                
                # Show reward breakdown for first agent
                if 'agent_0' in infos and 'reward_breakdown' in infos['agent_0']:
                    breakdown = infos['agent_0']['reward_breakdown']
                    if 'team_components' in breakdown:
                        team_comps = breakdown['team_components']
                        logger.info(f"    Team reward components: {list(team_comps.keys())}")
            
            # Check if episode is done
            if any(terminated.values()) or any(truncated.values()):
                logger.info(f"📈 Episode completed at step {step}")
                break
            
            step += 1
        
        # Final statistics
        logger.info("🎯 Final Results:")
        logger.info(f"  Episode length: {step} steps")
        
        for agent_id, total_reward in total_rewards.items():
            logger.info(f"  {agent_id}: {total_reward:.2f} total reward")
        
        # Team statistics
        final_stats = env.get_team_statistics()
        team_stats = final_stats['team_stats']['team_stats']
        
        logger.info("📊 Team Performance:")
        logger.info(f"  Survival time: {team_stats.get('team_survival_time', 0):.1f} seconds")
        logger.info(f"  Communication messages: {team_stats.get('communication_messages', 0)}")
        logger.info(f"  Coordination events: {team_stats.get('coordination_events', 0)}")
        logger.info(f"  Successful coordinations: {team_stats.get('successful_coordinations', 0)}")
        logger.info(f"  Max team mass: {team_stats.get('max_team_mass', 0):.1f}")
        
        # Communication statistics
        total_messages_sent = sum(
            stats['messages_sent'] 
            for stats in final_stats['communication_stats'].values()
        )
        total_messages_received = sum(
            stats['messages_received'] 
            for stats in final_stats['communication_stats'].values()
        )
        
        logger.info("📡 Communication Performance:")
        logger.info(f"  Total messages sent: {total_messages_sent}")
        logger.info(f"  Total messages received: {total_messages_received}")
        logger.info(f"  Communication efficiency: {total_messages_received / max(total_messages_sent, 1):.2f}")
        
        # Show coordination success rate
        coord_events = team_stats.get('coordination_events', 0)
        successful_coords = team_stats.get('successful_coordinations', 0)
        if coord_events > 0:
            success_rate = successful_coords / coord_events * 100
            logger.info(f"  Coordination success rate: {success_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}", exc_info=True)
    
    finally:
        # Cleanup
        env.close()
        logger.info("🏁 Multi-Agent Demo completed")


def run_demo():
    """Run the demo (sync wrapper)"""
    asyncio.run(run_multi_agent_demo())


if __name__ == "__main__":
    run_demo()
