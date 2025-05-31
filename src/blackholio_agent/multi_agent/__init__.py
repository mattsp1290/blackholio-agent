"""
Multi-agent coordination system for Blackholio agents.

This package provides advanced multi-agent capabilities including:
- Team coordination and communication
- Shared observation and action spaces
- Team-based reward calculations
- Advanced strategic behaviors
"""

from .multi_agent_env import MultiAgentBlackholioEnv, MultiAgentEnvConfig
from .agent_communication import AgentCommunication, CommunicationProtocol, MessageType
from .team_observation_space import TeamObservationSpace, TeamObservationConfig
from .coordination_action_space import CoordinationActionSpace, CoordinationActionConfig
from .team_reward_calculator import TeamRewardCalculator, TeamRewardConfig

__all__ = [
    "MultiAgentBlackholioEnv",
    "MultiAgentEnvConfig", 
    "AgentCommunication",
    "CommunicationProtocol",
    "MessageType",
    "TeamObservationSpace",
    "TeamObservationConfig",
    "CoordinationActionSpace",
    "CoordinationActionConfig",
    "TeamRewardCalculator",
    "TeamRewardConfig",
]
