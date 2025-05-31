"""
Coordination action space for multi-agent teams.

This module extends the single-agent action space to include
communication actions and team coordination signals.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging

from ..environment.action_space import ActionSpace, ActionConfig
from .agent_communication import AgentCommunication, MessageType

logger = logging.getLogger(__name__)


@dataclass
class CoordinationActionConfig:
    """Configuration for coordination action space"""
    # Base action config
    base_config: ActionConfig = None
    
    # Communication actions
    enable_communication: bool = True
    max_comm_actions_per_step: int = 3
    comm_action_dim: int = 8  # Type, target, urgency, position, etc.
    
    # Coordination signals
    enable_coordination_signals: bool = True
    coordination_signal_dim: int = 6  # Formation, strategy, role, etc.
    
    # Formation control
    enable_formation_control: bool = True
    formation_types: List[str] = None
    
    # Strategic commands
    enable_strategic_commands: bool = True
    strategy_types: List[str] = None
    
    def __post_init__(self):
        if self.base_config is None:
            self.base_config = ActionConfig()
        
        if self.formation_types is None:
            self.formation_types = [
                "spread",      # Spread out formation
                "compact",     # Tight formation
                "line",        # Linear formation
                "surround",    # Surround target
                "pincer",      # Pincer movement
                "retreat",     # Retreat formation
            ]
        
        if self.strategy_types is None:
            self.strategy_types = [
                "passive",     # Passive food collection
                "aggressive",  # Aggressive hunting
                "defensive",   # Defensive play
                "opportunistic", # Opportunistic strikes
                "support",     # Support teammates
                "solo",        # Individual play
            ]


class CoordinationActionSpace:
    """
    Extended action space for multi-agent coordination.
    
    Combines individual agent actions with:
    - Communication actions (messages to teammates)
    - Coordination signals (formation, strategy)
    - Team-level commands
    """
    
    def __init__(self, config: CoordinationActionConfig = None):
        """
        Initialize coordination action space.
        
        Args:
            config: Coordination action configuration
        """
        self.config = config or CoordinationActionConfig()
        
        # Create base action space
        self.base_action_space = ActionSpace(self.config.base_config)
        
        # Calculate total action dimensions
        self._calculate_dimensions()
        
        # Action components for parsing
        self._setup_action_components()
        
        logger.info(f"CoordinationActionSpace initialized with {self.shape[0]} dimensions")
        logger.info(f"Base actions: {self.base_action_space.shape[0]}")
        logger.info(f"Coordination extensions: {self.shape[0] - self.base_action_space.shape[0]}")
    
    def _calculate_dimensions(self):
        """Calculate total action space dimensions"""
        base_dim = self.base_action_space.shape[0]  # movement (2) + split (1) = 3
        
        # Communication actions
        comm_dim = 0
        if self.config.enable_communication:
            comm_dim = self.config.comm_action_dim
        
        # Coordination signals
        coord_dim = 0
        if self.config.enable_coordination_signals:
            coord_dim = self.config.coordination_signal_dim
        
        # Formation control
        formation_dim = 0
        if self.config.enable_formation_control:
            formation_dim = len(self.config.formation_types) + 2  # formation type + intensity + timeout
        
        # Strategic commands
        strategy_dim = 0
        if self.config.enable_strategic_commands:
            strategy_dim = len(self.config.strategy_types) + 1  # strategy type + priority
        
        total_dim = base_dim + comm_dim + coord_dim + formation_dim + strategy_dim
        self.shape = (total_dim,)
        
        # Store component dimensions
        self.component_dims = {
            "base": base_dim,
            "communication": comm_dim,
            "coordination": coord_dim,
            "formation": formation_dim,
            "strategy": strategy_dim,
            "total": total_dim
        }
    
    def _setup_action_components(self):
        """Setup action component indices for parsing"""
        start_idx = 0
        
        # Base actions (movement + split)
        self.base_slice = slice(start_idx, start_idx + self.component_dims["base"])
        start_idx += self.component_dims["base"]
        
        # Communication actions
        if self.component_dims["communication"] > 0:
            self.comm_slice = slice(start_idx, start_idx + self.component_dims["communication"])
            start_idx += self.component_dims["communication"]
        else:
            self.comm_slice = slice(0, 0)
        
        # Coordination signals
        if self.component_dims["coordination"] > 0:
            self.coord_slice = slice(start_idx, start_idx + self.component_dims["coordination"])
            start_idx += self.component_dims["coordination"]
        else:
            self.coord_slice = slice(0, 0)
        
        # Formation control
        if self.component_dims["formation"] > 0:
            self.formation_slice = slice(start_idx, start_idx + self.component_dims["formation"])
            start_idx += self.component_dims["formation"]
        else:
            self.formation_slice = slice(0, 0)
        
        # Strategy commands
        if self.component_dims["strategy"] > 0:
            self.strategy_slice = slice(start_idx, start_idx + self.component_dims["strategy"])
        else:
            self.strategy_slice = slice(0, 0)
    
    async def execute_coordination_action(self,
                                        connection,
                                        action: Union[np.ndarray, Dict[str, Any]],
                                        agent_communication: AgentCommunication,
                                        teammates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a coordination action including movement and team coordination.
        
        Args:
            connection: Game connection
            action: Action vector or dictionary
            agent_communication: Communication system
            teammates: Current teammate information
            
        Returns:
            Action execution results
        """
        # Parse action components
        if isinstance(action, dict):
            action_components = action
        else:
            action_components = self._parse_action_vector(action)
        
        results = {}
        
        # Execute base movement action
        base_action = action_components.get("base", np.zeros(self.component_dims["base"]))
        base_result = await self.base_action_space.execute_action(connection, base_action)
        results["base"] = base_result
        
        # Execute communication actions
        if self.config.enable_communication and "communication" in action_components:
            comm_result = await self._execute_communication_action(
                action_components["communication"], 
                agent_communication,
                teammates
            )
            results["communication"] = comm_result
        
        # Execute coordination signals
        if self.config.enable_coordination_signals and "coordination" in action_components:
            coord_result = await self._execute_coordination_signals(
                action_components["coordination"],
                agent_communication,
                teammates
            )
            results["coordination"] = coord_result
        
        # Execute formation control
        if self.config.enable_formation_control and "formation" in action_components:
            formation_result = await self._execute_formation_control(
                action_components["formation"],
                agent_communication,
                teammates
            )
            results["formation"] = formation_result
        
        # Execute strategic commands
        if self.config.enable_strategic_commands and "strategy" in action_components:
            strategy_result = await self._execute_strategic_commands(
                action_components["strategy"],
                agent_communication,
                teammates
            )
            results["strategy"] = strategy_result
        
        return results
    
    def _parse_action_vector(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Parse action vector into components"""
        components = {}
        
        # Base actions
        components["base"] = action[self.base_slice]
        
        # Communication actions
        if self.component_dims["communication"] > 0:
            components["communication"] = action[self.comm_slice]
        
        # Coordination signals
        if self.component_dims["coordination"] > 0:
            components["coordination"] = action[self.coord_slice]
        
        # Formation control
        if self.component_dims["formation"] > 0:
            components["formation"] = action[self.formation_slice]
        
        # Strategy commands
        if self.component_dims["strategy"] > 0:
            components["strategy"] = action[self.strategy_slice]
        
        return components
    
    async def _execute_communication_action(self,
                                          comm_action: np.ndarray,
                                          agent_comm: AgentCommunication,
                                          teammates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute communication actions"""
        results = {"messages_sent": 0, "message_types": []}
        
        # Parse communication action
        # Format: [message_type, target_agent, urgency, pos_x, pos_y, mass_info, action_type, extra]
        if len(comm_action) < self.config.comm_action_dim:
            return results
        
        # Determine message type
        msg_type_idx = int(np.argmax(comm_action[:len(MessageType)]))
        message_types = list(MessageType)
        if msg_type_idx < len(message_types):
            message_type = message_types[msg_type_idx]
        else:
            return results  # Invalid message type
        
        # Extract message parameters
        target_agent_idx = int(comm_action[len(MessageType)] * len(teammates))
        urgency = max(1, min(5, int(comm_action[len(MessageType) + 1] * 5 + 1)))
        pos_x = comm_action[len(MessageType) + 2] * 1000.0  # Denormalize
        pos_y = comm_action[len(MessageType) + 3] * 1000.0
        mass_info = comm_action[len(MessageType) + 4] * 1000.0
        action_type = comm_action[len(MessageType) + 5]
        
        # Only send message if action is above threshold
        if np.max(comm_action) < 0.5:  # Threshold for sending message
            return results
        
        # Build message content
        content = {
            "x": pos_x,
            "y": pos_y,
            "mass": mass_info,
        }
        
        # Add message-specific content
        if message_type == MessageType.SPLIT_COORDINATION:
            content["target_x"] = pos_x
            content["target_y"] = pos_y
            content["action"] = "request" if action_type > 0.5 else "confirm"
        elif message_type == MessageType.DANGER_ALERT:
            content["threat_level"] = urgency
        elif message_type == MessageType.TARGET_DESIGNATION:
            content["target_mass"] = mass_info
        
        # Determine recipient
        recipient_id = None
        if target_agent_idx < len(teammates) and comm_action[len(MessageType)] > 0.5:
            recipient_id = teammates[target_agent_idx].get("agent_id")
        
        # Send message
        try:
            success = await agent_comm.send_message(
                message_type=message_type,
                content=content,
                recipient_id=recipient_id,
                priority=urgency
            )
            
            if success:
                results["messages_sent"] += 1
                results["message_types"].append(message_type.value)
        except Exception as e:
            logger.warning(f"Failed to send message: {e}")
        
        return results
    
    async def _execute_coordination_signals(self,
                                          coord_action: np.ndarray,
                                          agent_comm: AgentCommunication,
                                          teammates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute coordination signals"""
        results = {"signals_sent": 0}
        
        if len(coord_action) < self.config.coordination_signal_dim:
            return results
        
        # Parse coordination signals
        # Format: [role_request, formation_request, priority_signal, position_signal, timing_signal, coordination_type]
        
        role_request = coord_action[0]
        formation_request = coord_action[1]
        priority_signal = coord_action[2]
        position_signal_x = coord_action[3]
        position_signal_y = coord_action[4]
        coordination_type = coord_action[5]
        
        # Send coordination signals if above threshold
        if role_request > 0.7:
            # Signal role change request
            await agent_comm.send_message(
                MessageType.STATUS_UPDATE,
                {"role_request": role_request, "type": "role_change"},
                priority=3
            )
            results["signals_sent"] += 1
        
        if formation_request > 0.7:
            # Signal formation change request
            await agent_comm.send_message(
                MessageType.AREA_CLAIM,
                {
                    "formation_request": formation_request,
                    "target_x": position_signal_x * 1000.0,
                    "target_y": position_signal_y * 1000.0
                },
                priority=int(priority_signal * 3 + 2)
            )
            results["signals_sent"] += 1
        
        return results
    
    async def _execute_formation_control(self,
                                       formation_action: np.ndarray,
                                       agent_comm: AgentCommunication,
                                       teammates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute formation control commands"""
        results = {"formation_commands": 0}
        
        if len(formation_action) < len(self.config.formation_types) + 2:
            return results
        
        # Parse formation action
        formation_probs = formation_action[:len(self.config.formation_types)]
        formation_intensity = formation_action[len(self.config.formation_types)]
        formation_timeout = formation_action[len(self.config.formation_types) + 1]
        
        # Determine desired formation
        formation_idx = np.argmax(formation_probs)
        formation_strength = np.max(formation_probs)
        
        # Only execute if formation signal is strong enough
        if formation_strength > 0.6:
            formation_type = self.config.formation_types[formation_idx]
            
            # Send formation command
            await agent_comm.send_message(
                MessageType.ATTACK_COORDINATION,
                {
                    "formation_type": formation_type,
                    "intensity": formation_intensity,
                    "timeout": formation_timeout * 10.0,  # Scale to seconds
                    "action": "formation_request"
                },
                priority=4
            )
            results["formation_commands"] += 1
            results["formation_type"] = formation_type
        
        return results
    
    async def _execute_strategic_commands(self,
                                        strategy_action: np.ndarray,
                                        agent_comm: AgentCommunication,
                                        teammates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute strategic commands"""
        results = {"strategy_commands": 0}
        
        if len(strategy_action) < len(self.config.strategy_types) + 1:
            return results
        
        # Parse strategy action
        strategy_probs = strategy_action[:len(self.config.strategy_types)]
        strategy_priority = strategy_action[len(self.config.strategy_types)]
        
        # Determine desired strategy
        strategy_idx = np.argmax(strategy_probs)
        strategy_strength = np.max(strategy_probs)
        
        # Only execute if strategy signal is strong enough
        if strategy_strength > 0.7:
            strategy_type = self.config.strategy_types[strategy_idx]
            
            # Send strategy command
            await agent_comm.send_message(
                MessageType.STATUS_UPDATE,
                {
                    "strategy_type": strategy_type,
                    "priority": strategy_priority,
                    "action": "strategy_change"
                },
                priority=int(strategy_priority * 3 + 2)
            )
            results["strategy_commands"] += 1
            results["strategy_type"] = strategy_type
        
        return results
    
    def get_action_mask(self, 
                       current_state: Dict[str, Any],
                       teammates: List[Dict[str, Any]]) -> np.ndarray:
        """
        Get action mask for invalid actions.
        
        Args:
            current_state: Current game state
            teammates: Current teammates
            
        Returns:
            Binary mask (1 = valid, 0 = invalid)
        """
        mask = np.ones(self.shape[0])
        
        # Base action mask (always valid for movement)
        # Communication mask - invalid if no teammates
        if self.component_dims["communication"] > 0 and len(teammates) == 0:
            mask[self.comm_slice] = 0.0
        
        # Formation mask - invalid if team too small
        if self.component_dims["formation"] > 0 and len(teammates) < 2:
            mask[self.formation_slice] = 0.0
        
        return mask
    
    def sample_random_action(self) -> np.ndarray:
        """Sample a random valid action"""
        action = np.random.uniform(-1.0, 1.0, self.shape[0])
        
        # Ensure base actions are in valid range
        action[self.base_slice] = np.random.uniform(-1.0, 1.0, self.component_dims["base"])
        
        # Communication actions - sparse (mostly inactive)
        if self.component_dims["communication"] > 0:
            comm_action = np.random.uniform(0.0, 1.0, self.component_dims["communication"])
            # Make communication sparse
            if np.random.random() < 0.9:  # 90% chance of no communication
                comm_action = np.zeros(self.component_dims["communication"])
            action[self.comm_slice] = comm_action
        
        # Other components - sparse activation
        for component, slice_obj in [
            ("coordination", self.coord_slice),
            ("formation", self.formation_slice),
            ("strategy", self.strategy_slice)
        ]:
            if self.component_dims[component] > 0:
                if np.random.random() < 0.8:  # 80% chance of no action
                    action[slice_obj] = np.zeros(self.component_dims[component])
                else:
                    action[slice_obj] = np.random.uniform(0.0, 1.0, self.component_dims[component])
        
        return action
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get action space statistics"""
        return {
            "total_dimensions": self.shape[0],
            "base_dimensions": self.component_dims["base"],
            "communication_dimensions": self.component_dims["communication"],
            "coordination_dimensions": self.component_dims["coordination"],
            "formation_dimensions": self.component_dims["formation"],
            "strategy_dimensions": self.component_dims["strategy"],
            "formation_types": len(self.config.formation_types),
            "strategy_types": len(self.config.strategy_types),
        }
    
    def get_action_breakdown(self, action: np.ndarray) -> Dict[str, Any]:
        """Break down action into interpretable components"""
        components = self._parse_action_vector(action)
        breakdown = {}
        
        # Base actions
        if "base" in components:
            base = components["base"]
            breakdown["movement"] = {
                "x": float(base[0]),
                "y": float(base[1]),
                "split": float(base[2]) if len(base) > 2 else 0.0
            }
        
        # Communication
        if "communication" in components and len(components["communication"]) > 0:
            comm = components["communication"]
            breakdown["communication"] = {
                "active": float(np.max(comm) > 0.5),
                "urgency": float(comm[len(MessageType) + 1] if len(comm) > len(MessageType) + 1 else 0),
                "strongest_signal": float(np.max(comm))
            }
        
        # Formation
        if "formation" in components and len(components["formation"]) > 0:
            formation = components["formation"]
            if len(formation) >= len(self.config.formation_types):
                formation_idx = np.argmax(formation[:len(self.config.formation_types)])
                breakdown["formation"] = {
                    "type": self.config.formation_types[formation_idx],
                    "strength": float(np.max(formation[:len(self.config.formation_types)])),
                    "active": float(np.max(formation[:len(self.config.formation_types)]) > 0.6)
                }
        
        # Strategy
        if "strategy" in components and len(components["strategy"]) > 0:
            strategy = components["strategy"]
            if len(strategy) >= len(self.config.strategy_types):
                strategy_idx = np.argmax(strategy[:len(self.config.strategy_types)])
                breakdown["strategy"] = {
                    "type": self.config.strategy_types[strategy_idx],
                    "strength": float(np.max(strategy[:len(self.config.strategy_types)])),
                    "active": float(np.max(strategy[:len(self.config.strategy_types)]) > 0.7)
                }
        
        return breakdown
