"""
Agent communication system for multi-agent coordination.

This module provides communication protocols and message passing
between coordinated Blackholio agents.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages agents can send to each other"""
    # Positional information
    POSITION_UPDATE = "position_update"
    MOVEMENT_INTENTION = "movement_intention"
    TARGET_DESIGNATION = "target_designation"
    
    # Tactical coordination
    SPLIT_REQUEST = "split_request"
    SPLIT_COORDINATION = "split_coordination"
    MERGE_REQUEST = "merge_request"
    
    # Threat and opportunity
    DANGER_ALERT = "danger_alert"
    OPPORTUNITY_SIGNAL = "opportunity_signal"
    ENEMY_SPOTTED = "enemy_spotted"
    
    # Strategic coordination
    AREA_CLAIM = "area_claim"
    RETREAT_SIGNAL = "retreat_signal"
    ATTACK_COORDINATION = "attack_coordination"
    
    # Resource coordination
    FOOD_LOCATION = "food_location"
    MASS_TRANSFER_REQUEST = "mass_transfer_request"
    
    # General
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"


@dataclass
class CommunicationMessage:
    """A single communication message between agents"""
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    priority: int = 1  # 1-5, higher is more urgent
    expires_at: Optional[float] = None
    
    def __post_init__(self):
        if self.expires_at is None:
            # Default expiration: 2 seconds for most messages
            ttl_map = {
                MessageType.HEARTBEAT: 1.0,
                MessageType.POSITION_UPDATE: 0.5,
                MessageType.MOVEMENT_INTENTION: 1.0,
                MessageType.DANGER_ALERT: 3.0,
                MessageType.SPLIT_COORDINATION: 2.0,
            }
            ttl = ttl_map.get(self.message_type, 2.0)
            self.expires_at = self.timestamp + ttl
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "expires_at": self.expires_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationMessage':
        """Create message from dictionary"""
        return cls(
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=data["timestamp"],
            priority=data["priority"],
            expires_at=data["expires_at"],
        )


@dataclass
class CommunicationProtocol:
    """Protocol settings for agent communication"""
    # Bandwidth limitations
    max_messages_per_second: float = 10.0
    max_message_size_bytes: int = 512
    max_recipients_per_message: int = 8
    
    # Reliability settings
    enable_acknowledgments: bool = True
    retransmit_timeout: float = 0.5
    max_retransmits: int = 2
    
    # Priority handling
    priority_queue_size: int = 100
    drop_low_priority_when_full: bool = True
    
    # Communication range (game units)
    max_communication_range: float = 500.0
    range_affects_reliability: bool = True
    
    # Compression
    compress_messages: bool = True
    compression_threshold: int = 128


class MessageQueue:
    """Priority queue for managing messages"""
    
    def __init__(self, max_size: int = 100, drop_low_priority: bool = True):
        self.max_size = max_size
        self.drop_low_priority = drop_low_priority
        self._messages: List[CommunicationMessage] = []
        self._lock = asyncio.Lock()
    
    async def put(self, message: CommunicationMessage) -> bool:
        """Add message to queue. Returns True if successful."""
        async with self._lock:
            # Remove expired messages
            self._cleanup_expired()
            
            # Check if queue is full
            if len(self._messages) >= self.max_size:
                if self.drop_low_priority:
                    # Find lowest priority message
                    min_priority = min(m.priority for m in self._messages)
                    if message.priority > min_priority:
                        # Remove lowest priority message
                        for i, m in enumerate(self._messages):
                            if m.priority == min_priority:
                                del self._messages[i]
                                break
                    else:
                        return False  # Message dropped
                else:
                    return False  # Queue full
            
            # Insert message in priority order
            inserted = False
            for i, existing in enumerate(self._messages):
                if message.priority > existing.priority:
                    self._messages.insert(i, message)
                    inserted = True
                    break
            
            if not inserted:
                self._messages.append(message)
            
            return True
    
    async def get(self) -> Optional[CommunicationMessage]:
        """Get highest priority message"""
        async with self._lock:
            self._cleanup_expired()
            if self._messages:
                return self._messages.pop(0)
            return None
    
    async def get_all(self) -> List[CommunicationMessage]:
        """Get all messages and clear queue"""
        async with self._lock:
            self._cleanup_expired()
            messages = self._messages.copy()
            self._messages.clear()
            return messages
    
    def _cleanup_expired(self):
        """Remove expired messages"""
        current_time = time.time()
        self._messages = [m for m in self._messages if not m.is_expired()]
    
    def size(self) -> int:
        """Get current queue size"""
        return len(self._messages)


class AgentCommunication:
    """
    Communication system for coordinating multiple agents.
    
    Provides:
    - Message passing between agents
    - Priority queuing
    - Bandwidth limitations
    - Range-based communication
    - Message reliability
    """
    
    def __init__(self, 
                 agent_id: str,
                 protocol: CommunicationProtocol = None):
        """
        Initialize communication system for an agent.
        
        Args:
            agent_id: Unique identifier for this agent
            protocol: Communication protocol settings
        """
        self.agent_id = agent_id
        self.protocol = protocol or CommunicationProtocol()
        
        # Message queues
        self.incoming_queue = MessageQueue(
            max_size=self.protocol.priority_queue_size,
            drop_low_priority=self.protocol.drop_low_priority_when_full
        )
        self.outgoing_queue = MessageQueue(
            max_size=self.protocol.priority_queue_size,
            drop_low_priority=self.protocol.drop_low_priority_when_full
        )
        
        # Rate limiting
        self.message_timestamps: List[float] = []
        self.bytes_sent_recent: List[Tuple[float, int]] = []
        
        # Network simulation
        self.other_agents: Dict[str, 'AgentCommunication'] = {}
        self.agent_positions: Dict[str, Tuple[float, float]] = {}
        self.my_position: Tuple[float, float] = (0.0, 0.0)
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_dropped": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }
        
        logger.info(f"AgentCommunication initialized for agent {agent_id}")
    
    def register_agent(self, other_agent: 'AgentCommunication'):
        """Register another agent for direct communication"""
        self.other_agents[other_agent.agent_id] = other_agent
        logger.debug(f"Agent {self.agent_id} registered agent {other_agent.agent_id}")
    
    def update_position(self, x: float, y: float):
        """Update this agent's position for range calculations"""
        self.my_position = (x, y)
        
        # Broadcast position to other agents
        asyncio.create_task(self.send_message(
            message_type=MessageType.POSITION_UPDATE,
            content={"x": x, "y": y},
            recipient_id=None,  # Broadcast
            priority=2
        ))
    
    def update_agent_position(self, agent_id: str, x: float, y: float):
        """Update another agent's position"""
        self.agent_positions[agent_id] = (x, y)
    
    def get_communication_range(self, recipient_id: str) -> float:
        """Calculate communication range to another agent"""
        if recipient_id not in self.agent_positions:
            return float('inf')  # Unknown position, assume in range
        
        my_x, my_y = self.my_position
        other_x, other_y = self.agent_positions[recipient_id]
        
        distance = np.sqrt((my_x - other_x)**2 + (my_y - other_y)**2)
        return distance
    
    def can_communicate_with(self, recipient_id: str) -> bool:
        """Check if we can communicate with another agent"""
        if recipient_id not in self.other_agents:
            return False
        
        distance = self.get_communication_range(recipient_id)
        return distance <= self.protocol.max_communication_range
    
    async def send_message(self,
                          message_type: MessageType,
                          content: Dict[str, Any],
                          recipient_id: Optional[str] = None,
                          priority: int = 1) -> bool:
        """
        Send a message to another agent or broadcast.
        
        Args:
            message_type: Type of message
            content: Message content
            recipient_id: Target agent ID (None for broadcast)
            priority: Message priority (1-5)
            
        Returns:
            True if message was sent successfully
        """
        # Rate limiting check
        if not self._check_rate_limit():
            self.stats["messages_dropped"] += 1
            logger.debug(f"Message dropped due to rate limiting: {message_type}")
            return False
        
        # Create message
        message = CommunicationMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            priority=priority
        )
        
        # Check message size
        message_size = len(json.dumps(message.to_dict()).encode())
        if message_size > self.protocol.max_message_size_bytes:
            self.stats["messages_dropped"] += 1
            logger.warning(f"Message dropped due to size: {message_size} bytes")
            return False
        
        # Add to outgoing queue
        if not await self.outgoing_queue.put(message):
            self.stats["messages_dropped"] += 1
            logger.debug(f"Message dropped due to full outgoing queue")
            return False
        
        # Process outgoing messages
        await self._process_outgoing_messages()
        
        return True
    
    async def _process_outgoing_messages(self):
        """Process messages in outgoing queue"""
        messages = await self.outgoing_queue.get_all()
        
        for message in messages:
            if message.recipient_id is None:
                # Broadcast message
                await self._broadcast_message(message)
            else:
                # Direct message
                await self._send_direct_message(message)
    
    async def _broadcast_message(self, message: CommunicationMessage):
        """Broadcast message to all agents in range"""
        sent_count = 0
        max_recipients = self.protocol.max_recipients_per_message
        
        for agent_id, agent_comm in self.other_agents.items():
            if sent_count >= max_recipients:
                break
            
            if self.can_communicate_with(agent_id):
                await self._deliver_message(agent_comm, message)
                sent_count += 1
        
        self.stats["messages_sent"] += 1
        self.stats["bytes_sent"] += len(json.dumps(message.to_dict()).encode())
    
    async def _send_direct_message(self, message: CommunicationMessage):
        """Send message to specific agent"""
        recipient_id = message.recipient_id
        
        if recipient_id not in self.other_agents:
            logger.warning(f"Unknown recipient: {recipient_id}")
            return
        
        if not self.can_communicate_with(recipient_id):
            logger.debug(f"Agent {recipient_id} out of communication range")
            return
        
        recipient_comm = self.other_agents[recipient_id]
        await self._deliver_message(recipient_comm, message)
        
        self.stats["messages_sent"] += 1
        self.stats["bytes_sent"] += len(json.dumps(message.to_dict()).encode())
    
    async def _deliver_message(self, recipient_comm: 'AgentCommunication', message: CommunicationMessage):
        """Deliver message to recipient's incoming queue"""
        # Simulate network unreliability based on distance
        if self.protocol.range_affects_reliability:
            distance = self.get_communication_range(recipient_comm.agent_id)
            max_range = self.protocol.max_communication_range
            reliability = max(0.5, 1.0 - (distance / max_range) * 0.5)
            
            if np.random.random() > reliability:
                logger.debug(f"Message lost due to distance: {distance}")
                return
        
        # Deliver message
        if await recipient_comm.incoming_queue.put(message):
            recipient_comm.stats["messages_received"] += 1
            recipient_comm.stats["bytes_received"] += len(json.dumps(message.to_dict()).encode())
        else:
            logger.debug(f"Recipient queue full, message dropped")
    
    async def receive_messages(self) -> List[CommunicationMessage]:
        """Get all received messages"""
        return await self.incoming_queue.get_all()
    
    async def receive_messages_by_type(self, message_type: MessageType) -> List[CommunicationMessage]:
        """Get all received messages of a specific type"""
        all_messages = await self.incoming_queue.get_all()
        return [msg for msg in all_messages if msg.message_type == message_type]
    
    def _check_rate_limit(self) -> bool:
        """Check if we can send a message based on rate limiting"""
        current_time = time.time()
        
        # Clean old timestamps
        self.message_timestamps = [
            ts for ts in self.message_timestamps 
            if current_time - ts < 1.0
        ]
        
        # Check rate limit
        if len(self.message_timestamps) >= self.protocol.max_messages_per_second:
            return False
        
        # Add current timestamp
        self.message_timestamps.append(current_time)
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            **self.stats,
            "incoming_queue_size": self.incoming_queue.size(),
            "outgoing_queue_size": self.outgoing_queue.size(),
            "connected_agents": len(self.other_agents),
            "agents_in_range": sum(1 for agent_id in self.other_agents.keys() 
                                 if self.can_communicate_with(agent_id)),
        }
    
    # Convenience methods for common message types
    
    async def broadcast_position(self, x: float, y: float):
        """Broadcast current position"""
        await self.send_message(
            MessageType.POSITION_UPDATE,
            {"x": x, "y": y},
            priority=2
        )
    
    async def signal_danger(self, danger_x: float, danger_y: float, threat_level: int = 3):
        """Signal danger at a location"""
        await self.send_message(
            MessageType.DANGER_ALERT,
            {"x": danger_x, "y": danger_y, "threat_level": threat_level},
            priority=5
        )
    
    async def request_split_coordination(self, target_x: float, target_y: float):
        """Request coordinated split maneuver"""
        await self.send_message(
            MessageType.SPLIT_COORDINATION,
            {"target_x": target_x, "target_y": target_y, "action": "request"},
            priority=4
        )
    
    async def confirm_split_coordination(self, recipient_id: str):
        """Confirm participation in split coordination"""
        await self.send_message(
            MessageType.SPLIT_COORDINATION,
            {"action": "confirm"},
            recipient_id=recipient_id,
            priority=4
        )
    
    async def report_food_location(self, food_x: float, food_y: float, food_mass: float):
        """Report food location to team"""
        await self.send_message(
            MessageType.FOOD_LOCATION,
            {"x": food_x, "y": food_y, "mass": food_mass},
            priority=2
        )
    
    async def designate_target(self, target_x: float, target_y: float, target_mass: float):
        """Designate a target for coordinated attack"""
        await self.send_message(
            MessageType.TARGET_DESIGNATION,
            {"x": target_x, "y": target_y, "mass": target_mass},
            priority=3
        )
