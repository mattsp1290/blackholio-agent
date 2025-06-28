"""
Minimal connection module that redirects to the fixed v1.1.2 implementation.
"""

import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConnectionConfig:
    """Configuration for Blackholio connection"""
    host: str
    database: str = "blackholio"  # Module name
    db_identity: Optional[str] = None  # v1.1.2 requires database identity
    auth_token: Optional[str] = None
    ssl_enabled: bool = False  # Default to False for localhost
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    subscription_timeout: float = 30.0
    
    def __post_init__(self):
        """Parse host URL and set SSL enabled based on protocol"""
        if self.host.startswith("wss://") or self.host.startswith("https://"):
            self.ssl_enabled = True

@dataclass
class GameState:
    """Current game state snapshot"""
    player_id: Optional[int] = None
    player_identity: Optional[str] = None
    entities: Dict[int, Any] = None  # entity_id -> entity data
    players: Dict[int, Any] = None   # player_id -> player data
    circles: Dict[int, Any] = None   # entity_id -> circle data
    food: List[Any] = None           # list of food entities
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = {}
        if self.players is None:
            self.players = {}
        if self.circles is None:
            self.circles = {}
        if self.food is None:
            self.food = []

class BlackholioConnection:
    """
    DISABLED: This class uses the old SpacetimeDB SDK which sends messages with 'type' field
    that cause server errors. Use BlackholioConnectionV112 instead.
    """
    
    def __init__(self, config: ConnectionConfig):
        raise RuntimeError(
            "BlackholioConnection is disabled due to incompatible message format. "
            "Use BlackholioConnectionV112 instead. The old SpacetimeDB SDK sends "
            "messages with 'type' field that cause server errors."
        )