"""
Configuration for inference system.

This module provides configuration management for the inference agent,
with support for environment variables and Docker deployment.
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..environment import BlackholioEnvConfig, ConnectionConfig, ObservationConfig, ActionConfig


@dataclass
class InferenceConfig:
    """Configuration for inference agent"""
    
    # Model settings
    model_path: str = field(default_factory=lambda: os.getenv('MODEL_PATH', 'checkpoints/best_model.pth'))
    device: str = field(default_factory=lambda: os.getenv('DEVICE', 'cpu'))
    
    # Connection settings (from env vars for Docker)
    host: str = field(default_factory=lambda: os.getenv('SPACETIMEDB_HOST', 'localhost:3000'))
    database: str = field(default_factory=lambda: os.getenv('SPACETIMEDB_DB', 'blackholio'))
    auth_token: Optional[str] = field(default_factory=lambda: os.getenv('SPACETIMEDB_TOKEN', None))
    ssl_enabled: bool = field(default_factory=lambda: os.getenv('SSL_ENABLED', 'false').lower() == 'true')
    
    # Agent settings
    player_name: str = field(default_factory=lambda: os.getenv('AGENT_NAME', f'ML_Agent_{uuid.uuid4().hex[:8]}'))
    
    # Performance settings
    cpu_threads: int = field(default_factory=lambda: int(os.getenv('CPU_THREADS', '4')))
    rate_limit_ms: float = field(default_factory=lambda: float(os.getenv('RATE_LIMIT_MS', '50')))
    
    # Monitoring settings
    log_interval: int = field(default_factory=lambda: int(os.getenv('LOG_INTERVAL', '100')))
    metrics_interval: int = field(default_factory=lambda: int(os.getenv('METRICS_INTERVAL', '1000')))
    verbose: bool = field(default_factory=lambda: os.getenv('VERBOSE', 'false').lower() == 'true')
    
    # Episode settings
    max_episodes: int = field(default_factory=lambda: int(os.getenv('MAX_EPISODES', '0')))  # 0 = unlimited
    max_steps_per_episode: int = field(default_factory=lambda: int(os.getenv('MAX_STEPS_PER_EPISODE', '10000')))
    
    # Success criteria
    success_survival_time: float = field(default_factory=lambda: float(os.getenv('SUCCESS_SURVIVAL_TIME', '60.0')))
    success_mass_threshold: float = field(default_factory=lambda: float(os.getenv('SUCCESS_MASS_THRESHOLD', '100.0')))
    
    # Warmup settings
    warmup_steps: int = field(default_factory=lambda: int(os.getenv('WARMUP_STEPS', '10')))
    
    def create_env_config(self) -> BlackholioEnvConfig:
        """Create environment config from inference config"""
        return BlackholioEnvConfig(
            host=self.host,
            database=self.database,
            auth_token=self.auth_token,
            ssl_enabled=self.ssl_enabled,
            player_name=self.player_name,
            render_mode=None,  # No rendering during inference
            max_episode_steps=self.max_steps_per_episode,
            step_interval=self.rate_limit_ms / 1000.0,  # Convert ms to seconds
            connection_config=ConnectionConfig(
                host=self.host,
                database=self.database,
                auth_token=self.auth_token,
                ssl_enabled=self.ssl_enabled,
                reconnect_delay=5.0,
                max_reconnect_attempts=10
            )
        )
    
    def validate(self):
        """Validate configuration"""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        if self.cpu_threads < 1:
            raise ValueError(f"CPU threads must be >= 1, got {self.cpu_threads}")
        
        if self.rate_limit_ms < 10:
            raise ValueError(f"Rate limit must be >= 10ms, got {self.rate_limit_ms}")
        
        if self.device not in ['cpu', 'cuda']:
            raise ValueError(f"Device must be 'cpu' or 'cuda', got {self.device}")
        
        # Force CPU for this deployment
        if self.device == 'cuda':
            self.device = 'cpu'
            print("WARNING: CUDA requested but forcing CPU for deployment")
    
    def __post_init__(self):
        """Post-initialization validation"""
        self.validate()
