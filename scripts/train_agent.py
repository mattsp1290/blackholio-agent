#!/usr/bin/env python
"""
Main training script for Blackholio RL agent.

This script provides a command-line interface to train a PPO agent
to play Blackholio using reinforcement learning.
"""

import argparse
import logging
import sys
import os
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from typing import Dict, Any, Optional
import socket
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.blackholio_agent.training import PPOTrainer, PPOConfig

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, experiment_name: str):
    """Setup logging with both console and file handlers"""
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup log file path
    log_file = Path(log_dir) / f"{experiment_name}_training.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    logger.info(f"Setting random seeds to: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_experiment_name() -> str:
    """Generate a unique experiment name based on timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"blackholio_training_{timestamp}"


def validate_server_connection(host: str) -> bool:
    """Validate that we can connect to the Blackholio server"""
    try:
        # Parse host:port
        if ':' in host:
            hostname, port = host.split(':')
            port = int(port)
        else:
            hostname = host
            port = 3000
        
        # Test connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((hostname, port))
        sock.close()
        
        if result == 0:
            logger.info(f"✅ Successfully connected to Blackholio server at {host}")
            return True
        else:
            logger.error(f"❌ Cannot connect to Blackholio server at {host}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error validating server connection: {e}")
        return False


def validate_configuration(config: PPOConfig) -> bool:
    """Validate training configuration parameters"""
    logger.info("Validating configuration...")
    
    # Check required parameters
    if config.total_timesteps <= 0:
        logger.error("❌ total_timesteps must be positive")
        return False
    
    if config.n_envs <= 0:
        logger.error("❌ n_envs must be positive")
        return False
    
    if config.learning_rate <= 0:
        logger.error("❌ learning_rate must be positive")
        return False
    
    if config.batch_size <= 0:
        logger.error("❌ batch_size must be positive")
        return False
    
    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        logger.error("❌ CUDA device requested but not available")
        return False
    
    if config.device == "mps" and not torch.backends.mps.is_available():
        logger.error("❌ MPS device requested but not available")
        return False
    
    logger.info("✅ Configuration validation passed")
    return True


def calculate_training_eta(total_timesteps: int, n_envs: int, estimated_fps: float = 100.0) -> timedelta:
    """Calculate estimated training completion time"""
    # Estimate steps per second based on parallel environments
    steps_per_second = estimated_fps * n_envs
    total_seconds = total_timesteps / steps_per_second
    return timedelta(seconds=total_seconds)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent to play Blackholio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment arguments
    parser.add_argument(
        "--host",
        type=str,
        default="localhost:3000",
        help="Blackholio server host:port"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="SpacetimeDB database name (uses default if not specified)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments"
    )
    
    # Training arguments
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total number of training timesteps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for PPO updates"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps per environment before update"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (auto-generated if not provided)"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, mps, cuda, or auto)"
    )
    
    # Validation options
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip server connectivity validation"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock connection instead of SpacetimeDB (for testing/offline training)"
    )
    
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    experiment_name = args.experiment_name or generate_experiment_name()
    
    # Create experiment-specific directories
    log_dir = os.path.join(args.log_dir, experiment_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_name)
    
    # Setup logging
    setup_logging(log_dir, experiment_name)
    logger.info("=" * 80)
    logger.info(f"🚀 Starting Blackholio RL Agent Training")
    logger.info(f"📁 Experiment: {experiment_name}")
    logger.info("=" * 80)
    
    # Set random seed if provided
    if args.seed is not None:
        set_random_seeds(args.seed)
    else:
        # Generate and use a random seed
        seed = random.randint(0, 2**32 - 1)
        set_random_seeds(seed)
        logger.info(f"Generated random seed: {seed}")
    
    # Load configuration
    if hasattr(args, 'config') and args.config:
        logger.info(f"📄 Loading configuration from: {args.config}")
        config_dict = load_config(args.config)
    else:
        config_dict = {}
    
    # Handle mock mode
    if args.mock:
        logger.info("🤖 Mock mode enabled - using simulated environment")
        config_dict["env_host"] = "mock://localhost"  # Special host to trigger mock mode
        args.skip_server_check = True  # Skip server check in mock mode
    
    # Override with command-line arguments
    if args.host != "localhost:3000":
        config_dict["env_host"] = args.host
    if args.database is not None:
        config_dict["env_database"] = args.database
    if args.n_envs != 8:
        config_dict["n_envs"] = args.n_envs
    if args.total_timesteps != 1_000_000:
        config_dict["total_timesteps"] = args.total_timesteps
    if args.learning_rate != 3e-4:
        config_dict["learning_rate"] = args.learning_rate
    if args.batch_size != 256:
        config_dict["batch_size"] = args.batch_size
    if args.n_steps != 2048:
        config_dict["n_steps"] = args.n_steps
    
    # Set experiment-specific directories
    config_dict["log_dir"] = log_dir
    config_dict["checkpoint_dir"] = checkpoint_dir
    
    # Handle device selection
    if args.device == "auto":
        if torch.backends.mps.is_available():
            config_dict["device"] = "mps"
        elif torch.cuda.is_available():
            config_dict["device"] = "cuda"
        else:
            config_dict["device"] = "cpu"
    else:
        config_dict["device"] = args.device
    
    # Create PPO config
    try:
        ppo_config = PPOConfig(**config_dict)
    except Exception as e:
        logger.error(f"❌ Failed to create configuration: {e}")
        return 1
    
    # Validate configuration
    if not validate_configuration(ppo_config):
        logger.error("❌ Configuration validation failed")
        return 1
    
    # Validate server connection (unless skipped)
    if not args.skip_server_check:
        if not validate_server_connection(ppo_config.env_host):
            logger.error("❌ Server validation failed. Use --skip-server-check to bypass.")
            return 1
    
    # Log configuration
    logger.info("📋 Training Configuration:")
    logger.info(f"  🌐 Server: {ppo_config.env_host}")
    logger.info(f"  🗄️  Database: {ppo_config.env_database}")
    logger.info(f"  🏃 Parallel environments: {ppo_config.n_envs}")
    logger.info(f"  📊 Total timesteps: {ppo_config.total_timesteps:,}")
    logger.info(f"  🧠 Learning rate: {ppo_config.learning_rate}")
    logger.info(f"  📦 Batch size: {ppo_config.batch_size}")
    logger.info(f"  💻 Device: {ppo_config.device}")
    logger.info(f"  📁 Log directory: {ppo_config.log_dir}")
    logger.info(f"  💾 Checkpoint directory: {ppo_config.checkpoint_dir}")
    
    # Calculate ETA
    eta = calculate_training_eta(ppo_config.total_timesteps, ppo_config.n_envs)
    logger.info(f"  ⏰ Estimated training time: {eta}")
    
    # TensorBoard information
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    logger.info(f"  📈 TensorBoard logs: {tensorboard_dir}")
    logger.info(f"  💡 Start TensorBoard with: tensorboard --logdir {tensorboard_dir}")
    
    # Create directories
    Path(ppo_config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(ppo_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    logger.info("🔧 Initializing PPO trainer...")
    try:
        trainer = PPOTrainer(ppo_config)
    except Exception as e:
        logger.error(f"❌ Failed to create trainer: {e}")
        return 1
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"🔄 Resuming from checkpoint: {args.resume}")
        if not trainer.load_checkpoint(args.resume):
            logger.error("❌ Failed to load checkpoint!")
            return 1
    
    # Record training start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    # Start training
    logger.info("🎯 Starting training...")
    logger.info(f"⏰ Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    try:
        trainer.train()
        
        # Calculate training duration
        end_time = time.time()
        duration = timedelta(seconds=end_time - start_time)
        
        logger.info("=" * 80)
        logger.info("🎉 Training completed successfully!")
        logger.info(f"⏱️  Total training time: {duration}")
        logger.info(f"💾 Final model saved in: {ppo_config.checkpoint_dir}")
        logger.info(f"📊 Training logs available in: {ppo_config.log_dir}")
        logger.info("=" * 80)
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("⏸️  Training interrupted by user")
        logger.info(f"💾 Latest checkpoint saved in: {ppo_config.checkpoint_dir}")
        logger.info("🔄 Resume training with: --resume <checkpoint_path>")
        logger.info("=" * 80)
        return 0
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        logger.error(f"📁 Check logs in: {ppo_config.log_dir}")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
