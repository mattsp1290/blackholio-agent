#!/usr/bin/env python3
"""
Run a trained Blackholio agent in inference mode.

This script loads a trained model and runs it in the Blackholio game,
with performance monitoring and metrics collection.

Usage:
    python run_agent.py --model checkpoints/best_model.pth
    python run_agent.py --model model.pth --host localhost:3000 --name MyAgent
    
Environment variables can also be used:
    MODEL_PATH=model.pth SPACETIMEDB_HOST=localhost:3000 python run_agent.py
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blackholio_agent.inference import InferenceConfig, InferenceAgent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run trained Blackholio agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model arguments
    parser.add_argument(
        '--model', '--model-path',
        type=str,
        default=None,
        help='Path to trained model checkpoint (default: from MODEL_PATH env var)'
    )
    
    # Connection arguments
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='SpacetimeDB host (default: localhost:3000 or SPACETIMEDB_HOST env var)'
    )
    
    parser.add_argument(
        '--database', '--db',
        type=str,
        default=None,
        help='SpacetimeDB database name (default: blackholio or SPACETIMEDB_DB env var)'
    )
    
    parser.add_argument(
        '--auth-token',
        type=str,
        default=None,
        help='SpacetimeDB auth token (default: from SPACETIMEDB_TOKEN env var)'
    )
    
    parser.add_argument(
        '--ssl',
        action='store_true',
        help='Enable SSL for SpacetimeDB connection'
    )
    
    # Agent arguments
    parser.add_argument(
        '--name', '--player-name',
        type=str,
        default=None,
        help='Player name in game (default: ML_Agent_<random> or AGENT_NAME env var)'
    )
    
    # Performance arguments
    parser.add_argument(
        '--cpu-threads',
        type=int,
        default=None,
        help='Number of CPU threads to use (default: 4 or CPU_THREADS env var)'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=None,
        help='Minimum milliseconds between actions (default: 50 or RATE_LIMIT_MS env var)'
    )
    
    # Episode arguments
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=None,
        help='Maximum number of episodes to run (default: unlimited)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Maximum steps per episode (default: 10000)'
    )
    
    # Monitoring arguments
    parser.add_argument(
        '--log-interval',
        type=int,
        default=None,
        help='Episodes between summary logs (default: 100)'
    )
    
    parser.add_argument(
        '--metrics-interval',
        type=int,
        default=None,
        help='Episodes between metrics saves (default: 1000)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Success criteria
    parser.add_argument(
        '--success-time',
        type=float,
        default=None,
        help='Survival time for success (seconds, default: 60)'
    )
    
    parser.add_argument(
        '--success-mass',
        type=float,
        default=None,
        help='Mass threshold for success (default: 100)'
    )
    
    # Other arguments
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=None,
        help='Model warmup steps (default: 10)'
    )
    
    parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Skip model warmup'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> InferenceConfig:
    """Create InferenceConfig from command line arguments"""
    # Start with defaults (which will read from env vars)
    config = InferenceConfig()
    
    # Override with command line arguments if provided
    if args.model:
        config.model_path = args.model
    
    if args.host:
        config.host = args.host
    
    if args.database:
        config.database = args.database
    
    if args.auth_token:
        config.auth_token = args.auth_token
    
    if args.ssl:
        config.ssl_enabled = True
    
    if args.name:
        config.player_name = args.name
    
    if args.cpu_threads is not None:
        config.cpu_threads = args.cpu_threads
    
    if args.rate_limit is not None:
        config.rate_limit_ms = args.rate_limit
    
    if args.max_episodes is not None:
        config.max_episodes = args.max_episodes
    
    if args.max_steps is not None:
        config.max_steps_per_episode = args.max_steps
    
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    
    if args.metrics_interval is not None:
        config.metrics_interval = args.metrics_interval
    
    if args.verbose:
        config.verbose = True
    
    if args.success_time is not None:
        config.success_survival_time = args.success_time
    
    if args.success_mass is not None:
        config.success_mass_threshold = args.success_mass
    
    if args.no_warmup:
        config.warmup_steps = 0
    elif args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    
    return config


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Create config
    try:
        config = create_config_from_args(args)
    except Exception as e:
        print(f"Error creating configuration: {e}", file=sys.stderr)
        return 1
    
    # Setup logging
    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('inference.log')
        ]
    )
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info("Starting Blackholio inference agent")
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Host: {config.host}")
    logger.info(f"Player: {config.player_name}")
    logger.info(f"CPU Threads: {config.cpu_threads}")
    logger.info(f"Rate Limit: {config.rate_limit_ms}ms")
    
    try:
        # Create and run agent
        agent = InferenceAgent(config)
        agent.run()
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
