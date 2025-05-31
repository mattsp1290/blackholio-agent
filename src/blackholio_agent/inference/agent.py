"""
Main inference agent for running trained models.

This module provides the InferenceAgent class that handles the
complete inference pipeline for playing Blackholio.
"""

import asyncio
import torch
import numpy as np
import time
import logging
import signal
import sys
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .config import InferenceConfig
from .model_loader import ModelLoader
from .metrics import InferenceMetrics, detect_failure_mode
from .rate_limiter import RateLimiter
from ..environment import BlackholioEnv
from ..models import BlackholioModel

logger = logging.getLogger(__name__)


class InferenceAgent:
    """
    Main inference agent for running trained Blackholio models.
    
    This agent handles:
    - Model loading and initialization
    - Environment interaction
    - Rate-limited inference loop
    - Performance monitoring
    - Graceful shutdown
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize inference agent.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        
        # Initialize components
        logger.info("Initializing inference agent...")
        
        # Load model
        self.model = ModelLoader.load_for_inference(
            model_path=config.model_path,
            device=config.device,
            num_threads=config.cpu_threads
        )
        
        # Validate model
        if not ModelLoader.validate_model(self.model):
            raise ValueError("Model validation failed")
        
        # Log model info
        model_info = ModelLoader.get_model_info(self.model)
        logger.info(f"Model info: {model_info}")
        
        # Create environment
        env_config = config.create_env_config()
        self.env = BlackholioEnv(env_config)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(min_interval_ms=config.rate_limit_ms)
        
        # Initialize metrics
        self.metrics = InferenceMetrics()
        
        # State tracking
        self.running = False
        self.current_obs = None
        self.episodes_completed = 0
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info(f"Inference agent initialized for player '{config.player_name}'")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run(self):
        """
        Run the inference agent.
        
        This is the main entry point that runs the game loop.
        """
        logger.info("Starting inference agent...")
        
        try:
            # Warmup model
            if self.config.warmup_steps > 0:
                warmup_stats = ModelLoader.warmup_model(
                    self.model, 
                    warmup_steps=self.config.warmup_steps,
                    device=self.config.device
                )
                logger.info(f"Model warmup stats: {warmup_stats}")
            
            # Run main loop
            self.running = True
            self._run_loop()
            
        except Exception as e:
            logger.error(f"Error in inference agent: {e}", exc_info=True)
            raise
        
        finally:
            self._cleanup()
    
    def _run_loop(self):
        """Main inference loop"""
        while self.running:
            try:
                # Check episode limit
                if self.config.max_episodes > 0 and self.episodes_completed >= self.config.max_episodes:
                    logger.info(f"Reached max episodes limit ({self.config.max_episodes})")
                    break
                
                # Run episode
                episode_metrics = self._run_episode()
                self.episodes_completed += 1
                
                # Log episode results
                logger.info(
                    f"[Episode {episode_metrics.episode_id}] "
                    f"Survived: {episode_metrics.survival_time:.1f}s | "
                    f"Max Mass: {episode_metrics.max_mass:.1f} | "
                    f"Avg Inference: {episode_metrics.avg_inference_latency*1000:.1f}ms | "
                    f"Failure: {episode_metrics.failure_mode}"
                )
                
                # Log summary periodically
                self.metrics.log_summary(episode_interval=self.config.log_interval)
                
                # Save metrics periodically
                if self.episodes_completed % self.config.metrics_interval == 0:
                    self.metrics.save_to_file(f"inference_metrics_{self.episodes_completed}.json")
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.running = False
                break
            
            except Exception as e:
                logger.error(f"Error in episode: {e}", exc_info=True)
                # Continue to next episode after error
                time.sleep(1)
    
    def _run_episode(self) -> Any:
        """
        Run a single episode.
        
        Returns:
            Episode metrics
        """
        # Start episode
        self.metrics.start_episode()
        
        # Reset environment
        obs, info = self.env.reset()
        self.current_obs = obs
        
        # Track episode start
        episode_start_time = time.time()
        
        # Run episode steps
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and self.running:
            # Rate limit
            self.rate_limiter.wait()
            
            # Get action from model
            start_time = time.time()
            action = self._get_action(obs)
            inference_time = time.time() - start_time
            
            # Execute action in environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.current_obs = obs
            
            # Get current mass
            current_mass = self._get_current_mass(info)
            
            # Log step metrics
            self.metrics.log_step(
                inference_latency=inference_time,
                reward=reward,
                info=info,
                current_mass=current_mass
            )
            
            # Verbose logging
            if self.config.verbose and self.metrics.current_episode_steps % 100 == 0:
                logger.debug(
                    f"Step {self.metrics.current_episode_steps}: "
                    f"Mass={current_mass:.1f}, "
                    f"Reward={reward:.2f}, "
                    f"Inference={inference_time*1000:.1f}ms"
                )
        
        # Determine success and failure mode
        survival_time = time.time() - episode_start_time
        max_mass = self.metrics.current_episode_max_mass
        
        success = (
            survival_time >= self.config.success_survival_time and
            max_mass >= self.config.success_mass_threshold
        )
        
        failure_mode = detect_failure_mode(info) if not success else "success"
        
        # End episode
        episode_metrics = self.metrics.end_episode(
            failure_mode=failure_mode,
            success=success,
            final_info=info
        )
        
        return episode_metrics
    
    def _get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action from model.
        
        Args:
            observation: Current observation
            
        Returns:
            Action array
        """
        # Convert to tensor
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        
        # Get action from model
        with torch.no_grad():
            actions, _ = self.model.get_action(
                obs_tensor.to(self.config.device),
                deterministic=True  # Use deterministic actions for inference
            )
        
        # Convert to numpy array for environment
        movement = actions['movement'].cpu().numpy().squeeze()
        split = actions['split'].cpu().numpy().squeeze()
        
        # Combine into single action array
        action = np.concatenate([movement, [split]])
        
        return action
    
    def _get_current_mass(self, info: Dict[str, Any]) -> float:
        """Extract current total mass from info"""
        # Try different possible keys
        if 'player_mass' in info:
            return info['player_mass']
        elif 'total_mass' in info:
            return info['total_mass']
        elif 'game_state' in info and 'player_mass' in info['game_state']:
            return info['game_state']['player_mass']
        
        # Calculate from entities if available
        if 'player_entities' in info:
            entities = info['player_entities']
            if entities:
                return sum(e.get('mass', 0) for e in entities)
        
        return 0.0
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up inference agent...")
        
        # Save final metrics
        if self.metrics.total_episodes > 0:
            self.metrics.save_to_file("inference_metrics_final.json")
            
            # Log final summary
            summary = self.metrics.get_summary()
            logger.info("Final metrics summary:")
            logger.info(f"  Total Episodes: {summary['total_episodes']}")
            logger.info(f"  Success Rate: {summary['success_rate']}%")
            logger.info(f"  Avg Survival Time: {summary['avg_survival_time']}s")
            logger.info(f"  Avg Inference Latency: {summary['avg_inference_latency_ms']}ms")
            
            # Log failure modes
            if summary['failure_modes']:
                logger.info("  Failure Modes:")
                for mode, percentage in summary['failure_modes'].items():
                    logger.info(f"    {mode}: {percentage}%")
        
        # Close environment
        self.env.close()
        
        logger.info("Inference agent shutdown complete")


def run_inference_agent(config: Optional[InferenceConfig] = None):
    """
    Convenience function to run the inference agent.
    
    Args:
        config: Optional configuration (uses defaults if None)
    """
    if config is None:
        config = InferenceConfig()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if config.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run agent
    agent = InferenceAgent(config)
    agent.run()
