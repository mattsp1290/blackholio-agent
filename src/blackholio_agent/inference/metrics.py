"""
Metrics tracking for inference system.

This module provides lightweight metrics collection for analyzing
agent performance and failure modes.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode"""
    episode_id: int
    start_time: float
    end_time: float
    survival_time: float
    max_mass: float
    total_reward: float
    num_steps: int
    failure_mode: str
    avg_inference_latency: float
    max_inference_latency: float
    
    @property
    def duration(self) -> float:
        """Episode duration in seconds"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'episode_id': self.episode_id,
            'survival_time': round(self.survival_time, 2),
            'max_mass': round(self.max_mass, 2),
            'total_reward': round(self.total_reward, 2),
            'num_steps': self.num_steps,
            'failure_mode': self.failure_mode,
            'avg_inference_latency_ms': round(self.avg_inference_latency * 1000, 2),
            'max_inference_latency_ms': round(self.max_inference_latency * 1000, 2),
            'duration': round(self.duration, 2)
        }


class InferenceMetrics:
    """
    Lightweight metrics tracker for inference system.
    
    Tracks key performance indicators and failure modes without
    storing detailed episode data.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of rolling window for recent metrics
        """
        self.window_size = window_size
        
        # Episode tracking
        self.total_episodes = 0
        self.current_episode_id = 0
        self.episode_start_time = None
        self.episode_first_step_time = None
        
        # Current episode metrics
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        self.current_episode_max_mass = 0.0
        self.current_episode_latencies = []
        
        # Rolling window of recent episodes
        self.recent_episodes: deque[EpisodeMetrics] = deque(maxlen=window_size)
        
        # Aggregate metrics
        self.total_steps = 0
        self.total_survival_time = 0.0
        self.successful_episodes = 0  # Based on config thresholds
        
        # Failure mode tracking
        self.failure_modes: Dict[str, int] = defaultdict(int)
        
        # Performance tracking
        self.inference_latencies: deque[float] = deque(maxlen=1000)
        self.step_times: deque[float] = deque(maxlen=1000)
        
        logger.info("Inference metrics tracker initialized")
    
    def start_episode(self):
        """Start tracking a new episode"""
        self.current_episode_id += 1
        self.total_episodes += 1
        self.episode_start_time = time.time()
        self.episode_first_step_time = None
        
        # Reset current episode metrics
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        self.current_episode_max_mass = 0.0
        self.current_episode_latencies.clear()
    
    def log_step(self, 
                 inference_latency: float,
                 reward: float,
                 info: Dict[str, any],
                 current_mass: Optional[float] = None):
        """
        Log metrics for a single step.
        
        Args:
            inference_latency: Time taken for model inference (seconds)
            reward: Reward received this step
            info: Step info from environment
            current_mass: Current total mass of agent
        """
        # Track first step time for survival calculation
        if self.episode_first_step_time is None:
            self.episode_first_step_time = time.time()
        
        # Update counters
        self.current_episode_steps += 1
        self.total_steps += 1
        
        # Update episode metrics
        self.current_episode_reward += reward
        if current_mass is not None:
            self.current_episode_max_mass = max(self.current_episode_max_mass, current_mass)
        
        # Track latencies
        self.inference_latencies.append(inference_latency)
        self.current_episode_latencies.append(inference_latency)
        
        # Track step time
        self.step_times.append(time.time())
    
    def end_episode(self, 
                    failure_mode: str = "unknown",
                    success: bool = False,
                    final_info: Optional[Dict] = None) -> EpisodeMetrics:
        """
        End current episode and record metrics.
        
        Args:
            failure_mode: How the agent died
            success: Whether episode was successful
            final_info: Final step info from environment
            
        Returns:
            Metrics for the completed episode
        """
        end_time = time.time()
        
        # Calculate survival time
        if self.episode_first_step_time:
            survival_time = end_time - self.episode_first_step_time
        else:
            survival_time = 0.0
        
        self.total_survival_time += survival_time
        
        # Track success
        if success:
            self.successful_episodes += 1
        
        # Track failure mode
        self.failure_modes[failure_mode] += 1
        
        # Calculate latency stats
        if self.current_episode_latencies:
            avg_latency = sum(self.current_episode_latencies) / len(self.current_episode_latencies)
            max_latency = max(self.current_episode_latencies)
        else:
            avg_latency = 0.0
            max_latency = 0.0
        
        # Create episode metrics
        episode_metrics = EpisodeMetrics(
            episode_id=self.current_episode_id,
            start_time=self.episode_start_time,
            end_time=end_time,
            survival_time=survival_time,
            max_mass=self.current_episode_max_mass,
            total_reward=self.current_episode_reward,
            num_steps=self.current_episode_steps,
            failure_mode=failure_mode,
            avg_inference_latency=avg_latency,
            max_inference_latency=max_latency
        )
        
        # Add to recent episodes
        self.recent_episodes.append(episode_metrics)
        
        return episode_metrics
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of all metrics"""
        if self.total_episodes == 0:
            return {
                'total_episodes': 0,
                'status': 'No episodes completed yet'
            }
        
        # Calculate aggregate stats
        success_rate = self.successful_episodes / self.total_episodes
        avg_survival_time = self.total_survival_time / self.total_episodes
        
        # Recent episode stats
        if self.recent_episodes:
            recent_survival = [ep.survival_time for ep in self.recent_episodes]
            recent_mass = [ep.max_mass for ep in self.recent_episodes]
            recent_avg_survival = sum(recent_survival) / len(recent_survival)
            recent_avg_mass = sum(recent_mass) / len(recent_mass)
        else:
            recent_avg_survival = 0.0
            recent_avg_mass = 0.0
        
        # Latency stats
        if self.inference_latencies:
            avg_latency = sum(self.inference_latencies) / len(self.inference_latencies)
            max_latency = max(self.inference_latencies)
        else:
            avg_latency = 0.0
            max_latency = 0.0
        
        # FPS calculation
        fps = self._calculate_fps()
        
        # Failure mode percentages
        failure_percentages = {}
        for mode, count in self.failure_modes.items():
            failure_percentages[mode] = round(count / self.total_episodes * 100, 1)
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'success_rate': round(success_rate * 100, 1),
            'avg_survival_time': round(avg_survival_time, 1),
            'recent_avg_survival': round(recent_avg_survival, 1),
            'recent_avg_mass': round(recent_avg_mass, 1),
            'avg_inference_latency_ms': round(avg_latency * 1000, 2),
            'max_inference_latency_ms': round(max_latency * 1000, 2),
            'current_fps': round(fps, 1),
            'failure_modes': failure_percentages
        }
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS based on recent step times"""
        if len(self.step_times) < 2:
            return 0.0
        
        # Use recent steps for FPS calculation
        recent_times = list(self.step_times)[-20:]  # Last 20 steps
        if len(recent_times) < 2:
            return 0.0
        
        time_span = recent_times[-1] - recent_times[0]
        if time_span > 0:
            return (len(recent_times) - 1) / time_span
        return 0.0
    
    def log_summary(self, episode_interval: int = 10):
        """Log summary metrics"""
        if self.total_episodes % episode_interval == 0 and self.total_episodes > 0:
            summary = self.get_summary()
            logger.info(f"[Episode {self.total_episodes}] Metrics Summary:")
            logger.info(f"  Success Rate: {summary['success_rate']}%")
            logger.info(f"  Avg Survival: {summary['avg_survival_time']}s")
            logger.info(f"  Recent Avg Mass: {summary['recent_avg_mass']}")
            logger.info(f"  FPS: {summary['current_fps']}")
            logger.info(f"  Avg Inference: {summary['avg_inference_latency_ms']}ms")
            
            # Log top failure modes
            if summary['failure_modes']:
                top_failures = sorted(summary['failure_modes'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                failures_str = ", ".join([f"{mode}: {pct}%" for mode, pct in top_failures])
                logger.info(f"  Top Failures: {failures_str}")
    
    def save_to_file(self, filepath: str):
        """Save metrics summary to JSON file"""
        summary = self.get_summary()
        
        # Add recent episode details
        if self.recent_episodes:
            summary['recent_episodes'] = [ep.to_dict() for ep in self.recent_episodes][-10:]
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")


def detect_failure_mode(info: Dict[str, any]) -> str:
    """
    Detect failure mode from environment info.
    
    Args:
        info: Step info from environment
        
    Returns:
        String describing failure mode
    """
    # Check various failure conditions
    if info.get('terminated', False):
        # Check specific failure reasons
        if 'eaten_by' in info:
            return 'eaten_by_larger'
        elif info.get('out_of_bounds', False):
            return 'out_of_bounds'
        elif info.get('no_entities', False):
            return 'no_entities'
        else:
            return 'died_unknown'
    
    # Check for truncation (max steps)
    if info.get('truncated', False):
        if info.get('TimeLimit.truncated', False):
            return 'time_limit'
        else:
            return 'truncated'
    
    # Check game events
    events = info.get('game_events', {})
    if events.get('killed_by_player'):
        return 'killed_by_player'
    elif events.get('killed_by_virus'):
        return 'killed_by_virus'
    elif events.get('starved'):
        return 'starvation'
    
    # Check reward components for clues
    reward_components = info.get('reward_components', {})
    if reward_components.get('death_penalty', 0) < 0:
        return 'death_penalty'
    
    return 'unknown'
