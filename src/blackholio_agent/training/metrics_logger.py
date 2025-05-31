"""
Metrics logging for training monitoring.

Supports TensorBoard and extensible to other logging backends.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from pathlib import Path
import logging

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Handles logging of training metrics to various backends.
    
    Currently supports:
    - TensorBoard
    - Console output
    - JSON logs (for custom processing)
    """
    
    def __init__(self,
                 log_dir: str,
                 experiment_name: Optional[str] = None,
                 console_log_interval: int = 10,
                 use_tensorboard: bool = True,
                 metrics_window: int = 100):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
            console_log_interval: Steps between console logs
            use_tensorboard: Whether to use TensorBoard
            metrics_window: Window size for rolling averages
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or f"blackholio_{int(time.time())}"
        self.console_log_interval = console_log_interval
        self.metrics_window = metrics_window
        
        # TensorBoard writer
        self.writer = None
        if use_tensorboard and HAS_TENSORBOARD:
            tb_dir = self.log_dir / "tensorboard" / self.experiment_name
            self.writer = SummaryWriter(str(tb_dir))
            logger.info(f"TensorBoard logging to: {tb_dir}")
        elif use_tensorboard and not HAS_TENSORBOARD:
            logger.warning("TensorBoard requested but not installed. Install with: pip install tensorboard")
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=metrics_window))
        self.step_metrics: Dict[str, float] = {}
        self.episode_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.total_steps = 0
        self.total_episodes = 0
        
    def log_step(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics for a single step.
        
        Args:
            step: Current training step
            metrics: Dictionary of metric values
        """
        self.total_steps = step
        
        # Store metrics
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.step_metrics[key] = value
        
        # Log to TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"step/{key}", value, step)
        
        # Console logging
        if step % self.console_log_interval == 0:
            self._log_to_console(step)
    
    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """
        Log metrics for a completed episode.
        
        Args:
            episode: Episode number
            metrics: Episode metrics
        """
        self.total_episodes = episode
        
        # Store episode metrics
        for key, value in metrics.items():
            self.episode_metrics[key].append(value)
        
        # Log to TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"episode/{key}", value, episode)
    
    def log_training_update(self, step: int, update_metrics: Dict[str, float]):
        """
        Log metrics from a training update (e.g., PPO update).
        
        Args:
            step: Current step
            update_metrics: Metrics from the update
        """
        # Log to TensorBoard
        if self.writer:
            for key, value in update_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, step)
        
        # Store for console logging
        for key, value in update_metrics.items():
            self.step_metrics[f"train/{key}"] = value
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log a histogram of values"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """Log hyperparameters and optionally their associated metrics"""
        if self.writer:
            # Filter hparams to only include scalar values
            scalar_hparams = {}
            for key, value in hparams.items():
                if isinstance(value, (int, float, bool, str)):
                    scalar_hparams[key] = value
                else:
                    scalar_hparams[key] = str(value)
            
            if metrics:
                self.writer.add_hparams(scalar_hparams, metrics)
            else:
                # Just log the hyperparameters
                for key, value in scalar_hparams.items():
                    self.writer.add_text(f"hparam/{key}", str(value))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        stats = {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "elapsed_time": time.time() - self.start_time,
            "steps_per_second": self.total_steps / (time.time() - self.start_time),
        }
        
        # Add rolling averages
        for key, values in self.metrics.items():
            if values:
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
                stats[f"{key}_last"] = values[-1]
        
        return stats
    
    def close(self):
        """Close the logger and flush any pending writes"""
        if self.writer:
            self.writer.close()
        
        # Save final statistics
        self._save_final_stats()
    
    def _log_to_console(self, step: int):
        """Log current metrics to console"""
        elapsed_time = time.time() - self.start_time
        steps_per_second = step / elapsed_time if elapsed_time > 0 else 0
        
        # Build log message
        msg_parts = [
            f"\n{'='*60}",
            f"Step: {step:,} | Episodes: {self.total_episodes:,}",
            f"Elapsed: {self._format_time(elapsed_time)} | SPS: {steps_per_second:.1f}",
        ]
        
        # Add key metrics
        if "reward" in self.step_metrics:
            msg_parts.append(f"Reward: {self.step_metrics['reward']:.2f}")
        
        if "mean_reward" in self.step_metrics:
            mean_reward = np.mean(list(self.metrics["mean_reward"]))
            msg_parts.append(f"Mean Reward (100): {mean_reward:.2f}")
        
        # Add training metrics
        train_metrics = []
        for key, value in self.step_metrics.items():
            if key.startswith("train/"):
                metric_name = key.replace("train/", "")
                train_metrics.append(f"{metric_name}: {value:.4f}")
        
        if train_metrics:
            msg_parts.append("Training: " + " | ".join(train_metrics[:4]))  # Limit to 4 metrics
        
        # Add episode metrics if available
        if self.episode_metrics:
            episode_parts = []
            for key, values in self.episode_metrics.items():
                if values and key in ["reward", "length"]:
                    recent_mean = np.mean(values[-10:])  # Last 10 episodes
                    episode_parts.append(f"{key}: {recent_mean:.1f}")
            
            if episode_parts:
                msg_parts.append("Episodes (last 10): " + " | ".join(episode_parts))
        
        msg_parts.append(f"{'='*60}")
        
        logger.info("\n".join(msg_parts))
        self.last_log_time = time.time()
    
    def _format_time(self, seconds: float) -> str:
        """Format time in a human-readable way"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _convert_to_native_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native_types(item) for item in obj]
        else:
            return obj
    
    def _save_final_stats(self):
        """Save final training statistics to a JSON file"""
        import json
        
        stats_file = self.log_dir / f"{self.experiment_name}_final_stats.json"
        
        final_stats = {
            "experiment_name": self.experiment_name,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "training_time": time.time() - self.start_time,
            "final_metrics": self._convert_to_native_types(self.step_metrics),
            "episode_metrics_summary": {}
        }
        
        # Summarize episode metrics
        for key, values in self.episode_metrics.items():
            if values:
                final_stats["episode_metrics_summary"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "final": float(values[-1])
                }
        
        # Convert entire final_stats to ensure all numpy types are converted
        final_stats = self._convert_to_native_types(final_stats)
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info(f"Saved final statistics to: {stats_file}")
