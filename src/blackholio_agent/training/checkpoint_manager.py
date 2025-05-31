"""
Checkpoint manager for saving and loading training state.

Handles model checkpoints with configurable retention policy.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging
import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint"""
    step: int
    timestamp: float
    episode: int
    mean_reward: float
    path: str
    is_best: bool = False
    metadata: Optional[Dict[str, Any]] = None


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
    - Save checkpoints at regular intervals
    - Keep best performing model
    - Maintain rolling window of recent checkpoints
    - Save complete training state for resuming
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 save_interval_minutes: float = 20.0,
                 keep_best: bool = True,
                 keep_recent: int = 5,
                 metrics_to_track: Optional[List[str]] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval_minutes: Save interval in minutes
            keep_best: Whether to keep best performing model
            keep_recent: Number of recent checkpoints to keep
            metrics_to_track: Metrics to track for best model selection
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval_seconds = save_interval_minutes * 60
        self.keep_best = keep_best
        self.keep_recent = keep_recent
        self.metrics_to_track = metrics_to_track or ["mean_reward"]
        
        # State
        self.last_save_time = 0.0
        self.best_metric_value = -float('inf')
        self.checkpoints: List[CheckpointInfo] = []
        
        # Load existing checkpoints
        self._load_checkpoint_history()
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
        logger.info(f"Save interval: {save_interval_minutes} minutes, "
                   f"Keep best: {keep_best}, Keep recent: {keep_recent}")
    
    def should_save(self) -> bool:
        """Check if it's time to save a checkpoint"""
        return time.time() - self.last_save_time >= self.save_interval_seconds
    
    def save(self,
             step: int,
             episode: int,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             metrics: Dict[str, float],
             additional_state: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Save a checkpoint.
        
        Args:
            step: Current training step
            episode: Current episode number
            model: Model to save
            optimizer: Optimizer state
            metrics: Current metrics
            additional_state: Additional state to save
            
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        # Check if we should save
        if not self.should_save() and not self._is_best(metrics):
            return None
        
        # Create checkpoint data
        checkpoint = {
            "step": step,
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": time.time(),
        }
        
        if additional_state:
            checkpoint.update(additional_state)
        
        # Determine checkpoint path
        checkpoint_name = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Create checkpoint info
        mean_reward = metrics.get("mean_reward", 0.0)
        is_best = self._is_best(metrics)
        
        info = CheckpointInfo(
            step=step,
            timestamp=checkpoint["timestamp"],
            episode=episode,
            mean_reward=mean_reward,
            path=str(checkpoint_path),
            is_best=is_best,
            metadata=metrics
        )
        
        self.checkpoints.append(info)
        self.last_save_time = time.time()
        
        # Handle best model
        if is_best and self.keep_best:
            self._save_best_model(checkpoint_path, metrics)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save checkpoint history
        self._save_checkpoint_history()
        
        return str(checkpoint_path)
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint"""
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda c: c.step)
        return self._load_checkpoint(latest.path)
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best performing checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return self._load_checkpoint(str(best_path))
        
        # Fallback to best from history
        best_checkpoints = [c for c in self.checkpoints if c.is_best]
        if best_checkpoints:
            return self._load_checkpoint(best_checkpoints[-1].path)
        
        return None
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint"""
        return self._load_checkpoint(checkpoint_path)
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about saved checkpoints"""
        return {
            "total_checkpoints": len(self.checkpoints),
            "best_checkpoint": next((c for c in self.checkpoints if c.is_best), None),
            "latest_checkpoint": max(self.checkpoints, key=lambda c: c.step) if self.checkpoints else None,
            "checkpoint_dir": str(self.checkpoint_dir)
        }
    
    def _is_best(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are best so far"""
        if not self.keep_best:
            return False
        
        # Calculate combined metric
        metric_value = 0.0
        for metric_name in self.metrics_to_track:
            if metric_name in metrics:
                metric_value += metrics[metric_name]
        
        if metric_value > self.best_metric_value:
            self.best_metric_value = metric_value
            return True
        
        return False
    
    def _save_best_model(self, checkpoint_path: Path, metrics: Dict[str, float]):
        """Save the best model"""
        best_path = self.checkpoint_dir / "best_model.pt"
        shutil.copy2(checkpoint_path, best_path)
        
        # Save best metrics
        best_info_path = self.checkpoint_dir / "best_model_info.json"
        with open(best_info_path, 'w') as f:
            json.dump({
                "source_checkpoint": str(checkpoint_path),
                "metrics": metrics,
                "timestamp": time.time()
            }, f, indent=2)
        
        logger.info(f"Saved best model: {best_path} (metrics: {metrics})")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints according to retention policy"""
        if self.keep_recent <= 0:
            return
        
        # Sort checkpoints by step
        sorted_checkpoints = sorted(self.checkpoints, key=lambda c: c.step)
        
        # Identify checkpoints to remove
        to_remove = []
        
        # Keep only recent checkpoints (excluding best)
        non_best = [c for c in sorted_checkpoints if not c.is_best]
        if len(non_best) > self.keep_recent:
            to_remove.extend(non_best[:-self.keep_recent])
        
        # Remove checkpoints
        for checkpoint in to_remove:
            try:
                if os.path.exists(checkpoint.path):
                    os.remove(checkpoint.path)
                    logger.info(f"Removed old checkpoint: {checkpoint.path}")
                self.checkpoints.remove(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint.path}: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint from disk"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to disk"""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        
        history = {
            "checkpoints": [asdict(c) for c in self.checkpoints],
            "best_metric_value": self.best_metric_value,
            "last_save_time": self.last_save_time
        }
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from disk"""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        
        if not history_path.exists():
            return
        
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            self.checkpoints = [
                CheckpointInfo(**c) for c in history.get("checkpoints", [])
            ]
            self.best_metric_value = history.get("best_metric_value", -float('inf'))
            self.last_save_time = history.get("last_save_time", 0.0)
            
            # Verify checkpoint files exist
            self.checkpoints = [
                c for c in self.checkpoints 
                if os.path.exists(c.path)
            ]
            
            logger.info(f"Loaded {len(self.checkpoints)} checkpoints from history")
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint history: {e}")
