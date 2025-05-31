"""
Self-play manager for training agents against themselves or past versions.

This module implements infrastructure for agents to play against
each other or historical versions of themselves, which is crucial
for learning competitive strategies.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import random
import time
import logging
from pathlib import Path
import pickle

from ..models import BlackholioModel
from ..environment import BlackholioEnv, BlackholioEnvConfig

logger = logging.getLogger(__name__)


@dataclass
class OpponentConfig:
    """Configuration for opponent selection"""
    # Opponent pool settings
    pool_size: int = 10  # Max number of historical models to keep
    save_interval: int = 10000  # Steps between saving to pool
    
    # Opponent selection strategy
    selection_strategy: str = "uniform"  # uniform, weighted, latest
    self_play_prob: float = 0.5  # Probability of playing against current self
    
    # Performance tracking
    track_win_rates: bool = True
    min_games_for_stats: int = 100


class ModelPool:
    """
    Pool of historical models for opponent selection.
    
    Maintains a collection of past model checkpoints that can be
    used as opponents during training.
    """
    
    def __init__(self, pool_dir: str, max_size: int = 10):
        """
        Initialize model pool.
        
        Args:
            pool_dir: Directory to store model checkpoints
            max_size: Maximum number of models to keep
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        
        # Model metadata
        self.models: List[Dict[str, Any]] = []
        self.load_existing_models()
        
    def add_model(self, model: BlackholioModel, step: int, 
                  performance_stats: Optional[Dict[str, float]] = None):
        """
        Add a model to the pool.
        
        Args:
            model: Model to add
            step: Training step when model was saved
            performance_stats: Optional performance metrics
        """
        # Create unique filename
        timestamp = int(time.time())
        filename = f"opponent_step{step}_{timestamp}.pth"
        filepath = self.pool_dir / filename
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.config,
            'step': step,
            'timestamp': timestamp,
            'performance_stats': performance_stats or {}
        }, filepath)
        
        # Add to metadata
        self.models.append({
            'filename': filename,
            'step': step,
            'timestamp': timestamp,
            'performance_stats': performance_stats or {},
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0
        })
        
        # Remove oldest if over capacity
        if len(self.models) > self.max_size:
            oldest = min(self.models, key=lambda x: x['timestamp'])
            self._remove_model(oldest)
        
        self._save_metadata()
        logger.info(f"Added model to pool: {filename} (pool size: {len(self.models)})")
    
    def get_opponent(self, strategy: str = "uniform", 
                    exclude_latest: bool = False) -> Optional[Tuple[BlackholioModel, Dict[str, Any]]]:
        """
        Get an opponent model from the pool.
        
        Args:
            strategy: Selection strategy (uniform, weighted, latest)
            exclude_latest: Whether to exclude the most recent model
            
        Returns:
            Tuple of (model, metadata) or None if pool is empty
        """
        if not self.models:
            return None
        
        candidates = self.models.copy()
        if exclude_latest and len(candidates) > 1:
            # Sort by step and remove latest
            candidates.sort(key=lambda x: x['step'])
            candidates = candidates[:-1]
        
        if strategy == "uniform":
            # Uniform random selection
            metadata = random.choice(candidates)
        elif strategy == "weighted":
            # Weight by inverse of games played (explore less-used models)
            weights = [1.0 / (m['games_played'] + 1) for m in candidates]
            metadata = random.choices(candidates, weights=weights)[0]
        elif strategy == "latest":
            # Always pick the most recent
            metadata = max(candidates, key=lambda x: x['step'])
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        # Load model
        filepath = self.pool_dir / metadata['filename']
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model = BlackholioModel(checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, metadata
    
    def update_stats(self, model_metadata: Dict[str, Any], 
                    result: str, opponent_rating: Optional[float] = None):
        """
        Update statistics for a model after a game.
        
        Args:
            model_metadata: Metadata of the model that played
            result: Game result ('win', 'loss', 'draw')
            opponent_rating: Optional rating of the opponent
        """
        # Find model in pool
        for model in self.models:
            if model['filename'] == model_metadata['filename']:
                model['games_played'] += 1
                if result == 'win':
                    model['wins'] += 1
                elif result == 'loss':
                    model['losses'] += 1
                else:
                    model['draws'] += 1
                
                # Update win rate
                if model['games_played'] > 0:
                    model['win_rate'] = model['wins'] / model['games_played']
                
                break
        
        self._save_metadata()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        if not self.models:
            return {
                'pool_size': 0,
                'total_games': 0,
                'avg_win_rate': 0.0
            }
        
        total_games = sum(m['games_played'] for m in self.models)
        avg_win_rate = np.mean([m.get('win_rate', 0.0) for m in self.models])
        
        return {
            'pool_size': len(self.models),
            'total_games': total_games,
            'avg_win_rate': avg_win_rate,
            'model_steps': [m['step'] for m in self.models],
            'games_distribution': [m['games_played'] for m in self.models]
        }
    
    def load_existing_models(self):
        """Load existing model metadata from disk"""
        metadata_file = self.pool_dir / "pool_metadata.pkl"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    self.models = pickle.load(f)
                logger.info(f"Loaded {len(self.models)} models from pool")
            except Exception as e:
                logger.warning(f"Failed to load model pool metadata: {e}")
                self.models = []
    
    def _save_metadata(self):
        """Save model metadata to disk"""
        metadata_file = self.pool_dir / "pool_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.models, f)
    
    def _remove_model(self, model_metadata: Dict[str, Any]):
        """Remove a model from the pool"""
        filepath = self.pool_dir / model_metadata['filename']
        if filepath.exists():
            filepath.unlink()
        self.models.remove(model_metadata)


class SelfPlayManager:
    """
    Manages self-play training for Blackholio agents.
    
    Coordinates multiple agents playing against each other or
    historical versions, tracking performance and managing the
    opponent pool.
    """
    
    def __init__(self, 
                 pool_dir: str,
                 config: Optional[OpponentConfig] = None):
        """
        Initialize self-play manager.
        
        Args:
            pool_dir: Directory for model pool
            config: Opponent configuration
        """
        self.config = config or OpponentConfig()
        self.model_pool = ModelPool(pool_dir, self.config.pool_size)
        
        # Performance tracking
        self.current_model_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': 0.0,
            'recent_results': deque(maxlen=100)
        }
        
        # Matchmaking queue
        self.matchmaking_queue = []
        
        logger.info(f"SelfPlayManager initialized with pool at: {pool_dir}")
    
    def should_save_to_pool(self, step: int) -> bool:
        """Check if current model should be saved to pool"""
        return step % self.config.save_interval == 0
    
    def save_current_model(self, model: BlackholioModel, step: int):
        """Save current model to opponent pool"""
        performance_stats = {
            'win_rate': self.current_model_stats['win_rate'],
            'games_played': self.current_model_stats['games_played']
        }
        self.model_pool.add_model(model, step, performance_stats)
    
    def get_opponent_model(self, current_model: Optional[BlackholioModel] = None) -> Tuple[BlackholioModel, str]:
        """
        Get an opponent for training.
        
        Args:
            current_model: Current model (for self-play)
            
        Returns:
            Tuple of (opponent_model, opponent_type)
        """
        # Decide whether to use self-play or historical opponent
        use_self_play = random.random() < self.config.self_play_prob
        
        if use_self_play and current_model is not None:
            # Play against current self
            opponent = current_model
            opponent_type = "self"
        else:
            # Try to get historical opponent
            result = self.model_pool.get_opponent(
                strategy=self.config.selection_strategy,
                exclude_latest=True
            )
            
            if result is not None:
                opponent, metadata = result
                opponent_type = f"historical_step{metadata['step']}"
            elif current_model is not None:
                # Fallback to self-play if no historical models
                opponent = current_model
                opponent_type = "self"
            else:
                raise ValueError("No opponent available")
        
        return opponent, opponent_type
    
    def create_match(self, 
                    env_config: BlackholioEnvConfig,
                    models: List[BlackholioModel],
                    agent_names: List[str]) -> Dict[str, Any]:
        """
        Create a match between agents.
        
        Args:
            env_config: Environment configuration
            models: List of models for each agent
            agent_names: Names for each agent
            
        Returns:
            Match configuration dictionary
        """
        match_id = f"match_{int(time.time())}_{random.randint(1000, 9999)}"
        
        match_config = {
            'match_id': match_id,
            'env_config': env_config,
            'agents': []
        }
        
        for i, (model, name) in enumerate(zip(models, agent_names)):
            match_config['agents'].append({
                'id': i,
                'name': name,
                'model': model,
                'type': 'self' if i == 0 else 'opponent'
            })
        
        return match_config
    
    def update_game_result(self, 
                         agent_id: int,
                         result: str,
                         opponent_type: str,
                         game_stats: Optional[Dict[str, Any]] = None):
        """
        Update statistics after a game.
        
        Args:
            agent_id: ID of the agent (0 for current model)
            result: Game result ('win', 'loss', 'draw')
            opponent_type: Type of opponent faced
            game_stats: Optional additional game statistics
        """
        if agent_id == 0:  # Current model
            self.current_model_stats['games_played'] += 1
            if result == 'win':
                self.current_model_stats['wins'] += 1
            elif result == 'loss':
                self.current_model_stats['losses'] += 1
            else:
                self.current_model_stats['draws'] += 1
            
            # Update win rate
            if self.current_model_stats['games_played'] > 0:
                self.current_model_stats['win_rate'] = (
                    self.current_model_stats['wins'] / 
                    self.current_model_stats['games_played']
                )
            
            # Track recent results
            self.current_model_stats['recent_results'].append({
                'result': result,
                'opponent_type': opponent_type,
                'timestamp': time.time(),
                'game_stats': game_stats or {}
            })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-play statistics"""
        pool_stats = self.model_pool.get_statistics()
        
        # Calculate recent performance
        recent_results = list(self.current_model_stats['recent_results'])
        recent_wins = sum(1 for r in recent_results if r['result'] == 'win')
        recent_win_rate = recent_wins / len(recent_results) if recent_results else 0.0
        
        return {
            'current_model': {
                'games_played': self.current_model_stats['games_played'],
                'win_rate': self.current_model_stats['win_rate'],
                'recent_win_rate': recent_win_rate,
                'wins': self.current_model_stats['wins'],
                'losses': self.current_model_stats['losses'],
                'draws': self.current_model_stats['draws']
            },
            'model_pool': pool_stats,
            'opponent_distribution': self._get_opponent_distribution()
        }
    
    def _get_opponent_distribution(self) -> Dict[str, int]:
        """Get distribution of opponent types faced"""
        distribution = {}
        for result in self.current_model_stats['recent_results']:
            opponent_type = result['opponent_type']
            distribution[opponent_type] = distribution.get(opponent_type, 0) + 1
        return distribution
    
    def reset_current_stats(self):
        """Reset current model statistics (e.g., after saving to pool)"""
        self.current_model_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': 0.0,
            'recent_results': deque(maxlen=100)
        }
