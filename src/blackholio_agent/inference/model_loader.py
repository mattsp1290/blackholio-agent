"""
Model loading and optimization for inference.

This module handles loading trained models and optimizing them
for CPU inference.
"""

import torch
import logging
import time
from typing import Optional, Dict, Any
import numpy as np

from ..models import BlackholioModel, BlackholioModelConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles model loading and optimization for inference.
    
    Provides CPU-specific optimizations and warmup procedures.
    """
    
    @staticmethod
    def load_for_inference(model_path: str, 
                          device: str = 'cpu',
                          num_threads: int = 4) -> BlackholioModel:
        """
        Load and optimize model for inference.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to load model on ('cpu' or 'cuda')
            num_threads: Number of CPU threads to use
            
        Returns:
            Optimized model ready for inference
        """
        logger.info(f"Loading model from {model_path}")
        
        # Set CPU threading
        if device == 'cpu':
            torch.set_num_threads(num_threads)
            logger.info(f"Set CPU threads to {num_threads}")
        
        # Load model
        try:
            model = BlackholioModel.load(model_path, device=device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Put in eval mode
        model.eval()
        
        # Optimize for inference
        model = ModelLoader._optimize_model(model, device)
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {param_count:,} parameters")
        
        return model
    
    @staticmethod
    def _optimize_model(model: BlackholioModel, device: str) -> BlackholioModel:
        """
        Apply optimizations to the model.
        
        Args:
            model: Model to optimize
            device: Target device
            
        Returns:
            Optimized model
        """
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # CPU-specific optimizations
        if device == 'cpu':
            # Use torch.jit.script for CPU if available
            try:
                # Create dummy input for tracing
                dummy_input = torch.randn(1, model.config.observation_dim)
                
                # Try to compile with torch.compile if available (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    logger.info("Compiling model with torch.compile")
                    model = torch.compile(model, mode='reduce-overhead')
                else:
                    logger.info("torch.compile not available, using standard model")
                
            except Exception as e:
                logger.warning(f"Model optimization failed: {e}")
                logger.info("Using unoptimized model")
        
        return model
    
    @staticmethod
    def warmup_model(model: BlackholioModel, 
                    warmup_steps: int = 10,
                    device: str = 'cpu') -> Dict[str, float]:
        """
        Warmup model with dummy inputs.
        
        This ensures all lazy initializations are done and 
        caches are warmed up before real inference.
        
        Args:
            model: Model to warmup
            warmup_steps: Number of warmup iterations
            device: Device to run on
            
        Returns:
            Warmup statistics
        """
        logger.info(f"Warming up model with {warmup_steps} steps")
        
        # Create dummy input
        dummy_obs = torch.randn(1, model.config.observation_dim, device=device)
        
        warmup_times = []
        
        with torch.no_grad():
            for i in range(warmup_steps):
                start_time = time.time()
                
                # Run inference
                actions, output = model.get_action(dummy_obs, deterministic=True)
                
                # Force computation
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start_time
                warmup_times.append(elapsed)
                
                if i == 0:
                    logger.debug(f"First warmup step: {elapsed*1000:.2f}ms")
        
        # Calculate statistics
        avg_time = np.mean(warmup_times)
        std_time = np.std(warmup_times)
        min_time = np.min(warmup_times)
        max_time = np.max(warmup_times)
        
        stats = {
            'avg_ms': avg_time * 1000,
            'std_ms': std_time * 1000,
            'min_ms': min_time * 1000,
            'max_ms': max_time * 1000
        }
        
        logger.info(f"Warmup complete. Avg inference: {stats['avg_ms']:.2f}ms")
        
        return stats
    
    @staticmethod
    def validate_model(model: BlackholioModel, 
                      observation_dim: int = 456) -> bool:
        """
        Validate that model works correctly.
        
        Args:
            model: Model to validate
            observation_dim: Expected observation dimension
            
        Returns:
            True if model is valid
        """
        try:
            # Check observation dimension
            if model.config.observation_dim != observation_dim:
                logger.error(f"Model expects {model.config.observation_dim} dims, "
                           f"but environment has {observation_dim}")
                return False
            
            # Test forward pass
            with torch.no_grad():
                test_obs = torch.randn(1, observation_dim)
                actions, output = model.get_action(test_obs, deterministic=True)
            
            # Validate output shapes
            if actions['movement'].shape != (1, 2):
                logger.error(f"Invalid movement shape: {actions['movement'].shape}")
                return False
            
            if actions['split'].shape != (1,):
                logger.error(f"Invalid split shape: {actions['split'].shape}")
                return False
            
            # Check value bounds
            if not torch.all((actions['movement'] >= -1) & (actions['movement'] <= 1)):
                logger.error("Movement actions out of bounds")
                return False
            
            logger.info("Model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    @staticmethod
    def get_model_info(model: BlackholioModel) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Args:
            model: Model to inspect
            
        Returns:
            Dictionary with model information
        """
        config = model.config
        
        info = {
            'architecture': {
                'observation_dim': config.observation_dim,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'use_attention': config.use_attention,
                'use_lstm': config.use_lstm,
                'use_spatial_features': config.use_spatial_features
            },
            'parameters': {
                'total': sum(p.numel() for p in model.parameters()),
                'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)
            },
            'device': str(next(model.parameters()).device),
            'dtype': str(next(model.parameters()).dtype)
        }
        
        return info
