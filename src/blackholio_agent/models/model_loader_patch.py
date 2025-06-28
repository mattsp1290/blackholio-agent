"""
Patch for BlackholioModel to handle both training and inference checkpoint formats.

This module provides a patched version of the load method that can handle:
1. Direct model saves (with BlackholioModelConfig)
2. Training checkpoints (with PPOConfig)
"""

import torch
import logging
from typing import Optional, Union
from pathlib import Path

from .blackholio_model import BlackholioModel, BlackholioModelConfig
from ..training.ppo_trainer import PPOConfig

logger = logging.getLogger(__name__)


def load_model_universal(checkpoint_path: str, device: Optional[str] = None) -> BlackholioModel:
    """
    Load a BlackholioModel from either training or inference checkpoint format.
    
    This function handles:
    1. Inference format: checkpoint with 'config' containing BlackholioModelConfig
    2. Training format: checkpoint with 'config' containing PPOConfig or in additional_state
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on (optional)
        
    Returns:
        BlackholioModel instance
    """
    logger.info(f"Loading model from {checkpoint_path} (universal loader)")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model config
    model_config = None
    
    # Case 1: Direct model save (inference format)
    if 'config' in checkpoint and isinstance(checkpoint['config'], BlackholioModelConfig):
        model_config = checkpoint['config']
        logger.info("Found BlackholioModelConfig in checkpoint (inference format)")
    
    # Case 2: Training checkpoint with PPOConfig
    elif 'config' in checkpoint and isinstance(checkpoint['config'], PPOConfig):
        ppo_config = checkpoint['config']
        if hasattr(ppo_config, 'model_config') and ppo_config.model_config:
            model_config = ppo_config.model_config
            logger.info("Extracted BlackholioModelConfig from PPOConfig")
    
    # Case 3: Check additional_state
    elif 'additional_state' in checkpoint and 'config' in checkpoint['additional_state']:
        additional_config = checkpoint['additional_state']['config']
        if isinstance(additional_config, PPOConfig):
            if hasattr(additional_config, 'model_config') and additional_config.model_config:
                model_config = additional_config.model_config
                logger.info("Extracted BlackholioModelConfig from additional_state")
    
    # Case 4: No config found, use defaults
    if model_config is None:
        logger.warning("No model config found in checkpoint, using default BlackholioModelConfig")
        model_config = BlackholioModelConfig()
    
    # Override device if specified
    if device is not None:
        model_config.device = device
    
    # Create model
    model = BlackholioModel(model_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume the checkpoint IS the state dict (old format)
        model.load_state_dict(checkpoint)
    
    # Move to device
    model.to(model.device)
    
    logger.info(f"Model loaded successfully (device: {model.device})")
    return model


# Monkey patch the original load method to use our universal loader
_original_load = BlackholioModel.load

@classmethod
def patched_load(cls, path: str, device: Optional[str] = None) -> 'BlackholioModel':
    """Patched load method that handles both checkpoint formats"""
    try:
        # Try universal loader first
        return load_model_universal(path, device)
    except Exception as e:
        logger.warning(f"Universal loader failed: {e}, trying original loader")
        # Fall back to original loader
        return _original_load(path, device)

# Apply the patch
BlackholioModel.load = patched_load


def save_for_inference(model: BlackholioModel, path: str):
    """
    Save model in inference-ready format.
    
    This ensures the saved checkpoint can be loaded by the standard
    BlackholioModel.load() method without the patch.
    
    Args:
        model: Model to save
        path: Path to save the model
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config
    }
    torch.save(checkpoint, path)
    logger.info(f"Model saved in inference format to {path}")