#!/usr/bin/env python3
"""
Example demonstrating advanced training features:
- Self-play training
- Adaptive curriculum learning
- Performance monitoring
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.blackholio_agent.training import (
    PPOTrainer, PPOConfig,
    CurriculumManager, SelfPlayManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run advanced training example"""
    
    # Create training configuration with all features enabled
    config = PPOConfig(
        # Short training for demo
        total_timesteps=100_000,
        n_envs=4,
        n_steps=512,
        batch_size=128,
        
        # Enable curriculum learning
        use_curriculum=True,
        adaptive_curriculum=True,
        
        # Enable self-play
        use_self_play=True,
        self_play_config={
            "pool_size": 5,
            "save_interval": 10000,
            "selection_strategy": "uniform",
            "self_play_prob": 0.5
        },
        opponent_pool_dir="demo_opponent_pool",
        
        # Logging
        log_dir="logs/advanced_demo",
        checkpoint_dir="checkpoints/advanced_demo",
        console_log_interval=5
    )
    
    print("=" * 60)
    print("Blackholio Advanced Training Demo")
    print("=" * 60)
    print("\nFeatures enabled:")
    print("✓ Self-play training with opponent pool")
    print("✓ Adaptive curriculum learning (5 stages)")
    print("✓ Performance-based stage transitions")
    print("✓ TensorBoard logging")
    print("✓ Automatic checkpointing")
    print("\n" + "=" * 60)
    
    # Create trainer
    print("\nInitializing PPO trainer...")
    trainer = PPOTrainer(config)
    
    # Print initial state
    if trainer.curriculum_manager:
        curr_stats = trainer.curriculum_manager.get_statistics()
        print(f"\nStarting curriculum stage: {curr_stats['current_stage']}")
    
    if trainer.self_play_manager:
        sp_stats = trainer.self_play_manager.get_statistics()
        print(f"Opponent pool size: {sp_stats['model_pool']['pool_size']}")
    
    print("\nStarting training...")
    print("Monitor progress with: tensorboard --logdir logs/advanced_demo")
    print("\nPress Ctrl+C to stop training\n")
    
    try:
        # Run training
        trainer.train()
        
        print("\nTraining completed!")
        
        # Print final statistics
        if trainer.curriculum_manager:
            curr_stats = trainer.curriculum_manager.get_statistics()
            print(f"\nFinal curriculum stage: {curr_stats['current_stage']}")
            print(f"Stage transitions: {curr_stats['promotions']} promotions, "
                  f"{curr_stats['demotions']} demotions")
        
        if trainer.self_play_manager:
            sp_stats = trainer.self_play_manager.get_statistics()
            print(f"\nSelf-play statistics:")
            print(f"- Current model win rate: {sp_stats['current_model']['win_rate']:.2%}")
            print(f"- Games played: {sp_stats['current_model']['games_played']}")
            print(f"- Opponent pool size: {sp_stats['model_pool']['pool_size']}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
