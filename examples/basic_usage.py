"""
Basic usage example for Blackholio ML environment.

This script demonstrates how to:
1. Create and configure the environment
2. Run random actions
3. Collect observations and rewards
"""

import asyncio
import numpy as np
import logging
from blackholio_agent import BlackholioEnv, BlackholioEnvConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """Run a basic example of the Blackholio environment."""
    
    # Create environment configuration
    config = BlackholioEnvConfig(
        host="localhost:3000",
        database="blackholio",
        player_name="ML_Agent_Example",
        render_mode="human",  # Print game state to console
        max_episode_steps=1000
    )
    
    # Create environment
    env = BlackholioEnv(config)
    
    try:
        # Run a few episodes
        for episode in range(3):
            print(f"\n{'='*50}")
            print(f"Starting Episode {episode + 1}")
            print(f"{'='*50}\n")
            
            # Reset environment
            obs, info = await env.async_reset()
            print(f"Initial observation shape: {obs.shape}")
            print(f"Player ID: {info.get('player_id')}")
            
            episode_reward = 0.0
            done = False
            step = 0
            
            while not done:
                # Take random action
                action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = await env.async_step(action)
                done = terminated or truncated
                
                episode_reward += reward
                step += 1
                
                # Render every 20 steps
                if step % 20 == 0:
                    env.render()
                
                # Log significant events
                if reward > 1.0 or reward < -1.0:
                    print(f"Step {step}: Significant reward: {reward:.2f}")
                    print(f"  Reward components: {info.get('reward_components', {})}")
                
                if done:
                    print(f"\nEpisode ended after {step} steps")
                    print(f"Total reward: {episode_reward:.2f}")
                    if 'episode_stats' in info:
                        stats = info['episode_stats']
                        print(f"Episode statistics:")
                        print(f"  - Mass gained: {stats.get('mass_gained', 0):.1f}")
                        print(f"  - Food collected: {stats.get('food_collected', 0)}")
                        print(f"  - Kills: {stats.get('kills', 0)}")
                        print(f"  - Deaths: {stats.get('deaths', 0)}")
                        print(f"  - Max mass: {stats.get('max_mass', 0):.1f}")
                        print(f"  - Survival time: {stats.get('survival_time', 0):.1f}s")
            
            # Small delay between episodes
            await asyncio.sleep(2.0)
    
    finally:
        # Always close the environment
        env.close()
        print("\nEnvironment closed.")


def run_sync_example():
    """Run a synchronous example (for environments without async support)."""
    config = BlackholioEnvConfig(
        host="localhost:3000",
        database="blackholio",
        player_name="ML_Agent_Sync",
        max_episode_steps=500
    )
    
    env = BlackholioEnv(config)
    
    try:
        # Reset environment
        obs, info = env.reset()
        
        # Run for a few steps
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
            
            if step % 20 == 0:
                print(f"Step {step}, Reward: {reward:.2f}")
    
    finally:
        env.close()


if __name__ == "__main__":
    print("Running Blackholio Environment Example")
    print("Make sure Blackholio game server is running on localhost:3000")
    print()
    
    # Run async example
    asyncio.run(main())
    
    # Uncomment to run sync example instead
    # run_sync_example()
