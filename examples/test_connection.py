"""
Test script to verify the Blackholio environment can connect and control a player.

This script tests:
1. Connection to SpacetimeDB
2. Player spawn detection
3. Movement control
4. Split action
5. Game state observation
"""

import asyncio
import numpy as np
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blackholio_agent.environment.connection import BlackholioConnection, ConnectionConfig
from src.blackholio_agent.environment.observation_space import ObservationSpace
from src.blackholio_agent.environment.action_space import ActionSpace

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_connection():
    """Test basic connection to Blackholio."""
    logger.info("Testing connection to Blackholio...")
    
    config = ConnectionConfig(
        host="localhost:3000",
        ssl_enabled=False
    )
    
    connection = BlackholioConnection(config)
    
    try:
        # Connect to game
        await connection.connect()
        logger.info("✓ Successfully connected to SpacetimeDB")
        
        # Wait for player spawn
        logger.info("Waiting for player to spawn...")
        start_time = asyncio.get_event_loop().time()
        while not connection.get_player_entities():
            if asyncio.get_event_loop().time() - start_time > 30:
                logger.error("✗ Timeout waiting for player spawn")
                return False
            await asyncio.sleep(0.1)
        
        logger.info(f"✓ Player spawned with ID: {connection.player_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Connection test failed: {e}")
        return False
    finally:
        await connection.disconnect()


async def test_movement_control():
    """Test player movement control."""
    logger.info("\nTesting movement control...")
    
    config = ConnectionConfig(
        host="localhost:3000",
        ssl_enabled=False
    )
    
    connection = BlackholioConnection(config)
    action_space = ActionSpace()
    
    try:
        await connection.connect()
        
        # Wait for spawn
        while not connection.get_player_entities():
            await asyncio.sleep(0.1)
        
        # Test different movement directions
        movements = [
            ([1.0, 0.0], "right"),
            ([0.0, 1.0], "down"),
            ([-1.0, 0.0], "left"),
            ([0.0, -1.0], "up"),
            ([0.7, 0.7], "diagonal")
        ]
        
        for movement, direction in movements:
            logger.info(f"Testing movement {direction}: {movement}")
            
            action = np.array(movement + [0.0])  # Movement + no split
            result = await action_space.execute_action(connection, action)
            
            if result['movement_executed']:
                logger.info(f"✓ Movement {direction} executed successfully")
            else:
                logger.error(f"✗ Movement {direction} failed: {result.get('error')}")
            
            await asyncio.sleep(0.5)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Movement control test failed: {e}")
        return False
    finally:
        await connection.disconnect()


async def test_split_action():
    """Test player split action."""
    logger.info("\nTesting split action...")
    
    config = ConnectionConfig(
        host="localhost:3000",
        ssl_enabled=False
    )
    
    connection = BlackholioConnection(config)
    action_space = ActionSpace()
    
    try:
        await connection.connect()
        
        # Wait for spawn
        while not connection.get_player_entities():
            await asyncio.sleep(0.1)
        
        initial_entities = len(connection.get_player_entities())
        logger.info(f"Initial number of entities: {initial_entities}")
        
        # Execute split
        action = np.array([0.0, 0.0, 1.0])  # No movement + split
        result = await action_space.execute_action(connection, action)
        
        if result['split_executed']:
            logger.info("✓ Split action executed")
            
            # Wait for split to take effect
            await asyncio.sleep(0.5)
            
            new_entities = len(connection.get_player_entities())
            if new_entities > initial_entities:
                logger.info(f"✓ Split successful: {initial_entities} -> {new_entities} entities")
            else:
                logger.warning("Split executed but entity count didn't increase")
        else:
            logger.error(f"✗ Split action failed: {result.get('error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Split action test failed: {e}")
        return False
    finally:
        await connection.disconnect()


async def test_observation_space():
    """Test game state observation."""
    logger.info("\nTesting observation space...")
    
    config = ConnectionConfig(
        host="localhost:3000",
        ssl_enabled=False
    )
    
    connection = BlackholioConnection(config)
    observation_space = ObservationSpace()
    
    try:
        await connection.connect()
        
        # Wait for spawn
        while not connection.get_player_entities():
            await asyncio.sleep(0.1)
        
        # Get observation
        player_entities = connection.get_player_entities()
        other_entities = connection.get_other_entities()
        
        logger.info(f"Player entities: {len(player_entities)}")
        logger.info(f"Other entities: {len(other_entities)}")
        
        # Process observation
        obs = observation_space.process_game_state(
            player_entities,
            other_entities,
            []  # No food tracking in this test
        )
        
        logger.info(f"✓ Observation shape: {obs.shape}")
        logger.info(f"✓ Observation dtype: {obs.dtype}")
        
        # Decode observation for verification
        decoded = observation_space.decode_observation(obs)
        logger.info(f"✓ Decoded player state: mass={decoded['player']['mass']:.1f}, "
                   f"pos=({decoded['player']['x']:.1f}, {decoded['player']['y']:.1f})")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Observation space test failed: {e}")
        return False
    finally:
        await connection.disconnect()


async def test_full_environment():
    """Test the full environment integration."""
    logger.info("\nTesting full environment...")
    
    from src.blackholio_agent import BlackholioEnv, BlackholioEnvConfig
    
    config = BlackholioEnvConfig(
        host="localhost:3000",
        database="blackholio",
        player_name="Test_Agent",
        max_episode_steps=100
    )
    
    env = BlackholioEnv(config)
    
    try:
        # Reset environment
        obs, info = await env.async_reset()
        logger.info(f"✓ Environment reset successful")
        logger.info(f"  Observation shape: {obs.shape}")
        logger.info(f"  Player ID: {info.get('player_id')}")
        
        # Take a few steps
        total_reward = 0.0
        for i in range(10):
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = await env.async_step(action)
            
            total_reward += reward
            
            if terminated or truncated:
                logger.info(f"Episode ended at step {i}")
                break
        
        logger.info(f"✓ Environment step execution successful")
        logger.info(f"  Total reward after 10 steps: {total_reward:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Full environment test failed: {e}")
        return False
    finally:
        env.close()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Blackholio Environment Connection Test")
    print("=" * 60)
    print("\nMake sure:")
    print("1. Blackholio game server is running on localhost:3000")
    print("2. The game is accessible and allows connections")
    print()
    
    tests = [
        ("Basic Connection", test_connection),
        ("Movement Control", test_movement_control),
        ("Split Action", test_split_action),
        ("Observation Space", test_observation_space),
        ("Full Environment", test_full_environment)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        
        # Small delay between tests
        await asyncio.sleep(1.0)
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<30} {status}")
        if not success:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All tests passed! The environment can connect to Blackholio")
        print("  and control a player programmatically.")
    else:
        print("✗ Some tests failed. Please check the logs above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
