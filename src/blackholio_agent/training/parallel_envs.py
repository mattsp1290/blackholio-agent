"""
Parallel environment wrapper for running multiple Blackholio agents.

This module manages multiple environment instances that connect to
the same Blackholio server for efficient training.
"""

import asyncio
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import threading
from queue import Queue

from ..environment import BlackholioEnv, BlackholioEnvConfig

logger = logging.getLogger(__name__)


@dataclass
class ParallelEnvConfig:
    """Configuration for parallel environments"""
    n_envs: int = 8
    host: str = "localhost:3000"
    database: Optional[str] = None  # Use BlackholioEnvConfig's default
    agent_name_prefix: str = "ML_Agent"
    max_episode_steps: int = 10000
    step_interval: float = 0.05  # 20Hz
    reset_on_error: bool = True
    connection_timeout: float = 30.0


class AsyncEnvWorker:
    """
    Worker that runs a single environment in its own thread with event loop.
    """
    def __init__(self, env_id: int, config: ParallelEnvConfig):
        self.env_id = env_id
        self.config = config
        self.env: Optional[BlackholioEnv] = None
        
        # Communication queues
        self.command_queue = Queue()
        self.result_queue = Queue()
        
        # Thread and loop
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.running = False
        
        # State
        self.last_obs: Optional[np.ndarray] = None
        self.episode_reward = 0.0
        self.episode_length = 0
        
    def start(self):
        """Start the worker thread"""
        self.running = True
        self.thread.start()
        
        # Wait for initialization
        result = self.result_queue.get()
        if result["type"] == "error":
            raise RuntimeError(f"Worker {self.env_id} failed to start: {result['error']}")
            
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        self.command_queue.put({"type": "stop"})
        self.thread.join(timeout=5.0)
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment"""
        self.command_queue.put({"type": "reset"})
        result = self.result_queue.get()
        
        if result["type"] == "error":
            raise RuntimeError(f"Reset failed: {result['error']}")
            
        self.last_obs = result["obs"]
        self.episode_reward = 0.0
        self.episode_length = 0
        
        return result["obs"], result["info"]
        
    def step(self, action: Union[np.ndarray, Dict[str, Any]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        self.command_queue.put({"type": "step", "action": action})
        result = self.result_queue.get()
        
        if result["type"] == "error":
            if self.config.reset_on_error:
                logger.warning(f"Worker {self.env_id} step failed, resetting: {result['error']}")
                return self.reset()[0], 0.0, True, False, {"error": result["error"]}
            else:
                raise RuntimeError(f"Step failed: {result['error']}")
                
        self.last_obs = result["obs"]
        self.episode_reward += result["reward"]
        self.episode_length += 1
        
        # Add episode stats to info
        result["info"]["episode_reward"] = self.episode_reward
        result["info"]["episode_length"] = self.episode_length
        
        return result["obs"], result["reward"], result["terminated"], result["truncated"], result["info"]
        
    def _run_loop(self):
        """Run the async event loop in this thread"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop()
        
        try:
            # Create environment with unique name
            agent_name = f"{self.config.agent_name_prefix}_{self.env_id}_{int(time.time())}"
            env_config = BlackholioEnvConfig(
                host=self.config.host,
                database=self.config.database,
                player_name=agent_name,
                max_episode_steps=self.config.max_episode_steps,
                step_interval=self.config.step_interval
            )
            self.env = BlackholioEnv(env_config)
            
            # Signal successful initialization
            self.result_queue.put({"type": "initialized"})
            
            # Run command processing loop
            self.loop.run_until_complete(self._process_commands())
            
        except Exception as e:
            logger.error(f"Worker {self.env_id} failed: {e}")
            self.result_queue.put({"type": "error", "error": str(e)})
        finally:
            if self.env:
                self.env.close()
            self.loop.close()
            
    async def _process_commands(self):
        """Process commands from the main thread"""
        while self.running:
            try:
                # Check for commands (non-blocking)
                if not self.command_queue.empty():
                    command = self.command_queue.get_nowait()
                    
                    if command["type"] == "stop":
                        break
                    elif command["type"] == "reset":
                        await self._handle_reset()
                    elif command["type"] == "step":
                        await self._handle_step(command["action"])
                        
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Worker {self.env_id} command processing error: {e}")
                self.result_queue.put({"type": "error", "error": str(e)})
                
    async def _handle_reset(self):
        """Handle reset command"""
        try:
            obs, info = await self.env.async_reset()
            self.result_queue.put({
                "type": "reset",
                "obs": obs,
                "info": info
            })
        except Exception as e:
            self.result_queue.put({"type": "error", "error": str(e)})
            
    async def _handle_step(self, action):
        """Handle step command"""
        try:
            obs, reward, terminated, truncated, info = await self.env.async_step(action)
            self.result_queue.put({
                "type": "step",
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            })
        except Exception as e:
            self.result_queue.put({"type": "error", "error": str(e)})


class ParallelBlackholioEnv:
    """
    Manages multiple Blackholio environments running in parallel.
    
    Each environment runs in its own thread with async support,
    connecting to the same Blackholio server with unique agent names.
    """
    
    def __init__(self, config: Union[ParallelEnvConfig, Dict[str, Any]] = None):
        """
        Initialize parallel environments.
        
        Args:
            config: Configuration for parallel environments
        """
        if config is None:
            self.config = ParallelEnvConfig()
        elif isinstance(config, dict):
            self.config = ParallelEnvConfig(**config)
        else:
            self.config = config
            
        self.n_envs = self.config.n_envs
        self.workers: List[AsyncEnvWorker] = []
        
        # Observation and action shapes
        self.observation_shape = (456,)  # From ObservationSpace
        self.action_shape = (3,)  # 2D movement + split
        
        # Episode tracking
        self.episode_rewards = np.zeros(self.n_envs)
        self.episode_lengths = np.zeros(self.n_envs, dtype=int)
        
        logger.info(f"Initializing {self.n_envs} parallel Blackholio environments")
        
    def start(self):
        """Start all environment workers"""
        logger.info(f"Starting {self.n_envs} environment workers...")
        
        for i in range(self.n_envs):
            worker = AsyncEnvWorker(i, self.config)
            worker.start()
            self.workers.append(worker)
            logger.info(f"Worker {i} started")
            
        logger.info("All environment workers started successfully")
        
    def reset(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Reset all environments.
        
        Returns:
            Tuple of (observations, infos) where observations is shape [n_envs, obs_dim]
        """
        observations = np.zeros((self.n_envs, *self.observation_shape), dtype=np.float32)
        infos = []
        
        # Reset all environments in parallel
        for i, worker in enumerate(self.workers):
            obs, info = worker.reset()
            observations[i] = obs
            infos.append(info)
            
        # Reset episode tracking
        self.episode_rewards.fill(0)
        self.episode_lengths.fill(0)
        
        return observations, infos
        
    def step(self, actions: Union[np.ndarray, List[Dict[str, Any]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments with given actions.
        
        Args:
            actions: Either array of shape [n_envs, action_dim] or list of action dicts
            
        Returns:
            Tuple of (observations, rewards, dones, truncated, infos)
        """
        observations = np.zeros((self.n_envs, *self.observation_shape), dtype=np.float32)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        infos = []
        
        # Convert actions to appropriate format
        if isinstance(actions, np.ndarray):
            # Convert numpy array to action dicts
            action_dicts = []
            for i in range(self.n_envs):
                if actions.shape[1] == 3:  # movement + split
                    action_dict = {
                        "movement": actions[i, :2],
                        "split": actions[i, 2]
                    }
                else:  # just movement
                    action_dict = actions[i]
                action_dicts.append(action_dict)
        else:
            action_dicts = actions
            
        # Step all environments
        for i, (worker, action) in enumerate(zip(self.workers, action_dicts)):
            obs, reward, terminated, trunc, info = worker.step(action)
            
            observations[i] = obs
            rewards[i] = reward
            dones[i] = terminated
            truncated[i] = trunc
            infos.append(info)
            
            # Track episodes
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1
            
            # Reset tracking on episode end
            if terminated or trunc:
                info["episode"] = {
                    "r": self.episode_rewards[i],
                    "l": self.episode_lengths[i]
                }
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0
                
        return observations, rewards, dones, truncated, infos
        
    def close(self):
        """Close all environments"""
        logger.info("Closing parallel environments...")
        
        for i, worker in enumerate(self.workers):
            try:
                worker.stop()
                logger.info(f"Worker {i} stopped")
            except Exception as e:
                logger.error(f"Error stopping worker {i}: {e}")
                
        self.workers.clear()
        logger.info("All environments closed")
        
    def get_env_stats(self) -> Dict[str, Any]:
        """Get statistics from all environments"""
        active_envs = sum(1 for w in self.workers if w.running)
        
        return {
            "n_envs": self.n_envs,
            "active_envs": active_envs,
            "mean_episode_reward": np.mean(self.episode_rewards),
            "mean_episode_length": np.mean(self.episode_lengths),
            "total_steps": np.sum([w.episode_length for w in self.workers])
        }
