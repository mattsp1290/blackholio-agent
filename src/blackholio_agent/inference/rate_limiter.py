"""
Rate limiter for inference system.

This module provides rate limiting to ensure we don't exceed
the desired update frequency and spam the API.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter to enforce minimum time between actions.
    
    This ensures we don't spam the SpacetimeDB API and maintains
    a consistent update rate.
    """
    
    def __init__(self, min_interval_ms: float = 50.0):
        """
        Initialize rate limiter.
        
        Args:
            min_interval_ms: Minimum time between calls in milliseconds
        """
        self.min_interval = min_interval_ms / 1000.0  # Convert to seconds
        self.last_call: Optional[float] = None
        self.total_waits = 0
        self.total_wait_time = 0.0
        
        logger.info(f"Rate limiter initialized with {min_interval_ms}ms minimum interval")
    
    def wait(self) -> float:
        """
        Wait if necessary to maintain rate limit.
        
        Returns:
            Time waited in seconds
        """
        current_time = time.time()
        
        if self.last_call is None:
            self.last_call = current_time
            return 0.0
        
        elapsed = current_time - self.last_call
        wait_time = 0.0
        
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            time.sleep(wait_time)
            self.total_waits += 1
            self.total_wait_time += wait_time
        
        self.last_call = time.time()
        return wait_time
    
    def reset(self):
        """Reset the rate limiter"""
        self.last_call = None
        self.total_waits = 0
        self.total_wait_time = 0.0
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        return {
            'total_waits': self.total_waits,
            'total_wait_time': self.total_wait_time,
            'average_wait_time': self.total_wait_time / max(1, self.total_waits),
            'min_interval_ms': self.min_interval * 1000
        }


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on system performance.
    
    This can increase the interval if we're consistently hitting
    the rate limit, or decrease it if we have headroom.
    """
    
    def __init__(self, 
                 min_interval_ms: float = 50.0,
                 max_interval_ms: float = 200.0,
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive rate limiter.
        
        Args:
            min_interval_ms: Minimum allowed interval
            max_interval_ms: Maximum allowed interval
            adaptation_rate: How quickly to adapt (0-1)
        """
        super().__init__(min_interval_ms)
        self.min_interval_ms = min_interval_ms
        self.max_interval_ms = max_interval_ms
        self.current_interval_ms = min_interval_ms
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.recent_latencies = []
        self.max_history = 100
    
    def add_latency_sample(self, latency_ms: float):
        """Add a latency sample for adaptation"""
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) > self.max_history:
            self.recent_latencies.pop(0)
        
        # Adapt if we have enough samples
        if len(self.recent_latencies) >= 10:
            self._adapt()
    
    def _adapt(self):
        """Adapt the rate limit based on recent performance"""
        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
        
        # If latency is high, increase interval
        if avg_latency > self.current_interval_ms * 0.8:
            new_interval = self.current_interval_ms * (1 + self.adaptation_rate)
            self.current_interval_ms = min(new_interval, self.max_interval_ms)
            self.min_interval = self.current_interval_ms / 1000.0
            logger.debug(f"Increased rate limit to {self.current_interval_ms:.1f}ms")
        
        # If latency is low, decrease interval
        elif avg_latency < self.current_interval_ms * 0.5:
            new_interval = self.current_interval_ms * (1 - self.adaptation_rate)
            self.current_interval_ms = max(new_interval, self.min_interval_ms)
            self.min_interval = self.current_interval_ms / 1000.0
            logger.debug(f"Decreased rate limit to {self.current_interval_ms:.1f}ms")
    
    def get_stats(self) -> dict:
        """Get adaptive rate limiter statistics"""
        stats = super().get_stats()
        stats.update({
            'current_interval_ms': self.current_interval_ms,
            'recent_avg_latency': sum(self.recent_latencies) / max(1, len(self.recent_latencies)),
            'adaptation_samples': len(self.recent_latencies)
        })
        return stats
