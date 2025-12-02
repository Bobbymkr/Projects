"""
Week 7: Distributed State Management with Redis.

Implements distributed state manager for agent state persistence and distributed locking.
"""

import json
import time
import threading
from typing import Any, Dict, Optional
from contextlib import contextmanager

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class DistributedStateManager:
    """Distributed state manager using Redis."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError:
            raise ConnectionError(f"Failed to connect to Redis at {redis_host}:{redis_port}")
    
    def save_agent_state(self, agent_id: str, state: Dict[str, Any], ttl: Optional[int] = None):
        """Save agent state to Redis."""
        key = f"agent_state:{agent_id}"
        value = json.dumps(state)
        
        if ttl:
            self.redis_client.setex(key, ttl, value)
        else:
            self.redis_client.set(key, value)
    
    def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent state from Redis."""
        key = f"agent_state:{agent_id}"
        value = self.redis_client.get(key)
        
        if value:
            return json.loads(value)
        return None
    
    def delete_agent_state(self, agent_id: str):
        """Delete agent state from Redis."""
        key = f"agent_state:{agent_id}"
        self.redis_client.delete(key)
    
    @contextmanager
    def distributed_lock(self, lock_key: str, timeout: int = 30):
        """Distributed locking using Redis."""
        lock = self.redis_client.lock(
            f"lock:{lock_key}",
            timeout=timeout,
            blocking_timeout=5
        )
        
        acquired = lock.acquire(blocking=True)
        if not acquired:
            raise RuntimeError(f"Failed to acquire lock: {lock_key}")
        
        try:
            yield lock
        finally:
            lock.release()
    
    def synchronize_state(self, key: str, value: Any, ttl: Optional[int] = None):
        """Synchronize state across instances."""
        state_key = f"sync:{key}"
        state_value = json.dumps(value)
        
        if ttl:
            self.redis_client.setex(state_key, ttl, state_value)
        else:
            self.redis_client.set(state_key, state_value)
    
    def get_synchronized_state(self, key: str) -> Optional[Any]:
        """Get synchronized state."""
        state_key = f"sync:{key}"
        value = self.redis_client.get(state_key)
        
        if value:
            return json.loads(value)
        return None


# Global state manager instance
_state_manager: Optional[DistributedStateManager] = None


def get_state_manager() -> DistributedStateManager:
    """Get or create global state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = DistributedStateManager()
    return _state_manager

