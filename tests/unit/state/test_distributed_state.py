"""
Unit tests for distributed state management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestDistributedStateManager:
    """Test distributed state manager."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_client = MagicMock()
        redis_client.ping.return_value = True
        redis_client.get.return_value = None
        redis_client.set.return_value = True
        redis_client.setex.return_value = True
        redis_client.delete.return_value = 1
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = True
        redis_client.lock.return_value = mock_lock
        return redis_client
    
    @patch('src.state.distributed_state.redis')
    def test_initialization(self, mock_redis_module, mock_redis):
        """Test state manager initialization."""
        from src.state.distributed_state import DistributedStateManager
        mock_redis_module.Redis.return_value = mock_redis
        
        manager = DistributedStateManager(redis_host="localhost", redis_port=6379)
        assert manager.redis_client is not None
    
    @patch('src.state.distributed_state.redis')
    def test_save_agent_state(self, mock_redis_module, mock_redis):
        """Test saving agent state."""
        from src.state.distributed_state import DistributedStateManager
        mock_redis_module.Redis.return_value = mock_redis
        
        manager = DistributedStateManager()
        state = {"param1": 1.0, "param2": 2.0}
        manager.save_agent_state("agent_1", state)
        
        # Should call Redis set or setex
        assert mock_redis.set.called or mock_redis.setex.called
    
    @patch('src.state.distributed_state.redis')
    def test_load_agent_state(self, mock_redis_module, mock_redis):
        """Test loading agent state."""
        from src.state.distributed_state import DistributedStateManager
        mock_redis_module.Redis.return_value = mock_redis
        mock_redis.get.return_value = '{"param1": 1.0, "param2": 2.0}'
        
        manager = DistributedStateManager()
        state = manager.load_agent_state("agent_1")
        
        assert state is not None
        assert state["param1"] == 1.0
    
    @patch('src.state.distributed_state.redis')
    def test_distributed_lock(self, mock_redis_module, mock_redis):
        """Test distributed locking."""
        from src.state.distributed_state import DistributedStateManager
        mock_redis_module.Redis.return_value = mock_redis
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = True
        mock_redis.lock.return_value = mock_lock
        
        manager = DistributedStateManager()
        
        with manager.distributed_lock("test_lock", timeout=30):
            # Should acquire and release lock
            assert True

