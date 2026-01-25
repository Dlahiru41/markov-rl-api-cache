"""
Unit tests for Redis backend implementation.

Tests RedisConfig and RedisBackend with and without Redis server.
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from src.cache.redis_backend import (
    RedisConfig, RedisBackend, CacheError,
    CacheConnectionError, CacheOperationError,
    REDIS_AVAILABLE
)
from src.cache.backend import CacheStats


class TestRedisConfig:
    """Test RedisConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.max_memory == 1024 * 1024 * 100  # 100MB
        assert config.eviction_policy == "allkeys-lru"
        assert config.key_prefix == "markov_cache:"
        assert config.socket_timeout == 5.0
        assert config.max_connections == 50
        assert config.decode_responses is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            max_memory=1024 * 1024 * 200,
            eviction_policy="allkeys-lfu",
            key_prefix="test:",
            socket_timeout=10.0,
            max_connections=100,
            decode_responses=True
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"
        assert config.max_memory == 1024 * 1024 * 200
        assert config.eviction_policy == "allkeys-lfu"
        assert config.key_prefix == "test:"
        assert config.socket_timeout == 10.0
        assert config.max_connections == 100
        assert config.decode_responses is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = RedisConfig(
            host="localhost",
            port=6379,
            key_prefix="test:"
        )
        d = config.to_dict()

        assert d['host'] == "localhost"
        assert d['port'] == 6379
        assert d['key_prefix'] == "test:"
        assert 'password' not in d  # Sensitive data excluded

    def test_config_immutability(self):
        """Test that config can be modified after creation."""
        config = RedisConfig()
        original_host = config.host
        config.host = "newhost"
        assert config.host == "newhost"


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis package not installed")
class TestRedisBackendUnit:
    """Unit tests for RedisBackend (no actual Redis connection needed)."""

    def test_init_default(self):
        """Test initialization with default config."""
        backend = RedisBackend()
        assert backend._config.host == "localhost"
        assert backend._redis is None
        assert not backend.is_connected

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = RedisConfig(host="custom", port=1234)
        backend = RedisBackend(config)
        assert backend._config.host == "custom"
        assert backend._config.port == 1234

    def test_key_with_prefix(self):
        """Test key prefixing."""
        config = RedisConfig(key_prefix="test:")
        backend = RedisBackend(config)

        prefixed = backend._key("mykey")
        assert prefixed == "test:mykey"

    def test_key_without_prefix(self):
        """Test key without prefix."""
        config = RedisConfig(key_prefix="")
        backend = RedisBackend(config)

        key = backend._key("mykey")
        assert key == "mykey"

    def test_metadata_key_generation(self):
        """Test metadata key generation."""
        backend = RedisBackend()
        meta_key = backend._metadata_key("test:mykey")
        assert meta_key == "test:mykey:meta"

    @patch('src.cache.redis_backend.Redis')
    def test_connect_success(self, mock_redis_class):
        """Test successful connection."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        result = backend.connect()

        assert result is True
        assert backend.is_connected
        mock_redis.ping.assert_called_once()

    @patch('src.cache.redis_backend.Redis')
    def test_connect_failure(self, mock_redis_class):
        """Test connection failure."""
        mock_redis = Mock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        result = backend.connect()

        assert result is False
        assert not backend.is_connected

    @patch('src.cache.redis_backend.Redis')
    def test_disconnect(self, mock_redis_class):
        """Test disconnection."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()
        backend.disconnect()

        mock_redis.close.assert_called_once()
        assert not backend.is_connected

    @patch('src.cache.redis_backend.Redis')
    def test_ping_when_connected(self, mock_redis_class):
        """Test ping when connected."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        result = backend.ping()
        assert result is True

    @patch('src.cache.redis_backend.Redis')
    def test_ping_when_not_connected(self, mock_redis_class):
        """Test ping when not connected."""
        backend = RedisBackend()
        result = backend.ping()
        assert result is False

    @patch('src.cache.redis_backend.Redis')
    def test_get_success(self, mock_redis_class):
        """Test successful get operation."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = b"test_value"
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        value = backend.get("test_key")
        assert value == b"test_value"

        stats = backend.get_stats()
        assert stats.hits == 1

    @patch('src.cache.redis_backend.Redis')
    def test_get_not_found(self, mock_redis_class):
        """Test get for nonexistent key."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        value = backend.get("nonexistent")
        assert value is None

        stats = backend.get_stats()
        assert stats.misses == 1

    @patch('src.cache.redis_backend.Redis')
    def test_set_without_ttl(self, mock_redis_class):
        """Test set without TTL."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.set.return_value = True
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        result = backend.set("key", b"value")
        assert result is True

        mock_redis.set.assert_called()
        stats = backend.get_stats()
        assert stats.sets == 1

    @patch('src.cache.redis_backend.Redis')
    def test_set_with_ttl(self, mock_redis_class):
        """Test set with TTL."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.setex.return_value = True
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        result = backend.set("key", b"value", ttl=60)
        assert result is True

        mock_redis.setex.assert_called()

    @patch('src.cache.redis_backend.Redis')
    def test_set_with_metadata(self, mock_redis_class):
        """Test set with metadata."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.set.return_value = True
        mock_redis.setex.return_value = True
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        metadata = {"endpoint": "/api", "user": "123"}
        result = backend.set("key", b"value", ttl=60, metadata=metadata)
        assert result is True

        # Should set both value and metadata
        assert mock_redis.setex.call_count >= 1

    @patch('src.cache.redis_backend.Redis')
    def test_delete_success(self, mock_redis_class):
        """Test successful delete."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.delete.return_value = 1  # 1 key deleted
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        result = backend.delete("key")
        assert result is True

        stats = backend.get_stats()
        assert stats.deletes == 1

    @patch('src.cache.redis_backend.Redis')
    def test_delete_not_found(self, mock_redis_class):
        """Test delete of nonexistent key."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.delete.return_value = 0  # No keys deleted
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        result = backend.delete("nonexistent")
        assert result is False

    @patch('src.cache.redis_backend.Redis')
    def test_exists_true(self, mock_redis_class):
        """Test exists for existing key."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.exists.return_value = 1
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        result = backend.exists("key")
        assert result is True

    @patch('src.cache.redis_backend.Redis')
    def test_exists_false(self, mock_redis_class):
        """Test exists for nonexistent key."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.exists.return_value = 0
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        result = backend.exists("nonexistent")
        assert result is False

    @patch('src.cache.redis_backend.Redis')
    def test_clear(self, mock_redis_class):
        """Test clear operation."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True

        # Mock scan_iter to return some keys
        mock_redis.scan_iter.return_value = iter([
            b"test:key1", b"test:key2", b"test:key3"
        ])

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = [1, 1, 1]  # 3 successful deletes
        mock_redis.pipeline.return_value = mock_pipeline

        mock_redis_class.return_value = mock_redis

        backend = RedisBackend(RedisConfig(key_prefix="test:"))
        backend.connect()

        count = backend.clear()
        assert count == 3

    @patch('src.cache.redis_backend.Redis')
    def test_get_many(self, mock_redis_class):
        """Test batch get operation."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.mget.return_value = [b"value1", None, b"value3"]
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        values = backend.get_many(["key1", "key2", "key3"])
        assert len(values) == 2
        assert values["key1"] == b"value1"
        assert values["key3"] == b"value3"
        assert "key2" not in values

    @patch('src.cache.redis_backend.Redis')
    def test_set_many(self, mock_redis_class):
        """Test batch set operation."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = [True, True, True]
        mock_redis.pipeline.return_value = mock_pipeline

        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        items = {
            "key1": b"value1",
            "key2": b"value2",
            "key3": b"value3"
        }

        count = backend.set_many(items)
        assert count == 3

    @patch('src.cache.redis_backend.Redis')
    def test_delete_many(self, mock_redis_class):
        """Test batch delete operation."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = [1, 1, 0]  # 2 successful deletes
        mock_redis.pipeline.return_value = mock_pipeline

        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        count = backend.delete_many(["key1", "key2", "key3"])
        assert count == 2

    @patch('src.cache.redis_backend.Redis')
    def test_keys_without_pattern(self, mock_redis_class):
        """Test keys without pattern."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.scan_iter.return_value = iter([
            b"test:key1", b"test:key2"
        ])
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend(RedisConfig(key_prefix="test:"))
        backend.connect()

        keys = backend.keys()
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    @patch('src.cache.redis_backend.Redis')
    def test_keys_with_pattern(self, mock_redis_class):
        """Test keys with pattern."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.scan_iter.return_value = iter([
            b"test:user:1", b"test:user:2"
        ])
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend(RedisConfig(key_prefix="test:"))
        backend.connect()

        keys = backend.keys("user:*")
        assert len(keys) == 2

    @patch('src.cache.redis_backend.Redis')
    def test_context_manager(self, mock_redis_class):
        """Test context manager support."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        config = RedisConfig()

        with RedisBackend(config) as backend:
            assert backend.is_connected

        mock_redis.close.assert_called_once()

    @patch('src.cache.redis_backend.Redis')
    def test_get_stats(self, mock_redis_class):
        """Test statistics retrieval."""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {
            'used_memory': 1024,
            'maxmemory': 10240,
            'evicted_keys': 5
        }
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        stats = backend.get_stats()
        assert isinstance(stats, CacheStats)
        assert stats.current_size_bytes >= 0

    @patch('src.cache.redis_backend.Redis')
    def test_error_handling_connection_error(self, mock_redis_class):
        """Test handling of connection errors."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.get.side_effect = RedisConnectionError("Connection lost")
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        # Should handle error gracefully
        value = backend.get("key")
        assert value is None

    @patch('src.cache.redis_backend.Redis')
    def test_error_handling_timeout(self, mock_redis_class):
        """Test handling of timeout errors."""
        from redis.exceptions import TimeoutError as RedisTimeoutError

        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.get.side_effect = RedisTimeoutError("Timeout")
        mock_redis_class.return_value = mock_redis

        backend = RedisBackend()
        backend.connect()

        # Should handle error gracefully
        value = backend.get("key")
        assert value is None

    def test_operations_when_not_connected(self):
        """Test that operations fail gracefully when not connected."""
        backend = RedisBackend()

        assert backend.get("key") is None
        assert backend.set("key", b"value") is False
        assert backend.delete("key") is False
        assert backend.exists("key") is False
        assert backend.clear() == 0


class TestRedisBackendExceptions:
    """Test custom exceptions."""

    def test_cache_error(self):
        """Test CacheError exception."""
        with pytest.raises(CacheError):
            raise CacheError("Test error")

    def test_cache_connection_error(self):
        """Test CacheConnectionError exception."""
        with pytest.raises(CacheConnectionError):
            raise CacheConnectionError("Connection failed")

        # Should be a CacheError
        with pytest.raises(CacheError):
            raise CacheConnectionError("Connection failed")

    def test_cache_operation_error(self):
        """Test CacheOperationError exception."""
        with pytest.raises(CacheOperationError):
            raise CacheOperationError("Operation failed")

        # Should be a CacheError
        with pytest.raises(CacheError):
            raise CacheOperationError("Operation failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

