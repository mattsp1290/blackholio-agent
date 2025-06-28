"""
Unit tests for BlackholioConnection component.

Tests connection management, game state handling, and SpacetimeDB interaction
using mock clients.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, call
from typing import Dict, Any
import time

from ...environment.connection import BlackholioConnection, ConnectionConfig, GameState
from ..fixtures.mock_spacetimedb import MockSpacetimeDBClient, MockConfig


class TestBlackholioConnection:
    """Test suite for BlackholioConnection class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock SpacetimeDB client."""
        """Create a mock SpacetimeDB client."""
        return MockSpacetimeDBClient()
    
    @pytest.fixture
    def connection_config(self):
        """Create a basic connection config."""
        return ConnectionConfig(
            host="localhost:3000",
            database="blackholio"
        )
    
    def test_initialization(self, connection_config):
        """Test BlackholioConnection initialization."""
        connection = BlackholioConnection(connection_config)
        
        assert connection.config == connection_config
        assert connection.client is None
        assert not connection.is_connected
        assert connection.player_id is None
        assert isinstance(connection.game_state, GameState)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, connection_config, monkeypatch):
        """Test successful connection to server."""
        connection = BlackholioConnection(connection_config)
        
        # Mock the client initialization
        mock_client = Mock()
        mock_client.is_connected = True
        mock_client.update = Mock()
        mock_client._register_row_update = Mock()
        mock_client.subscribe = Mock()
        
        # Patch the executor to set client when called
        async def mock_run_in_executor(executor, func, *args):
            # When initializing client, set it on the connection
            connection.client = mock_client
            return mock_client
            
        monkeypatch.setattr(connection, '_executor', Mock())
        monkeypatch.setattr(connection, '_loop', asyncio.get_event_loop())
        connection._loop.run_in_executor = mock_run_in_executor
        
        # Mock waiting methods
        async def mock_wait_for_connection():
            connection._connected = True
            
        async def mock_subscribe_to_tables():
            connection._subscribed = True
            
        monkeypatch.setattr(connection, '_wait_for_connection', mock_wait_for_connection)
        monkeypatch.setattr(connection, '_subscribe_to_tables', mock_subscribe_to_tables)
        monkeypatch.setattr(connection, '_register_table_handlers', Mock())
        
        # Connect
        await connection.connect()
        
        # Check that connection was established
        assert connection._connected is True
        assert connection.client is not None
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, connection_config):
        """Test handling connection failure."""
        connection = BlackholioConnection(connection_config)
        
        # Make connection fail
        async def failing_connect():
            raise ConnectionError("Failed to connect")
            
        with pytest.raises(ConnectionError):
            await failing_connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, connection_config, mock_client):
        """Test disconnection from server."""
        connection = BlackholioConnection(connection_config)
        connection.client = mock_client
        connection._connected = True
        connection._loop = asyncio.get_event_loop()
        
        # Mock executor
        async def mock_run_in_executor(executor, func, *args):
            if hasattr(func, '__name__') and 'disconnect' in func.__name__:
                mock_client.disconnect()
            return None
            
        connection._loop.run_in_executor = mock_run_in_executor
        connection._executor = Mock()
        
        # Disconnect
        await connection.disconnect()
        
        assert not connection.is_connected
        assert connection.client is None
    
    def test_get_player_entities(self, connection_config):
        """Test getting player entities."""
        connection = BlackholioConnection(connection_config)
        connection.game_state.player_id = 100
        
        # Add entities to game state
        connection.game_state.entities = {
            1: {"entity_id": 1, "mass": 30.0},
            2: {"entity_id": 2, "mass": 20.0},
            3: {"entity_id": 3, "mass": 50.0},
        }
        
        # Add circles to indicate ownership
        connection.game_state.circles = {
            1: {"entity_id": 1, "player_id": 100},
            2: {"entity_id": 2, "player_id": 100},
            3: {"entity_id": 3, "player_id": 101},
        }
        
        player_entities = connection.get_player_entities()
        
        assert len(player_entities) == 2
        assert all(entity["entity_id"] in [1, 2] for entity in player_entities)
    
    def test_get_other_entities(self, connection_config):
        """Test getting other entities (enemies)."""
        connection = BlackholioConnection(connection_config)
        connection.game_state.player_id = 100
        
        # Add entities
        connection.game_state.entities = {
            1: {"entity_id": 1, "mass": 30.0},
            2: {"entity_id": 2, "mass": 50.0},
            3: {"entity_id": 3, "mass": 20.0},
        }
        
        # Add circles
        connection.game_state.circles = {
            1: {"entity_id": 1, "player_id": 100},
            2: {"entity_id": 2, "player_id": 101},
            3: {"entity_id": 3, "player_id": 102},
        }
        
        other_entities = connection.get_other_entities()
        
        assert len(other_entities) == 2
        assert all(entity["entity_id"] in [2, 3] for entity in other_entities)
    
    @pytest.mark.asyncio
    async def test_update_player_input(self, connection_config):
        """Test sending player input."""
        connection = BlackholioConnection(connection_config)
        connection._connected = True
        
        # Mock call_reducer
        async def mock_call_reducer(name, *args):
            assert name == "UpdatePlayerInput"
            assert args == ([5.0, -3.0],)
            return True
            
        connection.call_reducer = mock_call_reducer
        
        # Send movement
        await connection.update_player_input([5.0, -3.0])
    
    @pytest.mark.asyncio
    async def test_call_reducer(self, connection_config):
        """Test calling game reducers."""
        connection = BlackholioConnection(connection_config)
        connection._connected = True
        connection.client = Mock()
        connection.client._reducer_call = Mock()
        connection._loop = asyncio.get_event_loop()
        connection._executor = Mock()
        
        # Mock executor
        async def mock_run_in_executor(executor, func, *args):
            func(*args)
            return True
            
        connection._loop.run_in_executor = mock_run_in_executor
        
        # Call split
        result = await connection.call_reducer("PlayerSplit")
        assert result is True
        connection.client._reducer_call.assert_called_with("PlayerSplit")
    
    @pytest.mark.asyncio
    async def test_ensure_connected(self, connection_config, monkeypatch):
        """Test ensure_connected functionality."""
        connection = BlackholioConnection(connection_config)
        connection._connected = False
        
        # Mock connect
        connect_called = False
        async def mock_connect():
            nonlocal connect_called
            connect_called = True
            connection._connected = True
            
        monkeypatch.setattr(connection, 'connect', mock_connect)
        
        # Ensure connected
        await connection.ensure_connected()
        
        assert connect_called
        assert connection._connected is True
    
    def test_game_state_listeners(self, connection_config):
        """Test game state listener functionality."""
        connection = BlackholioConnection(connection_config)
        
        # Track calls
        listener_calls = []
        
        def listener(game_state):
            listener_calls.append(game_state)
        
        # Add listener
        connection.add_game_state_listener(listener)
        
        # Trigger update
        connection._notify_game_state_update()
        
        assert len(listener_calls) == 1
        assert isinstance(listener_calls[0], GameState)
    
    def test_get_update_rate(self, connection_config):
        """Test getting server update rate."""
        connection = BlackholioConnection(connection_config)
        
        # Initially 0
        assert connection.get_update_rate() == 0.0
        
        # Simulate updates
        connection._last_update_time = time.time() - 0.05  # 50ms ago
        connection._update_count = 2
        
        rate = connection.get_update_rate()
        assert rate > 0  # Should be approximately 20 Hz
    
    def test_connection_config_parsing(self):
        """Test ConnectionConfig URL parsing."""
        # Test with wss://
        config = ConnectionConfig(host="wss://example.com:3000", database="test")
        assert config.ssl_enabled is True
        assert config.host == "example.com:3000"
        
        # Test with ws://
        config = ConnectionConfig(host="ws://localhost:3000", database="test")
        assert config.ssl_enabled is False
        assert config.host == "localhost:3000"
        
        # Test without protocol
        config = ConnectionConfig(host="localhost:3000", database="test")
        assert config.ssl_enabled is False
        assert config.host == "localhost:3000"
    
    def test_identity_handling(self, connection_config):
        """Test identity assignment handling."""
        connection = BlackholioConnection(connection_config)
        
        # Handle identity
        connection._handle_identity("token123", "identity456", "conn789")
        
        assert connection.game_state.player_identity == "identity456"
    
    def test_connection_callbacks(self, connection_config):
        """Test connection event callbacks."""
        connection = BlackholioConnection(connection_config)
        
        connected_called = False
        disconnected_called = False
        
        def on_connected():
            nonlocal connected_called
            connected_called = True
            
        def on_disconnected(reason):
            nonlocal disconnected_called
            disconnected_called = True
            
        connection.add_connection_listener(on_connected)
        connection.add_disconnection_listener(on_disconnected)
        
        # Trigger events
        connection._handle_connect()
        assert connected_called
        
        connection._handle_disconnect("test reason")
        assert disconnected_called
    
    @pytest.mark.asyncio
    async def test_auto_reconnect(self, connection_config, monkeypatch):
        """Test automatic reconnection on connection loss."""
        config = ConnectionConfig(
            host="localhost:3000",
            database="test",
            auto_reconnect=True,
            max_reconnect_attempts=2,
            reconnect_delay=0.1
        )
        connection = BlackholioConnection(config)
        
        connect_attempts = 0
        
        async def mock_connect():
            nonlocal connect_attempts
            connect_attempts += 1
            if connect_attempts < 2:
                raise ConnectionError("Failed")
            connection._connected = True
            
        monkeypatch.setattr(connection, 'connect', mock_connect)
        
        # Trigger auto reconnect
        await connection._auto_reconnect()
        
        assert connect_attempts == 2
        assert connection._connected is True
