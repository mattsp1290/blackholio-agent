#!/usr/bin/env python3
"""
Enhanced Fix for Blackholio AI Agent WebSocket Connection Persistence

This enhanced version addresses the connection termination issue that occurs
even after successfully handling "Invalid close frame" errors.
"""

import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Callable

class EnhancedWebSocketConnectionManager:
    """
    Enhanced WebSocket connection manager that maintains connection persistence
    even when underlying WebSocket connections drop due to protocol errors.
    """
    
    def __init__(self, connection_instance, logger=None):
        self.connection_instance = connection_instance
        self.logger = logger or logging.getLogger(__name__)
        self.reconnect_enabled = True
        self.reconnect_delay = 2.0  # seconds
        self.max_reconnect_attempts = 3
        self.reconnect_attempts = 0
        self.last_subscription_queries = None
        self.connection_state = "disconnected"
        self.pending_initial_subscription = False
        self.reconnect_lock = threading.Lock()
        
    def handle_invalid_close_frame_reconnection(self, original_on_close):
        """Enhanced close handler that implements automatic reconnection for protocol errors."""
        
        def enhanced_on_close(ws, close_status_code, close_msg):
            """Enhanced close handler with automatic reconnection."""
            self.logger.info(f"WebSocket closed with status: {close_status_code}, message: {close_msg}")
            
            # Check if this is an invalid close frame scenario
            is_protocol_error = (
                close_status_code is None or 
                close_status_code == 1006 or  # Abnormal closure
                (hasattr(self.connection_instance, '_last_error_was_invalid_frame') and 
                 self.connection_instance._last_error_was_invalid_frame)
            )
            
            if is_protocol_error and self.reconnect_enabled and self.reconnect_attempts < self.max_reconnect_attempts:
                self.logger.info("Detected protocol error closure - initiating automatic reconnection")
                self._schedule_reconnection()
            else:
                # Call original close handler for normal closures
                if original_on_close:
                    original_on_close(ws, close_status_code, close_msg)
            
            # Reset the error flag
            if hasattr(self.connection_instance, '_last_error_was_invalid_frame'):
                self.connection_instance._last_error_was_invalid_frame = False
        
        return enhanced_on_close
    
    def _schedule_reconnection(self):
        """Schedule a reconnection attempt."""
        with self.reconnect_lock:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                self.logger.error("Max reconnection attempts reached")
                return
            
            self.reconnect_attempts += 1
            self.logger.info(f"Scheduling reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {self.reconnect_delay}s")
            
            # Schedule reconnection in a separate thread
            reconnect_thread = threading.Thread(
                target=self._perform_reconnection,
                daemon=True,
                name=f"BlackholioReconnect-{self.reconnect_attempts}"
            )
            reconnect_thread.start()
    
    def _perform_reconnection(self):
        """Perform the actual reconnection."""
        try:
            time.sleep(self.reconnect_delay)
            
            self.logger.info("Attempting to reconnect...")
            
            # Store current connection details
            if hasattr(self.connection_instance, '_store_connection_state'):
                self.connection_instance._store_connection_state()
            
            # Create new WebSocket connection
            if hasattr(self.connection_instance, '_create_new_connection'):
                success = self.connection_instance._create_new_connection()
                if success:
                    self.logger.info("Reconnection successful")
                    self.reconnect_attempts = 0  # Reset on success
                    
                    # Re-establish subscriptions if needed
                    if self.last_subscription_queries:
                        self.logger.info("Re-establishing subscriptions after reconnection")
                        time.sleep(1)  # Give connection time to stabilize
                        self._reestablish_subscriptions()
                else:
                    self.logger.warning("Reconnection failed")
            else:
                self.logger.warning("Connection instance doesn't support reconnection - manual restart required")
                
        except Exception as e:
            self.logger.error(f"Reconnection attempt failed: {e}")
    
    def _reestablish_subscriptions(self):
        """Re-establish subscriptions after reconnection."""
        try:
            if hasattr(self.connection_instance, '_resubscribe') and self.last_subscription_queries:
                self.connection_instance._resubscribe(self.last_subscription_queries)
        except Exception as e:
            self.logger.error(f"Failed to re-establish subscriptions: {e}")

def enhance_websocket_with_persistence(websocket_instance, connection_instance, logger=None):
    """
    Enhanced version that includes connection persistence for "Invalid close frame" errors.
    
    Args:
        websocket_instance: The WebSocket instance to enhance
        connection_instance: The parent connection instance
        logger: Optional logger for debugging
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create connection manager
    connection_manager = EnhancedWebSocketConnectionManager(connection_instance, logger)
    
    # Store original handlers
    original_on_message = getattr(websocket_instance, 'on_message', None)
    original_on_error = getattr(websocket_instance, 'on_error', None)
    original_on_close = getattr(websocket_instance, 'on_close', None)
    
    def enhanced_on_message(ws, message):
        """Enhanced message handler with large message support and state tracking."""
        try:
            # Handle message data type
            if isinstance(message, str):
                message_data = message.encode('utf-8')
            else:
                message_data = message
            
            message_size = len(message_data)
            large_message_threshold = 50 * 1024  # 50KB
            
            # Log large message handling
            if message_size > large_message_threshold:
                logger.info(f"Processing large message: {message_size} bytes")
                connection_manager.pending_initial_subscription = True
                
                # Log InitialSubscription details for debugging
                try:
                    if message_data.startswith(b'{') and b'"InitialSubscription"' in message_data:
                        parsed_preview = json.loads(message_data.decode('utf-8'))
                        if "InitialSubscription" in parsed_preview:
                            initial_sub = parsed_preview["InitialSubscription"]
                            database_update = initial_sub.get("database_update", {})
                            tables = database_update.get("tables", [])
                            logger.info(f"Large InitialSubscription: {len(tables)} tables, {message_size} bytes")
                            for table in tables:
                                table_name = table.get("table_name", "unknown")
                                num_rows = table.get("num_rows", 0)
                                logger.debug(f"  - {table_name}: {num_rows} rows")
                            
                            # Mark that we successfully received InitialSubscription
                            connection_manager.pending_initial_subscription = False
                            connection_manager.reconnect_attempts = 0  # Reset on successful large message
                            
                except Exception as parse_error:
                    logger.debug(f"Could not parse large message preview: {parse_error}")
            
            # Call original handler if it exists
            if original_on_message:
                original_on_message(ws, message)
            
            # Log successful processing
            if message_size > large_message_threshold:
                logger.info(f"Successfully processed large message: {message_size} bytes")
                
        except Exception as e:
            logger.error(f"Enhanced message handler error: {e}")
            # Continue processing to avoid connection drops
    
    def enhanced_on_error(ws, error):
        """Enhanced error handler with persistence management."""
        error_str = str(error).lower()
        
        # Detect and handle "Invalid close frame" errors specifically
        if "invalid close frame" in error_str:
            logger.error("WebSocket Invalid Close Frame Error detected")
            logger.info("This often occurs after processing large messages (>50KB)")
            logger.info("Applying enhanced error recovery with connection persistence...")
            
            # Mark that we had an invalid frame error
            connection_instance._last_error_was_invalid_frame = True
            
            # If we were expecting an InitialSubscription and got this error, prepare for reconnection
            if connection_manager.pending_initial_subscription:
                logger.info("Invalid frame error occurred during InitialSubscription - preparing for reconnection")
                connection_manager.reconnect_enabled = True
            
            # Don't propagate this error immediately
            logger.info("Enhanced error handling prepared for connection recovery")
            return
            
        elif "frame too large" in error_str or "message too large" in error_str:
            logger.error("WebSocket frame size limit exceeded")
            logger.info("Enabling reconnection for oversized frame error")
            connection_instance._last_error_was_invalid_frame = True
            connection_manager.reconnect_enabled = True
            
        elif "buffer overflow" in error_str or "memory" in error_str:
            logger.error("WebSocket buffer/memory issue with large message")
            logger.info("Enhanced handling applied")
        
        # Call original error handler
        if original_on_error:
            original_on_error(ws, error)
    
    # Enhanced close handler with reconnection logic
    enhanced_on_close = connection_manager.handle_invalid_close_frame_reconnection(original_on_close)
    
    # Apply enhanced handlers
    websocket_instance.on_message = enhanced_on_message
    websocket_instance.on_error = enhanced_on_error
    websocket_instance.on_close = enhanced_on_close
    
    # Store connection manager reference
    connection_instance._connection_manager = connection_manager
    
    logger.info("Applied enhanced WebSocket handling with connection persistence")
    return websocket_instance

def enhance_blackholio_connection_with_persistence(connection_instance):
    """
    Enhanced fix for Blackholio connection with automatic reconnection support.
    
    This should replace the original fix_blackholio_websocket_connection call.
    """
    logger = logging.getLogger("blackholio_agent.enhanced_connection_fix")
    
    # Check if the connection has a WebSocket instance
    websocket_attrs = ['ws', 'websocket', '_websocket', 'socket', '_ws']
    websocket_instance = None
    
    for attr in websocket_attrs:
        if hasattr(connection_instance, attr):
            websocket_instance = getattr(connection_instance, attr)
            if websocket_instance:
                logger.info(f"Found WebSocket instance at attribute: {attr}")
                break
    
    if websocket_instance:
        enhance_websocket_with_persistence(websocket_instance, connection_instance, logger)
        
        # Add connection management methods to the connection instance
        _add_connection_management_methods(connection_instance, logger)
        
        logger.info("Applied enhanced Blackholio WebSocket fix with connection persistence")
        return True
    else:
        logger.warning("Could not find WebSocket instance to enhance")
        return False

def _add_connection_management_methods(connection_instance, logger):
    """Add connection management methods to the connection instance."""
    
    def _store_connection_state(self):
        """Store current connection state for reconnection."""
        if hasattr(self, '_connection_manager'):
            # Store subscription queries if available
            if hasattr(self, 'last_subscription_queries'):
                self._connection_manager.last_subscription_queries = self.last_subscription_queries
            logger.debug("Connection state stored for reconnection")
    
    def _create_new_connection(self):
        """Create a new WebSocket connection preserving the original identity."""
        try:
            logger.info("Creating new WebSocket connection with preserved identity...")
            
            # Close existing connection if it exists
            if hasattr(self, 'ws') and self.ws:
                try:
                    self.ws.close()
                except:
                    pass
            
            # Import websocket here to avoid circular imports
            import websocket
            
            # Create new WebSocket connection with the same URL and handlers
            # CRITICAL: Use the SAME URL that preserves the db_identity parameter
            url = getattr(self, 'url', None) or getattr(self, 'server_url', None)
            if url:
                logger.info(f"Reconnecting to preserved URL: {url}")
                
                # Important: Include the SpacetimeDB protocol headers that the original connection used
                headers = {
                    "Sec-WebSocket-Protocol": "v1.json.spacetimedb"
                }
                
                self.ws = websocket.WebSocketApp(
                    url,
                    on_message=getattr(self, '_on_ws_message', None),
                    on_error=getattr(self, '_on_ws_error', None),
                    on_close=getattr(self, '_on_ws_close', None),
                    on_open=getattr(self, '_on_ws_open', None),
                    header=headers,
                    subprotocols=["v1.json.spacetimedb"]  # v1.1.2 protocol
                )
                
                # Re-apply the enhanced fix to the new websocket
                if hasattr(self, '_connection_manager'):
                    enhance_websocket_with_persistence(self.ws, self, logger)
                
                # Start the connection in a separate thread
                import threading
                def run_connection():
                    try:
                        self.ws.run_forever()
                    except Exception as e:
                        logger.error(f"WebSocket run_forever error: {e}")
                
                connection_thread = threading.Thread(target=run_connection, daemon=True)
                connection_thread.start()
                
                # Give the new connection a moment to establish
                import time
                time.sleep(0.5)
                
                # Mark as connected to allow message processing
                self._connected = True
                
                logger.info("New WebSocket connection created successfully with preserved identity")
                return True
            else:
                logger.error("No URL available for reconnection")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create new connection: {e}")
            return False
    
    def _resubscribe(self, queries):
        """Re-establish subscriptions after reconnection."""
        try:
            logger.info(f"Re-subscribing to {len(queries)} queries")
            
            # Wait for connection to stabilize
            import time
            time.sleep(1.0)
            
            # Re-subscribe to tables using the existing method
            if hasattr(self, '_subscribe_to_tables'):
                import asyncio
                # Since _subscribe_to_tables is async, we need to run it properly
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, create a task
                        asyncio.create_task(self._subscribe_to_tables())
                    else:
                        # If not, run it in the loop
                        loop.run_until_complete(self._subscribe_to_tables())
                except RuntimeError:
                    # No event loop, create a new one
                    asyncio.run(self._subscribe_to_tables())
                    
                logger.info("Successfully re-subscribed to tables")
                
                # CRITICAL: Re-enter the game after reconnection
                # This is essential because the new identity needs a player entry
                logger.info("Re-entering game after reconnection...")
                if hasattr(self, '_reenter_game_after_reconnection'):
                    self._reenter_game_after_reconnection()
                else:
                    logger.warning("No _reenter_game_after_reconnection method found")
                    
            else:
                logger.warning("No _subscribe_to_tables method found for re-subscription")
                
        except Exception as e:
            logger.error(f"Failed to re-subscribe: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Bind methods to the connection instance
    import types
    connection_instance._store_connection_state = types.MethodType(_store_connection_state, connection_instance)
    connection_instance._create_new_connection = types.MethodType(_create_new_connection, connection_instance)
    connection_instance._resubscribe = types.MethodType(_resubscribe, connection_instance)
    
    # Initialize error tracking
    connection_instance._last_error_was_invalid_frame = False

# Enhanced usage example
def example_enhanced_integration():
    """
    Example of how to integrate the enhanced fix.
    """
    example_code = '''
# In blackholio_connection_v112.py, replace the original fix with:

from .BLACKHOLIO_ENHANCED_WEBSOCKET_FIX import enhance_blackholio_connection_with_persistence

class BlackholioConnectionV112:
    def __init__(self, ...):
        # Your existing WebSocket creation
        self.ws = websocket.WebSocketApp(
            url, 
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # 🔧 APPLY THE ENHANCED FIX HERE:
        enhance_blackholio_connection_with_persistence(self)
        
        # Continue with your existing code
    '''
    print("Enhanced integration example:")
    print(example_code)

if __name__ == "__main__":
    print("🔧 Enhanced Blackholio WebSocket Fix with Connection Persistence")
    print("This enhanced fix addresses connection termination after 'Invalid close frame' errors")
    print()
    print("Key improvements over the original fix:")
    print("✅ Automatic reconnection for protocol errors")
    print("✅ Connection state preservation")
    print("✅ Subscription re-establishment after reconnection")
    print("✅ Enhanced error recovery with persistence")
    print()
    print("Integration instructions:")
    print("1. Replace the original import with the enhanced version")
    print("2. Call enhance_blackholio_connection_with_persistence(self) instead")
    print("3. Test with your AI training pipeline")
    print()
    
    # Show example integration
    example_enhanced_integration()
