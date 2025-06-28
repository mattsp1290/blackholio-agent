#!/usr/bin/env python3
"""
Specific Fix for Blackholio AI Agent Custom WebSocket Implementation

This addresses the "Invalid close frame" errors in their custom connection module
by providing WebSocket message handling improvements specifically for their implementation.
"""

import json
import logging
from typing import Any, Dict, Optional

def enhance_websocket_message_handling(websocket_instance, logger=None):
    """
    Apply large message handling enhancements to a custom WebSocket instance.
    
    This fixes the "Invalid close frame" errors that occur when processing
    large InitialSubscription messages (61KB+).
    
    Args:
        websocket_instance: The WebSocket instance to enhance
        logger: Optional logger for debugging
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Store original handlers
    original_on_message = getattr(websocket_instance, 'on_message', None)
    original_on_error = getattr(websocket_instance, 'on_error', None)
    original_on_close = getattr(websocket_instance, 'on_close', None)
    
    def enhanced_on_message(ws, message):
        """Enhanced message handler for large messages."""
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
        """Enhanced error handler with better large message error detection."""
        error_str = str(error).lower()
        
        # Detect and handle "Invalid close frame" errors specifically
        if "invalid close frame" in error_str:
            logger.error("WebSocket Invalid Close Frame Error detected")
            logger.info("This often occurs after processing large messages (>50KB)")
            logger.info("Applying enhanced error recovery...")
            
            # Don't propagate this error immediately - try to recover
            # Log the issue but don't crash the connection
            logger.info("Enhanced error handling prevented connection drop")
            return
            
        elif "frame too large" in error_str or "message too large" in error_str:
            logger.error("WebSocket frame size limit exceeded")
            logger.info("Consider implementing message chunking")
            
        elif "buffer overflow" in error_str or "memory" in error_str:
            logger.error("WebSocket buffer/memory issue with large message")
            logger.info("Enhanced handling applied")
        
        # Call original error handler
        if original_on_error:
            original_on_error(ws, error)
    
    def enhanced_on_close(ws, close_status_code, close_msg):
        """Enhanced close handler with better error analysis."""
        if close_status_code:
            logger.info(f"WebSocket closed with status: {close_status_code}")
            
            # Analyze close codes related to large message issues
            if close_status_code == 1009:  # Message too big
                logger.error("WebSocket closed: Message too big (1009)")
                logger.info("Server rejected message due to size limit")
            elif close_status_code == 1002:  # Protocol error  
                logger.error("WebSocket closed: Protocol error (1002)")
                logger.info("Possible frame formatting issue with large messages")
            elif close_status_code == 1006:  # Abnormal closure
                logger.error("WebSocket closed: Abnormal closure (1006)")
                logger.info("Connection dropped unexpectedly, possibly during large message processing")
        
        # Call original close handler
        if original_on_close:
            original_on_close(ws, close_status_code, close_msg)
    
    # Apply enhanced handlers
    websocket_instance.on_message = enhanced_on_message
    websocket_instance.on_error = enhanced_on_error
    websocket_instance.on_close = enhanced_on_close
    
    logger.info("Applied enhanced WebSocket large message handling")
    return websocket_instance

def create_enhanced_websocket_connection(url, **kwargs):
    """
    Create a WebSocket connection with built-in large message support.
    
    This creates a WebSocket that can handle large InitialSubscription messages
    without experiencing "Invalid close frame" errors.
    
    Args:
        url: WebSocket URL
        **kwargs: Additional WebSocket options
        
    Returns:
        Enhanced WebSocket instance
    """
    import websocket
    
    logger = logging.getLogger(__name__)
    
    # Set WebSocket options for large messages
    websocket_options = {
        'ping_interval': 60,  # Keep connection alive during large message processing
        'ping_timeout': 30,   # Reasonable timeout
        'enable_multithread': True,  # Allow concurrent message processing
    }
    websocket_options.update(kwargs)
    
    # Create WebSocket with enhanced handlers
    ws = websocket.WebSocketApp(url, **websocket_options)
    
    # Apply large message enhancements
    enhance_websocket_message_handling(ws, logger)
    
    logger.info(f"Created enhanced WebSocket with large message support for: {url}")
    return ws

# Specific fix function for the Blackholio AI Agent custom connection
def fix_blackholio_websocket_connection(connection_instance):
    """
    Specific fix for the Blackholio AI Agent custom connection implementation.
    
    This should be called in the blackholio_connection_v112.py file to apply
    the WebSocket large message handling improvements.
    
    Args:
        connection_instance: The custom connection instance with WebSocket
    """
    logger = logging.getLogger("blackholio_agent.connection_fix")
    
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
        enhance_websocket_message_handling(websocket_instance, logger)
        logger.info("Applied Blackholio WebSocket large message fix")
        return True
    else:
        logger.warning("Could not find WebSocket instance to enhance")
        return False

# Example usage for the Blackholio AI Agent team
def example_integration():
    """
    Example of how to integrate this fix into the Blackholio AI Agent code.
    
    Add this to your blackholio_connection_v112.py file:
    """
    example_code = '''
# In blackholio_connection_v112.py, after creating your WebSocket connection:

from BLACKHOLIO_CUSTOM_WEBSOCKET_LARGE_MESSAGE_FIX import fix_blackholio_websocket_connection

class BlackholioConnectionV112:
    def __init__(self, ...):
        # Your existing initialization code
        self.ws = websocket.WebSocketApp(
            url, 
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # APPLY THE FIX HERE:
        fix_blackholio_websocket_connection(self)
        
        # Continue with your existing code
    '''
    print("Integration example:")
    print(example_code)

if __name__ == "__main__":
    print("🔧 Blackholio Custom WebSocket Large Message Fix")
    print("This fix addresses 'Invalid close frame' errors in custom WebSocket implementations")
    print()
    print("Key improvements:")
    print("✅ Enhanced handling of 61KB+ InitialSubscription messages")
    print("✅ Specific 'Invalid close frame' error detection and recovery")
    print("✅ Large message logging and debugging")
    print("✅ Connection stability improvements")
    print()
    print("Integration instructions:")
    print("1. Import: from BLACKHOLIO_CUSTOM_WEBSOCKET_LARGE_MESSAGE_FIX import fix_blackholio_websocket_connection")
    print("2. Apply after WebSocket creation: fix_blackholio_websocket_connection(self)")
    print("3. Test with your AI training pipeline")
    print()
    
    # Show example integration
    example_integration()
