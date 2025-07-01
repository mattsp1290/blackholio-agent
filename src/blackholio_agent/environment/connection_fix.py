"""
Temporary fix for the blackholio-python-client connection issue.

This module provides a workaround for the rust:// prefix issue in the client configuration.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fix_connection_config():
    """
    Fix the connection configuration by setting proper environment variables.
    
    The blackholio-python-client seems to be adding 'rust://' prefix which causes
    WebSocket connection failures. This function ensures clean configuration.
    """
    # Remove any existing problematic environment variables
    problematic_vars = ['SPACETIMEDB_URI', 'SPACETIME_URI', 'BLACKHOLIO_SERVER']
    for var in problematic_vars:
        if var in os.environ:
            value = os.environ[var]
            if value.startswith('rust://'):
                os.environ[var] = value[7:]  # Remove 'rust://' prefix
                logger.info(f"Fixed {var}: {os.environ[var]}")
    
    # Set clean environment variables
    os.environ['SPACETIMEDB_HOST'] = 'localhost:3000'
    os.environ['SPACETIMEDB_DATABASE'] = 'blackholio'
    os.environ['SPACETIME_DB_IDENTITY'] = 'blackholio'
    
    # Also try to set some alternative variable names the client might use
    os.environ['BLACKHOLIO_HOST'] = 'localhost:3000'
    os.environ['BLACKHOLIO_DATABASE'] = 'blackholio'
    
    logger.info("Connection configuration fixed:")
    logger.info(f"  SPACETIMEDB_HOST={os.environ.get('SPACETIMEDB_HOST')}")
    logger.info(f"  SPACETIMEDB_DATABASE={os.environ.get('SPACETIMEDB_DATABASE')}")


def get_clean_host(host: str) -> str:
    """
    Clean the host string by removing any protocol prefixes.
    
    Args:
        host: The host string (e.g., "rust://localhost:3000" or "localhost:3000")
        
    Returns:
        Clean host string without protocol prefix
    """
    # Remove common protocol prefixes
    prefixes = ['rust://', 'python://', 'csharp://', 'go://', 'ws://', 'wss://', 'http://', 'https://']
    
    clean_host = host
    for prefix in prefixes:
        if clean_host.startswith(prefix):
            clean_host = clean_host[len(prefix):]
            logger.debug(f"Removed prefix '{prefix}' from host: {host} -> {clean_host}")
            break
    
    return clean_host


def create_fixed_game_client(host: str = "localhost:3000", database: str = "blackholio", **kwargs):
    """
    Create a game client with fixed configuration.
    
    This wrapper ensures the host doesn't have problematic prefixes.
    """
    # Fix environment first
    fix_connection_config()
    
    # Clean the host
    clean_host_str = get_clean_host(host)
    
    # Import here to ensure environment is set first
    from blackholio_client import BlackholioClient
    from blackholio_client.client import GameClient  # Try the GameClient instead
    from blackholio_client.config import EnvironmentConfig
    
    # Client has been fixed to use correct database name
    logger.info(f"Using database name: {database}")
    
    # Create config directly without rust:// prefix
    config = EnvironmentConfig(
        server_language='rust',  # Use 'rust' to match supported languages
        server_ip=clean_host_str.split(':')[0],
        server_port=int(clean_host_str.split(':')[1]) if ':' in clean_host_str else 3000,
        spacetime_db_identity=database,
        server_use_ssl=False
    )
    
    # WORKAROUND: Add missing attributes expected by BlackholioClient
    # The client expects config.language but EnvironmentConfig has server_language
    config.language = config.server_language
    # The client also expects config.host but EnvironmentConfig has server_ip/server_port
    config.host = f"{config.server_ip}:{config.server_port}"
    # The client also expects config.port but EnvironmentConfig has server_port
    config.port = config.server_port
    # The client also expects config.db_identity but EnvironmentConfig has spacetime_db_identity
    config.db_identity = config.spacetime_db_identity
    
    logger.info(f"Creating game client with config: server={clean_host_str}, database={database}")
    
    # Create the client with clean configuration
    # Try GameClient first which has a cleaner API
    try:
        logger.info("Attempting to create GameClient...")
        return GameClient(
            host=clean_host_str,
            database=database,
            server_language='rust',
            protocol='v1.json.spacetimedb',  # Use JSON protocol
            auto_reconnect=True
        )
    except Exception as e:
        logger.error(f"Failed to create GameClient: {e}")
        # Fallback to BlackholioClient with minimal parameters
        try:
            logger.info("Falling back to BlackholioClient...")
            return BlackholioClient(
                server_language='rust'
            )
        except Exception as e2:
            logger.error(f"Failed to create BlackholioClient: {e2}")
            # Final fallback - return None to trigger mock mode
            logger.error("All client creation attempts failed, will use mock connection")
            return None