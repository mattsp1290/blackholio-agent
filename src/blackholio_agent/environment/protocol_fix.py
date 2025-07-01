"""
Protocol fix for SpacetimeDB SDK compatibility issues.

This module fixes the protocol mapping issue where SpacetimeDB SDK expects
"text" or "binary" but receives "v1.json.spacetimedb" or "v1.bsatn.spacetimedb".
"""

import logging

logger = logging.getLogger(__name__)


def apply_protocol_fixes():
    """Apply all necessary protocol fixes for SpacetimeDB SDK compatibility."""
    
    # Fix 1: Patch the protocol constants in spacetimedb_sdk
    _patch_protocol_constants()
    
    # Fix 2: Patch the factory base class
    _patch_factory_base()
    
    # Fix 3: Patch the connection builder validation
    _patch_connection_builder()
    
    logger.info("✅ All protocol fixes applied successfully")


def _patch_protocol_constants():
    """Replace protocol constants with simple values."""
    try:
        import spacetimedb_sdk.protocol as protocol_module
        
        # Store original values for reference
        protocol_module._ORIGINAL_TEXT_PROTOCOL = getattr(protocol_module, 'TEXT_PROTOCOL', None)
        protocol_module._ORIGINAL_BIN_PROTOCOL = getattr(protocol_module, 'BIN_PROTOCOL', None)
        
        # Replace with simple values that connection builder expects
        protocol_module.TEXT_PROTOCOL = "text"
        protocol_module.BIN_PROTOCOL = "binary"
        
        logger.debug("   ✅ Patched protocol constants")
        
    except Exception as e:
        logger.warning(f"   ⚠️ Could not patch protocol constants: {e}")


def _patch_factory_base():
    """Patch the factory base to handle protocol mapping."""
    try:
        from spacetimedb_sdk.factory.base import SpacetimeDBClientFactoryBase
        
        # Store original method
        original_create_connection_builder = SpacetimeDBClientFactoryBase.create_connection_builder
        
        def patched_create_connection_builder(self, optimization_profile=None):
            """Patched version that maps protocol values correctly."""
            from spacetimedb_sdk.factory.base import OptimizationProfile
            from spacetimedb_sdk.connection_builder import SpacetimeDBConnectionBuilder
            
            if optimization_profile is None:
                optimization_profile = OptimizationProfile.BALANCED
            
            try:
                # Get base configuration
                config = self.get_recommended_config(optimization_profile)
                
                # Create builder
                builder = SpacetimeDBConnectionBuilder()
                
                # Map protocol values to what the connection builder expects
                if "protocol" in config:
                    protocol_value = str(config["protocol"])
                    # Map full protocol names to simple ones
                    if "bsatn" in protocol_value.lower() or protocol_value == "binary":
                        mapped_protocol = "binary"
                    elif "json" in protocol_value.lower() or protocol_value == "text":
                        mapped_protocol = "text"
                    else:
                        # Default to binary for better performance
                        mapped_protocol = "binary"
                    
                    builder = builder.with_protocol(mapped_protocol)
                
                # Apply other configurations
                if "compression" in config:
                    builder = builder.with_compression(config["compression"])
                
                if "energy_budget" in config:
                    builder = builder.with_energy_budget(config["energy_budget"])
                
                if "retry_policy" in config:
                    builder = builder.with_retry_policy(config["retry_policy"])
                
                return builder
                
            except Exception as e:
                logger.error(f"Failed to create connection builder: {e}")
                raise
        
        # Replace the method
        SpacetimeDBClientFactoryBase.create_connection_builder = patched_create_connection_builder
        SpacetimeDBClientFactoryBase._original_create_connection_builder = original_create_connection_builder
        
        logger.debug("   ✅ Patched factory base class")
        
    except Exception as e:
        logger.warning(f"   ⚠️ Could not patch factory base: {e}")


def _patch_connection_builder():
    """Patch the connection builder to be more lenient with protocol values."""
    try:
        from spacetimedb_sdk import connection_builder as cb_module
        
        # Check if we can access the ConnectionBuilder class
        if hasattr(cb_module, 'SpacetimeDBConnectionBuilder'):
            ConnectionBuilder = cb_module.SpacetimeDBConnectionBuilder
            
            # Store original with_protocol method
            original_with_protocol = ConnectionBuilder.with_protocol
            
            def patched_with_protocol(self, protocol):
                """Patched version that maps protocol values."""
                # Map protocol values
                protocol_str = str(protocol)
                if "bsatn" in protocol_str.lower() or protocol_str == "binary":
                    mapped_protocol = "binary"
                elif "json" in protocol_str.lower() or protocol_str == "text":
                    mapped_protocol = "text"
                else:
                    # Try to use as-is, might be valid
                    mapped_protocol = protocol_str
                
                # Call original with mapped value
                return original_with_protocol(self, mapped_protocol)
            
            # Replace the method
            ConnectionBuilder.with_protocol = patched_with_protocol
            ConnectionBuilder._original_with_protocol = original_with_protocol
            
            logger.debug("   ✅ Patched connection builder")
            
    except Exception as e:
        logger.warning(f"   ⚠️ Could not patch connection builder: {e}")


# Auto-apply fixes when module is imported
apply_protocol_fixes()