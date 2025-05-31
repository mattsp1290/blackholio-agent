"""
SpacetimeDB Data Converter for Blackholio ML Agent

Converts SpacetimeDB data objects to dictionaries for the environment.
Based on the proven pygame client implementation.
"""

import logging

logger = logging.getLogger(__name__)


def convert_to_dict(obj):
    """
    Convert a SpacetimeDB data object to a dictionary.
    Handles nested objects and special types.
    """
    if obj is None:
        return None
    
    # If it's already a dict, return it
    if isinstance(obj, dict):
        return obj
    
    # If it has __dict__, convert it
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            # Skip private/internal attributes
            if key.startswith('_'):
                continue
            
            # Recursively convert nested objects
            if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool)):
                result[key] = convert_to_dict(value)
            else:
                result[key] = value
        
        return result
    
    # If it's a basic type, return as is
    return obj


def extract_entity_data(entity_obj):
    """
    Extract entity data from SpacetimeDB object.
    """
    if hasattr(entity_obj, 'data'):
        entity_obj = entity_obj.data
    
    # Convert to dict
    data = convert_to_dict(entity_obj)
    
    # Log the actual data for debugging
    logger.debug(f"Entity data: {data}")
    
    # Ensure we have the expected structure
    if 'position' in data and hasattr(data['position'], '__dict__'):
        data['position'] = convert_to_dict(data['position'])
    
    return data


def extract_circle_data(circle_obj):
    """
    Extract circle data from SpacetimeDB object.
    """
    if hasattr(circle_obj, 'data'):
        circle_obj = circle_obj.data
    
    # Convert to dict
    data = convert_to_dict(circle_obj)
    
    # Log the actual data for debugging
    logger.debug(f"Circle data: {data}")
    
    # Ensure we have the expected structure
    if 'direction' in data and hasattr(data['direction'], '__dict__'):
        data['direction'] = convert_to_dict(data['direction'])
    
    return data


def extract_player_data(player_obj):
    """
    Extract player data from SpacetimeDB object.
    """
    if hasattr(player_obj, 'data'):
        player_obj = player_obj.data
    
    # Convert to dict
    data = convert_to_dict(player_obj)
    
    # Convert identity to string if needed
    if 'identity' in data and hasattr(data['identity'], '__str__'):
        data['identity'] = str(data['identity'])
    
    return data


def extract_config_data(config_obj):
    """
    Extract config data from SpacetimeDB object.
    """
    if hasattr(config_obj, 'data'):
        config_obj = config_obj.data
    
    return convert_to_dict(config_obj)


def extract_food_data(food_obj):
    """
    Extract food data from SpacetimeDB object.
    """
    if hasattr(food_obj, 'data'):
        food_obj = food_obj.data
    
    # Convert to dict
    data = convert_to_dict(food_obj)
    
    # Ensure we have the expected structure
    if 'position' in data and hasattr(data['position'], '__dict__'):
        data['position'] = convert_to_dict(data['position'])
    
    return data
