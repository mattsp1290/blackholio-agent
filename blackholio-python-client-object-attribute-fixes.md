# BlackHolio Python Client - Object Attribute Fixes Needed

## Critical Issue Summary
The protocol handlers are treating structured objects as dictionaries, causing `AttributeError: 'DatabaseUpdate' object has no attribute 'get'`.

## Root Cause Analysis

### Error Details
```
ERROR - Error processing v1.1.2 message: 'DatabaseUpdate' object has no attribute 'get'
```

**Location**: `blackholio_client/connection/protocol_handlers.py:255`

**Problem**: The code assumes `DatabaseUpdate` is a dictionary and calls `.get()` method:
```python
def _handle_database_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle database update response"""
    return {
        'type': 'DatabaseUpdate',
        'tables': data.get('database_update', {}).get('tables', []),  # ❌ FAILS HERE
        'request_id': data.get('request_id'),                         # ❌ FAILS HERE
        'execution_duration': data.get('total_host_execution_duration') # ❌ FAILS HERE
    }
```

**Actual structure**: `DatabaseUpdate` is a structured object with attributes, not a dictionary:
```python
{
    'database_update': DatabaseUpdate(tables=[]),  # ← This is an object, not dict
    'request_id': 411302231,
    'total_host_execution_duration': TimeDuration(nanos={'__time_duration_micros__': 527})
}
```

## Required Fixes

### Fix 1: Handle Structured Objects in Protocol Handlers
**File**: `blackholio_client/connection/protocol_handlers.py`

**Replace the current handlers with object-aware versions**:

```python
def _handle_database_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle database update response with proper object handling"""
    # Handle both dict and object formats
    database_update = data.get('database_update')
    
    # Extract tables from DatabaseUpdate object
    tables = []
    if database_update:
        if hasattr(database_update, 'tables'):
            tables = database_update.tables
        elif isinstance(database_update, dict):
            tables = database_update.get('tables', [])
    
    # Extract request_id safely
    request_id = None
    if hasattr(data, 'request_id'):
        request_id = data.request_id
    elif isinstance(data, dict):
        request_id = data.get('request_id')
    
    # Extract execution duration safely
    execution_duration = None
    duration_obj = data.get('total_host_execution_duration')
    if duration_obj:
        if hasattr(duration_obj, 'nanos'):
            execution_duration = duration_obj.nanos
        elif isinstance(duration_obj, dict):
            execution_duration = duration_obj.get('__time_duration_micros__')
    
    return {
        'type': 'DatabaseUpdate',
        'tables': tables,
        'request_id': request_id,
        'execution_duration': execution_duration
    }

def _handle_transaction_commit(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle transaction commit response with proper object handling"""
    # Handle Timestamp object
    timestamp = data.get('timestamp')
    timestamp_value = None
    if timestamp and hasattr(timestamp, 'nanos_since_epoch'):
        timestamp_value = timestamp.nanos_since_epoch
    
    # Handle Identity object
    identity = data.get('caller_identity')
    identity_value = None
    if identity and hasattr(identity, 'data'):
        identity_value = identity.data
    
    # Handle EnergyQuanta object
    energy = data.get('energy_quanta_used')
    energy_value = None
    if energy and hasattr(energy, 'quanta'):
        energy_value = energy.quanta
    
    # Handle TimeDuration object
    duration = data.get('total_host_execution_duration')
    duration_value = None
    if duration and hasattr(duration, 'nanos'):
        duration_value = duration.nanos
    
    return {
        'type': 'TransactionCommit',
        'status': data.get('status'),
        'timestamp': timestamp_value,
        'caller_identity': identity_value,
        'energy_used': energy_value,
        'execution_duration': duration_value
    }

def _handle_identity_token(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle identity token response with proper object handling"""
    # Handle Identity object
    identity = data.get('identity')
    identity_value = None
    if identity and hasattr(identity, 'data'):
        identity_value = identity.data
    
    # Handle ConnectionId object
    connection_id = data.get('connection_id')
    connection_id_value = None
    if connection_id and hasattr(connection_id, 'data'):
        connection_id_value = connection_id.data
    
    return {
        'type': 'IdentityToken',
        'identity': identity_value,
        'token': data.get('token'),
        'connection_id': connection_id_value
    }
```

### Fix 2: Add Object Type Detection
**File**: `blackholio_client/connection/protocol_handlers.py`

Add a utility method to safely extract values:

```python
def _safe_extract(self, obj, attr_name, default=None):
    """Safely extract attribute from object or dict"""
    if obj is None:
        return default
    
    # Try attribute access first (for objects)
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)
    
    # Fall back to dict access
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    
    return default

def _extract_nested_value(self, obj, path, default=None):
    """Extract nested values like obj.nanos.micros"""
    current = obj
    for part in path:
        current = self._safe_extract(current, part)
        if current is None:
            return default
    return current
```

### Fix 3: Update Message Processing Logic
**File**: `blackholio_client/connection/protocol_handlers.py`

Add debugging to understand object structures:

```python
def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        # Debug: Log object types for troubleshooting
        if 'database_update' in data:
            db_update = data['database_update']
            logger.debug(f"DatabaseUpdate type: {type(db_update)}, attrs: {dir(db_update)}")
        
        # Existing message processing logic...
        message_type = self._get_message_type(data)
        
        if not message_type:
            # Enhanced debugging for unknown messages
            logger.warning(f"Unknown message type in data keys: {list(data.keys())}")
            for key, value in data.items():
                logger.debug(f"  {key}: {type(value)} = {str(value)[:100]}...")
            return None
            
        # Rest of processing...
        
    except AttributeError as e:
        logger.error(f"AttributeError in message processing: {e}")
        logger.debug(f"Data structure: {data}")
        return None
    except Exception as e:
        logger.error(f"Error processing v1.1.2 message: {e}")
        logger.debug(f"Full data: {data}")
        return None
```

## Object Structure Reference

Based on the logs, these are the actual object types received:

```python
# Message structure
{
    'database_update': DatabaseUpdate(tables=[]),
    'request_id': 411302231,
    'total_host_execution_duration': TimeDuration(nanos={'__time_duration_micros__': 527})
}

# TransactionCommit structure
{
    'status': 'Committed',
    'timestamp': Timestamp(nanos_since_epoch={'__timestamp_micros_since_unix_epoch__': 1751326253127110}),
    'caller_identity': Identity(data=b"{'__identity__': '0x...'}"),
    'caller_connection_id': ConnectionId(data=b"{'__connection_id__': 246634077236934735698079173074425116788}"),
    'reducer_call': ReducerCallInfo(reducer_name='', reducer_id=0, args=b'', request_id=0),
    'energy_quanta_used': EnergyQuanta({'quanta': 1134000}),
    'total_host_execution_duration': TimeDuration(nanos={'__time_duration_micros__': 1163})
}
```

## Testing Requirements

1. **Test object attribute access** for all SpacetimeDB objects
2. **Verify backward compatibility** with dict-based messages
3. **Test error handling** for missing attributes
4. **Integration test** with actual SpacetimeDB responses

## Implementation Priority

1. **Phase 1**: Fix `_handle_database_update` method - **CRITICAL** (blocking training)
2. **Phase 2**: Fix other object handlers - **HIGH**
3. **Phase 3**: Add comprehensive object debugging - **MEDIUM**

This fix resolves the `'DatabaseUpdate' object has no attribute 'get'` error blocking the training process.