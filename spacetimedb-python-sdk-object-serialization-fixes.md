# SpacetimeDB Python SDK - Object Serialization Fixes Needed

## Issue Summary
The SpacetimeDB Python SDK is sending structured objects (`DatabaseUpdate`, `TimeDuration`, `Identity`, etc.) that don't behave like dictionaries, causing attribute errors in downstream consumers.

## Root Cause Analysis

### Object vs Dictionary Mismatch
**Problem**: SDK sends objects like `DatabaseUpdate(tables=[])` but client code expects dictionaries that support `.get()` method.

**Current SDK behavior**:
```python
# SDK sends this:
{
    'database_update': DatabaseUpdate(tables=[]),              # Object, not dict
    'total_host_execution_duration': TimeDuration(nanos={...}) # Object, not dict
}

# But clients expect this:
{
    'database_update': {'tables': []},                         # Dict
    'total_host_execution_duration': {'nanos': {...}}          # Dict
}
```

### Specific Object Types Causing Issues

1. **DatabaseUpdate**: `DatabaseUpdate(tables=[])`
2. **TimeDuration**: `TimeDuration(nanos={'__time_duration_micros__': 527})`
3. **Timestamp**: `Timestamp(nanos_since_epoch={'__timestamp_micros_since_unix_epoch__': 1751326253127110})`
4. **Identity**: `Identity(data=b"{'__identity__': '0x...'}")`
5. **ConnectionId**: `ConnectionId(data=b"{'__connection_id__': 246634077236934735698079173074425116788}")`
6. **EnergyQuanta**: `EnergyQuanta({'quanta': 1134000})`
7. **ReducerCallInfo**: `ReducerCallInfo(reducer_name='', reducer_id=0, ...)`

## Required Fixes

### Fix 1: Add Dictionary-Like Behavior to Objects
**File**: `spacetimedb/database_update.py` (or similar)

Make objects behave like dictionaries by implementing `__getitem__` and `get` methods:

```python
class DatabaseUpdate:
    def __init__(self, tables=None):
        self.tables = tables or []
    
    def get(self, key, default=None):
        """Dictionary-like get method for backward compatibility"""
        if key == 'tables':
            return self.tables
        return default
    
    def __getitem__(self, key):
        """Dictionary-like access"""
        if key == 'tables':
            return self.tables
        raise KeyError(key)
    
    def keys(self):
        """Dictionary-like keys method"""
        return ['tables']

class TimeDuration:
    def __init__(self, nanos=None):
        self.nanos = nanos or {}
    
    def get(self, key, default=None):
        """Dictionary-like get method"""
        if key == 'nanos':
            return self.nanos
        elif key == '__time_duration_micros__':
            return self.nanos.get('__time_duration_micros__', default)
        return default
    
    def __getitem__(self, key):
        """Dictionary-like access"""
        if key == 'nanos':
            return self.nanos
        elif key == '__time_duration_micros__':
            return self.nanos['__time_duration_micros__']
        raise KeyError(key)

class Identity:
    def __init__(self, data=None):
        self.data = data
    
    def get(self, key, default=None):
        """Dictionary-like get method"""
        if key == 'data':
            return self.data
        return default
    
    def __getitem__(self, key):
        """Dictionary-like access"""
        if key == 'data':
            return self.data
        raise KeyError(key)

# Similar implementations for other object types...
```

### Fix 2: Implement Universal MixIn Class
**File**: `spacetimedb/base_objects.py` (new file)

Create a base class that provides dictionary-like behavior:

```python
class DictLikeMixin:
    """Mixin to provide dictionary-like behavior to objects"""
    
    def get(self, key, default=None):
        """Dictionary-like get method"""
        if hasattr(self, key):
            return getattr(self, key)
        return default
    
    def __getitem__(self, key):
        """Dictionary-like access"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)
    
    def __contains__(self, key):
        """Dictionary-like 'in' operator"""
        return hasattr(self, key)
    
    def keys(self):
        """Dictionary-like keys method"""
        return [attr for attr in dir(self) if not attr.startswith('_')]
    
    def items(self):
        """Dictionary-like items method"""
        return [(key, getattr(self, key)) for key in self.keys()]
    
    def values(self):
        """Dictionary-like values method"""
        return [getattr(self, key) for key in self.keys()]

# Apply to existing classes
class DatabaseUpdate(DictLikeMixin):
    def __init__(self, tables=None):
        self.tables = tables or []

class TimeDuration(DictLikeMixin):
    def __init__(self, nanos=None):
        self.nanos = nanos or {}

class Identity(DictLikeMixin):
    def __init__(self, data=None):
        self.data = data

# ... apply to all other classes
```

### Fix 3: Serialization Consistency
**File**: `spacetimedb/serialization.py`

Ensure consistent serialization format:

```python
def serialize_for_client(obj):
    """Serialize SpacetimeDB objects for client consumption"""
    if isinstance(obj, DatabaseUpdate):
        return {
            'tables': obj.tables
        }
    elif isinstance(obj, TimeDuration):
        return {
            'nanos': obj.nanos,
            '__time_duration_micros__': obj.nanos.get('__time_duration_micros__')
        }
    elif isinstance(obj, Identity):
        return {
            'data': obj.data
        }
    elif isinstance(obj, ConnectionId):
        return {
            'data': obj.data
        }
    elif isinstance(obj, EnergyQuanta):
        return {
            'quanta': obj.quanta if hasattr(obj, 'quanta') else obj.get('quanta')
        }
    elif isinstance(obj, Timestamp):
        return {
            'nanos_since_epoch': obj.nanos_since_epoch
        }
    elif isinstance(obj, ReducerCallInfo):
        return {
            'reducer_name': obj.reducer_name,
            'reducer_id': obj.reducer_id,
            'args': obj.args,
            'request_id': obj.request_id
        }
    else:
        return obj

def prepare_message_for_client(message_data):
    """Prepare entire message for client consumption"""
    result = {}
    for key, value in message_data.items():
        if hasattr(value, '__dict__'):  # It's an object
            result[key] = serialize_for_client(value)
        else:
            result[key] = value
    return result
```

### Fix 4: Protocol Version Compatibility
**File**: `spacetimedb/protocol_handler.py`

Add version-specific serialization:

```python
class ProtocolHandler:
    def __init__(self, version="v1.json.spacetimedb"):
        self.version = version
    
    def format_message(self, message_data):
        """Format message based on protocol version"""
        if self.version == "v1.json.spacetimedb":
            # Ensure all objects are dictionary-serializable
            return self._ensure_dict_compatible(message_data)
        return message_data
    
    def _ensure_dict_compatible(self, data):
        """Ensure all objects in data support dictionary operations"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if hasattr(value, 'get'):  # Already dict-like
                    result[key] = value
                elif hasattr(value, '__dict__'):  # Object that needs conversion
                    result[key] = serialize_for_client(value)
                else:
                    result[key] = value
            return result
        return data
```

## Integration Points

### Update Message Sending
**File**: `spacetimedb/connection.py`

```python
def send_message(self, message_data):
    """Send message with proper serialization"""
    # Ensure compatibility with client expectations
    compatible_data = prepare_message_for_client(message_data)
    
    # Send the compatible data
    self._send_raw(compatible_data)
```

## Testing Requirements

1. **Test dictionary operations** on all SpacetimeDB objects:
   - `.get()` method
   - `['key']` access
   - `'key' in object` checks
   - `.keys()`, `.values()`, `.items()` methods

2. **Backward compatibility testing**:
   - Existing clients should still work
   - New clients get proper object behavior

3. **Serialization consistency**:
   - Objects serialize to expected dictionary format
   - No loss of data during serialization

## Implementation Priority

1. **Phase 1**: Add `DictLikeMixin` to all object classes - **CRITICAL**
2. **Phase 2**: Update serialization for client compatibility - **HIGH**
3. **Phase 3**: Add protocol version handling - **MEDIUM**

## Impact Assessment

- **Breaking Change**: No - these are additive compatibility features
- **Performance**: Minimal overhead for dictionary-like operations
- **Client Impact**: Fixes existing AttributeError issues in blackholio-python-client

This fix ensures SpacetimeDB objects behave like dictionaries when accessed by client code, resolving the `'DatabaseUpdate' object has no attribute 'get'` errors.