# BlackHolio Python Client - Protocol Message Format Fixes Needed

## Issue Summary
The blackholio-python-client has message format detection issues causing "Unknown message type" warnings and preventing proper handling of SpacetimeDB messages.

## Root Cause Analysis

### 1. Message Type Detection Mismatch
**Location**: `blackholio_client/connection/protocol_handlers.py:85`

**Problem**: SpacetimeDB sends CamelCase message types but protocol handlers expect snake_case:
- **SpacetimeDB sends**: `TransactionUpdate`, `IdentityToken`, `InitialSubscription` 
- **Protocol handler expects**: `transaction_update`, `subscription_update`, etc.

**Current Code**:
```python
def _get_message_type(self, data: Dict[str, Any]) -> Optional[str]:
    # Looks for patterns like 'transaction_update' but receives 'TransactionUpdate'
    if 'transaction_update' in str(data).lower():
        return 'TransactionUpdate'
```

### 2. Unrecognized Message Structures
**Location**: `blackholio_client/connection/spacetimedb_connection.py:815`

**Problem**: New SpacetimeDB message formats aren't recognized:

**Actual messages received**:
```python
# Transaction commit response
{
    'status': 'Committed', 
    'timestamp': Timestamp(...),
    'caller_identity': Identity(...),
    'caller_connection_id': ConnectionId(...),
    'reducer_call': ReducerCallInfo(...),
    'energy_quanta_used': EnergyQuanta(...),
    'total_host_execution_duration': TimeDuration(...)
}

# Identity token response  
{
    'identity': Identity(...),
    'token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9...',
    'connection_id': ConnectionId(...)
}

# Database update response
{
    'database_update': DatabaseUpdate(tables=[]),
    'request_id': 411302231,
    'total_host_execution_duration': TimeDuration(...)
}
```

### 3. Binary vs JSON Protocol Confusion
**Location**: `blackholio_client/connection/spacetimedb_connection.py:638`

**Problem**: Client negotiates JSON protocol but receives binary frames, causing protocol mismatch warnings.

## Required Fixes

### Fix 1: Update Message Type Detection
**File**: `blackholio_client/connection/protocol_handlers.py`

Add recognition for the new message structures:
```python
def _get_message_type(self, data: Dict[str, Any]) -> Optional[str]:
    # Handle transaction commit responses
    if 'status' in data and 'timestamp' in data and 'caller_identity' in data:
        return 'TransactionCommit'
    
    # Handle identity token responses
    if 'identity' in data and 'token' in data and 'connection_id' in data:
        return 'IdentityToken'
    
    # Handle database update responses
    if 'database_update' in data and 'request_id' in data:
        return 'DatabaseUpdate'
    
    # Existing logic...
```

### Fix 2: Add New Message Handlers
**File**: `blackholio_client/connection/protocol_handlers.py`

```python
self.message_handlers = {
    'TransactionUpdate': self._handle_transaction_update,
    'TransactionCommit': self._handle_transaction_commit,  # NEW
    'SubscriptionUpdate': self._handle_subscription_update,
    'DatabaseUpdate': self._handle_database_update,        # NEW
    'Error': self._handle_error,
    'Connected': self._handle_connected,
    'Disconnected': self._handle_disconnected,
    'IdentityToken': self._handle_identity_token,          # NEW
}

def _handle_transaction_commit(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle transaction commit response"""
    return {
        'type': 'TransactionCommit',
        'status': data.get('status'),
        'timestamp': data.get('timestamp'),
        'energy_used': data.get('energy_quanta_used'),
        'execution_duration': data.get('total_host_execution_duration')
    }

def _handle_database_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle database update response"""
    return {
        'type': 'DatabaseUpdate',
        'tables': data.get('database_update', {}).get('tables', []),
        'request_id': data.get('request_id'),
        'execution_duration': data.get('total_host_execution_duration')
    }

def _handle_identity_token(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle identity token response"""
    return {
        'type': 'IdentityToken',
        'identity': data.get('identity'),
        'token': data.get('token'),
        'connection_id': data.get('connection_id')
    }
```

### Fix 3: Protocol Negotiation Fix
**File**: `blackholio_client/connection/spacetimedb_connection.py`

Ensure consistent protocol usage:
```python
# In connection setup, verify protocol consistency
if self.protocol == "v1.json.spacetimedb" and isinstance(message, bytes):
    logger.error("Protocol mismatch: negotiated JSON but received binary frame")
    # Convert or renegotiate protocol
```

## Testing Requirements

1. **Test message parsing** for all new message types
2. **Verify protocol consistency** - no binary frames with JSON protocol
3. **Test backward compatibility** with existing message handlers
4. **Integration test** with SpacetimeDB to ensure proper message flow

## Impact Assessment

- **High Priority**: These fixes are blocking proper game state synchronization
- **Breaking Change**: No - these are additive fixes
- **Dependencies**: May need spacetimedb-python-sdk updates for protocol handling

## Current Error Messages Being Fixed

```
WARNING - Unknown message type in data: {'status': 'Committed', 'timestamp': ...}
WARNING - Unknown message type in data: {'identity': Identity(...), 'token': '...'}
INFO - Received unrecognized message format: ['status', 'timestamp', 'caller_identity', ...]
INFO - Received unrecognized message format: ['identity', 'token', 'connection_id']
INFO - Received unrecognized message format: ['database_update', 'request_id', ...]
```

## Implementation Priority

1. **Phase 1**: Add new message type detection (Fix 1) - **CRITICAL**
2. **Phase 2**: Add new message handlers (Fix 2) - **HIGH** 
3. **Phase 3**: Protocol negotiation consistency (Fix 3) - **MEDIUM**

This should resolve the "Unknown message type" and "Received unrecognized message format" warnings currently preventing proper game state synchronization.