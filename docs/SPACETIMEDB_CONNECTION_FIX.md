# SpacetimeDB Connection Issue Resolution

## Problem Summary

The Blackholio agent training script was failing to connect to SpacetimeDB with the error:
```
[Errno 8] nodename nor servname provided, or not known
```

## Root Cause Analysis

After extensive debugging, we discovered:

1. **SpacetimeDB v1.1.2 API Changes**: The WebSocket endpoints that the Python SDK expects no longer exist in SpacetimeDB v1.1.2. All WebSocket URLs return 404 errors.

2. **SDK Incompatibility**: The SpacetimeDB Python SDK (from `/Users/punk1290/git/spacetimedb-python-sdk/src`) is incompatible with SpacetimeDB v1.1.2.

3. **Server Status**: The SpacetimeDB server itself is running correctly and the module publishes successfully.

## Testing Results

### What Works:
- ✅ Docker container starts successfully
- ✅ SpacetimeDB server runs on port 3000
- ✅ Module builds and publishes using the CLI
- ✅ Training works perfectly in mock mode

### What Doesn't Work:
- ❌ WebSocket endpoints (all return 404)
- ❌ Python SDK connection to SpacetimeDB v1.1.2

## Solution

### Immediate Solution: Use Mock Mode

Run the training script with the `--mock` flag:

```bash
python scripts/train_agent.py \
  --total-timesteps 1000000 \
  --n-envs 8 \
  --experiment-name my_experiment \
  --mock
```

This uses a simulated environment that mimics the Blackholio game mechanics without requiring a SpacetimeDB connection.

### Long-term Solutions

1. **Update the Python SDK**: Wait for or contribute to an updated Python SDK that supports SpacetimeDB v1.1.2's new API.

2. **Use a Different SpacetimeDB Version**: Downgrade to a version compatible with the current SDK.

3. **Implement Direct WebSocket Connection**: Create a custom connection implementation that works with the new API.

## Fixed Issues

During debugging, we also fixed:

1. **Curriculum Update Bug**: Fixed a slicing error in `ppo_trainer.py` where metrics were being incorrectly accessed.

2. **Start Script**: Created `start_blackholio_fixed.sh` that properly builds and publishes the module.

## Verification

To verify the mock training is working:

```bash
# Check that training starts and progresses
python scripts/train_agent.py --mock --total-timesteps 10000 --n-envs 2

# Look for:
# - Training steps progressing
# - Rewards being logged
# - Checkpoints being saved
```

## Next Steps

1. Monitor SpacetimeDB Python SDK updates for v1.1.2 compatibility
2. Continue development using mock mode
3. Test with real SpacetimeDB connection once SDK is updated

## Technical Details

The connection issue occurs because:
- SDK expects WebSocket at: `ws://localhost:3000/database/ws/{database_id}`
- Server returns 404 for all WebSocket endpoints tested
- The API structure has likely changed in v1.1.2

Tested endpoints that all returned 404:
- `/ws`
- `/database/ws/blackholio`
- `/database/ws/{database_id}`
- Various other WebSocket URL patterns

This indicates a fundamental API change in SpacetimeDB v1.1.2 that the Python SDK hasn't been updated to handle.
