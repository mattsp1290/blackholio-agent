# Blackholio Server Docker Setup

This document explains how to run a local Blackholio server using Docker for development and testing.

## Prerequisites

1. **Docker**: Must be installed and running
2. **SpacetimeDB**: Located at `/Users/punk1290/git/SpacetimeDB`
3. **Blackholio Module**: Located at `/Users/punk1290/git/Blackholio/server-rust`
4. **Rust**: Required for building the Blackholio module
5. **curl**: For health checks and API calls

## Quick Start

### Starting the Server

```bash
./start_blackholio.sh
```

This script will:
1. Check that Docker is running
2. Build and start a SpacetimeDB container
3. Build the Blackholio Rust module
4. Publish the module to SpacetimeDB
5. Create a `.blackholio.env` file with connection details

### Stopping the Server

```bash
./stop_blackholio.sh
```

This script will:
1. Stop the SpacetimeDB container
2. Optionally remove the container
3. Optionally remove the data volume (WARNING: deletes all data)
4. Clean up the environment file

## Server Details

When running, the server is accessible at:
- **Host**: `localhost:3000`
- **Module**: `blackholio`
- **Container Name**: `spacetimedb-blackholio`

## Using with the Agent

### Update Connection Configuration

When the server is running, you can test with real connections:

```python
from blackholio_agent.environment import ConnectionConfig, BlackholioConnection

config = ConnectionConfig(
    host="localhost:3000",
    database="blackholio",
    auth_token=None,  # Will be generated
    namespace="blackholio"
)

connection = BlackholioConnection(config)
await connection.connect()
```

### Running Tests Against Real Server

```bash
# Start the server
./start_blackholio.sh

# Run integration tests against real server
BLACKHOLIO_TEST_REAL_SERVER=1 pytest src/blackholio_agent/tests/integration -v

# Run connection test
python examples/test_connection.py

# Stop the server when done
./stop_blackholio.sh
```

## Docker Management

### View Container Logs

```bash
docker logs -f spacetimedb-blackholio
```

### Check Container Status

```bash
docker ps | grep spacetimedb-blackholio
```

### Access Container Shell

```bash
docker exec -it spacetimedb-blackholio /bin/bash
```

## Troubleshooting

### Docker Not Running

If you see "Error: Docker is not running", start Docker Desktop or the Docker daemon.

### Port Already in Use

If port 3000 is already in use, edit `start_blackholio.sh` and change the `DB_PORT` variable.

### Build Failures

If the Rust module fails to build:
1. Ensure you have Rust installed: `rustup --version`
2. Install the WASM target: `rustup target add wasm32-unknown-unknown`
3. Check the build logs in the Blackholio module directory

### Connection Issues

If the agent can't connect:
1. Check the server is running: `docker ps`
2. Verify the health endpoint: `curl http://localhost:3000/health`
3. Check container logs: `docker logs spacetimedb-blackholio`

## Environment Variables

The scripts create a `.blackholio.env` file with:
- `SPACETIMEDB_HOST`: Server host and port
- `BLACKHOLIO_MODULE`: Module name
- `SPACETIMEDB_URI`: Full URI for connections
- `CONTAINER_NAME`: Docker container name

Source this file to use the variables:
```bash
source .blackholio.env
```

## Data Persistence

- Data is stored in a Docker volume named `spacetimedb-data`
- This persists between container restarts
- Only deleted if you explicitly choose to remove it

## Development Workflow

1. **Start server**: `./start_blackholio.sh`
2. **Develop and test**: Your agent code with real connections
3. **View logs**: `docker logs -f spacetimedb-blackholio`
4. **Stop server**: `./stop_blackholio.sh`

## Integration with CI/CD

For CI/CD pipelines, you can use these scripts in GitHub Actions:

```yaml
- name: Start Blackholio Server
  run: |
    ./start_blackholio.sh
    
- name: Run Integration Tests
  run: |
    BLACKHOLIO_TEST_REAL_SERVER=1 pytest tests/integration
    
- name: Stop Blackholio Server
  if: always()
  run: |
    ./stop_blackholio.sh
```

## Next Steps

With the local server running, you can:
1. Test the complete training pipeline with real connections
2. Validate agent behaviors in the actual game
3. Debug connection and synchronization issues
4. Benchmark real-world performance
