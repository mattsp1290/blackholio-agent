# Blackholio Agent Testing Documentation

## Overview

The Blackholio Agent project has a comprehensive testing framework covering unit tests, integration tests, behavior validation, and performance benchmarks. All tests can run without a live Blackholio server using the mock infrastructure.

## Test Categories

### 1. Unit Tests (`src/blackholio_agent/tests/unit/`)
- **Coverage**: Individual components and classes
- **Purpose**: Verify correct behavior of isolated units
- **Components tested**:
  - ObservationSpace
  - ActionSpace
  - RewardCalculator
  - BlackholioModel
  - RolloutBuffer
  - CheckpointManager
  - MetricsLogger
  - BlackholioConnection

### 2. Integration Tests (`src/blackholio_agent/tests/integration/`)
- **Coverage**: Component interactions
- **Purpose**: Verify system behavior when components work together
- **Tests include**:
  - Full environment workflow
  - Training pipeline integration
  - Model save/load with environment

### 3. Behavior Tests (`src/blackholio_agent/tests/behavior/`)
- **Coverage**: Agent decision-making patterns
- **Purpose**: Validate that trained agents exhibit expected behaviors
- **Scenarios tested**:
  - Food collection behavior
  - Threat evasion
  - Hunting smaller entities
  - Strategic splitting
  - Crowded area navigation
  - Exploration patterns

### 4. Performance Tests (`src/blackholio_agent/tests/performance/`)
- **Coverage**: System performance characteristics
- **Purpose**: Ensure acceptable performance metrics
- **Benchmarks include**:
  - Inference latency
  - Memory usage
  - Parallel environment scaling
  - Training throughput
  - Model operations

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python scripts/run_tests.py all

# Run specific test suite
python scripts/run_tests.py unit
python scripts/run_tests.py integration
python scripts/run_tests.py behavior
python scripts/run_tests.py performance

# Run tests with coverage
python scripts/test_coverage.py --suite all --analyze

# Run quick tests (exclude slow tests)
python scripts/run_tests.py quick
```

### Advanced Options

```bash
# Run tests in parallel
python scripts/run_tests.py unit --parallel 4

# Stop on first failure
python scripts/run_tests.py all --failfast

# Run specific test by keyword
python scripts/run_tests.py all --keyword "test_food_collection"

# Run tests with specific markers
python scripts/run_tests.py all --markers "not slow and not requires_gpu"
```

### Coverage Analysis

```bash
# Generate comprehensive coverage report
python scripts/test_coverage.py --analyze --badge

# Coverage for specific suites
python scripts/test_coverage.py --suite unit integration

# Set minimum coverage threshold
python scripts/test_coverage.py --minimum 85
```

## Mock Infrastructure

The project includes a sophisticated mock SpacetimeDB infrastructure that simulates:
- Game state updates
- Entity movements and interactions
- Network delays
- Connection lifecycle

This allows comprehensive testing without requiring a live game server.

### Using Mock Infrastructure

```python
from blackholio_agent.tests.fixtures.mock_spacetimedb import MockSpacetimeDBClient
from blackholio_agent.tests.fixtures.game_states import EARLY_GAME_SOLO

# Create mock client with scenario
mock_client = MockSpacetimeDBClient()
mock_client.setup_scenario(EARLY_GAME_SOLO)

# Use with environment
env = BlackholioEnv(mock_client)
```

## Continuous Integration

Tests run automatically on GitHub Actions for:
- Every push to main/develop branches
- Every pull request
- Multiple Python versions (3.8, 3.9, 3.10)

### CI Pipeline

1. **Test Job**: Runs unit, integration, and behavior tests
2. **Performance Job**: Runs performance benchmarks
3. **Lint Job**: Code quality checks (flake8, black, mypy)

## Performance Benchmarks

### Running Benchmarks

```bash
# Run all benchmarks
pytest src/blackholio_agent/tests/performance -m benchmark

# Generate performance report
python -c "from blackholio_agent.tests.performance.test_comprehensive_benchmarks import generate_performance_report; generate_performance_report()"
```

### Benchmark Metrics

- **Inference Latency**: < 100ms for batch size 1
- **Throughput**: > 10 FPS minimum
- **Memory Usage**: Tracked for different batch sizes
- **Parallel Scaling**: Near-linear for up to 8 environments

## Behavior Validation

### Running Behavior Tests

```bash
# Test specific behaviors
pytest src/blackholio_agent/tests/behavior/test_agent_behaviors.py::test_food_collection_behavior

# Generate behavior report
python -c "from blackholio_agent.tests.behavior.test_agent_behaviors import generate_behavior_report; generate_behavior_report('model.pt', 'behavior_report.json')"
```

### Behavior Metrics

- Movement patterns
- Split frequency
- Food collection rate
- Survival time
- Mass growth rate
- Aggression/evasion scores

## Test Fixtures

### Available Game Scenarios

- `EARLY_GAME_SOLO`: Single player, scattered food
- `EARLY_GAME_WITH_THREAT`: Player with larger threat nearby
- `MID_GAME_MULTI_ENTITY`: Multiple entities of various sizes
- `LATE_GAME_DOMINANT`: Large player dominating smaller ones
- `SPLIT_DECISION`: Scenario optimal for splitting
- `CROWDED_AREA`: Dense population of entities

### Creating Custom Scenarios

```python
from blackholio_agent.tests.fixtures.game_states import create_custom_scenario

scenario = create_custom_scenario(
    n_player_entities=2,
    n_enemies=5,
    n_food=20,
    arena_width=1000,
    arena_height=800
)
```

## Writing New Tests

### Unit Test Template

```python
import pytest
from blackholio_agent.component import Component

class TestComponent:
    @pytest.fixture
    def component(self):
        return Component()
    
    def test_basic_functionality(self, component):
        result = component.process(input_data)
        assert result == expected_output
    
    @pytest.mark.parametrize("input,expected", [
        (case1, result1),
        (case2, result2),
    ])
    def test_edge_cases(self, component, input, expected):
        assert component.process(input) == expected
```

### Integration Test Template

```python
@pytest.mark.integration
class TestSystemIntegration:
    def test_full_workflow(self, mock_connection):
        env = BlackholioEnv(mock_connection)
        model = BlackholioModel()
        
        obs = env.reset()
        action = model.get_action(obs)
        next_obs, reward, done, info = env.step(action)
        
        assert next_obs.shape == (456,)
        assert isinstance(reward, float)
```

## Test Configuration

### pytest.ini Settings

- Test discovery paths
- Marker definitions
- Output formatting
- Timeout settings
- Log configuration

### Coverage Configuration

- Minimum coverage: 80% (configurable)
- Branch coverage included
- HTML and XML reports
- Per-file analysis

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **Async Tests**: Install pytest-asyncio for async test support
3. **GPU Tests**: Use `--gpu` flag to include GPU-requiring tests
4. **Slow Tests**: Use `--markers "not slow"` to skip

### Debug Options

```bash
# Verbose output
pytest -vv

# Show print statements
pytest -s

# Debug on failure
pytest --pdb

# Show local variables on failure
pytest -l
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Mock External Dependencies**: Use mock SpacetimeDB for all tests
3. **Descriptive Names**: Test names should explain what they verify
4. **Fast Tests**: Keep unit tests under 1 second each
5. **Deterministic**: Use fixed seeds for reproducibility
6. **Coverage**: Aim for >80% coverage on new code

## Future Improvements

- [ ] Property-based testing with Hypothesis
- [ ] Mutation testing for test quality
- [ ] Load testing for production scenarios
- [ ] Visual regression testing for rendered gameplay
- [ ] Fuzzing for robustness testing
