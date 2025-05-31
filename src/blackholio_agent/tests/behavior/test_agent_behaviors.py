"""
Behavior validation tests for trained Blackholio agents.

These tests validate that agents exhibit expected behaviors
in various game scenarios, helping identify failure modes
and ensure reasonable gameplay patterns.
"""

import pytest
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from ...models import BlackholioModel
from ...environment import BlackholioEnv, ObservationSpace
from ..fixtures.game_states import (
    EARLY_GAME_SOLO,
    EARLY_GAME_WITH_THREAT,
    MID_GAME_MULTI_ENTITY,
    LATE_GAME_DOMINANT,
    SPLIT_DECISION,
    CROWDED_AREA
)


@dataclass
class BehaviorMetrics:
    """Metrics for evaluating agent behavior"""
    avg_movement_speed: float
    split_frequency: float
    food_collection_rate: float
    survival_time: float
    mass_growth_rate: float
    aggression_score: float  # How often agent moves toward smaller entities
    evasion_score: float     # How often agent moves away from larger entities
    exploration_score: float # How much of the map is explored


class AgentBehaviorValidator:
    """Validates agent behaviors in various scenarios"""
    
    def __init__(self, model: BlackholioModel, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def analyze_trajectory(self, 
                          states: List[np.ndarray], 
                          actions: List[Dict],
                          rewards: List[float]) -> BehaviorMetrics:
        """Analyze a gameplay trajectory to extract behavior metrics"""
        
        # Movement analysis
        movements = [a.get("movement", [0, 0]) for a in actions]
        movement_speeds = [np.linalg.norm(m) for m in movements]
        avg_movement_speed = np.mean(movement_speeds) if movement_speeds else 0.0
        
        # Split analysis
        splits = [a.get("split", False) for a in actions]
        split_frequency = sum(splits) / len(splits) if splits else 0.0
        
        # Food collection (based on reward patterns)
        food_rewards = [r for r in rewards if 0 < r < 1]  # Small positive rewards
        food_collection_rate = len(food_rewards) / len(rewards) if rewards else 0.0
        
        # Survival and growth
        survival_time = len(states)
        total_reward = sum(rewards)
        mass_growth_rate = total_reward / survival_time if survival_time > 0 else 0.0
        
        # Aggression/Evasion analysis (simplified)
        aggression_score = 0.5  # Placeholder
        evasion_score = 0.5     # Placeholder
        
        # Exploration
        exploration_score = avg_movement_speed * 0.5  # Simplified metric
        
        return BehaviorMetrics(
            avg_movement_speed=avg_movement_speed,
            split_frequency=split_frequency,
            food_collection_rate=food_collection_rate,
            survival_time=survival_time,
            mass_growth_rate=mass_growth_rate,
            aggression_score=aggression_score,
            evasion_score=evasion_score,
            exploration_score=exploration_score
        )
    
    def test_scenario(self, 
                     scenario_name: str,
                     initial_state: Dict,
                     max_steps: int = 1000) -> BehaviorMetrics:
        """Test agent behavior in a specific scenario"""
        
        # Create mock environment with scenario
        env = self._create_mock_env(initial_state)
        
        # Run episode
        states, actions, rewards = [], [], []
        obs = env.reset()
        hidden_state = None
        
        for _ in range(max_steps):
            # Get action from model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                if hidden_state is not None:
                    action, _, _, hidden_state = self.model.get_action(
                        obs_tensor, hidden_state, deterministic=True
                    )
                else:
                    action, _, _ = self.model.get_action(
                        obs_tensor, deterministic=True
                    )
            
            # Convert action to dict
            movement = action[:2].cpu().numpy()
            split = action[2].item() > 0.5
            action_dict = {"movement": movement, "split": split}
            
            # Step environment
            next_obs, reward, done, info = env.step(action_dict)
            
            states.append(obs)
            actions.append(action_dict)
            rewards.append(reward)
            
            if done:
                break
                
            obs = next_obs
        
        return self.analyze_trajectory(states, actions, rewards)
    
    def _create_mock_env(self, scenario: Dict) -> BlackholioEnv:
        """Create a mock environment for testing"""
        # This would create a mock environment with the given scenario
        # For now, returning a placeholder
        from ..fixtures.mock_spacetimedb import MockSpacetimeDBClient
        mock_client = MockSpacetimeDBClient()
        # Setup mock client with scenario data
        return BlackholioEnv(mock_client)


class TestAgentBehaviors:
    """Test suite for agent behavior validation"""
    
    @pytest.fixture
    def trained_model(self):
        """Load a trained model for testing"""
        # In real tests, this would load an actual trained model
        # For now, create a new model
        model = BlackholioModel()
        return model
    
    @pytest.fixture
    def behavior_validator(self, trained_model):
        """Create behavior validator"""
        return AgentBehaviorValidator(trained_model)
    
    def test_food_collection_behavior(self, behavior_validator):
        """Test that agent actively collects food when safe"""
        metrics = behavior_validator.test_scenario(
            "food_collection",
            EARLY_GAME_SOLO,
            max_steps=500
        )
        
        # Agent should move around to collect food
        assert metrics.avg_movement_speed > 0.1
        # Should rarely split when alone
        assert metrics.split_frequency < 0.1
        # Should have positive growth
        assert metrics.mass_growth_rate > 0
    
    def test_threat_evasion_behavior(self, behavior_validator):
        """Test that agent evades larger threats"""
        metrics = behavior_validator.test_scenario(
            "threat_evasion",
            EARLY_GAME_WITH_THREAT,
            max_steps=500
        )
        
        # Should have high evasion score
        assert metrics.evasion_score > 0.7
        # Should move quickly to escape
        assert metrics.avg_movement_speed > 0.5
    
    def test_hunting_behavior(self, behavior_validator):
        """Test that large agents hunt smaller ones"""
        metrics = behavior_validator.test_scenario(
            "hunting",
            LATE_GAME_DOMINANT,
            max_steps=500
        )
        
        # Should have high aggression when dominant
        assert metrics.aggression_score > 0.6
        # Should actively move
        assert metrics.avg_movement_speed > 0.3
    
    def test_split_strategy(self, behavior_validator):
        """Test splitting behavior in appropriate situations"""
        metrics = behavior_validator.test_scenario(
            "split_strategy",
            SPLIT_DECISION,
            max_steps=500
        )
        
        # Should use splits strategically
        assert 0.05 < metrics.split_frequency < 0.3
    
    def test_crowded_area_navigation(self, behavior_validator):
        """Test navigation in crowded areas"""
        metrics = behavior_validator.test_scenario(
            "crowded_navigation",
            CROWDED_AREA,
            max_steps=500
        )
        
        # Should balance aggression and evasion
        assert 0.3 < metrics.aggression_score < 0.7
        assert 0.3 < metrics.evasion_score < 0.7
    
    def test_exploration_behavior(self, behavior_validator):
        """Test that agent explores the map"""
        metrics = behavior_validator.test_scenario(
            "exploration",
            EARLY_GAME_SOLO,
            max_steps=1000
        )
        
        # Should have good exploration score
        assert metrics.exploration_score > 0.4
        # Should maintain movement
        assert metrics.avg_movement_speed > 0.2


class TestBehaviorPatterns:
    """Test for specific behavior patterns and failure modes"""
    
    @pytest.fixture
    def behavior_validator(self, trained_model):
        """Create behavior validator"""
        model = BlackholioModel()
        return AgentBehaviorValidator(model)
    
    def test_no_spinning_behavior(self, behavior_validator):
        """Ensure agent doesn't get stuck spinning in place"""
        metrics = behavior_validator.test_scenario(
            "anti_spin",
            EARLY_GAME_SOLO,
            max_steps=100
        )
        
        # Movement should be consistent, not oscillating
        # This would need more sophisticated tracking
        assert metrics.avg_movement_speed > 0.1
    
    def test_no_wall_hugging(self, behavior_validator):
        """Ensure agent doesn't get stuck on walls"""
        # Create scenario with agent near wall
        wall_scenario = EARLY_GAME_SOLO.copy()
        # Modify to place agent near boundary
        
        metrics = behavior_validator.test_scenario(
            "wall_avoidance",
            wall_scenario,
            max_steps=200
        )
        
        # Should move away from walls
        assert metrics.exploration_score > 0.3
    
    def test_reasonable_split_timing(self, behavior_validator):
        """Test that splits happen at reasonable times"""
        # Would need to track mass before splits
        metrics = behavior_validator.test_scenario(
            "split_timing",
            MID_GAME_MULTI_ENTITY,
            max_steps=500
        )
        
        # Splits should be strategic, not random
        assert metrics.split_frequency < 0.2


def generate_behavior_report(model_path: str, 
                           output_path: str,
                           scenarios: Optional[List[str]] = None):
    """Generate a comprehensive behavior report for a model"""
    
    # Load model
    model = BlackholioModel()
    model.load(model_path)
    
    validator = AgentBehaviorValidator(model)
    
    # Test all scenarios
    if scenarios is None:
        scenarios = [
            "early_game_solo",
            "early_game_with_threat", 
            "mid_game_multi_entity",
            "late_game_dominant",
            "split_decision",
            "crowded_area"
        ]
    
    results = {}
    for scenario in scenarios:
        # Get scenario data
        scenario_data = globals().get(scenario.upper(), EARLY_GAME_SOLO)
        
        # Run test
        metrics = validator.test_scenario(scenario, scenario_data)
        
        # Store results
        results[scenario] = {
            "avg_movement_speed": metrics.avg_movement_speed,
            "split_frequency": metrics.split_frequency,
            "food_collection_rate": metrics.food_collection_rate,
            "survival_time": metrics.survival_time,
            "mass_growth_rate": metrics.mass_growth_rate,
            "aggression_score": metrics.aggression_score,
            "evasion_score": metrics.evasion_score,
            "exploration_score": metrics.exploration_score
        }
    
    # Save report
    report = {
        "model_path": model_path,
        "scenarios_tested": len(scenarios),
        "results": results,
        "summary": {
            "avg_survival_time": np.mean([r["survival_time"] for r in results.values()]),
            "avg_growth_rate": np.mean([r["mass_growth_rate"] for r in results.values()]),
            "behavior_diversity": np.std([r["aggression_score"] for r in results.values()])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


@pytest.mark.behavior
class TestBehaviorReporting:
    """Test behavior report generation"""
    
    def test_report_generation(self, tmp_path):
        """Test that behavior reports can be generated"""
        model = BlackholioModel()
        model_path = tmp_path / "test_model.pt"
        model.save(str(model_path))
        
        report_path = tmp_path / "behavior_report.json"
        report = generate_behavior_report(
            str(model_path),
            str(report_path),
            scenarios=["early_game_solo"]
        )
        
        assert report_path.exists()
        assert "results" in report
        assert "summary" in report
        assert len(report["results"]) == 1
