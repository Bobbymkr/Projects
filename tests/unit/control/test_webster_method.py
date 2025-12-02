"""
Unit tests for Webster method.
"""

import pytest
import numpy as np
from src.control.webster_method import WebsterMethod


class TestWebsterMethod:
    """Test Webster method."""
    
    @pytest.fixture
    def webster(self):
        """Create Webster method instance."""
        return WebsterMethod()
    
    def test_initialization(self, webster):
        """Test Webster method initialization."""
        assert webster is not None
        assert webster.lost_time == 5
        assert webster.saturation_flow == 1800
        assert webster.min_cycle == 60
        assert webster.max_cycle == 180
    
    def test_calculate_cycle_length(self, webster):
        """Test cycle length calculation."""
        flow_ratios = [0.3, 0.25, 0.35, 0.2]
        cycle_length = webster.calculate_cycle_length(flow_ratios)
        
        assert cycle_length is not None
        assert isinstance(cycle_length, (int, float))
        assert webster.min_cycle <= cycle_length <= webster.max_cycle
    
    def test_calculate_green_times(self, webster):
        """Test green time calculation."""
        cycle_length = 120
        flow_ratios = [0.3, 0.25, 0.35, 0.2]
        green_times = webster.calculate_green_times(cycle_length, flow_ratios)
        
        assert green_times is not None
        assert len(green_times) == len(flow_ratios)
        assert all(g > 0 for g in green_times)
    
    def test_get_action(self, webster):
        """Test getting action from state."""
        state = {"volumes": [100, 150, 120, 80]}
        action = webster.get_action(state)
        
        assert action is not None
        assert isinstance(action, dict)

