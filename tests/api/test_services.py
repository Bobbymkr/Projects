"""
Unit Tests for Service Layer Components.

Tests for traffic controller service and business logic.
"""

import pytest
from unittest.mock import AsyncMock, patch


class TestTrafficControllerService:
    """Tests for traffic controller service."""
    
    @pytest.mark.asyncio
    async def test_make_decision(
        self,
    ):
        """Test traffic decision making."""
        from src.api.services.traffic_controller import TrafficController
        
        controller = TrafficController()
        
        decision = await controller.make_decision(
            intersection_id="int-001",
            queue_lengths=[12.5, 8.3, 15.2, 10.1],
            wait_times=[25.3, 18.7, 32.1, 22.5],
            throughput=450.0,
            current_phase=1,
        )
        
        assert "phase" in decision
        assert "green_time" in decision
        assert "confidence" in decision
        assert "algorithm" in decision
        
        # Validate ranges
        assert 0 <= decision["phase"] <= 3
        assert 5.0 <= decision["green_time"] <= 60.0
        assert 0.0 <= decision["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_make_decision_error_handling(
        self,
    ):
        """Test error handling in decision making."""
        from src.api.services.traffic_controller import TrafficController
        
        controller = TrafficController()
        
        # Test with invalid data (should still return safe default)
        decision = await controller.make_decision(
            intersection_id="int-001",
            queue_lengths=[],  # Empty list
            wait_times=[],
            throughput=0,
            current_phase=1,
        )
        
        # Should return a decision (even if fallback)
        assert "phase" in decision
        assert "green_time" in decision
    
    @pytest.mark.asyncio
    async def test_get_all_intersections(
        self,
    ):
        """Test getting all intersections."""
        from src.api.services.traffic_controller import TrafficController
        
        controller = TrafficController()
        intersections = await controller.get_all_intersections()
        
        assert isinstance(intersections, list)
    
    @pytest.mark.asyncio
    async def test_get_intersection(
        self,
    ):
        """Test getting single intersection."""
        from src.api.services.traffic_controller import TrafficController
        
        controller = TrafficController()
        intersection = await controller.get_intersection("int-001")
        
        # Should return None if not found (or intersection data if found)
        assert intersection is None or isinstance(intersection, dict)

