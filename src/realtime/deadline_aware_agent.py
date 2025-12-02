"""
Week 9: Deadline-Aware Agent Wrapper.

Wraps agents with deadline awareness and fallback mechanisms.
"""

import time
from typing import Any, Optional, Callable
from src.realtime.scheduler import get_scheduler, Priority


class DeadlineAwareAgent:
    """Agent wrapper with deadline awareness."""
    
    def __init__(self, agent: Any, deadline_ms: int = 100, fallback_agent: Optional[Any] = None):
        self.agent = agent
        self.deadline_ms = deadline_ms
        self.fallback_agent = fallback_agent
        self.deadline_misses = 0
        self.fallback_activations = 0
    
    def select_action(self, state: Any, deadline_ms: Optional[int] = None) -> int:
        """Select action with deadline awareness."""
        deadline = deadline_ms or self.deadline_ms
        deadline_timestamp = time.time() + (deadline / 1000.0)
        
        try:
            # Try primary agent
            start_time = time.time()
            action = self.agent.select_action(state)
            elapsed_ms = (time.time() - start_time) * 1000
            
            if elapsed_ms > deadline:
                self.deadline_misses += 1
                # Use fallback if available
                if self.fallback_agent:
                    self.fallback_activations += 1
                    return self.fallback_agent.select_action(state)
            
            return action
        
        except Exception as e:
            # Fallback on error
            if self.fallback_agent:
                self.fallback_activations += 1
                return self.fallback_agent.select_action(state)
            raise
    
    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "deadline_misses": self.deadline_misses,
            "fallback_activations": self.fallback_activations,
            "deadline_ms": self.deadline_ms
        }

