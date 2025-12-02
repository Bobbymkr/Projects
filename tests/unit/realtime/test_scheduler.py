"""
Unit tests for real-time scheduler.
"""

import pytest
import time
from src.realtime.scheduler import RealTimeScheduler, Task, Priority, get_scheduler


class TestTask:
    """Test Task class."""
    
    def test_task_creation(self):
        """Test task creation."""
        def dummy_func():
            return 42
        
        task = Task(
            func=dummy_func,
            deadline=time.time() + 1.0,
            priority=Priority.HIGH,
            task_id="test_task"
        )
        assert task.func == dummy_func
        assert task.priority == Priority.HIGH
        assert task.task_id == "test_task"
    
    def test_task_comparison(self):
        """Test task priority comparison."""
        def dummy_func():
            return 42
        
        task1 = Task(dummy_func, deadline=time.time() + 1.0, priority=Priority.CRITICAL)
        task2 = Task(dummy_func, deadline=time.time() + 2.0, priority=Priority.NORMAL)
        
        assert task1 < task2  # CRITICAL < NORMAL


class TestRealTimeScheduler:
    """Test real-time scheduler."""
    
    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = RealTimeScheduler(max_workers=2)
        assert scheduler.max_workers == 2
        assert len(scheduler.task_queue) == 0
        assert scheduler.running == False
    
    def test_submit_task(self):
        """Test task submission."""
        scheduler = RealTimeScheduler()
        
        def dummy_func():
            return 42
        
        task = scheduler.submit(
            dummy_func,
            deadline=time.time() + 1.0,
            priority=Priority.NORMAL,
            task_id="test"
        )
        
        assert task is not None
        assert len(scheduler.task_queue) == 1
    
    def test_get_stats(self):
        """Test getting scheduler statistics."""
        scheduler = RealTimeScheduler()
        stats = scheduler.get_stats()
        
        assert "queued_tasks" in stats
        assert "completed_tasks" in stats
        assert "missed_deadlines" in stats
        assert "deadline_compliance_rate" in stats


class TestGetScheduler:
    """Test global scheduler."""
    
    def test_get_scheduler(self):
        """Test getting global scheduler."""
        scheduler = get_scheduler()
        assert scheduler is not None
        assert isinstance(scheduler, RealTimeScheduler)

