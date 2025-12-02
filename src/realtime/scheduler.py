"""
Week 9: Real-Time Scheduler.

Implements deadline-aware scheduling with priority queues for critical events.
"""

import heapq
import threading
import time
from typing import Callable, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum


class Priority(IntEnum):
    """Priority levels for tasks."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Task:
    """Task with deadline and priority."""
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    deadline: float = 0.0  # Absolute deadline timestamp
    priority: Priority = Priority.NORMAL
    task_id: str = ""
    
    def __lt__(self, other):
        """Compare tasks for priority queue."""
        # Critical priority first, then by deadline
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.deadline < other.deadline


class RealTimeScheduler:
    """Real-time scheduler with deadline awareness."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.task_queue = []
        self.lock = threading.Lock()
        self.workers = []
        self.running = False
        self.completed_tasks = 0
        self.missed_deadlines = 0
    
    def submit(self, func: Callable, deadline: float, priority: Priority = Priority.NORMAL,
               task_id: str = "", *args, **kwargs) -> Task:
        """Submit a task with deadline."""
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            deadline=deadline,
            priority=priority,
            task_id=task_id or f"task_{time.time()}"
        )
        
        with self.lock:
            heapq.heappush(self.task_queue, task)
        
        return task
    
    def start(self):
        """Start scheduler workers."""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(self.max_workers)
        ]
        
        for worker in self.workers:
            worker.start()
    
    def stop(self):
        """Stop scheduler workers."""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5)
    
    def _worker(self):
        """Worker thread that processes tasks."""
        while self.running:
            task = None
            
            with self.lock:
                if self.task_queue:
                    current_time = time.time()
                    
                    # Check if highest priority task can meet deadline
                    if self.task_queue[0].deadline < current_time:
                        # Deadline already missed
                        task = heapq.heappop(self.task_queue)
                        self.missed_deadlines += 1
                        print(f"Warning: Task {task.task_id} missed deadline")
                    else:
                        # Task can still meet deadline
                        task = heapq.heappop(self.task_queue)
            
            if task:
                try:
                    # Execute task
                    start_time = time.time()
                    result = task.func(*task.args, **task.kwargs)
                    execution_time = time.time() - start_time
                    
                    # Check if deadline was met
                    if time.time() > task.deadline:
                        self.missed_deadlines += 1
                        print(f"Warning: Task {task.task_id} exceeded deadline")
                    else:
                        self.completed_tasks += 1
                    
                    return result
                except Exception as e:
                    print(f"Error executing task {task.task_id}: {e}")
            else:
                time.sleep(0.01)  # Small sleep when no tasks
    
    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        with self.lock:
            return {
                "queued_tasks": len(self.task_queue),
                "completed_tasks": self.completed_tasks,
                "missed_deadlines": self.missed_deadlines,
                "deadline_compliance_rate": (
                    (self.completed_tasks - self.missed_deadlines) / max(self.completed_tasks, 1) * 100
                )
            }


# Global scheduler instance
_scheduler: Optional[RealTimeScheduler] = None


def get_scheduler() -> RealTimeScheduler:
    """Get or create global scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = RealTimeScheduler()
        _scheduler.start()
    return _scheduler

