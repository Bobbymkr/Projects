"""
Performance Optimization Utilities.

Async/await optimizations, concurrent execution helpers,
and performance enhancement utilities.
"""

from typing import List, TypeVar, Callable, Any, Coroutine, Optional
import asyncio
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def gather_with_limit(
    coros: List[Coroutine],
    limit: int = 10,
) -> List[Any]:
    """
    Execute coroutines with concurrency limit.
    
    Useful for limiting concurrent database queries or API calls.
    
    Args:
        coros: List of coroutines to execute
        limit: Maximum concurrent executions
        
    Returns:
        List of results in the same order as input
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[bounded_coro(coro) for coro in coros])


async def batch_process(
    items: List[Any],
        process_fn: Callable[[Any], Coroutine],
    batch_size: int = 10,
    max_concurrent: int = 5,
) -> List[Any]:
    """
    Process items in batches with concurrency control.
    
    Args:
        items: List of items to process
        process_fn: Async function to process each item
        batch_size: Number of items per batch
        max_concurrent: Maximum concurrent batches
        
    Returns:
        List of processed results
    """
    results = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch with concurrency limit
        batch_coros = [process_fn(item) for item in batch]
        batch_results = await gather_with_limit(batch_coros, limit=max_concurrent)
        results.extend(batch_results)
    
    return results


def async_timing(func: Callable) -> Callable:
    """
    Decorator to measure async function execution time.
    
    Logs execution time and can be used for performance monitoring.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.debug(
                f"{func.__name__} executed in {execution_time:.2f}ms",
                extra={
                    "function": func.__name__,
                    "execution_time_ms": execution_time,
                },
            )
    
    return wrapper


class AsyncBatchProcessor:
    """
    Efficient batch processor for async operations.
    
    Processes items in batches with configurable concurrency
    and automatic retry logic.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent: int = 5,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.retry_count = retry_count
        self.retry_delay = retry_delay
    
    async def process(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Coroutine],
    ) -> List[Any]:
        """
        Process items with automatic retry logic.
        
        Args:
            items: Items to process
            process_fn: Async processing function
            
        Returns:
            List of processed results
        """
        async def process_with_retry(item):
            last_error = None
            for attempt in range(self.retry_count):
                try:
                    return await process_fn(item)
                except Exception as e:
                    last_error = e
                    if attempt < self.retry_count - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"Failed to process item after {self.retry_count} attempts: {e}")
                        raise last_error
            
            return None
        
        return await batch_process(items, process_with_retry, self.batch_size, self.max_concurrent)


async def parallel_execute(
    tasks: List[Coroutine],
    timeout: Optional[float] = None,
) -> List[Any]:
    """
    Execute multiple async tasks in parallel with optional timeout.
    
    Args:
        tasks: List of coroutines to execute
        timeout: Optional timeout in seconds
        
    Returns:
        List of results
    """
    if timeout:
        return await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=timeout,
        )
    return await asyncio.gather(*tasks)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    async def time_execution(
        self,
        name: str,
        coro: Coroutine,
    ) -> Any:
        """
        Execute coroutine and track execution time.
        
        Args:
            name: Metric name
            coro: Coroutine to execute
            
        Returns:
            Result from coroutine
        """
        start_time = time.time()
        try:
            result = await coro
            execution_time = time.time() - start_time
            
            # Track metric
            if name not in self.metrics:
                self.metrics[name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "min_time": float("inf"),
                    "max_time": 0.0,
                }
            
            metric = self.metrics[name]
            metric["count"] += 1
            metric["total_time"] += execution_time
            metric["min_time"] = min(metric["min_time"], execution_time)
            metric["max_time"] = max(metric["max_time"], execution_time)
            
            return result
        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            raise
    
    def get_stats(self, name: str) -> dict:
        """Get performance statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        metric = self.metrics[name]
        avg_time = metric["total_time"] / metric["count"] if metric["count"] > 0 else 0
        
        return {
            "count": metric["count"],
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": metric["min_time"] * 1000,
            "max_time_ms": metric["max_time"] * 1000,
            "total_time_ms": metric["total_time"] * 1000,
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()

