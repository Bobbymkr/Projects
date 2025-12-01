"""
Real-Time Event Streaming System.

Provides event-driven architecture for real-time traffic updates,
system events, and notifications.
"""

from typing import AsyncIterator, Dict, Any, Optional, Set
from datetime import datetime
import asyncio
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event type enumeration."""
    TRAFFIC_UPDATE = "traffic_update"
    SYSTEM_STATUS = "system_status"
    ALERT = "alert"
    INTERSECTION_CHANGE = "intersection_change"
    DECISION_MADE = "decision_made"
    METRICS_UPDATE = "metrics_update"


class Event:
    """Event data structure."""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.event_type = event_type
        self.data = data
        self.source = source or "system"
        self.timestamp = timestamp or datetime.utcnow()
        self.id = f"{self.timestamp.isoformat()}-{id(self)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.event_type.value,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class EventStream:
    """Event stream for broadcasting events to subscribers."""
    
    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(
        self,
        event_type: EventType,
        queue: asyncio.Queue,
        subscriber_id: Optional[str] = None,
    ):
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            queue: Queue to receive events
            subscriber_id: Optional subscriber identifier
        """
        async with self._lock:
            key = event_type.value
            if key not in self._subscribers:
                self._subscribers[key] = set()
            self._subscribers[key].add(queue)
            logger.info(f"Subscriber {subscriber_id} subscribed to {event_type.value}")
    
    async def unsubscribe(
        self,
        event_type: EventType,
        queue: asyncio.Queue,
        subscriber_id: Optional[str] = None,
    ):
        """Unsubscribe from events."""
        async with self._lock:
            key = event_type.value
            if key in self._subscribers:
                self._subscribers[key].discard(queue)
                if not self._subscribers[key]:
                    del self._subscribers[key]
                logger.info(f"Subscriber {subscriber_id} unsubscribed from {event_type.value}")
    
    async def publish(self, event: Event):
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        async with self._lock:
            key = event.event_type.value
            subscribers = self._subscribers.get(key, set()).copy()
            
            # Also publish to wildcard subscribers
            all_subscribers = self._subscribers.get("*", set()).copy()
            subscribers.update(all_subscribers)
        
        if not subscribers:
            return
        
        # Broadcast to all subscribers
        failed_subscribers = []
        for queue in subscribers:
            try:
                await asyncio.wait_for(queue.put(event), timeout=0.1)
            except asyncio.TimeoutError:
                failed_subscribers.append(queue)
            except Exception as e:
                logger.error(f"Error publishing event to subscriber: {e}")
                failed_subscribers.append(queue)
        
        # Remove failed subscribers
        if failed_subscribers:
            async with self._lock:
                for queue in failed_subscribers:
                    for key in self._subscribers:
                        self._subscribers[key].discard(queue)
    
    async def subscribe_all(self, queue: asyncio.Queue, subscriber_id: Optional[str] = None):
        """Subscribe to all event types."""
        await self.subscribe(EventType.TRAFFIC_UPDATE, queue, subscriber_id)
        await self.subscribe(EventType.SYSTEM_STATUS, queue, subscriber_id)
        await self.subscribe(EventType.ALERT, queue, subscriber_id)
        await self.subscribe(EventType.INTERSECTION_CHANGE, queue, subscriber_id)
        await self.subscribe(EventType.DECISION_MADE, queue, subscriber_id)
        await self.subscribe(EventType.METRICS_UPDATE, queue, subscriber_id)


# Global event stream instance
event_stream = EventStream()


async def create_event_subscription(
    event_types: list[EventType],
    timeout: Optional[float] = None,
) -> AsyncIterator[Event]:
    """
    Create an async iterator for event subscription.
    
    Args:
        event_types: List of event types to subscribe to
        timeout: Optional timeout for iterator
        
    Yields:
        Events as they occur
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    subscriber_id = f"subscriber-{id(queue)}"
    
    try:
        # Subscribe to all requested event types
        for event_type in event_types:
            await event_stream.subscribe(event_type, queue, subscriber_id)
        
        # Yield events as they arrive
        while True:
            try:
                if timeout:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)
                else:
                    event = await queue.get()
                
                yield event
            except asyncio.TimeoutError:
                # Send heartbeat
                yield Event(
                    EventType.SYSTEM_STATUS,
                    {"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()},
                )
    finally:
        # Cleanup subscriptions
        for event_type in event_types:
            await event_stream.unsubscribe(event_type, queue, subscriber_id)

