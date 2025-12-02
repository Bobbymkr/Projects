"""
WebSocket Routes for Real-Time Updates.

Real-time bidirectional communication for live traffic updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Set, Optional
import json
import asyncio
import logging
from datetime import datetime

from ..schemas import TrafficUpdateMessage, SystemStatusMessage
from ..config import settings
from ..events.stream import event_stream, EventType, create_event_subscription, Event

logger = logging.getLogger(__name__)

router = APIRouter()

# Active WebSocket connections
active_connections: Set[WebSocket] = set()


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        """Accept and store new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all active connections."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@router.websocket("/traffic")
async def websocket_traffic_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time traffic updates.
    
    Enhanced with event streaming system for efficient real-time updates.
    
    Query Parameters:
        intersection_id: Optional filter for specific intersection
    """
    await manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Connected to traffic updates stream",
            },
            websocket,
        )
        
        # Subscribe to traffic events
        event_types = [
            EventType.TRAFFIC_UPDATE,
            EventType.DECISION_MADE,
            EventType.INTERSECTION_CHANGE,
        ]
        
        # Get intersection filter from query parameters
        intersection_id = websocket.query_params.get("intersection_id")
        
        # Filter events if intersection_id provided
        async for event in create_event_subscription(event_types, timeout=settings.WS_HEARTBEAT_INTERVAL):
            # Filter by intersection if specified
            if intersection_id:
                event_data = event.data
                if event_data.get("intersection_id") != intersection_id:
                    continue
            
            # Send event to client
            try:
                await manager.send_personal_message(event.to_dict(), websocket)
            except Exception as e:
                logger.error(f"Error sending event to WebSocket: {e}")
                break
            
            # Handle ping/pong
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                message = json.loads(data)
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                        websocket,
                    )
            except asyncio.TimeoutError:
                pass  # No message from client
            except json.JSONDecodeError:
                pass  # Invalid message
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e}")
                break
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)


@router.websocket("/system")
async def websocket_system_status(websocket: WebSocket):
    """WebSocket endpoint for real-time system status updates."""
    await manager.connect(websocket)
    
    try:
        # Subscribe to system events
        event_types = [
            EventType.SYSTEM_STATUS,
            EventType.ALERT,
            EventType.METRICS_UPDATE,
        ]
        
        async for event in create_event_subscription(event_types, timeout=10.0):
            # Send event to client
            try:
                await manager.send_personal_message(event.to_dict(), websocket)
            except Exception as e:
                logger.error(f"Error sending system event to WebSocket: {e}")
                break
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket system status error: {e}", exc_info=True)
        manager.disconnect(websocket)


@router.websocket("/events")
async def websocket_all_events(
    websocket: WebSocket,
    event_types: Optional[str] = None,
):
    """
    WebSocket endpoint for subscribing to all event types.
    
    Query Parameters:
        event_types: Comma-separated list of event types to subscribe to
                    (default: all events)
    """
    await manager.connect(websocket)
    
    try:
        # Parse event types
        if event_types:
            requested_types = [
                EventType(et.strip()) for et in event_types.split(",")
            ]
        else:
            requested_types = list(EventType)
        
        # Subscribe to requested event types
        async for event in create_event_subscription(requested_types, timeout=30.0):
            try:
                await manager.send_personal_message(event.to_dict(), websocket)
            except Exception as e:
                logger.error(f"Error sending event to WebSocket: {e}")
                break
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket events error: {e}", exc_info=True)
        manager.disconnect(websocket)

