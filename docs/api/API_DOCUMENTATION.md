# API Documentation Guide
## Adaptive Traffic Control System

Complete API reference documentation with interactive examples and usage patterns.

---

## Overview

The Adaptive Traffic Control API provides a comprehensive RESTful interface for:
- Real-time traffic signal control
- Multi-agent coordination
- Performance analytics and metrics
- System monitoring and health checks
- WebSocket-based real-time updates

**Base URL**: `http://localhost:8000/api/v1`

---

## Quick Access

- **Interactive API Docs (Swagger UI)**: http://localhost:8000/api/docs
- **ReDoc Documentation**: http://localhost:8000/api/redoc
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json

---

## Authentication

The API supports multiple authentication methods:

### 1. OAuth2 Bearer Token (Recommended)

```bash
# Get access token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret"

# Use token in requests
curl -X GET "http://localhost:8000/api/v1/traffic/intersections" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 2. API Key (Alternative)

```bash
curl -X GET "http://localhost:8000/api/v1/traffic/intersections" \
  -H "X-API-Key: YOUR_API_KEY"
```

---

## Traffic Control Endpoints

### POST /traffic/decision

Make a single traffic signal decision for an intersection.

**Request Body**:
```json
{
  "intersection_id": "intersection_1",
  "current_state": {
    "queue_lengths": [5, 3, 8, 2],
    "wait_times": [12.5, 8.3, 15.2, 6.1],
    "arrival_rates": [0.3, 0.2, 0.4, 0.15]
  },
  "algorithm": "dqn",
  "metadata": {
    "timestamp": "2024-11-30T12:00:00Z",
    "episode": 100
  }
}
```

**Response**:
```json
{
  "decision": {
    "phase": 0,
    "duration": 30,
    "confidence": 0.92
  },
  "performance": {
    "expected_wait_reduction": 0.35,
    "queue_reduction": 0.28
  },
  "metadata": {
    "algorithm": "dqn",
    "processing_time_ms": 45,
    "model_version": "v2.1.0"
  }
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/traffic/decision" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "intersection_id": "intersection_1",
    "current_state": {
      "queue_lengths": [5, 3, 8, 2],
      "wait_times": [12.5, 8.3, 15.2, 6.1],
      "arrival_rates": [0.3, 0.2, 0.4, 0.15]
    },
    "algorithm": "dqn"
  }'
```

---

### POST /traffic/batch

Process multiple intersections in a single request.

**Request Body**:
```json
{
  "requests": [
    {
      "intersection_id": "intersection_1",
      "current_state": { ... }
    },
    {
      "intersection_id": "intersection_2",
      "current_state": { ... }
    }
  ],
  "algorithm": "dqn",
  "parallel": true
}
```

**Response**:
```json
{
  "results": [
    {
      "intersection_id": "intersection_1",
      "decision": { ... },
      "performance": { ... }
    },
    {
      "intersection_id": "intersection_2",
      "decision": { ... },
      "performance": { ... }
    }
  ],
  "summary": {
    "total_intersections": 2,
    "average_processing_time_ms": 52,
    "success_rate": 1.0
  }
}
```

---

### GET /traffic/intersections

List all available intersections.

**Query Parameters**:
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 10)
- `status`: Filter by status (active, inactive)

**Response**:
```json
{
  "intersections": [
    {
      "id": "intersection_1",
      "name": "Main St & Oak Ave",
      "location": {
        "latitude": 37.7749,
        "longitude": -122.4194
      },
      "status": "active",
      "phases": 4,
      "last_update": "2024-11-30T12:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 10,
    "total": 25,
    "total_pages": 3
  }
}
```

---

### GET /traffic/intersections/{id}

Get detailed information about a specific intersection.

**Response**:
```json
{
  "id": "intersection_1",
  "name": "Main St & Oak Ave",
  "location": { ... },
  "configuration": {
    "phases": 4,
    "min_phase_duration": 15,
    "max_phase_duration": 120,
    "yellow_duration": 3
  },
  "current_state": {
    "active_phase": 0,
    "phase_start_time": "2024-11-30T12:00:00Z",
    "queue_lengths": [5, 3, 8, 2],
    "wait_times": [12.5, 8.3, 15.2, 6.1]
  },
  "performance": {
    "average_wait_time": 10.5,
    "throughput": 450,
    "efficiency": 0.92
  }
}
```

---

## System Endpoints

### GET /system/health

Comprehensive health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-30T12:00:00Z",
  "version": "2.0.0",
  "components": {
    "api": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "ml_models": "healthy"
  },
  "uptime_seconds": 86400,
  "metrics": {
    "request_rate": 125.5,
    "error_rate": 0.01,
    "average_response_time_ms": 45
  }
}
```

---

### GET /system/status

Current system status and operational state.

**Response**:
```json
{
  "operational": true,
  "mode": "production",
  "active_intersections": 25,
  "active_connections": 150,
  "system_load": 0.65,
  "alerts": []
}
```

---

## Metrics Endpoints

### GET /metrics/kpis

Key Performance Indicators for traffic control.

**Query Parameters**:
- `start_time`: Start timestamp (ISO 8601)
- `end_time`: End timestamp (ISO 8601)
- `intersection_id`: Filter by intersection

**Response**:
```json
{
  "period": {
    "start": "2024-11-30T00:00:00Z",
    "end": "2024-11-30T12:00:00Z"
  },
  "kpis": {
    "average_wait_time": 8.5,
    "total_throughput": 12500,
    "queue_reduction": 0.32,
    "wait_time_reduction": 0.28,
    "efficiency_score": 0.91
  },
  "intersections": [
    {
      "id": "intersection_1",
      "kpis": { ... }
    }
  ]
}
```

---

### GET /metrics/performance

Performance metrics and analytics.

**Response**:
```json
{
  "metrics": {
    "api_latency": {
      "p50": 35,
      "p95": 85,
      "p99": 120,
      "max": 250
    },
    "decision_quality": {
      "average_confidence": 0.89,
      "optimal_decisions": 0.85
    },
    "throughput": {
      "requests_per_second": 125.5,
      "decisions_per_minute": 1500
    }
  },
  "algorithm_performance": {
    "dqn": {
      "average_wait_reduction": 0.32,
      "usage_count": 12500
    },
    "fuzzy": {
      "average_wait_reduction": 0.25,
      "usage_count": 8500
    }
  }
}
```

---

### GET /metrics/dashboard

Aggregated dashboard data.

**Response**:
```json
{
  "summary": {
    "total_intersections": 25,
    "active_intersections": 23,
    "total_decisions": 125000,
    "average_efficiency": 0.91
  },
  "real_time": {
    "current_wait_times": [8.5, 7.2, 9.1, 6.8],
    "current_queues": [4, 3, 6, 2],
    "active_phases": [0, 2, 1, 3]
  },
  "trends": {
    "wait_time_trend": [8.2, 8.3, 8.5, 8.4, 8.3],
    "throughput_trend": [1200, 1250, 1300, 1280, 1250]
  }
}
```

---

## Analytics Endpoints

### GET /analytics/algorithm-performance

Compare performance across different algorithms.

**Query Parameters**:
- `start_time`: Start timestamp
- `end_time`: End timestamp
- `intersection_id`: Filter by intersection

**Response**:
```json
{
  "period": { ... },
  "algorithms": {
    "dqn": {
      "average_wait_time": 8.5,
      "wait_reduction": 0.32,
      "queue_reduction": 0.28,
      "efficiency": 0.91,
      "usage_count": 12500,
      "confidence": 0.89
    },
    "fuzzy": {
      "average_wait_time": 10.2,
      "wait_reduction": 0.25,
      "queue_reduction": 0.22,
      "efficiency": 0.87,
      "usage_count": 8500,
      "confidence": 0.82
    }
  },
  "recommendations": [
    {
      "intersection_id": "intersection_1",
      "recommended_algorithm": "dqn",
      "expected_improvement": 0.15
    }
  ]
}
```

---

### GET /analytics/traffic-patterns

Analyze traffic patterns and trends.

**Response**:
```json
{
  "patterns": {
    "hourly": {
      "peak_hours": [7, 8, 9, 17, 18],
      "average_demand": [0.3, 0.8, 0.9, 0.7, 0.4]
    },
    "daily": {
      "weekday_vs_weekend": {
        "weekday_avg_wait": 9.5,
        "weekend_avg_wait": 7.2
      }
    },
    "seasonal": {
      "trends": [ ... ]
    }
  },
  "predictions": {
    "next_hour": {
      "expected_demand": 0.75,
      "recommended_config": { ... }
    }
  }
}
```

---

## WebSocket Endpoints

### /ws/traffic

Real-time traffic updates via WebSocket.

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/traffic');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Traffic update:', data);
};
```

**Message Format**:
```json
{
  "type": "traffic_update",
  "timestamp": "2024-11-30T12:00:00Z",
  "intersection_id": "intersection_1",
  "data": {
    "queue_lengths": [5, 3, 8, 2],
    "wait_times": [12.5, 8.3, 15.2, 6.1],
    "current_phase": 0
  }
}
```

---

### /ws/system

Real-time system status updates.

**Message Format**:
```json
{
  "type": "system_status",
  "timestamp": "2024-11-30T12:00:00Z",
  "status": {
    "operational": true,
    "active_intersections": 25,
    "system_load": 0.65
  }
}
```

---

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "field": "Additional error details"
  },
  "request_id": "uuid-of-request",
  "timestamp": "2024-11-30T12:00:00Z"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

---

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default**: 100 requests per minute per IP
- **Authenticated**: 1000 requests per minute per user
- **Headers**:
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

---

## Code Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Get access token
response = requests.post(
    f"{BASE_URL}/auth/login",
    data={"username": "admin", "password": "secret"}
)
token = response.json()["access_token"]

# Make traffic decision
headers = {"Authorization": f"Bearer {token}"}
decision = requests.post(
    f"{BASE_URL}/traffic/decision",
    json={
        "intersection_id": "intersection_1",
        "current_state": {
            "queue_lengths": [5, 3, 8, 2],
            "wait_times": [12.5, 8.3, 15.2, 6.1],
            "arrival_rates": [0.3, 0.2, 0.4, 0.15]
        },
        "algorithm": "dqn"
    },
    headers=headers
)
print(decision.json())
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000/api/v1';

// Get access token
const authResponse = await axios.post(`${BASE_URL}/auth/login`, {
  username: 'admin',
  password: 'secret'
});
const token = authResponse.data.access_token;

// Make traffic decision
const decisionResponse = await axios.post(
  `${BASE_URL}/traffic/decision`,
  {
    intersection_id: 'intersection_1',
    current_state: {
      queue_lengths: [5, 3, 8, 2],
      wait_times: [12.5, 8.3, 15.2, 6.1],
      arrival_rates: [0.3, 0.2, 0.4, 0.15]
    },
    algorithm: 'dqn'
  },
  {
    headers: { Authorization: `Bearer ${token}` }
  }
);
console.log(decisionResponse.data);
```

### cURL

```bash
# Get token
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret" | jq -r .access_token)

# Make decision
curl -X POST "http://localhost:8000/api/v1/traffic/decision" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "intersection_id": "intersection_1",
    "current_state": {
      "queue_lengths": [5, 3, 8, 2],
      "wait_times": [12.5, 8.3, 15.2, 6.1]
    },
    "algorithm": "dqn"
  }'
```

---

## Best Practices

1. **Use Authentication**: Always authenticate requests in production
2. **Handle Errors**: Implement proper error handling with retries
3. **Respect Rate Limits**: Monitor rate limit headers and throttle accordingly
4. **Use WebSockets**: For real-time updates, use WebSocket endpoints
5. **Cache Responses**: Cache metrics and analytics data appropriately
6. **Validate Inputs**: Validate request data before sending
7. **Monitor Health**: Regularly check `/system/health` endpoint

---

## Additional Resources

- [Getting Started Guide](../developer/GETTING_STARTED.md)
- [Architecture Documentation](../architecture/README.md)
- [Testing Guide](../testing/strategy.md)
- [OpenAPI Schema](http://localhost:8000/api/openapi.json)

---

**Last Updated**: November 30, 2024  
**API Version**: v1  
**Documentation Version**: 2.0.0

