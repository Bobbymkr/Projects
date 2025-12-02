# Adaptive Traffic API

REST API and WebSocket service layer for the Adaptive Traffic Signal Control System.

## Overview

This package provides a complete API layer for interacting with the traffic control system:

- **REST API**: FastAPI-based REST endpoints
- **WebSocket**: Real-time traffic updates
- **GraphQL**: GraphQL API support
- **Authentication**: OAuth2 and JWT-based authentication
- **Monitoring**: System health and performance monitoring
- **Security**: Rate limiting, input validation, security headers

## Installation

```bash
pip install -e .
```

## Dependencies

- `adaptive-traffic-core` - Core traffic control functionality
- `adaptive-traffic-common` - Shared utilities

## Quick Start

```bash
# Start the API server
python -m adaptive_traffic_api.api.main

# Or use the script
python scripts/start_api.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /api/v1/traffic/status` - Get traffic status
- `POST /api/v1/traffic/control` - Control traffic signals
- `GET /api/v1/metrics` - Get performance metrics
- `WS /ws/traffic` - WebSocket for real-time updates

## Documentation

API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
adaptive-traffic-api/
├── src/
│   ├── api/              # API code
│   │   ├── routes/       # API routes
│   │   ├── services/     # Business logic
│   │   ├── middleware/   # Middleware
│   │   └── auth/         # Authentication
│   └── security/         # Security utilities
├── scripts/              # Utility scripts
└── README.md
```

## Configuration

Set environment variables:
- `DATABASE_URL` - Database connection string
- `REDIS_URL` - Redis connection string
- `SECRET_KEY` - Secret key for JWT
- `LOG_LEVEL` - Logging level

## License

MIT License

