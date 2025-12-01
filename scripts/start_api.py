#!/usr/bin/env python3
"""
Start the Production FastAPI Server.

Usage:
    python scripts/start_api.py
    python scripts/start_api.py --host 0.0.0.0 --port 8000
    python scripts/start_api.py --reload  # Development mode
"""

import argparse
import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Start the FastAPI application server."""
    parser = argparse.ArgumentParser(description="Start Adaptive Traffic Control API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Adaptive Traffic Control API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("=" * 60)
    print()
    print(f"API Documentation: http://{args.host}:{args.port}/api/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print()
    print("Starting server...")
    print()
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()

