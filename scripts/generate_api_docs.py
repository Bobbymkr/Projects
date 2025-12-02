#!/usr/bin/env python3
"""
Week 13: Generate API Documentation.

Generates OpenAPI specification and Swagger UI.
"""

import json
from pathlib import Path
from datetime import datetime


def generate_openapi_spec(output: Path):
    """Generate OpenAPI 3.0 specification."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Adaptive Traffic Control API",
            "version": "1.0.0",
            "description": "API for adaptive traffic signal control system",
            "contact": {
                "name": "API Support",
                "email": "support@adaptive-traffic.example.com"
            }
        },
        "servers": [
            {
                "url": "https://api.adaptive-traffic.example.com",
                "description": "Production server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ],
        "paths": {
            "/api/v1/traffic/decision": {
                "post": {
                    "summary": "Make traffic decision",
                    "description": "Get recommended traffic signal phase",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/TrafficDecisionRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/TrafficDecisionResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {
                        "200": {
                            "description": "Service is healthy"
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "TrafficDecisionRequest": {
                    "type": "object",
                    "required": ["intersection_id", "queue_lengths"],
                    "properties": {
                        "intersection_id": {
                            "type": "string",
                            "description": "Intersection identifier"
                        },
                        "queue_lengths": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Queue lengths per lane"
                        },
                        "wait_times": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Wait times per lane"
                        }
                    }
                },
                "TrafficDecisionResponse": {
                    "type": "object",
                    "properties": {
                        "recommended_phase": {
                            "type": "integer",
                            "description": "Recommended signal phase"
                        },
                        "green_time": {
                            "type": "integer",
                            "description": "Recommended green time in seconds"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Decision confidence"
                        }
                    }
                }
            }
        }
    }
    
    with open(output, 'w') as f:
        json.dump(spec, f, indent=2)
    
    print(f"OpenAPI specification saved to {output}")


def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Generate API documentation")
    parser.add_argument("--output", type=Path, default=Path("docs/api/openapi.json"), help="Output file")
    
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_openapi_spec(args.output)


if __name__ == "__main__":
    main()

