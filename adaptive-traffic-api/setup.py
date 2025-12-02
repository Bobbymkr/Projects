"""
Setup script for Adaptive Traffic API.
"""

from setuptools import setup, find_packages

setup(
    name="adaptive-traffic-api",
    version="1.0.0",
    description="REST API and WebSocket service layer for Adaptive Traffic",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.7.0",
        "sqlalchemy>=2.0.0",
        "redis>=5.0.0",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "python-multipart>=0.0.6",
        "adaptive-traffic-core",
        "adaptive-traffic-common",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-traffic-api=adaptive_traffic_api.api.main:main",
        ],
    },
)

