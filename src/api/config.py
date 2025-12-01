"""
API Configuration using Pydantic Settings.

Centralized configuration management with environment variable support.
"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        # Try pydantic v2
        from pydantic import BaseSettings
    except ImportError:
        # Fallback to basic class for testing
        class BaseSettings:
            class Config:
                env_file = ".env"
                env_file_encoding = "utf-8"
                case_sensitive = True
from typing import List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production
    LOG_LEVEL: str = "INFO"
    
    # API Configuration
    API_VERSION: str = "v1"
    API_TITLE: str = "Adaptive Traffic Control API"
    ENABLE_DOCS: bool = True
    ENABLE_CORS: bool = True
    CORS_ORIGINS: List[str] = ["*"]
    
    # Security Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Rate Limiting
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Redis Configuration (for caching and rate limiting)
    ENABLE_REDIS: bool = os.getenv("ENABLE_REDIS", "false").lower() == "true"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str | None = os.getenv("REDIS_PASSWORD", None)
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
    
    # Cache Configuration
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    DEFAULT_CACHE_TTL: int = int(os.getenv("DEFAULT_CACHE_TTL", "300"))  # 5 minutes
    RESPONSE_CACHE_TTL: int = int(os.getenv("RESPONSE_CACHE_TTL", "300"))  # 5 minutes
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/traffic_control"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Monitoring Configuration
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 9090
    ENABLE_SENTRY: bool = os.getenv("ENABLE_SENTRY", "false").lower() == "true"
    SENTRY_DSN: str | None = os.getenv("SENTRY_DSN", None)
    ENABLE_STRUCTURED_LOGGING: bool = True
    LOG_FILE: str | None = os.getenv("LOG_FILE", None)
    
    # Performance Configuration
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT: float = 30.0
    MAX_CONCURRENT_REQUESTS: int = 100
    
    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL: float = 30.0
    WS_MAX_CONNECTIONS: int = 1000
    
    # FastAPI Configuration
    FASTAPI_REQUIRED: bool = os.getenv("FASTAPI_REQUIRED", "false").lower() == "true"
    FASTAPI_FALLBACK_MODE: str = os.getenv("FASTAPI_FALLBACK_MODE", "warn")  # warn, error, silent
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

