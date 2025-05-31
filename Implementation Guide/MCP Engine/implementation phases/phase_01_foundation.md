# Phase 01: Project Foundation & Setup
**Duration**: Week 1-2 (Days 1-10)  
**Team Size**: 2-3 developers  
**Complexity**: Medium  

## Overview
Establish the foundational infrastructure, project structure, and core dependencies for the MCP Engine service. This phase sets up the development environment, basic configuration, and essential utilities that all subsequent phases will depend upon.

## Step 1: Project Structure & Dependencies (Days 1-3)

### Files to Create
```
mcp-engine/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── database.py
│   │   └── constants.py
│   ├── exceptions/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── flow_exceptions.py
│   │   └── state_exceptions.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── metrics.py
│       └── validators.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── unit/
├── scripts/
│   ├── setup_db.py
│   └── migrate.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

### `/requirements.txt`
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9
redis==5.0.1
kafka-python==2.0.2
grpcio==1.59.3
grpcio-tools==1.59.3
protobuf==4.25.1
asyncpg==0.29.0
motor==3.3.2
pymongo==4.6.0
celery==5.3.4
prometheus-client==0.19.0
structlog==23.2.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
aioredis==2.0.1
httpx==0.25.2
tenacity==8.2.3
jsonschema==4.20.0
jinja2==3.1.2
```

### `/src/config/settings.py`
**Purpose**: Central configuration management using Pydantic settings
```python
from pydantic import BaseSettings, Field
from typing import Optional, List
import os

class DatabaseSettings(BaseSettings):
    postgres_uri: str = Field(..., env="POSTGRES_URI")
    redis_url: str = Field(..., env="REDIS_URL") 
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    timescaledb_uri: Optional[str] = Field(None, env="TIMESCALEDB_URI")
    
    # Connection pool settings
    postgres_pool_size: int = Field(10, env="POSTGRES_POOL_SIZE")
    redis_pool_size: int = Field(20, env="REDIS_POOL_SIZE")

class ServiceSettings(BaseSettings):
    service_name: str = Field("mcp-engine", env="SERVICE_NAME")
    grpc_port: int = Field(50051, env="GRPC_PORT")
    http_port: int = Field(8002, env="HTTP_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    environment: str = Field("development", env="ENVIRONMENT")

class ExternalServiceSettings(BaseSettings):
    model_orchestrator_url: str = Field(..., env="MODEL_ORCHESTRATOR_URL")
    adaptor_service_url: str = Field(..., env="ADAPTOR_SERVICE_URL")
    security_hub_url: str = Field(..., env="SECURITY_HUB_URL")
    analytics_engine_url: str = Field(..., env="ANALYTICS_ENGINE_URL")

class PerformanceSettings(BaseSettings):
    max_parallel_executions: int = Field(100, env="MAX_PARALLEL_EXECUTIONS")
    state_execution_timeout_ms: int = Field(10000, env="STATE_EXECUTION_TIMEOUT_MS")
    context_lock_timeout_ms: int = Field(5000, env="CONTEXT_LOCK_TIMEOUT_MS")
    flow_cache_ttl_seconds: int = Field(300, env="FLOW_CACHE_TTL_SECONDS")

class FeatureFlags(BaseSettings):
    enable_ab_testing: bool = Field(True, env="ENABLE_AB_TESTING")
    enable_flow_versioning: bool = Field(True, env="ENABLE_FLOW_VERSIONING")
    enable_visual_designer: bool = Field(True, env="ENABLE_VISUAL_DESIGNER")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")

class Settings(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    service: ServiceSettings = ServiceSettings()
    external_services: ExternalServiceSettings = ExternalServiceSettings()
    performance: PerformanceSettings = PerformanceSettings()
    features: FeatureFlags = FeatureFlags()
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
```

### `/src/exceptions/base.py`
**Purpose**: Base exception classes for the MCP Engine
```python
from typing import Optional, Dict, Any
from enum import Enum

class ErrorCode(str, Enum):
    # Flow related errors
    FLOW_NOT_FOUND = "FLOW_NOT_FOUND"
    FLOW_INVALID = "FLOW_INVALID"
    FLOW_PARSE_ERROR = "FLOW_PARSE_ERROR"
    
    # State machine errors
    STATE_NOT_FOUND = "STATE_NOT_FOUND"
    STATE_EXECUTION_ERROR = "STATE_EXECUTION_ERROR"
    TRANSITION_ERROR = "TRANSITION_ERROR"
    
    # Context errors
    CONTEXT_LOCK_ERROR = "CONTEXT_LOCK_ERROR"
    CONTEXT_LOAD_ERROR = "CONTEXT_LOAD_ERROR"
    
    # Integration errors
    INTEGRATION_TIMEOUT = "INTEGRATION_TIMEOUT"
    INTEGRATION_ERROR = "INTEGRATION_ERROR"
    
    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"

class MCPBaseException(Exception):
    """Base exception for all MCP Engine errors"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details
        }

class ValidationError(MCPBaseException):
    """Raised when validation fails"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details={
                "field": field,
                "value": value,
                **(details or {})
            }
        )

class TimeoutError(MCPBaseException):
    """Raised when operations timeout"""
    
    def __init__(
        self,
        message: str,
        timeout_ms: int,
        operation: str
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTEGRATION_TIMEOUT,
            details={
                "timeout_ms": timeout_ms,
                "operation": operation
            }
        )
```

## Step 2: Logging & Metrics Infrastructure (Days 4-5)

### `/src/utils/logger.py`
**Purpose**: Structured logging configuration with tenant isolation
```python
import structlog
import logging
from typing import Any, Dict, Optional
from src.config.settings import settings

def configure_logging():
    """Configure structured logging for the application"""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.service.log_level.upper())
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

class MCPLogger:
    """Enhanced logger with tenant and conversation context"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def bind_context(
        self,
        tenant_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        flow_id: Optional[str] = None,
        **kwargs
    ) -> "MCPLogger":
        """Bind context to logger"""
        context = {}
        if tenant_id:
            context["tenant_id"] = tenant_id
        if conversation_id:
            context["conversation_id"] = conversation_id
        if user_id:
            context["user_id"] = user_id
        if flow_id:
            context["flow_id"] = flow_id
        context.update(kwargs)
        
        new_logger = MCPLogger(self.logger.name)
        new_logger.logger = self.logger.bind(**context)
        return new_logger
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message"""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)

def get_logger(name: str) -> MCPLogger:
    """Get logger instance"""
    return MCPLogger(name)
```

### `/src/utils/metrics.py`
**Purpose**: Prometheus metrics collection for monitoring
```python
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from typing import Dict, Any, Optional
from functools import wraps
import time
from src.config.settings import settings

# Create custom registry for the service
registry = CollectorRegistry()

# Define metrics
state_executions = Counter(
    'mcp_state_executions_total',
    'Total number of state executions',
    ['tenant_id', 'flow_id', 'state_name', 'status'],
    registry=registry
)

state_execution_duration = Histogram(
    'mcp_state_execution_duration_seconds',
    'Time spent executing states',
    ['tenant_id', 'flow_id', 'state_name'],
    registry=registry
)

flow_completions = Counter(
    'mcp_flow_completions_total',
    'Total number of flow completions',
    ['tenant_id', 'flow_id', 'completion_type'],
    registry=registry
)

integration_calls = Counter(
    'mcp_integration_calls_total',
    'Total number of integration calls',
    ['tenant_id', 'integration_id', 'status'],
    registry=registry
)

integration_duration = Histogram(
    'mcp_integration_duration_seconds',
    'Time spent on integration calls',
    ['tenant_id', 'integration_id'],
    registry=registry
)

active_conversations = Gauge(
    'mcp_active_conversations',
    'Number of active conversations',
    ['tenant_id'],
    registry=registry
)

context_cache_hits = Counter(
    'mcp_context_cache_hits_total',
    'Number of context cache hits',
    ['tenant_id'],
    registry=registry
)

context_cache_misses = Counter(
    'mcp_context_cache_misses_total',
    'Number of context cache misses',
    ['tenant_id'],
    registry=registry
)

class MetricsCollector:
    """Centralized metrics collection"""
    
    @staticmethod
    def record_state_execution(
        tenant_id: str,
        flow_id: str,
        state_name: str,
        status: str,
        duration_seconds: float
    ):
        """Record state execution metrics"""
        if not settings.features.enable_metrics:
            return
            
        state_executions.labels(
            tenant_id=tenant_id,
            flow_id=flow_id,
            state_name=state_name,
            status=status
        ).inc()
        
        state_execution_duration.labels(
            tenant_id=tenant_id,
            flow_id=flow_id,
            state_name=state_name
        ).observe(duration_seconds)
    
    @staticmethod
    def record_flow_completion(
        tenant_id: str,
        flow_id: str,
        completion_type: str
    ):
        """Record flow completion metrics"""
        if not settings.features.enable_metrics:
            return
            
        flow_completions.labels(
            tenant_id=tenant_id,
            flow_id=flow_id,
            completion_type=completion_type
        ).inc()
    
    @staticmethod
    def record_integration_call(
        tenant_id: str,
        integration_id: str,
        status: str,
        duration_seconds: float
    ):
        """Record integration call metrics"""
        if not settings.features.enable_metrics:
            return
            
        integration_calls.labels(
            tenant_id=tenant_id,
            integration_id=integration_id,
            status=status
        ).inc()
        
        integration_duration.labels(
            tenant_id=tenant_id,
            integration_id=integration_id
        ).observe(duration_seconds)
    
    @staticmethod
    def update_active_conversations(tenant_id: str, count: int):
        """Update active conversations count"""
        if not settings.features.enable_metrics:
            return
            
        active_conversations.labels(tenant_id=tenant_id).set(count)
    
    @staticmethod
    def record_cache_hit(tenant_id: str):
        """Record cache hit"""
        if not settings.features.enable_metrics:
            return
            
        context_cache_hits.labels(tenant_id=tenant_id).inc()
    
    @staticmethod
    def record_cache_miss(tenant_id: str):
        """Record cache miss"""
        if not settings.features.enable_metrics:
            return
            
        context_cache_misses.labels(tenant_id=tenant_id).inc()

def track_execution_time(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None
):
    """Decorator to track execution time"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success metrics based on metric_name
                if metric_name == "state_execution" and labels:
                    MetricsCollector.record_state_execution(
                        tenant_id=labels.get("tenant_id", "unknown"),
                        flow_id=labels.get("flow_id", "unknown"),
                        state_name=labels.get("state_name", "unknown"),
                        status="success",
                        duration_seconds=duration
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metrics
                if metric_name == "state_execution" and labels:
                    MetricsCollector.record_state_execution(
                        tenant_id=labels.get("tenant_id", "unknown"),
                        flow_id=labels.get("flow_id", "unknown"),
                        state_name=labels.get("state_name", "unknown"),
                        status="error",
                        duration_seconds=duration
                    )
                
                raise
        return wrapper
    return decorator
```

## Step 3: Database Connections & Base Repository (Days 6-8)

### `/src/config/database.py`
**Purpose**: Database connection management and session handling
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from motor.motor_asyncio import AsyncIOMotorClient
import aioredis
from typing import AsyncGenerator, Optional
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# SQLAlchemy setup for PostgreSQL
Base = declarative_base()
postgres_engine = None
async_session_maker = None

# MongoDB setup
mongodb_client: Optional[AsyncIOMotorClient] = None
mongodb_db = None

# Redis setup
redis_pool: Optional[aioredis.ConnectionPool] = None

async def init_postgres():
    """Initialize PostgreSQL connection"""
    global postgres_engine, async_session_maker
    
    try:
        postgres_engine = create_async_engine(
            settings.database.postgres_uri,
            pool_size=settings.database.postgres_pool_size,
            max_overflow=20,
            pool_pre_ping=True,
            echo=settings.service.log_level == "DEBUG"
        )
        
        async_session_maker = async_sessionmaker(
            postgres_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("PostgreSQL connection initialized")
        
    except Exception as e:
        logger.error("Failed to initialize PostgreSQL", error=e)
        raise

async def init_mongodb():
    """Initialize MongoDB connection"""
    global mongodb_client, mongodb_db
    
    try:
        mongodb_client = AsyncIOMotorClient(
            settings.database.mongodb_uri,
            maxPoolSize=50,
            minPoolSize=10,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        await mongodb_client.admin.command('ping')
        
        # Get database name from URI or use default
        db_name = settings.database.mongodb_uri.split('/')[-1] or "mcp_engine"
        mongodb_db = mongodb_client[db_name]
        
        logger.info("MongoDB connection initialized", database=db_name)
        
    except Exception as e:
        logger.error("Failed to initialize MongoDB", error=e)
        raise

async def init_redis():
    """Initialize Redis connection pool"""
    global redis_pool
    
    try:
        redis_pool = aioredis.ConnectionPool.from_url(
            settings.database.redis_url,
            max_connections=settings.database.redis_pool_size,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        redis = aioredis.Redis(connection_pool=redis_pool)
        await redis.ping()
        await redis.close()
        
        logger.info("Redis connection pool initialized")
        
    except Exception as e:
        logger.error("Failed to initialize Redis", error=e)
        raise

async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    """Get PostgreSQL async session"""
    if not async_session_maker:
        raise RuntimeError("PostgreSQL not initialized")
    
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def get_mongodb() -> AsyncIOMotorClient:
    """Get MongoDB database instance"""
    if not mongodb_db:
        raise RuntimeError("MongoDB not initialized")
    return mongodb_db

async def get_redis() -> aioredis.Redis:
    """Get Redis connection"""
    if not redis_pool:
        raise RuntimeError("Redis not initialized")
    return aioredis.Redis(connection_pool=redis_pool)

async def close_database_connections():
    """Close all database connections"""
    global postgres_engine, mongodb_client, redis_pool
    
    if postgres_engine:
        await postgres_engine.dispose()
        logger.info("PostgreSQL connection closed")
    
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")
    
    if redis_pool:
        await redis_pool.disconnect()
        logger.info("Redis connection pool closed")

class DatabaseManager:
    """Database connection manager"""
    
    @staticmethod
    async def initialize_all():
        """Initialize all database connections"""
        await init_postgres()
        await init_mongodb()
        await init_redis()
        logger.info("All database connections initialized")
    
    @staticmethod
    async def close_all():
        """Close all database connections"""
        await close_database_connections()
        logger.info("All database connections closed")
    
    @staticmethod
    async def health_check() -> Dict[str, bool]:
        """Check health of all database connections"""
        health = {}
        
        # Check PostgreSQL
        try:
            async with async_session_maker() as session:
                await session.execute("SELECT 1")
            health["postgres"] = True
        except Exception:
            health["postgres"] = False
        
        # Check MongoDB
        try:
            await mongodb_client.admin.command('ping')
            health["mongodb"] = True
        except Exception:
            health["mongodb"] = False
        
        # Check Redis
        try:
            redis = await get_redis()
            await redis.ping()
            await redis.close()
            health["redis"] = True
        except Exception:
            health["redis"] = False
        
        return health
```

## Step 4: Application Entry Point & Health Checks (Days 9-10)

### `/src/main.py`
**Purpose**: FastAPI application setup and lifecycle management
```python
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any
import signal
import asyncio

from src.config.settings import settings
from src.config.database import DatabaseManager
from src.utils.logger import configure_logging, get_logger
from src.utils.metrics import registry
from src.exceptions.base import MCPBaseException
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Configure logging before anything else
configure_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting MCP Engine service", version="2.0")
    
    try:
        # Initialize database connections
        await DatabaseManager.initialize_all()
        
        # Initialize other services here (will be added in later phases)
        
        logger.info(
            "MCP Engine service started successfully",
            grpc_port=settings.service.grpc_port,
            http_port=settings.service.http_port
        )
        
    except Exception as e:
        logger.error("Failed to start MCP Engine service", error=e)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Engine service")
    
    try:
        # Close database connections
        await DatabaseManager.close_all()
        
        # Cleanup other resources here
        
        logger.info("MCP Engine service shut down successfully")
        
    except Exception as e:
        logger.error("Error during service shutdown", error=e)

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="MCP Engine",
        description="Message Control Processor Engine for AI Chatbot Platform",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.service.environment != "production" else None,
        redoc_url="/redoc" if settings.service.environment != "production" else None
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    if settings.service.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure appropriately
        )
    
    # Global exception handler
    @app.exception_handler(MCPBaseException)
    async def mcp_exception_handler(request: Request, exc: MCPBaseException):
        """Handle MCP-specific exceptions"""
        logger.error(
            "MCP Exception occurred",
            error_code=exc.error_code,
            message=exc.message,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": exc.to_dict(),
                "meta": {
                    "service": "mcp-engine",
                    "version": "2.0.0"
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(
            "Unhandled exception occurred",
            error_type=type(exc).__name__,
            error_message=str(exc),
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred"
                },
                "meta": {
                    "service": "mcp-engine",
                    "version": "2.0.0"
                }
            }
        )
    
    # Health check endpoints
    @app.get("/health")
    async def health_check():
        """Basic health check"""
        return {
            "status": "healthy",
            "service": "mcp-engine",
            "version": "2.0.0"
        }
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check including database connections"""
        try:
            db_health = await DatabaseManager.health_check()
            
            overall_health = all(db_health.values())
            
            return {
                "status": "healthy" if overall_health else "unhealthy",
                "service": "mcp-engine",
                "version": "2.0.0",
                "components": {
                    "databases": db_health
                }
            }
        except Exception as e:
            logger.error("Health check failed", error=e)
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "service": "mcp-engine",
                    "version": "2.0.0",
                    "error": str(e)
                }
            )
    
    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint"""
        return Response(
            generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # TODO: Add API routes here in later phases
    
    return app

# Create the application instance
app = create_app()

async def start_grpc_server():
    """Start gRPC server (implementation in later phases)"""
    # This will be implemented in Phase 6
    pass

def main():
    """Main entry point"""
    try:
        # Start HTTP server
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=settings.service.http_port,
            reload=settings.service.environment == "development",
            log_config=None,  # Use our custom logging
            access_log=False  # Disable uvicorn access logs
        )
    except Exception as e:
        logger.error("Failed to start server", error=e)
        raise

if __name__ == "__main__":
    main()
```

### `/docker-compose.yml`
**Purpose**: Local development environment setup
```yaml
version: '3.8'

services:
  mcp-engine:
    build: .
    ports:
      - "8002:8002"
      - "50051:50051"
    environment:
      - POSTGRES_URI=postgresql+asyncpg://postgres:password@postgres:5432/mcp_engine
      - REDIS_URL=redis://redis:6379/0
      - MONGODB_URI=mongodb://mongo:27017/mcp_engine
      - MODEL_ORCHESTRATOR_URL=http://model-orchestrator:8003
      - ADAPTOR_SERVICE_URL=http://adaptor-service:8004
      - SECURITY_HUB_URL=http://security-hub:8001
      - ANALYTICS_ENGINE_URL=http://analytics-engine:8005
      - LOG_LEVEL=DEBUG
      - ENVIRONMENT=development
    depends_on:
      - postgres
      - redis
      - mongo
    volumes:
      - ./src:/app/src
    networks:
      - mcp-network

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mcp_engine
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - mcp-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - mcp-network

  mongo:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    networks:
      - mcp-network

volumes:
  postgres_data:
  redis_data:
  mongo_data:

networks:
  mcp-network:
    driver: bridge
```

## Success Criteria
- [x] Project structure established with all required folders
- [x] Configuration management system implemented
- [x] Logging infrastructure with structured logging
- [x] Metrics collection framework setup
- [x] Database connections configured (PostgreSQL, MongoDB, Redis)
- [x] Basic FastAPI application with health checks
- [x] Exception handling framework
- [x] Docker development environment
- [x] Development dependencies and tooling configured

## Key Error Handling & Performance Considerations
1. **Configuration**: Environment-based settings with validation
2. **Database Connections**: Connection pooling, health checks, graceful degradation
3. **Logging**: Structured logging with tenant isolation
4. **Metrics**: Prometheus integration with performance tracking
5. **Exception Handling**: Centralized error handling with proper HTTP status codes
6. **Docker**: Multi-stage builds for production optimization

## Technologies Used
- **Framework**: FastAPI, Uvicorn
- **Databases**: PostgreSQL (AsyncPG), MongoDB (Motor), Redis (aioredis)
- **Monitoring**: Prometheus, Structlog
- **Configuration**: Pydantic Settings
- **Containerization**: Docker, Docker Compose

## Next Phase Dependencies
Phase 2 will build upon:
- Database connection infrastructure
- Configuration system
- Logging and metrics framework
- Exception handling patterns
- Base repository patterns established here