# Phase 1: Foundation & Project Setup
**Duration:** Week 1  
**Steps:** 1-2 of 18

---

## 🎯 Objectives
- Set up project foundation and development environment
- Establish basic project structure and tooling
- Configure essential dependencies and development workflow

---

## 📋 Step 1: Project Infrastructure Setup

### What Will Be Implemented
- Project repository structure
- Development environment configuration
- Basic CI/CD pipeline setup
- Documentation framework

### Folders and Files Created

```
chat-service/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd-staging.yml
│       └── cd-production.yml
├── docs/
│   ├── README.md
│   ├── CONTRIBUTING.md
│   ├── API.md
│   └── DEPLOYMENT.md
├── scripts/
│   ├── setup.sh
│   ├── start.sh
│   ├── test.sh
│   └── health_check.sh
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

### File Documentation

#### `.github/workflows/ci.yml`
**Purpose:** Continuous Integration pipeline for automated testing  
**Usage:** Triggered on PR creation and commits to main branch

```yaml
name: CI Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      mongodb:
        image: mongo:7
        ports:
          - 27017:27017
      redis:
        image: redis:7
        ports:
          - 6379:6379
      postgres:
        image: postgres:15
        ports:
          - 5432:5432
        env:
          POSTGRES_PASSWORD: postgres
    
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      - name: Run linting
        run: |
          black --check src/
          isort --check-only src/
          flake8 src/
      - name: Run type checking
        run: mypy src/
      - name: Run tests
        run: pytest tests/ -v --cov=src/
```

#### `pyproject.toml`
**Purpose:** Python project configuration and build settings  
**Usage:** Defines project metadata, dependencies, and tool configurations

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "chat-service"
version = "2.0.0"
description = "Multi-tenant AI chatbot platform - Chat Service"
authors = [{name = "Development Team", email = "dev@company.com"}]
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "motor>=3.3.0",
    "redis>=5.0.0",
    "kafka-python>=2.0.2",
    "grpcio>=1.59.0",
    "grpcio-tools>=1.59.0",
    "pydantic>=2.4.0",
    "pydantic-settings>=2.0.0",
    "cryptography>=41.0.0",
    "prometheus-client>=0.18.0",
    "structlog>=23.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.6.0",
    "pre-commit>=3.5.0",
]

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=src --cov-report=html --cov-report=term-missing"
```

#### `scripts/setup.sh`
**Purpose:** Environment setup and dependency installation script  
**Usage:** Run once to set up development environment

```bash
#!/bin/bash
set -e

echo "🚀 Setting up Chat Service development environment..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
required_version="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "❌ Python 3.11+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Copy environment file
if [ ! -f .env ]; then
    echo "📄 Creating .env file..."
    cp .env.example .env
    echo "⚠️ Please update .env file with your configuration"
fi

echo "✅ Setup complete! Activate the environment with: source venv/bin/activate"
```

---

## 📋 Step 2: Basic Project Structure & Dependencies

### What Will Be Implemented
- Core Python package structure
- Essential configuration files
- Basic dependency injection framework
- Logging and monitoring setup

### Folders and Files Created

```
src/
├── __init__.py
├── main.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── constants.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── dependencies.py
└── exceptions/
    ├── __init__.py
    └── base_exceptions.py
```

### File Documentation

#### `src/main.py`
**Purpose:** FastAPI application entry point and server initialization  
**Usage:** Main application file that starts the Chat Service

**Methods:**

1. **create_app() -> FastAPI**
   - **Purpose:** Create and configure FastAPI application instance
   - **Parameters:** None
   - **Return:** FastAPI application instance
   - **Description:** Sets up middleware, routes, exception handlers, and startup/shutdown events

2. **startup_event() -> None**
   - **Purpose:** Initialize services and connections on application startup
   - **Parameters:** None
   - **Return:** None
   - **Description:** Establishes database connections, initializes caches, starts background tasks

3. **shutdown_event() -> None**
   - **Purpose:** Clean up resources on application shutdown
   - **Parameters:** None
   - **Return:** None
   - **Description:** Closes database connections, stops background tasks

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from src.config.settings import get_settings
from src.utils.logger import setup_logging
from src.exceptions.base_exceptions import setup_exception_handlers

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Chat Service...")
    await startup_event()
    yield
    # Shutdown
    logger.info("Shutting down Chat Service...")
    await shutdown_event()

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="Chat Service API",
        description="Multi-tenant AI chatbot platform - Chat Service",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Setup middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    return app

async def startup_event() -> None:
    """Initialize services and connections"""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    
    logger.info(
        "Chat Service starting",
        version="2.0.0",
        environment=settings.ENVIRONMENT
    )

async def shutdown_event() -> None:
    """Clean up resources"""
    logger.info("Chat Service shutdown complete")

app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
```

#### `src/config/settings.py`
**Purpose:** Application configuration management using Pydantic Settings  
**Usage:** Centralized configuration with environment variable support

**Classes:**

1. **Settings(BaseSettings)**
   - **Purpose:** Main configuration class with all application settings
   - **Fields:** Database URLs, service configurations, feature flags
   - **Methods:**
     - **get_database_url() -> str**: Returns formatted database connection string
     - **is_production() -> bool**: Checks if running in production environment

```python
from pydantic import BaseSettings, Field
from typing import List, Optional
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Environment
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = Field(default=False)
    
    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8001)
    
    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO
    
    # Database connections
    MONGODB_URI: str = Field(default="mongodb://localhost:27017")
    MONGODB_DATABASE: str = Field(default="chatbot_conversations")
    
    REDIS_URL: str = Field(default="redis://localhost:6379")
    REDIS_DB: int = Field(default=0)
    
    POSTGRESQL_URI: str = Field(default="postgresql://postgres:postgres@localhost:5432/chatbot_config")
    
    # Kafka
    KAFKA_BROKERS: List[str] = Field(default=["localhost:9092"])
    KAFKA_TOPIC_PREFIX: str = Field(default="chatbot.platform")
    
    # External services
    MCP_ENGINE_URL: str = Field(default="localhost:50051")
    SECURITY_HUB_URL: str = Field(default="localhost:50052")
    
    # Security
    JWT_SECRET_KEY: str = Field(default="dev-secret-change-in-production")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=60)
    
    # Performance
    MAX_CONNECTIONS_MONGO: int = Field(default=100)
    MAX_CONNECTIONS_REDIS: int = Field(default=50)
    REQUEST_TIMEOUT_MS: int = Field(default=30000)
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def get_database_url(self, db_type: str) -> str:
        """Get formatted database URL"""
        urls = {
            "mongodb": self.MONGODB_URI,
            "redis": self.REDIS_URL,
            "postgresql": self.POSTGRESQL_URI
        }
        return urls.get(db_type, "")
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == Environment.PRODUCTION

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

#### `src/utils/logger.py`
**Purpose:** Structured logging configuration using structlog  
**Usage:** Centralized logging setup with JSON formatting for production

**Functions:**

1. **setup_logging(log_level: str) -> None**
   - **Purpose:** Configure application-wide logging
   - **Parameters:** 
     - `log_level` (str): Logging level (DEBUG, INFO, WARNING, ERROR)
   - **Return:** None
   - **Description:** Sets up structured logging with appropriate formatters and processors

2. **get_logger(name: str = None) -> structlog.BoundLogger**
   - **Purpose:** Get a logger instance with optional name
   - **Parameters:**
     - `name` (str, optional): Logger name, defaults to caller module
   - **Return:** structlog.BoundLogger instance
   - **Description:** Returns a configured logger instance

```python
import structlog
import logging
import sys
from typing import Optional

def setup_logging(log_level: str = "INFO") -> None:
    """Setup structured logging configuration"""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
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

def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a configured logger instance"""
    return structlog.get_logger(name)
```

---

## 🔧 Technologies Used
- **Python 3.11+**: Main programming language
- **FastAPI**: Web framework for API development
- **Pydantic**: Data validation and settings management
- **structlog**: Structured logging
- **pytest**: Testing framework
- **GitHub Actions**: CI/CD pipeline
- **Docker**: Containerization (setup in next phase)

---

## ⚠️ Key Considerations

### Error Handling
- Comprehensive exception handling framework
- Structured error responses
- Request ID tracking for debugging

### Performance
- Async/await pattern throughout
- Connection pooling configuration
- Request timeout settings

### Security
- Environment-based configuration
- Secret management preparation
- CORS configuration

---

## 🎯 Success Criteria
- [ ] Project structure is established
- [ ] CI/CD pipeline is functional
- [ ] Development environment setup is automated
- [ ] Basic logging and configuration is working
- [ ] Code quality tools are configured and passing

---

## 📋 Next Phase Preview
Phase 2 will focus on implementing core data models and establishing database connections, building upon the foundation created in this phase.