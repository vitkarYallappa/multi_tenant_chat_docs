# Phase 1: Foundation & Database Setup
**Duration**: Week 1-2 (14 days)  
**Team**: 2-3 developers  
**Dependencies**: None  

## Overview
Establish the foundational infrastructure for the Security Hub service, including project structure, database schemas, configuration management, and basic health monitoring.

## Step 1: Project Structure & Environment Setup

### Folders to Create
```
security-hub/
├── src/
│   ├── api/v2/          # REST API endpoints
│   ├── config/          # Configuration management
│   ├── models/          # Data models (PostgreSQL, Redis, Pydantic)
│   ├── utils/           # Utilities and helpers
│   ├── exceptions/      # Custom exceptions
│   └── main.py          # Application entry point
├── tests/               # Test suite
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container definition
└── docker-compose.yml  # Local development setup
```

### Key Files Implementation

#### `/src/config/settings.py`
**Purpose**: Central configuration management with environment variables  
**Technology**: Pydantic Settings, Python-dotenv  

**Classes & Methods**:
- `DatabaseSettings`: Database connection configurations
  - `postgres_dsn` (property): Returns PostgreSQL connection string
- `SecuritySettings`: Security-related configurations (JWT, encryption, sessions)
- `ServiceSettings`: Service identity and network settings
- `Settings`: Main settings class combining all sections

**Key Features**: Environment variable validation, type checking, default values

#### `/src/config/database.py`
**Purpose**: Database connection management and session handling  
**Technology**: SQLAlchemy async, Redis asyncio  

**Classes & Methods**:
- `DatabaseManager`: Manages all database connections
  - `initialize()`: Setup PostgreSQL and Redis connections
  - `get_postgres_session()`: Context manager for PostgreSQL sessions
  - `get_redis_client()`: Returns Redis client instance
  - `close()`: Cleanup all connections
  - `_test_connections()`: Health check for databases

**Performance Considerations**: Connection pooling, automatic retries, graceful degradation

#### `/src/models/postgres/base.py`
**Purpose**: Base PostgreSQL model with common fields  
**Technology**: SQLAlchemy, UUID, soft delete pattern  

**Classes & Methods**:
- `BaseModel`: Abstract base class for all PostgreSQL models
  - `to_dict()`: Convert model to dictionary
  - `soft_delete()`: Mark record as deleted without removing
  - `restore()`: Restore soft-deleted record
  - `get_table_name()`: Class method to get table name

**Common Fields**: ID (UUID), timestamps, audit fields, soft delete flags, versioning

#### `/src/utils/logging_config.py`
**Purpose**: Structured logging configuration  
**Technology**: Python logging, JSON formatting  

**Classes & Methods**:
- `JSONFormatter`: Custom formatter for structured logs
  - `format()`: Format log records as JSON with metadata
- `setup_logging()`: Initialize logging configuration
- `get_logger()`: Factory function for loggers with context

**Features**: Service metadata injection, exception tracking, configurable formats

## Step 2: Health Monitoring & Basic API

#### `/src/api/v2/health_routes.py`
**Purpose**: Health check endpoints for monitoring  
**Technology**: FastAPI, async/await  

**Classes & Methods**:
- `HealthChecker`: Service for system health monitoring
  - `check_postgres()`: Verify PostgreSQL connectivity
  - `check_redis()`: Verify Redis connectivity  
  - `get_system_info()`: Return service metadata

**Endpoints**:
- `GET /health/`: Basic health check (200 if running)
- `GET /health/detailed`: Comprehensive health with dependency status
- `GET /health/ready`: Kubernetes readiness probe
- `GET /health/live`: Kubernetes liveness probe

**Error Handling**: Graceful degradation, timeout handling, detailed error responses

## Step 3: Core Data Models

#### `/src/models/postgres/user_model.py`
**Purpose**: User and tenant data models  
**Technology**: SQLAlchemy, PostgreSQL, JSON fields  

**Models**:
- `Tenant`: Multi-tenant organization model
  - Fields: name, subdomain, status, plan_type, billing info, features, quotas
  - Methods: `get_active_users()`, `check_quota()`, `update_billing()`

- `TenantUser`: User model with tenant association
  - Fields: email, username, password_hash, role, permissions, MFA settings
  - Methods: `verify_password()`, `enable_mfa()`, `update_permissions()`

**Relationships**: One tenant to many users, role-based permissions

#### `/src/models/postgres/api_key_model.py`
**Purpose**: API key management model  
**Technology**: SQLAlchemy, hashed keys, expiration  

**Models**:
- `APIKey`: API key storage and management
  - Fields: key_hash, permissions, scopes, rate_limits, expiration
  - Methods: `verify_key()`, `update_usage()`, `check_permissions()`

**Security Features**: Key hashing, scope limitation, usage tracking

#### `/src/models/redis/session_store.py`
**Purpose**: Session management in Redis  
**Technology**: Redis, JSON serialization, TTL  

**Classes & Methods**:
- `SessionStore`: Manages user sessions
  - `create_session()`: Create new session with metadata
  - `get_session()`: Retrieve session data
  - `update_activity()`: Update last activity timestamp
  - `invalidate_session()`: Remove session
  - `cleanup_expired()`: Remove expired sessions

**Features**: Automatic expiration, device tracking, concurrent session limits

## Step 4: Exception Handling & Validation

#### `/src/exceptions/base_exceptions.py`
**Purpose**: Custom exception hierarchy  
**Technology**: Python exceptions, HTTP status codes  

**Exception Classes**:
- `SecurityHubException`: Base exception class
- `AuthenticationError`: Authentication failures
- `AuthorizationError`: Permission denied
- `ValidationError`: Input validation failures
- `DatabaseError`: Database operation failures
- `ConfigurationError`: Configuration issues

**Features**: HTTP status mapping, error codes, detailed messages

#### `/src/utils/validators.py`
**Purpose**: Input validation utilities  
**Technology**: Pydantic, regex patterns  

**Functions**:
- `validate_email()`: Email format validation
- `validate_password()`: Password complexity check
- `validate_api_key()`: API key format validation
- `validate_permissions()`: Permission syntax validation
- `sanitize_input()`: Input sanitization

## Cross-Service Integration Points

### Database Dependencies
- **PostgreSQL**: User data, tenant configurations, API keys
- **Redis**: Sessions, rate limiting, temporary data
- **Connection Pooling**: Shared across all services

### Health Check Integration
- **Load Balancers**: Use `/health/` for routing decisions
- **Kubernetes**: Readiness and liveness probes
- **Monitoring**: Detailed health for alerting

### Configuration Management
- **Environment Variables**: 12-factor app compliance
- **Service Discovery**: Dynamic configuration updates
- **Secrets Management**: Encrypted sensitive values

## Performance Considerations

### Database Optimization
- Connection pooling with min/max limits
- Async operations for better concurrency
- Prepared statements and query optimization
- Database connection health monitoring

### Caching Strategy
- Redis for frequently accessed data
- Session caching for authentication
- Configuration caching with TTL
- Connection reuse and pooling

### Error Handling
- Circuit breaker pattern for external dependencies
- Graceful degradation when services unavailable
- Retry logic with exponential backoff
- Comprehensive error logging

## Security Considerations

### Data Protection
- Password hashing with salt rounds
- API key hashing and secure storage
- Soft delete for audit trails
- Data encryption at rest preparation

### Access Control
- Role-based permission model
- API key scope limitations
- Session timeout enforcement
- Audit logging preparation

## Testing Requirements

### Unit Tests
- Configuration loading and validation
- Database connection management
- Model CRUD operations
- Validation functions

### Integration Tests
- Database connectivity
- Health check endpoints
- Session management
- Error handling flows

## Deployment Configuration

### Environment Variables
```
# Database
POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
REDIS_URL

# Security
JWT_SECRET_KEY, ENCRYPTION_KEY
SESSION_TIMEOUT_MINUTES=60
MAX_CONCURRENT_SESSIONS=5

# Service
SERVICE_NAME=security-hub
PORT=8005
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Docker Configuration
- Multi-stage build for optimization
- Health check integration
- Non-root user execution
- Security scanning preparation

## Success Criteria
- [ ] All database connections working
- [ ] Health endpoints responding correctly
- [ ] Basic user and tenant models functional
- [ ] Session management operational
- [ ] Configuration loading from environment
- [ ] Logging structured and working
- [ ] Unit tests passing (>80% coverage)
- [ ] Docker container builds and runs