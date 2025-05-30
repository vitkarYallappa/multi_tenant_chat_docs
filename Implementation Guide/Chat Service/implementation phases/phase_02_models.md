# Phase 2: Core Models & Configuration
**Duration:** Week 2-3  
**Steps:** 3-4 of 18

---

## 🎯 Objectives
- Implement core data models and schemas
- Establish database connections and configurations
- Create foundational type definitions and validators
- Set up authentication and security models

---

## 📋 Step 3: Data Models & Type Definitions

### What Will Be Implemented
- Pydantic models for request/response validation
- MongoDB document models
- Redis data structures
- Common type definitions and enums

### Folders and Files Created

```
src/models/
├── __init__.py
├── base_model.py
├── types.py
├── mongo/
│   ├── __init__.py
│   ├── conversation_model.py
│   ├── message_model.py
│   └── session_model.py
├── redis/
│   ├── __init__.py
│   ├── session_cache.py
│   ├── rate_limit_cache.py
│   └── conversation_state.py
└── schemas/
    ├── __init__.py
    ├── request_schemas.py
    ├── response_schemas.py
    └── validation_schemas.py
```

### File Documentation

#### `src/models/types.py`
**Purpose:** Common type definitions, enums, and type aliases used across the application  
**Usage:** Centralized type definitions to ensure consistency

**Classes and Enums:**

1. **ChannelType(str, Enum)**
   - **Purpose:** Define supported communication channels
   - **Values:** WEB, WHATSAPP, MESSENGER, SLACK, TEAMS, SMS, VOICE

2. **MessageType(str, Enum)**
   - **Purpose:** Define supported message content types
   - **Values:** TEXT, IMAGE, FILE, AUDIO, VIDEO, LOCATION, QUICK_REPLY, CAROUSEL

3. **ConversationStatus(str, Enum)**
   - **Purpose:** Define conversation lifecycle states
   - **Values:** ACTIVE, COMPLETED, ABANDONED, ESCALATED, ERROR

```python
from enum import Enum
from typing import TypedDict, Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import UUID

# Enums for type safety
class ChannelType(str, Enum):
    WEB = "web"
    WHATSAPP = "whatsapp"
    MESSENGER = "messenger"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    VOICE = "voice"

class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    LOCATION = "location"
    QUICK_REPLY = "quick_reply"
    CAROUSEL = "carousel"

class ConversationStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"
    ERROR = "error"

class DeliveryStatus(str, Enum):
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"

class ProcessingStage(str, Enum):
    RECEIVED = "received"
    VALIDATED = "validated"
    NORMALIZED = "normalized"
    PROCESSED = "processed"
    RESPONDED = "responded"
    DELIVERED = "delivered"

# Type aliases for clarity
TenantId = str
UserId = str
ConversationId = str
MessageId = str
SessionId = str

# Base content models
class MediaContent(BaseModel):
    url: str = Field(..., regex=r'^https?://')
    type: str = Field(..., description="MIME type")
    size_bytes: int = Field(..., ge=0, le=52428800)  # Max 50MB
    alt_text: Optional[str] = Field(None, max_length=500)
    thumbnail_url: Optional[str] = None
    
    @validator('type')
    def validate_mime_type(cls, v):
        allowed_types = [
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'video/mp4', 'video/quicktime',
            'audio/mpeg', 'audio/wav', 'audio/ogg',
            'application/pdf', 'text/plain'
        ]
        if v not in allowed_types:
            raise ValueError(f'Unsupported media type: {v}')
        return v

class LocationContent(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy_meters: Optional[int] = Field(None, ge=0)
    address: Optional[str] = Field(None, max_length=500)

class QuickReply(BaseModel):
    title: str = Field(..., max_length=20)
    payload: str = Field(..., max_length=1000)
    content_type: str = Field(default="text")

class Button(BaseModel):
    type: str = Field(..., regex=r'^(postback|url|phone|share)$')
    title: str = Field(..., max_length=20)
    payload: Optional[str] = None
    url: Optional[str] = None

class MessageContent(BaseModel):
    type: MessageType
    text: Optional[str] = Field(None, max_length=4096)
    language: Optional[str] = Field(default="en", regex=r'^[a-z]{2}$')
    media: Optional[MediaContent] = None
    location: Optional[LocationContent] = None
    quick_replies: Optional[List[QuickReply]] = Field(None, max_items=10)
    buttons: Optional[List[Button]] = Field(None, max_items=3)
    
    @validator('text')
    def text_required_for_text_type(cls, v, values):
        if values.get('type') == MessageType.TEXT and not v:
            raise ValueError('Text content required for text messages')
        return v

class ChannelMetadata(BaseModel):
    platform_message_id: Optional[str] = None
    platform_user_id: Optional[str] = None
    thread_id: Optional[str] = None
    workspace_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ProcessingHints(BaseModel):
    priority: str = Field(default="normal", regex=r'^(low|normal|high|urgent)$')
    expected_response_type: Optional[MessageType] = None
    bypass_automation: bool = Field(default=False)
    require_human_review: bool = Field(default=False)
```

#### `src/models/mongo/conversation_model.py`
**Purpose:** MongoDB document model for conversation storage  
**Usage:** Represents conversation documents in MongoDB with all metadata

**Classes:**

1. **ConversationDocument(BaseModel)**
   - **Purpose:** MongoDB document structure for conversations
   - **Fields:** All conversation metadata, status, context, metrics
   - **Methods:**
     - **to_dict() -> Dict[str, Any]**: Convert to MongoDB document format
     - **from_dict(data: Dict[str, Any]) -> ConversationDocument**: Create from MongoDB data

```python
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from bson import ObjectId

from src.models.types import (
    ConversationStatus, ChannelType, TenantId, UserId, 
    ConversationId, SessionId, ChannelMetadata
)

class ConversationMetrics(BaseModel):
    message_count: int = Field(default=0)
    user_messages: int = Field(default=0)
    bot_messages: int = Field(default=0)
    response_time_avg_ms: Optional[int] = None
    response_time_max_ms: Optional[int] = None
    intent_switches: int = Field(default=0)
    escalation_triggers: int = Field(default=0)
    user_satisfaction: Optional[float] = Field(None, ge=1, le=5)
    completion_rate: Optional[float] = Field(None, ge=0, le=1)
    goal_achieved: bool = Field(default=False)

class ConversationContext(BaseModel):
    intent_history: List[str] = Field(default_factory=list)
    current_intent: Optional[str] = None
    intent_confidence: Optional[float] = Field(None, ge=0, le=1)
    entities: Dict[str, Any] = Field(default_factory=dict)
    slots: Dict[str, Any] = Field(default_factory=dict)
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    session_variables: Dict[str, Any] = Field(default_factory=dict)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)
    conversation_tags: List[str] = Field(default_factory=list)

class ConversationDocument(BaseModel):
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    conversation_id: ConversationId
    tenant_id: TenantId
    user_id: UserId
    session_id: Optional[SessionId] = None
    
    # Channel and metadata
    channel: ChannelType
    channel_metadata: Optional[ChannelMetadata] = None
    
    # Lifecycle
    status: ConversationStatus = ConversationStatus.ACTIVE
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    # Flow and state
    flow_id: Optional[str] = None
    flow_version: Optional[str] = None
    current_state: Optional[str] = None
    previous_states: List[str] = Field(default_factory=list)
    
    # Context and metrics
    context: ConversationContext = Field(default_factory=ConversationContext)
    metrics: ConversationMetrics = Field(default_factory=ConversationMetrics)
    
    # Business context
    business_context: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB document format"""
        data = self.dict(by_alias=True, exclude_none=True)
        if self.id:
            data["_id"] = self.id
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationDocument":
        """Create from MongoDB document"""
        if "_id" in data:
            data["id"] = data.pop("_id")
        return cls(**data)
    
    def update_last_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()
    
    def increment_message_count(self, is_user_message: bool = True) -> None:
        """Increment message counters"""
        self.metrics.message_count += 1
        if is_user_message:
            self.metrics.user_messages += 1
        else:
            self.metrics.bot_messages += 1
        self.update_last_activity()
```

#### `src/models/redis/session_cache.py`
**Purpose:** Redis data structures for session management  
**Usage:** Handle active session state and caching

**Classes:**

1. **SessionData(BaseModel)**
   - **Purpose:** Structure for session data stored in Redis
   - **Fields:** Session metadata, user info, conversation context
   - **Methods:**
     - **to_redis_hash() -> Dict[str, str]**: Convert to Redis hash format
     - **from_redis_hash(data: Dict[str, str]) -> SessionData**: Create from Redis data
     - **get_cache_key(tenant_id: str, session_id: str) -> str**: Generate Redis key

```python
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json

from src.models.types import (
    TenantId, UserId, ConversationId, SessionId, ChannelType
)

class SessionData(BaseModel):
    session_id: SessionId
    tenant_id: TenantId
    user_id: UserId
    conversation_id: Optional[ConversationId] = None
    channel: ChannelType
    
    # Session lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=1)
    )
    
    # Session context
    context: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # User information
    user_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Feature flags
    features: Dict[str, bool] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @staticmethod
    def get_cache_key(tenant_id: TenantId, session_id: SessionId) -> str:
        """Generate Redis cache key for session"""
        return f"session:{tenant_id}:{session_id}"
    
    def to_redis_hash(self) -> Dict[str, str]:
        """Convert to Redis hash format"""
        data = self.dict()
        
        # Convert complex fields to JSON strings
        for field in ['context', 'preferences', 'user_info', 'features']:
            if field in data:
                data[field] = json.dumps(data[field])
        
        # Convert datetime fields to ISO strings
        for field in ['created_at', 'last_activity', 'expires_at']:
            if field in data and data[field]:
                data[field] = data[field].isoformat()
        
        # Convert all values to strings for Redis
        return {k: str(v) for k, v in data.items() if v is not None}
    
    @classmethod
    def from_redis_hash(cls, data: Dict[str, str]) -> "SessionData":
        """Create SessionData from Redis hash"""
        processed_data = {}
        
        for key, value in data.items():
            if key in ['context', 'preferences', 'user_info', 'features']:
                processed_data[key] = json.loads(value) if value else {}
            elif key in ['created_at', 'last_activity', 'expires_at']:
                processed_data[key] = datetime.fromisoformat(value) if value else None
            else:
                processed_data[key] = value
        
        return cls(**processed_data)
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at
    
    def extend_session(self, hours: int = 1) -> None:
        """Extend session expiration"""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
```

---

## 📋 Step 4: Database Connections & Configuration

### What Will Be Implemented
- Database connection managers
- Configuration validation
- Connection pooling setup
- Health check endpoints

### Folders and Files Created

```
src/database/
├── __init__.py
├── mongodb.py
├── redis_client.py
├── postgresql.py
└── health_checks.py

src/config/
├── database_config.py
└── redis_config.py
```

### File Documentation

#### `src/database/mongodb.py`
**Purpose:** MongoDB connection manager and client setup  
**Usage:** Provides async MongoDB client with connection pooling

**Classes:**

1. **MongoDBManager**
   - **Purpose:** Manage MongoDB connections and operations
   - **Methods:**
     - **connect() -> None**: Establish MongoDB connection
     - **disconnect() -> None**: Close MongoDB connection
     - **get_database() -> AsyncIOMotorDatabase**: Get database instance
     - **health_check() -> bool**: Check MongoDB health

```python
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError
from typing import Optional
import structlog

from src.config.settings import get_settings

logger = structlog.get_logger()

class MongoDBManager:
    """MongoDB connection manager"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.settings = get_settings()
    
    async def connect(self) -> None:
        """Establish MongoDB connection"""
        try:
            self.client = AsyncIOMotorClient(
                self.settings.MONGODB_URI,
                maxPoolSize=self.settings.MAX_CONNECTIONS_MONGO,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
            )
            
            # Test connection
            await self.client.admin.command('ping')
            
            self.database = self.client[self.settings.MONGODB_DATABASE]
            
            logger.info(
                "MongoDB connected successfully",
                database=self.settings.MONGODB_DATABASE,
                max_pool_size=self.settings.MAX_CONNECTIONS_MONGO
            )
            
        except ServerSelectionTimeoutError as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            raise ConnectionError(f"MongoDB connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def get_database(self) -> AsyncIOMotorDatabase:
        """Get database instance"""
        if not self.database:
            raise RuntimeError("Database not connected")
        return self.database
    
    async def health_check(self) -> bool:
        """Check MongoDB health"""
        try:
            if self.client:
                await self.client.admin.command('ping')
                return True
            return False
        except Exception as e:
            logger.error("MongoDB health check failed", error=str(e))
            return False
    
    async def create_indexes(self) -> None:
        """Create required database indexes"""
        if not self.database:
            return
        
        # Conversations collection indexes
        conversations = self.database.conversations
        await conversations.create_index([
            ("tenant_id", 1), ("started_at", -1)
        ])
        await conversations.create_index("conversation_id", unique=True)
        await conversations.create_index([
            ("tenant_id", 1), ("user_id", 1), ("started_at", -1)
        ])
        await conversations.create_index([
            ("tenant_id", 1), ("status", 1), ("last_activity_at", -1)
        ])
        
        # Messages collection indexes
        messages = self.database.messages
        await messages.create_index([
            ("conversation_id", 1), ("sequence_number", 1)
        ])
        await messages.create_index("message_id", unique=True)
        await messages.create_index([
            ("tenant_id", 1), ("timestamp", -1)
        ])
        
        logger.info("MongoDB indexes created successfully")

# Global MongoDB manager instance
mongodb_manager = MongoDBManager()

async def get_mongodb() -> AsyncIOMotorDatabase:
    """Dependency to get MongoDB database"""
    return mongodb_manager.get_database()
```

#### `src/database/redis_client.py`
**Purpose:** Redis connection manager and client setup  
**Usage:** Provides Redis client with connection pooling and cluster support

**Classes:**

1. **RedisManager**
   - **Purpose:** Manage Redis connections and operations
   - **Methods:**
     - **connect() -> None**: Establish Redis connection
     - **disconnect() -> None**: Close Redis connection
     - **get_client() -> Redis**: Get Redis client instance
     - **health_check() -> bool**: Check Redis health

```python
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from typing import Optional
import structlog

from src.config.settings import get_settings

logger = structlog.get_logger()

class RedisManager:
    """Redis connection manager"""
    
    def __init__(self):
        self.client: Optional[Redis] = None
        self.settings = get_settings()
    
    async def connect(self) -> None:
        """Establish Redis connection"""
        try:
            self.client = redis.from_url(
                self.settings.REDIS_URL,
                max_connections=self.settings.MAX_CONNECTIONS_REDIS,
                socket_timeout=10,
                socket_connect_timeout=10,
                decode_responses=True,
                encoding="utf-8"
            )
            
            # Test connection
            await self.client.ping()
            
            logger.info(
                "Redis connected successfully",
                url=self.settings.REDIS_URL,
                max_connections=self.settings.MAX_CONNECTIONS_REDIS
            )
            
        except RedisConnectionError as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise ConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")
    
    def get_client(self) -> Redis:
        """Get Redis client instance"""
        if not self.client:
            raise RuntimeError("Redis not connected")
        return self.client
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            if self.client:
                await self.client.ping()
                return True
            return False
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False
    
    async def set_with_ttl(
        self, 
        key: str, 
        value: str, 
        ttl_seconds: int
    ) -> bool:
        """Set key with TTL"""
        try:
            result = await self.client.setex(key, ttl_seconds, value)
            return result
        except Exception as e:
            logger.error("Redis set operation failed", key=key, error=str(e))
            return False
    
    async def get_hash(self, key: str) -> dict:
        """Get hash from Redis"""
        try:
            return await self.client.hgetall(key)
        except Exception as e:
            logger.error("Redis hash get failed", key=key, error=str(e))
            return {}
    
    async def set_hash(self, key: str, mapping: dict, ttl_seconds: int = None) -> bool:
        """Set hash in Redis with optional TTL"""
        try:
            pipe = self.client.pipeline()
            pipe.hset(key, mapping=mapping)
            if ttl_seconds:
                pipe.expire(key, ttl_seconds)
            await pipe.execute()
            return True
        except Exception as e:
            logger.error("Redis hash set failed", key=key, error=str(e))
            return False

# Global Redis manager instance
redis_manager = RedisManager()

async def get_redis() -> Redis:
    """Dependency to get Redis client"""
    return redis_manager.get_client()
```

#### `src/database/health_checks.py`
**Purpose:** Database health check implementations  
**Usage:** Provides health check endpoints for monitoring

**Functions:**

1. **check_all_databases() -> Dict[str, bool]**
   - **Purpose:** Check health of all database connections
   - **Parameters:** None
   - **Return:** Dict with database names and health status
   - **Description:** Performs health checks on MongoDB, Redis, and PostgreSQL

2. **check_mongodb() -> bool**
   - **Purpose:** Check MongoDB connection health
   - **Parameters:** None
   - **Return:** True if healthy, False otherwise
   - **Description:** Performs ping operation to verify MongoDB connectivity

```python
from typing import Dict, Any
import structlog
import asyncio

from src.database.mongodb import mongodb_manager
from src.database.redis_client import redis_manager

logger = structlog.get_logger()

async def check_all_databases() -> Dict[str, Any]:
    """Check health of all database connections"""
    health_status = {
        "status": "healthy",
        "databases": {},
        "timestamp": None
    }
    
    # Run all health checks concurrently
    mongodb_task = check_mongodb()
    redis_task = check_redis()
    
    mongodb_healthy, redis_healthy = await asyncio.gather(
        mongodb_task, redis_task, return_exceptions=True
    )
    
    # Process results
    health_status["databases"]["mongodb"] = {
        "healthy": isinstance(mongodb_healthy, bool) and mongodb_healthy,
        "error": str(mongodb_healthy) if isinstance(mongodb_healthy, Exception) else None
    }
    
    health_status["databases"]["redis"] = {
        "healthy": isinstance(redis_healthy, bool) and redis_healthy,
        "error": str(redis_healthy) if isinstance(redis_healthy, Exception) else None
    }
    
    # Determine overall status
    all_healthy = all(
        db["healthy"] for db in health_status["databases"].values()
    )
    health_status["status"] = "healthy" if all_healthy else "unhealthy"
    
    from datetime import datetime
    health_status["timestamp"] = datetime.utcnow().isoformat()
    
    return health_status

async def check_mongodb() -> bool:
    """Check MongoDB health"""
    try:
        return await mongodb_manager.health_check()
    except Exception as e:
        logger.error("MongoDB health check exception", error=str(e))
        return False

async def check_redis() -> bool:
    """Check Redis health"""
    try:
        return await redis_manager.health_check()
    except Exception as e:
        logger.error("Redis health check exception", error=str(e))
        return False
```

---

## 🔧 Technologies Used
- **Motor**: Async MongoDB driver
- **Redis**: Async Redis client
- **Pydantic**: Data validation and modeling
- **structlog**: Structured logging
- **asyncio**: Asynchronous programming

---

## ⚠️ Key Considerations

### Error Handling
- Database connection failure handling
- Graceful degradation on database issues
- Comprehensive health checks

### Performance
- Connection pooling for all databases
- Async operations throughout
- Efficient data serialization

### Data Integrity
- Validation at model level
- Type safety with Pydantic
- Consistent data formats

---

## 🎯 Success Criteria
- [ ] All data models are implemented and validated
- [ ] Database connections are established and tested
- [ ] Health checks are functional
- [ ] Redis caching is operational
- [ ] MongoDB document operations work correctly

---

## 📋 Next Phase Preview
Phase 3 will focus on implementing the repository layer and establishing data access patterns, building upon the models and database connections created in this phase.