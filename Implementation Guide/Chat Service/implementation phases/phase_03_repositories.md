# Phase 3: Database Layer & Repositories
**Duration:** Week 3-4  
**Steps:** 5-6 of 18

---

## 🎯 Objectives
- Implement repository pattern for data access
- Create database-specific repositories
- Establish data consistency and transaction management
- Implement caching strategies at repository level

---

## 📋 Step 5: Base Repository Pattern & MongoDB Repositories

### What Will Be Implemented
- Abstract base repository pattern
- MongoDB conversation and message repositories
- Transaction management
- Error handling and retry logic

### Folders and Files Created

```
src/repositories/
├── __init__.py
├── base_repository.py
├── mongo_repository.py
├── conversation_repository.py
├── message_repository.py
└── exceptions.py
```

### File Documentation

#### `src/repositories/base_repository.py`
**Purpose:** Abstract base repository defining common data access patterns  
**Usage:** Foundation for all repository implementations with consistent interface

**Classes:**

1. **BaseRepository(ABC)**
   - **Purpose:** Abstract base class for all repositories
   - **Methods:**
     - **create(entity: T) -> T**: Create new entity
     - **get_by_id(id: str) -> Optional[T]**: Get entity by ID
     - **update(entity: T) -> T**: Update existing entity
     - **delete(id: str) -> bool**: Delete entity by ID
     - **list(filters: Dict, pagination: Pagination) -> List[T]**: List entities with filters

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Dict, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

T = TypeVar('T')

@dataclass
class Pagination:
    """Pagination parameters"""
    page: int = 1
    page_size: int = 20
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries"""
        return (self.page - 1) * self.page_size
    
    def validate(self) -> None:
        """Validate pagination parameters"""
        if self.page < 1:
            raise ValueError("Page must be >= 1")
        if self.page_size < 1 or self.page_size > 100:
            raise ValueError("Page size must be between 1 and 100")
        if self.sort_order not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")

@dataclass
class PaginatedResult(Generic[T]):
    """Paginated result container"""
    items: List[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def create(cls, items: List[T], total: int, pagination: Pagination) -> "PaginatedResult[T]":
        """Create paginated result from items and pagination"""
        has_next = pagination.offset + len(items) < total
        has_previous = pagination.page > 1
        
        return cls(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
            has_next=has_next,
            has_previous=has_previous
        )

class BaseRepository(ABC, Generic[T]):
    """Abstract base repository class"""
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity"""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update existing entity"""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID"""
        pass
    
    @abstractmethod
    async def list(
        self, 
        filters: Optional[Dict[str, Any]] = None,
        pagination: Optional[Pagination] = None
    ) -> PaginatedResult[T]:
        """List entities with optional filters and pagination"""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists"""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching filters"""
        pass
    
    def _build_sort_criteria(self, pagination: Optional[Pagination]) -> List[tuple]:
        """Build sort criteria for database queries"""
        if not pagination or not pagination.sort_by:
            return [("created_at", -1)]  # Default sort
        
        sort_direction = 1 if pagination.sort_order == "asc" else -1
        return [(pagination.sort_by, sort_direction)]
    
    def _log_operation(self, operation: str, **kwargs) -> None:
        """Log repository operation"""
        self.logger.info(
            f"Repository operation: {operation}",
            **kwargs
        )
```

#### `src/repositories/conversation_repository.py`
**Purpose:** MongoDB repository for conversation document operations  
**Usage:** Handle all conversation-related database operations with caching

**Classes:**

1. **ConversationRepository(BaseRepository[ConversationDocument])**
   - **Purpose:** Manage conversation documents in MongoDB
   - **Methods:**
     - **create_conversation(conversation: ConversationDocument) -> ConversationDocument**: Create new conversation
     - **get_by_conversation_id(conversation_id: str) -> Optional[ConversationDocument]**: Get by conversation ID
     - **get_active_conversations(tenant_id: str, user_id: str) -> List[ConversationDocument]**: Get active conversations
     - **update_conversation_status(conversation_id: str, status: ConversationStatus) -> bool**: Update status
     - **get_conversations_by_tenant(tenant_id: str, filters: Dict, pagination: Pagination) -> PaginatedResult**: List tenant conversations

```python
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from src.repositories.base_repository import BaseRepository, Pagination, PaginatedResult
from src.models.mongo.conversation_model import ConversationDocument
from src.models.types import ConversationStatus, ChannelType, TenantId, UserId
from src.database.mongodb import get_mongodb
from src.repositories.exceptions import (
    RepositoryError, EntityNotFoundError, DuplicateEntityError
)

class ConversationRepository(BaseRepository[ConversationDocument]):
    """Repository for conversation documents"""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        super().__init__()
        self.database = database
        self.collection = database.conversations
    
    async def create(self, conversation: ConversationDocument) -> ConversationDocument:
        """Create a new conversation"""
        try:
            document = conversation.to_dict()
            result = await self.collection.insert_one(document)
            conversation.id = result.inserted_id
            
            self._log_operation(
                "create_conversation",
                conversation_id=conversation.conversation_id,
                tenant_id=conversation.tenant_id,
                channel=conversation.channel
            )
            
            return conversation
            
        except DuplicateKeyError:
            raise DuplicateEntityError(
                f"Conversation {conversation.conversation_id} already exists"
            )
        except Exception as e:
            self.logger.error(
                "Failed to create conversation",
                conversation_id=conversation.conversation_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to create conversation: {e}")
    
    async def get_by_id(self, conversation_id: str) -> Optional[ConversationDocument]:
        """Get conversation by conversation_id"""
        try:
            document = await self.collection.find_one(
                {"conversation_id": conversation_id}
            )
            
            if document:
                return ConversationDocument.from_dict(document)
            return None
            
        except Exception as e:
            self.logger.error(
                "Failed to get conversation",
                conversation_id=conversation_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to get conversation: {e}")
    
    async def update(self, conversation: ConversationDocument) -> ConversationDocument:
        """Update existing conversation"""
        try:
            document = conversation.to_dict()
            document.pop("_id", None)  # Remove ID from update document
            
            result = await self.collection.update_one(
                {"conversation_id": conversation.conversation_id},
                {"$set": document}
            )
            
            if result.matched_count == 0:
                raise EntityNotFoundError(
                    f"Conversation {conversation.conversation_id} not found"
                )
            
            self._log_operation(
                "update_conversation",
                conversation_id=conversation.conversation_id,
                modified_count=result.modified_count
            )
            
            return conversation
            
        except EntityNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to update conversation",
                conversation_id=conversation.conversation_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to update conversation: {e}")
    
    async def delete(self, conversation_id: str) -> bool:
        """Delete conversation by ID"""
        try:
            result = await self.collection.delete_one(
                {"conversation_id": conversation_id}
            )
            
            success = result.deleted_count > 0
            self._log_operation(
                "delete_conversation",
                conversation_id=conversation_id,
                deleted=success
            )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Failed to delete conversation",
                conversation_id=conversation_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to delete conversation: {e}")
    
    async def list(
        self, 
        filters: Optional[Dict[str, Any]] = None,
        pagination: Optional[Pagination] = None
    ) -> PaginatedResult[ConversationDocument]:
        """List conversations with filters and pagination"""
        try:
            filters = filters or {}
            pagination = pagination or Pagination()
            pagination.validate()
            
            # Build MongoDB query
            query = self._build_query(filters)
            sort_criteria = self._build_sort_criteria(pagination)
            
            # Get total count
            total = await self.collection.count_documents(query)
            
            # Get paginated results
            cursor = self.collection.find(query)
            cursor = cursor.sort(sort_criteria)
            cursor = cursor.skip(pagination.offset)
            cursor = cursor.limit(pagination.page_size)
            
            documents = await cursor.to_list(length=pagination.page_size)
            conversations = [
                ConversationDocument.from_dict(doc) for doc in documents
            ]
            
            self._log_operation(
                "list_conversations",
                filters=filters,
                total=total,
                returned=len(conversations)
            )
            
            return PaginatedResult.create(conversations, total, pagination)
            
        except Exception as e:
            self.logger.error(
                "Failed to list conversations",
                filters=filters,
                error=str(e)
            )
            raise RepositoryError(f"Failed to list conversations: {e}")
    
    async def exists(self, conversation_id: str) -> bool:
        """Check if conversation exists"""
        try:
            count = await self.collection.count_documents(
                {"conversation_id": conversation_id}, limit=1
            )
            return count > 0
        except Exception as e:
            self.logger.error(
                "Failed to check conversation existence",
                conversation_id=conversation_id,
                error=str(e)
            )
            return False
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count conversations matching filters"""
        try:
            query = self._build_query(filters or {})
            return await self.collection.count_documents(query)
        except Exception as e:
            self.logger.error(
                "Failed to count conversations",
                filters=filters,
                error=str(e)
            )
            raise RepositoryError(f"Failed to count conversations: {e}")
    
    # Conversation-specific methods
    
    async def get_active_conversations(
        self, 
        tenant_id: TenantId, 
        user_id: UserId,
        limit: int = 10
    ) -> List[ConversationDocument]:
        """Get active conversations for a user"""
        try:
            query = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "status": ConversationStatus.ACTIVE
            }
            
            cursor = self.collection.find(query)
            cursor = cursor.sort("last_activity_at", -1)
            cursor = cursor.limit(limit)
            
            documents = await cursor.to_list(length=limit)
            return [ConversationDocument.from_dict(doc) for doc in documents]
            
        except Exception as e:
            self.logger.error(
                "Failed to get active conversations",
                tenant_id=tenant_id,
                user_id=user_id,
                error=str(e)
            )
            raise RepositoryError(f"Failed to get active conversations: {e}")
    
    async def update_conversation_status(
        self, 
        conversation_id: str, 
        status: ConversationStatus,
        completed_at: Optional[datetime] = None
    ) -> bool:
        """Update conversation status"""
        try:
            update_doc = {"status": status}
            if status == ConversationStatus.COMPLETED and completed_at:
                update_doc["completed_at"] = completed_at
            
            result = await self.collection.update_one(
                {"conversation_id": conversation_id},
                {"$set": update_doc}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            self.logger.error(
                "Failed to update conversation status",
                conversation_id=conversation_id,
                status=status,
                error=str(e)
            )
            raise RepositoryError(f"Failed to update conversation status: {e}")
    
    async def get_conversations_by_tenant(
        self,
        tenant_id: TenantId,
        filters: Optional[Dict[str, Any]] = None,
        pagination: Optional[Pagination] = None
    ) -> PaginatedResult[ConversationDocument]:
        """Get conversations for a specific tenant"""
        base_filters = {"tenant_id": tenant_id}
        if filters:
            base_filters.update(filters)
        
        return await self.list(filters=base_filters, pagination=pagination)
    
    def _build_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build MongoDB query from filters"""
        query = {}
        
        # Direct field mappings
        direct_fields = [
            "tenant_id", "user_id", "conversation_id", "status", "channel"
        ]
        for field in direct_fields:
            if field in filters:
                query[field] = filters[field]
        
        # Date range filters
        if "start_date" in filters or "end_date" in filters:
            date_query = {}
            if "start_date" in filters:
                date_query["$gte"] = filters["start_date"]
            if "end_date" in filters:
                date_query["$lte"] = filters["end_date"]
            query["started_at"] = date_query
        
        # Text search (simplified)
        if "search" in filters:
            query["$text"] = {"$search": filters["search"]}
        
        return query

# Dependency function
async def get_conversation_repository() -> ConversationRepository:
    """Get conversation repository instance"""
    database = await get_mongodb()
    return ConversationRepository(database)
```

---

## 📋 Step 6: Redis Repositories & Caching Layer

### What Will Be Implemented
- Redis-based repositories for session and state management
- Caching strategies and TTL management
- Rate limiting repository
- State synchronization mechanisms

### Folders and Files Created

```
src/repositories/
├── redis_repository.py
├── session_repository.py
├── rate_limit_repository.py
└── cache_repository.py
```

### File Documentation

#### `src/repositories/session_repository.py`
**Purpose:** Redis repository for session management and caching  
**Usage:** Handle session state storage, retrieval, and lifecycle management

**Classes:**

1. **SessionRepository**
   - **Purpose:** Manage session data in Redis
   - **Methods:**
     - **create_session(session_data: SessionData) -> bool**: Create new session
     - **get_session(tenant_id: str, session_id: str) -> Optional[SessionData]**: Get session data
     - **update_session(session_data: SessionData) -> bool**: Update session
     - **delete_session(tenant_id: str, session_id: str) -> bool**: Delete session
     - **extend_session(tenant_id: str, session_id: str, hours: int) -> bool**: Extend session TTL

```python
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from redis.asyncio import Redis
import json

from src.models.redis.session_cache import SessionData
from src.models.types import TenantId, SessionId, UserId
from src.database.redis_client import get_redis
from src.repositories.exceptions import RepositoryError, EntityNotFoundError

class SessionRepository:
    """Repository for session management in Redis"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def create_session(self, session_data: SessionData) -> bool:
        """Create a new session"""
        try:
            cache_key = SessionData.get_cache_key(
                session_data.tenant_id, 
                session_data.session_id
            )
            
            # Convert to Redis hash format
            hash_data = session_data.to_redis_hash()
            
            # Calculate TTL
            ttl_seconds = int((session_data.expires_at - datetime.utcnow()).total_seconds())
            if ttl_seconds <= 0:
                ttl_seconds = self.default_ttl
            
            # Store in Redis with TTL
            pipe = self.redis.pipeline()
            pipe.hset(cache_key, mapping=hash_data)
            pipe.expire(cache_key, ttl_seconds)
            results = await pipe.execute()
            
            success = all(results)
            
            if success:
                # Also maintain a user sessions index
                await self._add_to_user_sessions_index(
                    session_data.tenant_id,
                    session_data.user_id,
                    session_data.session_id
                )
            
            return success
            
        except Exception as e:
            raise RepositoryError(f"Failed to create session: {e}")
    
    async def get_session(
        self, 
        tenant_id: TenantId, 
        session_id: SessionId
    ) -> Optional[SessionData]:
        """Get session data by ID"""
        try:
            cache_key = SessionData.get_cache_key(tenant_id, session_id)
            hash_data = await self.redis.hgetall(cache_key)
            
            if not hash_data:
                return None
            
            session_data = SessionData.from_redis_hash(hash_data)
            
            # Check if session is expired
            if session_data.is_expired():
                await self.delete_session(tenant_id, session_id)
                return None
            
            return session_data
            
        except Exception as e:
            raise RepositoryError(f"Failed to get session: {e}")
    
    async def update_session(self, session_data: SessionData) -> bool:
        """Update existing session"""
        try:
            # Update last activity
            session_data.update_activity()
            
            cache_key = SessionData.get_cache_key(
                session_data.tenant_id, 
                session_data.session_id
            )
            
            # Check if session exists
            exists = await self.redis.exists(cache_key)
            if not exists:
                raise EntityNotFoundError(f"Session {session_data.session_id} not found")
            
            # Convert to Redis hash format
            hash_data = session_data.to_redis_hash()
            
            # Update in Redis
            await self.redis.hset(cache_key, mapping=hash_data)
            
            return True
            
        except EntityNotFoundError:
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to update session: {e}")
    
    async def delete_session(
        self, 
        tenant_id: TenantId, 
        session_id: SessionId
    ) -> bool:
        """Delete session by ID"""
        try:
            cache_key = SessionData.get_cache_key(tenant_id, session_id)
            
            # Get session data first to clean up indexes
            session_data = await self.get_session(tenant_id, session_id)
            
            # Delete from Redis
            deleted = await self.redis.delete(cache_key)
            
            # Clean up user sessions index
            if session_data:
                await self._remove_from_user_sessions_index(
                    tenant_id,
                    session_data.user_id,
                    session_id
                )
            
            return deleted > 0
            
        except Exception as e:
            raise RepositoryError(f"Failed to delete session: {e}")
    
    async def extend_session(
        self, 
        tenant_id: TenantId, 
        session_id: SessionId,
        hours: int = 1
    ) -> bool:
        """Extend session expiration"""
        try:
            session_data = await self.get_session(tenant_id, session_id)
            if not session_data:
                return False
            
            # Extend expiration
            session_data.extend_session(hours)
            
            # Update in Redis
            return await self.update_session(session_data)
            
        except Exception as e:
            raise RepositoryError(f"Failed to extend session: {e}")
    
    async def get_user_sessions(
        self, 
        tenant_id: TenantId, 
        user_id: UserId
    ) -> List[SessionData]:
        """Get all active sessions for a user"""
        try:
            sessions_key = f"user_sessions:{tenant_id}:{user_id}"
            session_ids = await self.redis.smembers(sessions_key)
            
            sessions = []
            for session_id in session_ids:
                session_data = await self.get_session(tenant_id, session_id)
                if session_data:
                    sessions.append(session_data)
            
            return sessions
            
        except Exception as e:
            raise RepositoryError(f"Failed to get user sessions: {e}")
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions (background task)"""
        try:
            # This is a simplified cleanup - in production, you'd want
            # a more sophisticated approach using Redis key expiration events
            cleaned_count = 0
            
            # Get all session keys
            pattern = "session:*"
            async for key in self.redis.scan_iter(match=pattern):
                ttl = await self.redis.ttl(key)
                if ttl == -2:  # Key doesn't exist
                    cleaned_count += 1
                elif ttl == -1:  # Key exists but no TTL
                    # Set default TTL
                    await self.redis.expire(key, self.default_ttl)
            
            return cleaned_count
            
        except Exception as e:
            raise RepositoryError(f"Failed to cleanup expired sessions: {e}")
    
    # Private helper methods
    
    async def _add_to_user_sessions_index(
        self, 
        tenant_id: TenantId, 
        user_id: UserId, 
        session_id: SessionId
    ) -> None:
        """Add session to user sessions index"""
        sessions_key = f"user_sessions:{tenant_id}:{user_id}"
        await self.redis.sadd(sessions_key, session_id)
        await self.redis.expire(sessions_key, 86400)  # 24 hours
    
    async def _remove_from_user_sessions_index(
        self, 
        tenant_id: TenantId, 
        user_id: UserId, 
        session_id: SessionId
    ) -> None:
        """Remove session from user sessions index"""
        sessions_key = f"user_sessions:{tenant_id}:{user_id}"
        await self.redis.srem(sessions_key, session_id)

# Dependency function
async def get_session_repository() -> SessionRepository:
    """Get session repository instance"""
    redis_client = await get_redis()
    return SessionRepository(redis_client)
```

#### `src/repositories/rate_limit_repository.py`
**Purpose:** Redis repository for rate limiting and quota management  
**Usage:** Handle API rate limiting, quota tracking, and abuse prevention

**Classes:**

1. **RateLimitRepository**
   - **Purpose:** Manage rate limiting data in Redis
   - **Methods:**
     - **check_rate_limit(tenant_id: str, identifier: str, limit: int, window_seconds: int) -> tuple**: Check rate limit
     - **increment_counter(tenant_id: str, identifier: str, window_seconds: int) -> int**: Increment rate counter
     - **get_remaining_quota(tenant_id: str, identifier: str, limit: int, window_seconds: int) -> int**: Get remaining quota
     - **reset_rate_limit(tenant_id: str, identifier: str) -> bool**: Reset rate limit counter

```python
from datetime import datetime, timedelta
from typing import Tuple, Optional
from redis.asyncio import Redis
import time
import math

from src.models.types import TenantId
from src.database.redis_client import get_redis
from src.repositories.exceptions import RepositoryError

class RateLimitRepository:
    """Repository for rate limiting using Redis"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def check_rate_limit(
        self,
        tenant_id: TenantId,
        identifier: str,  # API key, user ID, IP address, etc.
        limit: int,
        window_seconds: int
    ) -> Tuple[bool, int, int]:
        """
        Check if request is within rate limit using sliding window
        
        Returns:
            (allowed, current_count, reset_time)
        """
        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Use sorted set for sliding window
            key = f"rate_limit:{tenant_id}:{identifier}"
            
            # Remove expired entries
            await self.redis.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            current_count = await self.redis.zcard(key)
            
            # Check if limit exceeded
            allowed = current_count < limit
            
            if allowed:
                # Add current request
                await self.redis.zadd(key, {str(current_time): current_time})
                await self.redis.expire(key, window_seconds)
                current_count += 1
            
            # Calculate reset time
            reset_time = current_time + window_seconds
            
            return allowed, current_count, reset_time
            
        except Exception as e:
            raise RepositoryError(f"Failed to check rate limit: {e}")
    
    async def increment_counter(
        self,
        tenant_id: TenantId,
        identifier: str,
        window_seconds: int
    ) -> int:
        """Increment rate limit counter"""
        try:
            current_time = int(time.time())
            key = f"rate_limit:{tenant_id}:{identifier}"
            
            # Add current request
            await self.redis.zadd(key, {str(current_time): current_time})
            await self.redis.expire(key, window_seconds)
            
            # Return current count
            return await self.redis.zcard(key)
            
        except Exception as e:
            raise RepositoryError(f"Failed to increment counter: {e}")
    
    async def get_remaining_quota(
        self,
        tenant_id: TenantId,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> int:
        """Get remaining quota for identifier"""
        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            key = f"rate_limit:{tenant_id}:{identifier}"
            
            # Remove expired entries and count current
            await self.redis.zremrangebyscore(key, 0, window_start)
            current_count = await self.redis.zcard(key)
            
            return max(0, limit - current_count)
            
        except Exception as e:
            raise RepositoryError(f"Failed to get remaining quota: {e}")
    
    async def reset_rate_limit(
        self,
        tenant_id: TenantId,
        identifier: str
    ) -> bool:
        """Reset rate limit counter for identifier"""
        try:
            key = f"rate_limit:{tenant_id}:{identifier}"
            deleted = await self.redis.delete(key)
            return deleted > 0
            
        except Exception as e:
            raise RepositoryError(f"Failed to reset rate limit: {e}")
    
    async def get_rate_limit_info(
        self,
        tenant_id: TenantId,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> dict:
        """Get comprehensive rate limit information"""
        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            key = f"rate_limit:{tenant_id}:{identifier}"
            
            # Clean up and get count
            await self.redis.zremrangebyscore(key, 0, window_start)
            current_count = await self.redis.zcard(key)
            
            # Get oldest request time for reset calculation
            oldest_requests = await self.redis.zrange(key, 0, 0, withscores=True)
            
            if oldest_requests:
                oldest_time = int(oldest_requests[0][1])
                reset_time = oldest_time + window_seconds
            else:
                reset_time = current_time + window_seconds
            
            return {
                "limit": limit,
                "remaining": max(0, limit - current_count),
                "reset_time": reset_time,
                "window_seconds": window_seconds,
                "current_count": current_count
            }
            
        except Exception as e:
            raise RepositoryError(f"Failed to get rate limit info: {e}")

# Dependency function
async def get_rate_limit_repository() -> RateLimitRepository:
    """Get rate limit repository instance"""
    redis_client = await get_redis()
    return RateLimitRepository(redis_client)
```

#### `src/repositories/exceptions.py`
**Purpose:** Repository-specific exception definitions  
**Usage:** Provide specific error handling for repository operations

**Classes:**

1. **RepositoryError(Exception)**
   - **Purpose:** Base exception for repository operations
   - **Usage:** Generic repository error handling

2. **EntityNotFoundError(RepositoryError)**
   - **Purpose:** Exception for when requested entity is not found
   - **Usage:** 404-like errors in repository layer

3. **DuplicateEntityError(RepositoryError)**
   - **Purpose:** Exception for duplicate entity creation attempts
   - **Usage:** Conflict errors in repository layer

```python
"""Repository-specific exceptions"""

class RepositoryError(Exception):
    """Base exception for repository operations"""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error

class EntityNotFoundError(RepositoryError):
    """Exception raised when an entity is not found"""
    
    def __init__(self, entity_type: str, entity_id: str):
        message = f"{entity_type} with ID '{entity_id}' not found"
        super().__init__(message)
        self.entity_type = entity_type
        self.entity_id = entity_id

class DuplicateEntityError(RepositoryError):
    """Exception raised when trying to create a duplicate entity"""
    
    def __init__(self, entity_type: str, entity_id: str):
        message = f"{entity_type} with ID '{entity_id}' already exists"
        super().__init__(message)
        self.entity_type = entity_type
        self.entity_id = entity_id

class ValidationError(RepositoryError):
    """Exception raised when entity validation fails"""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(message)
        self.field = field

class ConnectionError(RepositoryError):
    """Exception raised when database connection fails"""
    
    def __init__(self, database_type: str, original_error: Exception = None):
        message = f"Failed to connect to {database_type}"
        super().__init__(message, original_error)
        self.database_type = database_type

class TransactionError(RepositoryError):
    """Exception raised when database transaction fails"""
    
    def __init__(self, message: str, operation: str = None):
        super().__init__(message)
        self.operation = operation
```

---

## 🔧 Technologies Used
- **Motor**: Async MongoDB operations
- **Redis**: Caching and session management
- **Async/await**: Asynchronous programming patterns
- **Pydantic**: Data validation and serialization
- **Python ABC**: Abstract base classes for interfaces

---

## ⚠️ Key Considerations

### Error Handling
- Comprehensive exception hierarchy
- Connection failure handling
- Transaction rollback mechanisms
- Retry logic for transient failures

### Performance
- Connection pooling utilization
- Efficient query patterns
- Proper indexing strategies
- Caching layer optimization

### Data Consistency
- Transaction management
- Optimistic locking patterns
- Cache invalidation strategies
- Data synchronization between stores

---

## 🎯 Success Criteria
- [ ] All repository interfaces are implemented
- [ ] MongoDB operations are working correctly
- [ ] Redis caching and session management is functional
- [ ] Rate limiting repository is operational
- [ ] Error handling is comprehensive
- [ ] Unit tests for repositories pass

---

## 📋 Next Phase Preview
Phase 4 will focus on implementing the core business logic layer including channels, processors, and normalizers, building upon the solid data access foundation established in this phase.