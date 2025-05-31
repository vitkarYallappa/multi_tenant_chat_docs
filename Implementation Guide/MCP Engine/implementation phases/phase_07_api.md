# Phase 07: API Layer & HTTP/gRPC Endpoints
**Duration**: Week 13-14 (Days 61-70)  
**Team Size**: 3-4 developers  
**Complexity**: High  

## Overview
Implement the complete API layer with FastAPI HTTP endpoints and gRPC services. This phase exposes all MCP Engine functionality through well-defined APIs with authentication, validation, rate limiting, and comprehensive documentation.

## Step 17: API Infrastructure & Middleware (Days 61-63)

### Files to Create
```
src/
├── api/
│   ├── __init__.py
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── authentication.py
│   │   ├── authorization.py
│   │   ├── rate_limiting.py
│   │   ├── request_validation.py
│   │   ├── tenant_isolation.py
│   │   └── error_handling.py
│   ├── v2/
│   │   ├── __init__.py
│   │   ├── execution_routes.py
│   │   ├── flow_routes.py
│   │   ├── context_routes.py
│   │   ├── analytics_routes.py
│   │   └── health_routes.py
│   ├── grpc/
│   │   ├── __init__.py
│   │   ├── mcp_service.py
│   │   ├── proto/
│   │   │   ├── mcp_engine.proto
│   │   │   ├── mcp_engine_pb2.py
│   │   │   └── mcp_engine_pb2_grpc.py
│   │   └── interceptors/
│   │       ├── __init__.py
│   │       ├── auth_interceptor.py
│   │       └── metrics_interceptor.py
│   └── dependencies/
│       ├── __init__.py
│       ├── auth.py
│       ├── rate_limit.py
│       └── validation.py
```

### `/src/api/middleware/authentication.py`
**Purpose**: Authentication middleware for API endpoints
```python
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime
import uuid

from src.utils.logger import get_logger
from src.config.settings import settings
from src.clients.security_hub_client import SecurityHubClient

logger = get_logger(__name__)

class AuthenticationMiddleware:
    """Authentication middleware for API requests"""
    
    def __init__(self):
        self.security_hub_client = SecurityHubClient()
        self.bearer_scheme = HTTPBearer(auto_error=False)
        
        # JWT configuration
        self.jwt_algorithm = "RS256"
        self.jwt_audience = "mcp-engine-api"
        
        # Cache for validated tokens (simple in-memory cache)
        self._token_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def authenticate_request(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = None
    ) -> Dict[str, Any]:
        """
        Authenticate API request
        
        Args:
            request: FastAPI request object
            credentials: Optional bearer credentials
            
        Returns:
            Authentication context
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # Extract token from request
            token = await self._extract_token(request, credentials)
            
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Validate token
            auth_context = await self._validate_token(token)
            
            # Add request metadata
            auth_context.update({
                "request_id": str(uuid.uuid4()),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent"),
                "authenticated_at": datetime.utcnow().isoformat()
            })
            
            logger.debug(
                "Request authenticated",
                tenant_id=auth_context.get("tenant_id"),
                user_id=auth_context.get("user_id"),
                auth_method=auth_context.get("auth_method")
            )
            
            return auth_context
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Authentication error", error=e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    async def _extract_token(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials]
    ) -> Optional[str]:
        """Extract authentication token from request"""
        
        # Try Bearer token first
        if credentials and credentials.scheme.lower() == "bearer":
            return credentials.credentials
        
        # Try Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ", 1)[1]
        
        # Try API key header
        api_key = request.headers.get("x-api-key")
        if api_key:
            return api_key
        
        # Try query parameter (for development/testing only)
        if settings.service.environment == "development":
            token = request.query_params.get("token")
            if token:
                return token
        
        return None
    
    async def _validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate authentication token
        
        Args:
            token: Authentication token
            
        Returns:
            Authentication context
        """
        # Check cache first
        cached_auth = self._get_cached_auth(token)
        if cached_auth:
            return cached_auth
        
        # Determine token type
        if token.startswith("cb_"):
            # API Key authentication
            auth_context = await self._validate_api_key(token)
        elif "." in token and len(token.split(".")) == 3:
            # JWT authentication
            auth_context = await self._validate_jwt_token(token)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format"
            )
        
        # Cache successful validation
        self._cache_auth(token, auth_context)
        
        return auth_context
    
    async def _validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        try:
            # Decode without verification first to get headers
            unverified_header = jwt.get_unverified_header(token)
            
            # Get public key from Security Hub
            public_key = await self._get_jwt_public_key(unverified_header.get("kid"))
            
            # Verify and decode token
            payload = jwt.decode(
                token,
                public_key,
                algorithms=[self.jwt_algorithm],
                audience=self.jwt_audience,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True
                }
            )
            
            # Extract authentication context
            return {
                "auth_method": "jwt",
                "user_id": payload.get("sub"),
                "tenant_id": payload.get("tenant_id"),
                "user_role": payload.get("user_role"),
                "permissions": payload.get("permissions", []),
                "scopes": payload.get("scopes", []),
                "rate_limit_tier": payload.get("rate_limit_tier", "standard"),
                "expires_at": datetime.fromtimestamp(payload.get("exp")).isoformat(),
                "issued_at": datetime.fromtimestamp(payload.get("iat")).isoformat()
            }
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    async def _validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key"""
        try:
            # Call Security Hub to validate API key
            validation_result = await self.security_hub_client.validate_api_key(api_key)
            
            if not validation_result.get("valid"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
            
            key_info = validation_result.get("key_info", {})
            
            return {
                "auth_method": "api_key",
                "api_key_id": key_info.get("key_id"),
                "tenant_id": key_info.get("tenant_id"),
                "permissions": key_info.get("permissions", []),
                "scopes": key_info.get("scopes", []),
                "rate_limit_tier": key_info.get("rate_limit_tier", "standard"),
                "rate_limit_per_minute": key_info.get("rate_limit_per_minute", 1000),
                "expires_at": key_info.get("expires_at"),
                "last_used_at": key_info.get("last_used_at")
            }
            
        except Exception as e:
            logger.error("API key validation failed", api_key_prefix=api_key[:16], error=e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key validation failed"
            )
    
    async def _get_jwt_public_key(self, key_id: Optional[str]) -> str:
        """Get JWT public key from Security Hub"""
        try:
            key_response = await self.security_hub_client.get_jwt_public_key(key_id)
            return key_response.get("public_key")
        except Exception as e:
            logger.error("Failed to get JWT public key", key_id=key_id, error=e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to verify token signature"
            )
    
    def _get_cached_auth(self, token: str) -> Optional[Dict[str, Any]]:
        """Get cached authentication result"""
        cache_key = f"auth_{hash(token)}"
        cached = self._token_cache.get(cache_key)
        
        if cached:
            # Check if cache entry is still valid
            if datetime.utcnow().timestamp() < cached.get("cache_expires", 0):
                return cached.get("auth_context")
            else:
                # Remove expired entry
                self._token_cache.pop(cache_key, None)
        
        return None
    
    def _cache_auth(self, token: str, auth_context: Dict[str, Any]):
        """Cache authentication result"""
        cache_key = f"auth_{hash(token)}"
        self._token_cache[cache_key] = {
            "auth_context": auth_context,
            "cache_expires": datetime.utcnow().timestamp() + self._cache_ttl
        }
        
        # Simple cache cleanup (remove expired entries)
        current_time = datetime.utcnow().timestamp()
        expired_keys = [
            key for key, value in self._token_cache.items()
            if value.get("cache_expires", 0) < current_time
        ]
        for key in expired_keys:
            self._token_cache.pop(key, None)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

class AuthenticationDependency:
    """FastAPI dependency for authentication"""
    
    def __init__(self):
        self.auth_middleware = AuthenticationMiddleware()
        self.bearer_scheme = HTTPBearer(auto_error=False)
    
    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = None
    ) -> Dict[str, Any]:
        """FastAPI dependency callable"""
        return await self.auth_middleware.authenticate_request(request, credentials)

# Create dependency instance
authenticate = AuthenticationDependency()

# Optional authentication dependency (doesn't raise exception if no auth)
class OptionalAuthenticationDependency:
    """FastAPI dependency for optional authentication"""
    
    def __init__(self):
        self.auth_middleware = AuthenticationMiddleware()
        self.bearer_scheme = HTTPBearer(auto_error=False)
    
    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = None
    ) -> Optional[Dict[str, Any]]:
        """FastAPI dependency callable for optional auth"""
        try:
            return await self.auth_middleware.authenticate_request(request, credentials)
        except HTTPException as e:
            if e.status_code == status.HTTP_401_UNAUTHORIZED:
                return None
            raise

optional_authenticate = OptionalAuthenticationDependency()
```

### `/src/api/middleware/rate_limiting.py`
**Purpose**: Rate limiting middleware for API protection
```python
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, status
from datetime import datetime, timedelta
import asyncio
import time

from src.utils.logger import get_logger
from src.config.database import get_redis
from src.config.settings import settings

logger = get_logger(__name__)

class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception"""
    
    def __init__(
        self,
        detail: str,
        limit: int,
        remaining: int,
        reset_time: int,
        retry_after: int
    ):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time),
                "Retry-After": str(retry_after)
            }
        )

class RateLimiter:
    """Advanced rate limiter with multiple algorithms"""
    
    def __init__(self):
        self.redis = None
        
        # Rate limit configurations by tier
        self.tier_limits = {
            "basic": {
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "requests_per_day": 10000,
                "burst_limit": 200
            },
            "standard": {
                "requests_per_minute": 1000,
                "requests_per_hour": 10000,
                "requests_per_day": 100000,
                "burst_limit": 2000
            },
            "premium": {
                "requests_per_minute": 10000,
                "requests_per_hour": 100000,
                "requests_per_day": 1000000,
                "burst_limit": 20000
            },
            "enterprise": {
                "requests_per_minute": 100000,
                "requests_per_hour": 1000000,
                "requests_per_day": 10000000,
                "burst_limit": 200000
            }
        }
    
    async def initialize(self):
        """Initialize rate limiter"""
        if not self.redis:
            self.redis = await get_redis()
    
    async def check_rate_limit(
        self,
        identifier: str,
        tier: str = "standard",
        tenant_id: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check rate limit for identifier
        
        Args:
            identifier: Rate limit identifier (user_id, api_key, etc.)
            tier: Rate limit tier
            tenant_id: Optional tenant context
            endpoint: Optional endpoint context
            
        Returns:
            Rate limit status
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        await self.initialize()
        
        tier_config = self.tier_limits.get(tier, self.tier_limits["standard"])
        
        # Check multiple time windows
        current_time = int(time.time())
        
        # Check minute limit
        minute_result = await self._check_sliding_window(
            identifier,
            "minute",
            tier_config["requests_per_minute"],
            60,
            current_time
        )
        
        # Check hour limit
        hour_result = await self._check_sliding_window(
            identifier,
            "hour",
            tier_config["requests_per_hour"],
            3600,
            current_time
        )
        
        # Check daily limit
        day_result = await self._check_sliding_window(
            identifier,
            "day",
            tier_config["requests_per_day"],
            86400,
            current_time
        )
        
        # Check burst limit using token bucket
        burst_result = await self._check_token_bucket(
            identifier,
            tier_config["burst_limit"],
            tier_config["requests_per_minute"] / 60.0,  # tokens per second
            current_time
        )
        
        # Determine most restrictive limit
        limits = [minute_result, hour_result, day_result, burst_result]
        most_restrictive = min(limits, key=lambda x: x["remaining"])
        
        if most_restrictive["allowed"]:
            # Increment counters for successful request
            await self._increment_counters(identifier, current_time)
            
            return {
                "allowed": True,
                "limit": most_restrictive["limit"],
                "remaining": most_restrictive["remaining"],
                "reset_time": most_restrictive["reset_time"],
                "tier": tier
            }
        else:
            # Rate limit exceeded
            raise RateLimitExceeded(
                detail=f"Rate limit exceeded for {most_restrictive['window']} window",
                limit=most_restrictive["limit"],
                remaining=most_restrictive["remaining"],
                reset_time=most_restrictive["reset_time"],
                retry_after=most_restrictive["reset_time"] - current_time
            )
    
    async def _check_sliding_window(
        self,
        identifier: str,
        window: str,
        limit: int,
        window_seconds: int,
        current_time: int
    ) -> Dict[str, Any]:
        """Check sliding window rate limit"""
        key = f"rate_limit:{identifier}:{window}"
        window_start = current_time - window_seconds
        
        # Remove expired entries and count current requests
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.expire(key, window_seconds)
        
        results = await pipe.execute()
        current_count = results[1]
        
        return {
            "window": window,
            "allowed": current_count < limit,
            "limit": limit,
            "remaining": max(0, limit - current_count),
            "reset_time": current_time + window_seconds,
            "current_count": current_count
        }
    
    async def _check_token_bucket(
        self,
        identifier: str,
        capacity: int,
        refill_rate: float,
        current_time: int
    ) -> Dict[str, Any]:
        """Check token bucket rate limit"""
        key = f"token_bucket:{identifier}"
        
        # Get current bucket state
        bucket_data = await self.redis.hgetall(key)
        
        if bucket_data:
            tokens = float(bucket_data.get("tokens", capacity))
            last_refill = float(bucket_data.get("last_refill", current_time))
        else:
            tokens = capacity
            last_refill = current_time
        
        # Calculate tokens to add
        time_passed = current_time - last_refill
        tokens_to_add = time_passed * refill_rate
        tokens = min(capacity, tokens + tokens_to_add)
        
        # Check if request is allowed
        allowed = tokens >= 1.0
        remaining_tokens = max(0, tokens - 1.0) if allowed else tokens
        
        # Update bucket state
        if allowed:
            await self.redis.hset(key, mapping={
                "tokens": str(remaining_tokens),
                "last_refill": str(current_time)
            })
            await self.redis.expire(key, 3600)  # 1 hour expiry
        
        return {
            "window": "burst",
            "allowed": allowed,
            "limit": capacity,
            "remaining": int(remaining_tokens),
            "reset_time": current_time + int((capacity - remaining_tokens) / refill_rate),
            "current_tokens": tokens
        }
    
    async def _increment_counters(self, identifier: str, current_time: int):
        """Increment rate limit counters"""
        # Add to sliding windows
        score = current_time
        value = f"{current_time}:{id(object())}"  # Unique value with timestamp
        
        pipe = self.redis.pipeline()
        
        # Increment sliding window counters
        for window in ["minute", "hour", "day"]:
            key = f"rate_limit:{identifier}:{window}"
            pipe.zadd(key, {value: score})
        
        await pipe.execute()
    
    async def get_rate_limit_status(
        self,
        identifier: str,
        tier: str = "standard"
    ) -> Dict[str, Any]:
        """
        Get current rate limit status without consuming quota
        
        Args:
            identifier: Rate limit identifier
            tier: Rate limit tier
            
        Returns:
            Current rate limit status
        """
        await self.initialize()
        
        tier_config = self.tier_limits.get(tier, self.tier_limits["standard"])
        current_time = int(time.time())
        
        # Check all windows without incrementing
        minute_status = await self._check_sliding_window(
            identifier, "minute", tier_config["requests_per_minute"], 60, current_time
        )
        hour_status = await self._check_sliding_window(
            identifier, "hour", tier_config["requests_per_hour"], 3600, current_time
        )
        day_status = await self._check_sliding_window(
            identifier, "day", tier_config["requests_per_day"], 86400, current_time
        )
        
        return {
            "tier": tier,
            "minute": {
                "limit": minute_status["limit"],
                "remaining": minute_status["remaining"],
                "reset_time": minute_status["reset_time"]
            },
            "hour": {
                "limit": hour_status["limit"],
                "remaining": hour_status["remaining"],
                "reset_time": hour_status["reset_time"]
            },
            "day": {
                "limit": day_status["limit"],
                "remaining": day_status["remaining"],
                "reset_time": day_status["reset_time"]
            }
        }

class RateLimitDependency:
    """FastAPI dependency for rate limiting"""
    
    def __init__(self, endpoint_specific: bool = False):
        self.rate_limiter = RateLimiter()
        self.endpoint_specific = endpoint_specific
    
    async def __call__(
        self,
        request: Request,
        auth_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """FastAPI dependency callable"""
        
        # Determine rate limit identifier and tier
        if auth_context.get("auth_method") == "api_key":
            identifier = f"api_key:{auth_context['api_key_id']}"
            tier = auth_context.get("rate_limit_tier", "standard")
        else:
            identifier = f"user:{auth_context['user_id']}"
            tier = auth_context.get("rate_limit_tier", "standard")
        
        # Add endpoint context if enabled
        endpoint = None
        if self.endpoint_specific:
            endpoint = f"{request.method}:{request.url.path}"
            identifier = f"{identifier}:{endpoint}"
        
        # Check rate limit
        rate_limit_status = await self.rate_limiter.check_rate_limit(
            identifier=identifier,
            tier=tier,
            tenant_id=auth_context.get("tenant_id"),
            endpoint=endpoint
        )
        
        # Log rate limit check
        logger.debug(
            "Rate limit checked",
            identifier=identifier,
            tier=tier,
            remaining=rate_limit_status["remaining"],
            endpoint=endpoint
        )
        
        return rate_limit_status

# Create dependency instances
rate_limit = RateLimitDependency(endpoint_specific=False)
endpoint_rate_limit = RateLimitDependency(endpoint_specific=True)
```

## Step 18: HTTP API Routes Implementation (Days 64-66)

### `/src/api/v2/execution_routes.py`
**Purpose**: HTTP endpoints for conversation execution
```python
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field, validator
import uuid

from src.api.dependencies.auth import authenticate
from src.api.dependencies.rate_limit import rate_limit
from src.services.execution_service import ExecutionService
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Request/Response Models
class MessageContent(BaseModel):
    """Message content model"""
    type: str = Field(default="text", description="Message type")
    text: Optional[str] = Field(None, description="Text content")
    payload: Optional[str] = Field(None, description="Quick reply payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ProcessMessageRequest(BaseModel):
    """Process message request model"""
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: str = Field(..., description="User identifier")
    channel: str = Field(..., description="Communication channel")
    content: MessageContent = Field(..., description="Message content")
    session_id: Optional[str] = Field(None, description="Session identifier")
    processing_hints: Optional[Dict[str, Any]] = Field(None, description="Processing hints")
    
    @validator('conversation_id', 'user_id')
    def validate_ids(cls, v):
        if not v or not v.strip():
            raise ValueError('ID cannot be empty')
        return v.strip()
    
    @validator('channel')
    def validate_channel(cls, v):
        allowed_channels = ['web', 'whatsapp', 'messenger', 'slack', 'teams', 'voice', 'sms']
        if v not in allowed_channels:
            raise ValueError(f'Channel must be one of: {", ".join(allowed_channels)}')
        return v

class ProcessMessageResponse(BaseModel):
    """Process message response model"""
    conversation_id: str
    current_state: str
    response: Dict[str, Any]
    context_updates: Dict[str, Any]
    actions_performed: list
    processing_time_ms: int
    success: bool
    ab_variant: Optional[str] = None

class ConversationStateResponse(BaseModel):
    """Conversation state response model"""
    conversation_id: str
    current_state: str
    flow_id: str
    user_id: str
    slots: Dict[str, Any]
    variables: Dict[str, Any]
    created_at: str
    last_activity: str
    message_count: int

class ResetConversationRequest(BaseModel):
    """Reset conversation request model"""
    reason: str = Field(default="manual_reset", description="Reason for reset")

# Create router
router = APIRouter(prefix="/execution", tags=["Execution"])

# Initialize service (will be dependency injected in production)
execution_service = ExecutionService()

@router.post(
    "/process",
    response_model=ProcessMessageResponse,
    summary="Process conversation message",
    description="Process a user message through the conversation flow engine"
)
async def process_message(
    request: ProcessMessageRequest,
    auth_context: Dict[str, Any] = Depends(authenticate),
    rate_limit_status: Dict[str, Any] = Depends(rate_limit),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> ProcessMessageResponse:
    """
    Process a conversation message through the MCP Engine
    
    This endpoint accepts a message from a user and processes it through
    the configured conversation flow, returning an appropriate response
    and updating the conversation state.
    """
    try:
        # Extract tenant ID from auth context
        tenant_id = auth_context.get("tenant_id")
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID not found in authentication context"
            )
        
        # Process the message
        result = await execution_service.process_message(
            tenant_id=tenant_id,
            conversation_id=request.conversation_id,
            message_content=request.content.dict(),
            user_id=request.user_id,
            channel=request.channel,
            session_id=request.session_id,
            processing_hints=request.processing_hints
        )
        
        # Convert to response model
        response = ProcessMessageResponse(
            conversation_id=result.conversation_id,
            current_state=result.current_state,
            response=result.response,
            context_updates=result.context_updates,
            actions_performed=result.actions_performed,
            processing_time_ms=result.processing_time_ms,
            success=result.success,
            ab_variant=result.ab_variant
        )
        
        # Add background task for analytics
        background_tasks.add_task(
            _track_message_processed,
            tenant_id,
            request.conversation_id,
            result.success,
            result.processing_time_ms
        )
        
        logger.info(
            "Message processed successfully",
            tenant_id=tenant_id,
            conversation_id=request.conversation_id,
            processing_time_ms=result.processing_time_ms
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Message processing failed",
            tenant_id=auth_context.get("tenant_id"),
            conversation_id=request.conversation_id,
            error=e
        )
        
        if isinstance(e, HTTPException):
            raise
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Message processing failed"
        )

@router.get(
    "/conversations/{conversation_id}/state",
    response_model=ConversationStateResponse,
    summary="Get conversation state",
    description="Retrieve the current state of a conversation"
)
async def get_conversation_state(
    conversation_id: str,
    auth_context: Dict[str, Any] = Depends(authenticate),
    rate_limit_status: Dict[str, Any] = Depends(rate_limit)
) -> ConversationStateResponse:
    """
    Get the current state of a conversation
    
    Returns detailed information about the conversation including
    current state, slots, variables, and metadata.
    """
    try:
        tenant_id = auth_context.get("tenant_id")
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID not found in authentication context"
            )
        
        # Get conversation state
        state_info = await execution_service.get_conversation_state(
            tenant_id=tenant_id,
            conversation_id=conversation_id
        )
        
        if not state_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return ConversationStateResponse(**state_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get conversation state",
            tenant_id=auth_context.get("tenant_id"),
            conversation_id=conversation_id,
            error=e
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation state"
        )

@router.post(
    "/conversations/{conversation_id}/reset",
    summary="Reset conversation",
    description="Reset a conversation to its initial state"
)
async def reset_conversation(
    conversation_id: str,
    request: ResetConversationRequest,
    auth_context: Dict[str, Any] = Depends(authenticate),
    rate_limit_status: Dict[str, Any] = Depends(rate_limit)
) -> Dict[str, Any]:
    """
    Reset a conversation to its initial state
    
    This will clear all conversation context, slots, and variables,
    returning the conversation to the beginning of the flow.
    """
    try:
        tenant_id = auth_context.get("tenant_id")
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID not found in authentication context"
            )
        
        # Reset conversation
        success = await execution_service.reset_conversation(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            reason=request.reason
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        logger.info(
            "Conversation reset",
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            reason=request.reason
        )
        
        return {
            "success": True,
            "message": "Conversation reset successfully",
            "conversation_id": conversation_id,
            "reason": request.reason
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to reset conversation",
            tenant_id=auth_context.get("tenant_id"),
            conversation_id=conversation_id,
            error=e
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset conversation"
        )

@router.get(
    "/health",
    summary="Execution service health check",
    description="Check the health of the execution service"
)
async def execution_health_check() -> Dict[str, Any]:
    """
    Health check for the execution service
    
    Returns health status and performance metrics for the execution engine.
    """
    try:
        # Perform health check
        health_info = await execution_service.health_check()
        
        return {
            "status": "healthy" if health_info.get("status") == "healthy" else "unhealthy",
            "service": "execution",
            "timestamp": health_info.get("timestamp"),
            "details": health_info
        }
        
    except Exception as e:
        logger.error("Execution health check failed", error=e)
        
        return {
            "status": "unhealthy",
            "service": "execution",
            "error": str(e)
        }

async def _track_message_processed(
    tenant_id: str,
    conversation_id: str,
    success: bool,
    processing_time_ms: int
):
    """Background task to track message processing analytics"""
    try:
        # This would normally call analytics service
        logger.info(
            "Message processing analytics",
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            success=success,
            processing_time_ms=processing_time_ms
        )
    except Exception as e:
        logger.error("Failed to track analytics", error=e)
```

### `/src/api/v2/flow_routes.py`
**Purpose**: HTTP endpoints for flow management
```python
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field, validator
import uuid

from src.api.dependencies.auth import authenticate
from src.api.dependencies.rate_limit import rate_limit
from src.services.flow_service import FlowService
from src.models.domain.enums import FlowStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Request/Response Models
class CreateFlowRequest(BaseModel):
    """Create flow request model"""
    name: str = Field(..., description="Flow name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Flow description")
    version: str = Field(default="1.0", description="Flow version")
    flow_definition: Dict[str, Any] = Field(..., description="Complete flow definition")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Flow name cannot be empty')
        return v.strip()

class UpdateFlowRequest(BaseModel):
    """Update flow request model"""
    name: Optional[str] = Field(None, description="Flow name")
    description: Optional[str] = Field(None, description="Flow description") 
    flow_definition: Optional[Dict[str, Any]] = Field(None, description="Flow definition")
    status: Optional[FlowStatus] = Field(None, description="Flow status")

class FlowResponse(BaseModel):
    """Flow response model"""
    flow_id: str
    tenant_id: str
    name: str
    version: str
    description: Optional[str]
    status: str
    is_default: bool
    flow_definition: Dict[str, Any]
    created_at: str
    updated_at: str
    usage_count: int
    last_used_at: Optional[str]

class FlowListResponse(BaseModel):
    """Flow list response model"""
    flows: List[FlowResponse]
    total_count: int
    page: int
    page_size: int

# Create router
router = APIRouter(prefix="/flows", tags=["Flows"])

# Initialize service
flow_service = FlowService()

@router.post(
    "/",
    response_model=FlowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create conversation flow",
    description="Create a new conversation flow"
)
async def create_flow(
    request: CreateFlowRequest,
    auth_context: Dict[str, Any] = Depends(authenticate),
    rate_limit_status: Dict[str, Any] = Depends(rate_limit)
) -> FlowResponse:
    """
    Create a new conversation flow
    
    Creates a new flow with the provided definition. The flow will be
    in 'draft' status initially and must be published before use.
    """
    try:
        tenant_id = auth_context.get("tenant_id")
        user_id = auth_context.get("user_id")
        
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID not found in authentication context"
            )
        
        # Create flow
        flow_definition = await flow_service.create_flow(
            tenant_id=tenant_id,
            name=request.name,
            flow_definition=request.flow_definition,
            description=request.description,
            version=request.version,
            created_by=user_id
        )
        
        return FlowResponse(
            flow_id=flow_definition.flow_id,
            tenant_id=flow_definition.tenant_id,
            name=flow_definition.name,
            version=flow_definition.version,
            description=flow_definition.description,
            status=FlowStatus.DRAFT.value,
            is_default=False,
            flow_definition=flow_definition.dict(),
            created_at=flow_definition.metadata.get("created_at", ""),
            updated_at=flow_definition.metadata.get("updated_at", ""),
            usage_count=0,
            last_used_at=None
        )
        
    except Exception as e:
        logger.error(
            "Failed to create flow",
            tenant_id=auth_context.get("tenant_id"),
            flow_name=request.name,
            error=e
        )
        
        if isinstance(e, HTTPException):
            raise
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create flow"
        )

@router.get(
    "/",
    response_model=FlowListResponse,
    summary="List flows",
    description="List flows for the authenticated tenant"
)
async def list_flows(
    status_filter: Optional[List[str]] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    auth_context: Dict[str, Any] = Depends(authenticate),
    rate_limit_status: Dict[str, Any] = Depends(rate_limit)
) -> FlowListResponse:
    """
    List conversation flows
    
    Returns a paginated list of flows for the authenticated tenant.
    Supports filtering by status.
    """
    try:
        tenant_id = auth_context.get("tenant_id")
        
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID not found in authentication context"
            )
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get flows
        flows = await flow_service.list_flows(
            tenant_id=tenant_id,
            status_filter=status_filter,
            limit=page_size,
            offset=offset
        )
        
        # Convert to response format
        flow_responses = []
        for flow in flows:
            flow_responses.append(FlowResponse(
                flow_id=flow.flow_id,
                tenant_id=flow.tenant_id,
                name=flow.name,
                version=flow.version,
                description=flow.description,
                status=flow.metadata.get("status", "draft"),
                is_default=flow.metadata.get("is_default", False),
                flow_definition=flow.dict(),
                created_at=flow.metadata.get("created_at", ""),
                updated_at=flow.metadata.get("updated_at", ""),
                usage_count=flow.metadata.get("usage_count", 0),
                last_used_at=flow.metadata.get("last_used_at")
            ))
        
        return FlowListResponse(
            flows=flow_responses,
            total_count=len(flow_responses),  # TODO: Get actual count
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(
            "Failed to list flows",
            tenant_id=auth_context.get("tenant_id"),
            error=e
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list flows"
        )

@router.get(
    "/{flow_id}",
    response_model=FlowResponse,
    summary="Get flow",
    description="Get a specific flow by ID"
)
async def get_flow(
    flow_id: str,
    auth_context: Dict[str, Any] = Depends(authenticate),
    rate_limit_status: Dict[str, Any] = Depends(rate_limit)
) -> FlowResponse:
    """
    Get a conversation flow by ID
    
    Returns the complete flow definition and metadata.
    """
    try:
        tenant_id = auth_context.get("tenant_id")
        
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID not found in authentication context"
            )
        
        # Get flow
        flow = await flow_service.get_flow_by_id(
            tenant_id=tenant_id,
            flow_id=flow_id
        )
        
        if not flow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Flow not found"
            )
        
        return FlowResponse(
            flow_id=flow.flow_id,
            tenant_id=flow.tenant_id,
            name=flow.name,
            version=flow.version,
            description=flow.description,
            status=flow.metadata.get("status", "draft"),
            is_default=flow.metadata.get("is_default", False),
            flow_definition=flow.dict(),
            created_at=flow.metadata.get("created_at", ""),
            updated_at=flow.metadata.get("updated_at", ""),
            usage_count=flow.metadata.get("usage_count", 0),
            last_used_at=flow.metadata.get("last_used_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get flow",
            tenant_id=auth_context.get("tenant_id"),
            flow_id=flow_id,
            error=e
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get flow"
        )

@router.put(
    "/{flow_id}",
    response_model=FlowResponse,
    summary="Update flow",
    description="Update an existing flow"
)
async def update_flow(
    flow_id: str,
    request: UpdateFlowRequest,
    auth_context: Dict[str, Any] = Depends(authenticate),
    rate_limit_status: Dict[str, Any] = Depends(rate_limit)
) -> FlowResponse:
    """
    Update a conversation flow
    
    Updates the specified flow with the provided changes.
    Only draft flows can be modified.
    """
    try:
        tenant_id = auth_context.get("tenant_id")
        user_id = auth_context.get("user_id")
        
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID not found in authentication context"
            )
        
        # Prepare updates
        updates = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.description is not None:
            updates["description"] = request.description
        if request.flow_definition is not None:
            updates["flow_definition"] = request.flow_definition
        if request.status is not None:
            updates["status"] = request.status.value
        
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No updates provided"
            )
        
        # Update flow
        updated_flow = await flow_service.update_flow(
            tenant_id=tenant_id,
            flow_id=flow_id,
            updates=updates,
            updated_by=user_id
        )
        
        if not updated_flow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Flow not found"
            )
        
        return FlowResponse(
            flow_id=updated_flow.flow_id,
            tenant_id=updated_flow.tenant_id,
            name=updated_flow.name,
            version=updated_flow.version,
            description=updated_flow.description,
            status=updated_flow.metadata.get("status", "draft"),
            is_default=updated_flow.metadata.get("is_default", False),
            flow_definition=updated_flow.dict(),
            created_at=updated_flow.metadata.get("created_at", ""),
            updated_at=updated_flow.metadata.get("updated_at", ""),
            usage_count=updated_flow.metadata.get("usage_count", 0),
            last_used_at=updated_flow.metadata.get("last_used_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to update flow",
            tenant_id=auth_context.get("tenant_id"),
            flow_id=flow_id,
            error=e
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update flow"
        )

@router.post(
    "/{flow_id}/publish",
    summary="Publish flow",
    description="Publish a flow to make it active"
)
async def publish_flow(
    flow_id: str,
    auth_context: Dict[str, Any] = Depends(authenticate),
    rate_limit_status: Dict[str, Any] = Depends(rate_limit)
) -> Dict[str, Any]:
    """
    Publish a conversation flow
    
    Changes the flow status to 'active' so it can be used in conversations.
    """
    try:
        tenant_id = auth_context.get("tenant_id")
        user_id = auth_context.get("user_id")
        
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID not found in authentication context"
            )
        
        # Publish flow
        success = await flow_service.publish_flow(
            tenant_id=tenant_id,
            flow_id=flow_id,
            published_by=user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Flow not found"
            )
        
        return {
            "success": True,
            "message": "Flow published successfully",
            "flow_id": flow_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to publish flow",
            tenant_id=auth_context.get("tenant_id"),
            flow_id=flow_id,
            error=e
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to publish flow"
        )

@router.post(
    "/{flow_id}/set-default",
    summary="Set default flow",
    description="Set a flow as the default for the tenant"
)
async def set_default_flow(
    flow_id: str,
    auth_context: Dict[str, Any] = Depends(authenticate),
    rate_limit_status: Dict[str, Any] = Depends(rate_limit)
) -> Dict[str, Any]:
    """
    Set a flow as the tenant's default flow
    
    The default flow is used when no specific flow is requested
    for a conversation.
    """
    try:
        tenant_id = auth_context.get("tenant_id")
        
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID not found in authentication context"
            )
        
        # Set default flow
        success = await flow_service.set_default_flow(
            tenant_id=tenant_id,
            flow_id=flow_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Flow not found"
            )
        
        return {
            "success": True,
            "message": "Default flow set successfully",
            "flow_id": flow_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to set default flow",
            tenant_id=auth_context.get("tenant_id"),
            flow_id=flow_id,
            error=e
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set default flow"
        )
```

## Success Criteria
- [x] Complete API infrastructure with middleware support
- [x] Authentication middleware with JWT and API key support
- [x] Advanced rate limiting with multiple algorithms
- [x] HTTP endpoints for execution and flow management
- [x] Request/response validation with Pydantic models
- [x] Comprehensive error handling and status codes
- [x] Background task support for analytics
- [x] Proper logging and monitoring integration

## Key Error Handling & Performance Considerations
1. **Authentication**: Multi-method auth with caching and validation
2. **Rate Limiting**: Sliding window and token bucket algorithms
3. **Request Validation**: Comprehensive input validation with Pydantic
4. **Error Handling**: Structured error responses with proper HTTP status codes
5. **Performance**: Background tasks for non-critical operations
6. **Security**: Tenant isolation and authorization checks
7. **Monitoring**: Request tracing and metrics collection

## Technologies Used
- **API Framework**: FastAPI with automatic OpenAPI documentation
- **Authentication**: JWT tokens and API key validation
- **Rate Limiting**: Redis-based sliding window and token bucket
- **Validation**: Pydantic models with custom validators
- **Background Tasks**: FastAPI background tasks for analytics
- **Documentation**: Automatic API documentation generation

## Cross-Service Integration
- **Execution Service**: Message processing and conversation management
- **Flow Service**: Flow CRUD operations and lifecycle management
- **Security Hub**: Authentication and authorization validation
- **Analytics**: Background event tracking and metrics
- **Rate Limiting**: Redis-based quota management
- **Caching**: Response caching and performance optimization

## Next Phase Dependencies
Phase 8 will build upon:
- HTTP API infrastructure and patterns
- Authentication and authorization framework
- Rate limiting and security measures
- Request/response validation systems
- Error handling and monitoring capabilities