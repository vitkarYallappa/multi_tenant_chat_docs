# Phase 6: API Layer & Middleware
**Duration:** Week 9-10  
**Steps:** 11-12 of 18

---

## 🎯 Objectives
- Implement REST API endpoints for chat operations
- Create middleware for authentication, rate limiting, and tenant isolation
- Build request/response validation and serialization
- Establish API versioning and documentation

---

## 📋 Step 11: API Endpoints & Route Handlers

### What Will Be Implemented
- Chat API endpoints for message processing
- Conversation management endpoints
- Health check and metrics endpoints
- Webhook handlers for external integrations

### Folders and Files Created

```
src/api/
├── __init__.py
├── v2/
│   ├── __init__.py
│   ├── chat_routes.py
│   ├── conversation_routes.py
│   ├── session_routes.py
│   ├── health_routes.py
│   └── webhook_routes.py
├── validators/
│   ├── __init__.py
│   ├── message_validators.py
│   ├── conversation_validators.py
│   └── common_validators.py
└── responses/
    ├── __init__.py
    ├── api_response.py
    └── error_responses.py
```

### File Documentation

#### `src/api/v2/chat_routes.py`
**Purpose:** REST API endpoints for chat message operations  
**Usage:** Handle incoming message requests and return bot responses

**Functions:**

1. **send_message(request: SendMessageRequest, ...) -> MessageResponse**
   - **Purpose:** Process incoming chat message and return response
   - **Parameters:**
     - `request` (SendMessageRequest): Message data and metadata
     - `tenant_id` (str): Tenant identifier from header
     - `auth_context` (AuthContext): Authentication context from middleware
     - `message_service` (MessageService): Injected message service
   - **Return:** MessageResponse with bot reply and conversation state
   - **HTTP Method:** POST /api/v2/chat/message

2. **get_conversation_history(conversation_id: str, ...) -> ConversationHistoryResponse**
   - **Purpose:** Retrieve conversation history with pagination
   - **Parameters:**
     - `conversation_id` (str): Conversation identifier
     - `page` (int): Page number for pagination
     - `page_size` (int): Number of messages per page
   - **Return:** Paginated conversation history
   - **HTTP Method:** GET /api/v2/chat/conversations/{conversation_id}/history

```python
from fastapi import APIRouter, Depends, Header, Query, HTTPException, status
from typing import Annotated, Optional
from datetime import datetime

from src.api.validators.message_validators import (
    SendMessageRequest, MessageResponse, ConversationHistoryResponse
)
from src.api.responses.api_response import APIResponse, create_success_response, create_error_response
from src.services.message_service import MessageService
from src.services.conversation_service import ConversationService
from src.dependencies import get_message_service, get_conversation_service
from src.api.middleware.auth_middleware import get_auth_context, AuthContext
from src.api.middleware.rate_limit_middleware import check_rate_limit
from src.models.types import ChannelType
from src.services.exceptions import (
    ServiceError, ValidationError, UnauthorizedError, NotFoundError
)

router = APIRouter(prefix="/api/v2/chat", tags=["chat"])

@router.post(
    "/message",
    response_model=APIResponse[MessageResponse],
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description="Process an incoming message and generate a bot response"
)
async def send_message(
    request: SendMessageRequest,
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    message_service: Annotated[MessageService, Depends(get_message_service)],
    rate_limit_check: Annotated[bool, Depends(check_rate_limit)]
) -> APIResponse[MessageResponse]:
    """
    Send a message through the chat system
    
    This endpoint processes incoming messages, generates appropriate responses,
    and handles the complete message lifecycle including:
    - Message validation and normalization
    - Conversation context management
    - Response generation
    - Channel delivery
    
    Args:
        request: Message content and metadata
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        message_service: Message processing service
        rate_limit_check: Rate limiting validation
    
    Returns:
        APIResponse containing MessageResponse with bot reply
        
    Raises:
        400: Invalid request data
        401: Authentication failed
        403: Access denied
        429: Rate limit exceeded
        500: Internal server error
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )
        
        # Add tenant_id to request if not present
        if not hasattr(request, 'tenant_id') or not request.tenant_id:
            request.tenant_id = tenant_id
        
        # Process message through service layer
        response = await message_service.process_message(
            request=request,
            user_context=auth_context.dict()
        )
        
        return create_success_response(
            data=response,
            message="Message processed successfully"
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Message processing failed"
        )

@router.get(
    "/conversations/{conversation_id}/history",
    response_model=APIResponse[ConversationHistoryResponse],
    summary="Get conversation history",
    description="Retrieve paginated conversation message history"
)
async def get_conversation_history(
    conversation_id: str,
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Messages per page")
) -> APIResponse[ConversationHistoryResponse]:
    """
    Get conversation message history with pagination
    
    Args:
        conversation_id: Conversation identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service
        page: Page number for pagination
        page_size: Number of messages per page
        
    Returns:
        APIResponse containing paginated conversation history
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )
        
        # Get conversation history
        history = await conversation_service.get_conversation_history(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            page=page,
            page_size=page_size,
            user_context=auth_context.dict()
        )
        
        return create_success_response(
            data=history,
            message="Conversation history retrieved successfully"
        )
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )

@router.get(
    "/conversations",
    response_model=APIResponse[ConversationHistoryResponse],
    summary="List conversations",
    description="Get a list of conversations for the authenticated user"
)
async def list_conversations(
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
    status: Optional[str] = Query(default=None, description="Filter by conversation status"),
    channel: Optional[ChannelType] = Query(default=None, description="Filter by channel"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of conversations"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination")
) -> APIResponse[ConversationHistoryResponse]:
    """
    List conversations for the authenticated user
    
    Args:
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service
        status: Optional status filter
        channel: Optional channel filter
        limit: Maximum number of conversations to return
        offset: Pagination offset
        
    Returns:
        APIResponse containing list of conversations
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )
        
        # Build filters
        filters = {}
        if status:
            filters["status"] = status
        if channel:
            filters["channel"] = channel
        
        # Get conversations
        conversations = await conversation_service.list_user_conversations(
            tenant_id=tenant_id,
            user_id=auth_context.user_id,
            filters=filters,
            limit=limit,
            offset=offset
        )
        
        return create_success_response(
            data=conversations,
            message="Conversations retrieved successfully"
        )
        
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )

@router.post(
    "/conversations/{conversation_id}/close",
    response_model=APIResponse[dict],
    summary="Close conversation",
    description="Mark a conversation as completed"
)
async def close_conversation(
    conversation_id: str,
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)]
) -> APIResponse[dict]:
    """
    Close an active conversation
    
    Args:
        conversation_id: Conversation identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service
        
    Returns:
        APIResponse confirming conversation closure
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )
        
        # Close conversation
        result = await conversation_service.close_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_context=auth_context.dict()
        )
        
        return create_success_response(
            data={"conversation_id": conversation_id, "status": "closed"},
            message="Conversation closed successfully"
        )
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to close conversation"
        )

@router.get(
    "/conversations/{conversation_id}/export",
    summary="Export conversation",
    description="Export conversation data in specified format"
)
async def export_conversation(
    conversation_id: str,
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID")],
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    conversation_service: Annotated[ConversationService, Depends(get_conversation_service)],
    format: str = Query(default="json", description="Export format (json, csv, txt)")
):
    """
    Export conversation data for download
    
    Args:
        conversation_id: Conversation identifier
        tenant_id: Tenant identifier from header
        auth_context: User authentication context
        conversation_service: Conversation management service
        format: Export format (json, csv, txt)
        
    Returns:
        File download response with conversation data
    """
    try:
        # Validate tenant access
        if auth_context.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )
        
        # Validate format
        if format not in ["json", "csv", "txt"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid export format. Supported: json, csv, txt"
            )
        
        # Export conversation
        export_data = await conversation_service.export_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            format=format,
            user_context=auth_context.dict()
        )
        
        # Return file response
        from fastapi.responses import Response
        
        media_type = {
            "json": "application/json",
            "csv": "text/csv",
            "txt": "text/plain"
        }[format]
        
        filename = f"conversation_{conversation_id}.{format}"
        
        return Response(
            content=export_data["content"],
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export conversation"
        )
```

#### `src/api/validators/message_validators.py`
**Purpose:** Pydantic models for request/response validation in chat endpoints  
**Usage:** Validate incoming requests and serialize outgoing responses

**Classes:**

1. **SendMessageRequest(BaseModel)**
   - **Purpose:** Validate incoming message requests
   - **Fields:** Message content, metadata, processing hints
   - **Usage:** Request validation in chat endpoints

2. **MessageResponse(BaseModel)**
   - **Purpose:** Structure for message processing responses
   - **Fields:** Response content, conversation state, metadata
   - **Usage:** Response serialization in chat endpoints

```python
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4

from src.models.types import (
    MessageContent, ChannelType, ChannelMetadata, ProcessingHints,
    TenantId, UserId, ConversationId, MessageId
)

class SendMessageRequest(BaseModel):
    """Request model for sending chat messages"""
    
    # Message identification
    message_id: MessageId = Field(default_factory=lambda: str(uuid4()))
    conversation_id: Optional[ConversationId] = None
    user_id: UserId
    session_id: Optional[str] = None
    tenant_id: Optional[TenantId] = None  # Set by middleware
    
    # Channel information
    channel: ChannelType
    channel_metadata: Optional[ChannelMetadata] = None
    
    # Message content
    content: MessageContent
    
    # Processing hints
    processing_hints: Optional[ProcessingHints] = None
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "message_id": "msg_123e4567-e89b-12d3-a456-426614174000",
                "user_id": "user_12345",
                "channel": "web",
                "content": {
                    "type": "text",
                    "text": "Hello, I need help with my order",
                    "language": "en"
                },
                "channel_metadata": {
                    "platform_user_id": "web_user_123",
                    "additional_data": {
                        "user_agent": "Mozilla/5.0...",
                        "ip_address": "192.168.1.1"
                    }
                },
                "processing_hints": {
                    "priority": "normal",
                    "expected_response_type": "text",
                    "bypass_automation": False
                }
            }
        }
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('User ID cannot be empty')
        if len(v) > 255:
            raise ValueError('User ID too long (max 255 characters)')
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        if not v:
            raise ValueError('Message content is required')
        
        # Additional content validation based on type
        if v.type.value == "text" and not v.text:
            raise ValueError('Text content is required for text messages')
        
        if v.type.value in ["image", "video", "audio", "file"] and not v.media:
            raise ValueError(f'Media content is required for {v.type} messages')
        
        if v.type.value == "location" and not v.location:
            raise ValueError('Location content is required for location messages')
        
        return v

class ConversationContext(BaseModel):
    """Conversation context in response"""
    current_intent: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    slots: Dict[str, Any] = Field(default_factory=dict)
    conversation_stage: Optional[str] = None
    next_expected_inputs: List[str] = Field(default_factory=list)

class ProcessingMetadata(BaseModel):
    """Processing metadata in response"""
    processing_time_ms: int
    model_used: Optional[str] = None
    model_provider: Optional[str] = None
    cost_cents: Optional[float] = None
    fallback_applied: bool = False
    confidence_score: Optional[float] = None

class MessageResponse(BaseModel):
    """Response model for processed messages"""
    
    # Message identification
    message_id: MessageId
    conversation_id: ConversationId
    
    # Bot response
    response: MessageContent
    
    # Conversation state
    conversation_state: ConversationContext
    
    # Processing metadata
    processing_metadata: ProcessingMetadata
    
    # Response metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "message_id": "msg_123e4567-e89b-12d3-a456-426614174000",
                "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174001",
                "response": {
                    "type": "text",
                    "text": "I'd be happy to help you with your order. Could you please provide your order number?",
                    "language": "en",
                    "quick_replies": [
                        {
                            "title": "I have my order number",
                            "payload": "provide_order_number"
                        },
                        {
                            "title": "I don't have it",
                            "payload": "no_order_number"
                        }
                    ]
                },
                "conversation_state": {
                    "current_intent": "order_inquiry",
                    "entities": {
                        "inquiry_type": "order_status"
                    },
                    "conversation_stage": "information_gathering",
                    "next_expected_inputs": ["order_number"]
                },
                "processing_metadata": {
                    "processing_time_ms": 287,
                    "model_used": "gpt-4-turbo",
                    "model_provider": "openai",
                    "cost_cents": 1.25,
                    "confidence_score": 0.92
                }
            }
        }

class ConversationSummary(BaseModel):
    """Summary information for conversation list"""
    conversation_id: ConversationId
    user_id: UserId
    channel: ChannelType
    status: str
    started_at: datetime
    last_activity_at: datetime
    message_count: int
    primary_intent: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history"""
    conversation_id: ConversationId
    summary: ConversationSummary
    messages: List[Dict[str, Any]]
    
    # Pagination
    page: int
    page_size: int
    total_messages: int
    has_next: bool
    has_previous: bool
    
    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174001",
                "summary": {
                    "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174001",
                    "user_id": "user_12345",
                    "channel": "web",
                    "status": "active",
                    "started_at": "2025-05-30T10:00:00Z",
                    "last_activity_at": "2025-05-30T10:05:00Z",
                    "message_count": 4,
                    "primary_intent": "order_inquiry"
                },
                "messages": [
                    {
                        "message_id": "msg_1",
                        "direction": "inbound",
                        "timestamp": "2025-05-30T10:00:00Z",
                        "content": {
                            "type": "text",
                            "text": "Hello, I need help with my order"
                        }
                    },
                    {
                        "message_id": "msg_2",
                        "direction": "outbound",
                        "timestamp": "2025-05-30T10:00:01Z",
                        "content": {
                            "type": "text",
                            "text": "I'd be happy to help! Could you provide your order number?"
                        }
                    }
                ],
                "page": 1,
                "page_size": 20,
                "total_messages": 4,
                "has_next": False,
                "has_previous": False
            }
        }

class BulkMessageRequest(BaseModel):
    """Request model for bulk message operations"""
    messages: List[SendMessageRequest] = Field(..., max_items=100)
    batch_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    
    @validator('messages')
    def validate_messages_limit(cls, v):
        if len(v) == 0:
            raise ValueError('At least one message is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 messages per batch')
        return v

class BulkMessageResponse(BaseModel):
    """Response model for bulk message operations"""
    batch_id: str
    total_messages: int
    successful_messages: int
    failed_messages: int
    results: List[Dict[str, Any]]
    processing_time_ms: int
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123e4567-e89b-12d3-a456-426614174000",
                "total_messages": 3,
                "successful_messages": 2,
                "failed_messages": 1,
                "results": [
                    {
                        "message_id": "msg_1",
                        "status": "success",
                        "response": "Message processed successfully"
                    },
                    {
                        "message_id": "msg_2",
                        "status": "failed",
                        "error": "Invalid content format"
                    }
                ],
                "processing_time_ms": 1250
            }
        }
```

---

## 📋 Step 12: Middleware Implementation

### What Will Be Implemented
- Authentication middleware for JWT validation
- Rate limiting middleware with Redis backend
- Tenant isolation middleware
- Error handling middleware with standardized responses

### Folders and Files Created

```
src/api/
├── middleware/
│   ├── __init__.py
│   ├── auth_middleware.py
│   ├── rate_limit_middleware.py
│   ├── tenant_middleware.py
│   ├── error_handler.py
│   └── logging_middleware.py
└── responses/
    ├── api_response.py
    └── error_responses.py
```

### File Documentation

#### `src/api/middleware/auth_middleware.py`
**Purpose:** JWT authentication and authorization middleware  
**Usage:** Validate tokens, extract user context, enforce permissions

**Classes:**

1. **AuthContext(BaseModel)**
   - **Purpose:** User authentication context data
   - **Fields:** User ID, tenant ID, permissions, token metadata
   - **Usage:** Pass authenticated user data between middleware and endpoints

**Functions:**

1. **get_auth_context(authorization: str = Header()) -> AuthContext**
   - **Purpose:** Extract and validate authentication from request headers
   - **Parameters:**
     - `authorization` (str): Authorization header with Bearer token
   - **Return:** AuthContext with validated user data
   - **Description:** Validates JWT tokens and extracts user context

2. **verify_jwt_token(token: str) -> Dict[str, Any]**
   - **Purpose:** Verify JWT token signature and extract payload
   - **Parameters:**
     - `token` (str): JWT token to verify
   - **Return:** Token payload if valid
   - **Description:** Validates token using configured secret and algorithm

```python
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import Header, HTTPException, status, Depends
from pydantic import BaseModel
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
import structlog

from src.config.settings import get_settings
from src.services.exceptions import UnauthorizedError

logger = structlog.get_logger()

class AuthContext(BaseModel):
    """Authentication context for requests"""
    user_id: str
    tenant_id: str
    email: Optional[str] = None
    role: str = "member"
    permissions: List[str] = []
    scopes: List[str] = []
    
    # Token metadata
    token_type: str = "bearer"
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Rate limiting info
    rate_limit_tier: str = "standard"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

async def get_auth_context(
    authorization: str = Header(alias="Authorization")
) -> AuthContext:
    """
    Extract and validate authentication context from request
    
    Args:
        authorization: Authorization header with Bearer token
        
    Returns:
        AuthContext with validated user data
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Validate authorization header format
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Extract token
        token = authorization.split(" ")[1]
        
        # Verify and decode token
        payload = await verify_jwt_token(token)
        
        # Extract user context from token
        auth_context = AuthContext(
            user_id=payload.get("sub"),
            tenant_id=payload.get("tenant_id"),
            email=payload.get("email"),
            role=payload.get("user_role", "member"),
            permissions=payload.get("permissions", []),
            scopes=payload.get("scopes", []),
            rate_limit_tier=payload.get("rate_limit_tier", "standard"),
            issued_at=datetime.fromtimestamp(payload.get("iat", 0)) if payload.get("iat") else None,
            expires_at=datetime.fromtimestamp(payload.get("exp", 0)) if payload.get("exp") else None
        )
        
        logger.info(
            "Authentication successful",
            user_id=auth_context.user_id,
            tenant_id=auth_context.tenant_id,
            role=auth_context.role
        )
        
        return auth_context
        
    except InvalidTokenError as e:
        logger.warning("Invalid JWT token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except ExpiredSignatureError:
        logger.warning("Expired JWT token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )

async def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and extract payload
    
    Args:
        token: JWT token to verify
        
    Returns:
        Token payload if valid
        
    Raises:
        InvalidTokenError: If token is invalid
        ExpiredSignatureError: If token is expired
    """
    settings = get_settings()
    
    try:
        # Decode and verify token
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Validate required fields
        required_fields = ["sub", "tenant_id", "exp"]
        missing_fields = [field for field in required_fields if field not in payload]
        
        if missing_fields:
            raise InvalidTokenError(f"Missing required fields: {missing_fields}")
        
        # Validate token hasn't expired
        current_timestamp = datetime.utcnow().timestamp()
        if payload.get("exp", 0) < current_timestamp:
            raise ExpiredSignatureError("Token has expired")
        
        return payload
        
    except (InvalidTokenError, ExpiredSignatureError):
        raise
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise InvalidTokenError(f"Token verification failed: {e}")

async def get_optional_auth_context(
    authorization: Optional[str] = Header(default=None, alias="Authorization")
) -> Optional[AuthContext]:
    """
    Get authentication context if provided, otherwise return None
    
    Args:
        authorization: Optional authorization header
        
    Returns:
        AuthContext if token provided and valid, None otherwise
    """
    if not authorization:
        return None
    
    try:
        return await get_auth_context(authorization)
    except HTTPException:
        return None

def require_permissions(*required_permissions: str):
    """
    Decorator to require specific permissions
    
    Args:
        required_permissions: List of required permissions
        
    Returns:
        Dependency function that validates permissions
    """
    def permission_checker(
        auth_context: AuthContext = Depends(get_auth_context)
    ) -> AuthContext:
        """Check if user has required permissions"""
        user_permissions = set(auth_context.permissions)
        missing_permissions = set(required_permissions) - user_permissions
        
        if missing_permissions:
            logger.warning(
                "Insufficient permissions",
                user_id=auth_context.user_id,
                required=list(required_permissions),
                missing=list(missing_permissions)
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {list(missing_permissions)}"
            )
        
        return auth_context
    
    return permission_checker

def require_role(required_role: str):
    """
    Decorator to require specific role
    
    Args:
        required_role: Required user role
        
    Returns:
        Dependency function that validates role
    """
    def role_checker(
        auth_context: AuthContext = Depends(get_auth_context)
    ) -> AuthContext:
        """Check if user has required role"""
        role_hierarchy = {
            "viewer": 0,
            "member": 1,
            "manager": 2,
            "developer": 3,
            "admin": 4,
            "owner": 5
        }
        
        user_level = role_hierarchy.get(auth_context.role, 0)
        required_level = role_hierarchy.get(required_role, 999)
        
        if user_level < required_level:
            logger.warning(
                "Insufficient role",
                user_id=auth_context.user_id,
                user_role=auth_context.role,
                required_role=required_role
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {required_role}"
            )
        
        return auth_context
    
    return role_checker
```

#### `src/api/middleware/rate_limit_middleware.py`
**Purpose:** Rate limiting middleware using Redis for tracking and enforcement  
**Usage:** Prevent API abuse and ensure fair usage across tenants

**Functions:**

1. **check_rate_limit(request: Request, auth_context: AuthContext) -> bool**
   - **Purpose:** Check and enforce rate limits for requests
   - **Parameters:**
     - `request` (Request): FastAPI request object
     - `auth_context` (AuthContext): User authentication context
   - **Return:** True if within rate limit
   - **Description:** Validates request against configured rate limits

2. **get_rate_limit_key(identifier: str, window: str) -> str**
   - **Purpose:** Generate Redis key for rate limit tracking
   - **Parameters:**
     - `identifier` (str): Unique identifier (user, IP, API key)
     - `window` (str): Time window (minute, hour, day)
   - **Return:** Redis key for rate limit counter
   - **Description:** Creates consistent keys for rate limit storage

```python
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status, Depends
from starlette.responses import Response
import structlog

from src.api.middleware.auth_middleware import AuthContext, get_auth_context, get_optional_auth_context
from src.repositories.rate_limit_repository import RateLimitRepository
from src.dependencies import get_rate_limit_repository
from src.config.settings import get_settings

logger = structlog.get_logger()

# Rate limit configuration
RATE_LIMITS = {
    "basic": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 10000
    },
    "standard": {
        "requests_per_minute": 200,
        "requests_per_hour": 5000,
        "requests_per_day": 50000
    },
    "premium": {
        "requests_per_minute": 1000,
        "requests_per_hour": 20000,
        "requests_per_day": 200000
    },
    "enterprise": {
        "requests_per_minute": 5000,
        "requests_per_hour": 100000,
        "requests_per_day": 1000000
    }
}

async def check_rate_limit(
    request: Request,
    rate_limit_repo: RateLimitRepository = Depends(get_rate_limit_repository),
    auth_context: Optional[AuthContext] = Depends(get_optional_auth_context)
) -> bool:
    """
    Check and enforce rate limits for requests
    
    Args:
        request: FastAPI request object
        rate_limit_repo: Rate limiting repository
        auth_context: Optional authentication context
        
    Returns:
        True if within rate limit
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    try:
        # Determine rate limit identifier and tier
        if auth_context:
            identifier = f"user:{auth_context.tenant_id}:{auth_context.user_id}"
            tier = auth_context.rate_limit_tier
            tenant_id = auth_context.tenant_id
        else:
            # Use IP address for unauthenticated requests
            client_ip = get_client_ip(request)
            identifier = f"ip:{client_ip}"
            tier = "basic"
            tenant_id = "anonymous"
        
        # Get rate limits for tier
        limits = RATE_LIMITS.get(tier, RATE_LIMITS["basic"])
        
        # Check rate limits for different windows
        rate_limit_results = {}
        
        # Check per-minute limit
        allowed_minute, count_minute, reset_minute = await rate_limit_repo.check_rate_limit(
            tenant_id=tenant_id,
            identifier=f"{identifier}:minute",
            limit=limits["requests_per_minute"],
            window_seconds=60
        )
        
        rate_limit_results["minute"] = {
            "allowed": allowed_minute,
            "count": count_minute,
            "limit": limits["requests_per_minute"],
            "reset": reset_minute
        }
        
        # Check per-hour limit
        allowed_hour, count_hour, reset_hour = await rate_limit_repo.check_rate_limit(
            tenant_id=tenant_id,
            identifier=f"{identifier}:hour",
            limit=limits["requests_per_hour"],
            window_seconds=3600
        )
        
        rate_limit_results["hour"] = {
            "allowed": allowed_hour,
            "count": count_hour,
            "limit": limits["requests_per_hour"],
            "reset": reset_hour
        }
        
        # Check per-day limit
        allowed_day, count_day, reset_day = await rate_limit_repo.check_rate_limit(
            tenant_id=tenant_id,
            identifier=f"{identifier}:day",
            limit=limits["requests_per_day"],
            window_seconds=86400
        )
        
        rate_limit_results["day"] = {
            "allowed": allowed_day,
            "count": count_day,
            "limit": limits["requests_per_day"],
            "reset": reset_day
        }
        
        # Check if any limit is exceeded
        exceeded_limits = []
        for window, result in rate_limit_results.items():
            if not result["allowed"]:
                exceeded_limits.append(window)
        
        if exceeded_limits:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                tier=tier,
                exceeded_limits=exceeded_limits,
                rate_limit_results=rate_limit_results
            )
            
            # Find the most restrictive limit for error response
            most_restrictive = rate_limit_results["minute"]
            for window in ["hour", "day"]:
                if window in exceeded_limits:
                    most_restrictive = rate_limit_results[window]
                    break
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {most_restrictive['reset'] - int(datetime.utcnow().timestamp())} seconds",
                headers={
                    "X-RateLimit-Limit": str(most_restrictive["limit"]),
                    "X-RateLimit-Remaining": str(max(0, most_restrictive["limit"] - most_restrictive["count"])),
                    "X-RateLimit-Reset": str(most_restrictive["reset"]),
                    "X-RateLimit-Type": tier,
                    "Retry-After": str(most_restrictive["reset"] - int(datetime.utcnow().timestamp()))
                }
            )
        
        # Add rate limit headers to response
        # Note: These will be added by response middleware
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(limits["requests_per_minute"]),
            "X-RateLimit-Remaining": str(max(0, limits["requests_per_minute"] - count_minute)),
            "X-RateLimit-Reset": str(reset_minute),
            "X-RateLimit-Type": tier
        }
        
        logger.debug(
            "Rate limit check passed",
            identifier=identifier,
            tier=tier,
            counts={
                "minute": count_minute,
                "hour": count_hour,
                "day": count_day
            }
        )
        
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Rate limit check failed", error=str(e))
        # Allow request on error to avoid blocking legitimate traffic
        return True

def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client IP address
    """
    # Check for IP in various headers (for reverse proxy setups)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct client IP
    return request.client.host if request.client else "unknown"

class RateLimitMiddleware:
    """Middleware class for rate limiting"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """Process request through rate limiting"""
        
        # Skip rate limiting for health checks and internal endpoints
        if request.url.path in ["/health", "/metrics", "/api/v2/health"]:
            response = await call_next(request)
            return response
        
        try:
            # Check rate limit (this will raise exception if exceeded)
            await check_rate_limit(request)
            
            # Continue with request processing
            response = await call_next(request)
            
            # Add rate limit headers if available
            if hasattr(request.state, "rate_limit_headers"):
                for header, value in request.state.rate_limit_headers.items():
                    response.headers[header] = value
            
            return response
            
        except HTTPException as e:
            # Rate limit exceeded - return 429 response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "status": "error",
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": e.detail
                    }
                },
                headers=e.headers
            )
        except Exception as e:
            logger.error("Rate limit middleware error", error=str(e))
            # Continue with request on middleware error
            response = await call_next(request)
            return response

# Utility functions for specific rate limit checks

async def check_api_key_rate_limit(
    api_key: str,
    tenant_id: str,
    rate_limit_repo: RateLimitRepository
) -> bool:
    """Check rate limit for API key usage"""
    # Implementation specific to API key rate limiting
    # This would have different limits than user-based limits
    pass

async def check_webhook_rate_limit(
    source_ip: str,
    webhook_type: str,
    rate_limit_repo: RateLimitRepository
) -> bool:
    """Check rate limit for webhook endpoints"""
    # Implementation specific to webhook rate limiting
    # This would have more permissive limits for legitimate webhooks
    pass
```

---

## 🔧 Technologies Used
- **FastAPI**: Web framework with dependency injection
- **Pydantic**: Request/response validation and serialization
- **JWT**: JSON Web Token authentication
- **Redis**: Rate limiting and session storage
- **structlog**: Structured logging throughout middleware

---

## ⚠️ Key Considerations

### Security
- JWT token validation and expiration
- Role-based access control (RBAC)
- Tenant isolation enforcement
- Rate limiting to prevent abuse
- Input validation and sanitization

### Performance
- Efficient middleware execution order
- Redis-based rate limiting for scalability
- Minimal overhead in auth validation
- Async operations throughout

### Reliability
- Graceful error handling in middleware
- Fallback behavior for external service failures
- Comprehensive logging for debugging
- Health check endpoint exclusions

### User Experience
- Clear error messages with proper HTTP status codes
- Rate limit headers for client awareness
- Consistent API response format
- Detailed validation error responses

---

## 🎯 Success Criteria
- [ ] All API endpoints are implemented and tested
- [ ] Authentication middleware validates JWT tokens correctly
- [ ] Rate limiting prevents abuse while allowing legitimate usage
- [ ] Request/response validation works properly
- [ ] Error handling provides clear, actionable messages
- [ ] API documentation is auto-generated and accurate

---

## 📋 Next Phase Preview
Phase 7 will focus on implementing event handling and Kafka integration for real-time messaging and analytics, building upon the solid API foundation established in this phase.

