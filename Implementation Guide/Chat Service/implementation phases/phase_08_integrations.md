# Phase 8: External Integrations & gRPC Clients
**Duration:** Week 13-14  
**Steps:** 15-16 of 18

---

## 🎯 Objectives
- Implement gRPC clients for MCP Engine and Security Hub
- Build webhook handling system for external platforms
- Create integration adapters for third-party services
- Establish secure communication patterns

---

## 📋 Step 15: gRPC Client Implementation

### What Will Be Implemented
- gRPC client for MCP Engine communication
- gRPC client for Security Hub authentication
- Connection management and health monitoring
- Request/response serialization and error handling

### Folders and Files Created

```
src/clients/
├── __init__.py
├── base_client.py
├── grpc/
│   ├── __init__.py
│   ├── mcp_client.py
│   ├── security_client.py
│   └── connection_manager.py
├── protos/
│   ├── mcp_service.proto
│   ├── security_service.proto
│   └── common.proto
└── exceptions.py
```

### File Documentation

#### `src/clients/base_client.py`
**Purpose:** Base gRPC client class with common functionality and patterns  
**Usage:** Foundation for all gRPC client implementations

**Classes:**

1. **BaseGRPCClient(ABC)**
   - **Purpose:** Abstract base class for gRPC clients
   - **Methods:**
     - **connect() -> None**: Establish gRPC connection
     - **disconnect() -> None**: Close gRPC connection
     - **health_check() -> bool**: Check service health
     - **_make_request(method, request, timeout) -> Any**: Make gRPC request with retry logic

```python
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any, Dict, Callable, TypeVar, Generic
import grpc
from grpc import aio
import structlog

from src.config.settings import get_settings
from src.clients.exceptions import ClientError, ConnectionError, TimeoutError

logger = structlog.get_logger()

T = TypeVar('T')
R = TypeVar('R')

class BaseGRPCClient(ABC, Generic[T]):
    """Base class for gRPC clients with common functionality"""
    
    def __init__(self, 
                 service_url: str,
                 service_name: str,
                 timeout_seconds: int = 30,
                 max_retries: int = 3):
        self.service_url = service_url
        self.service_name = service_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        self.channel: Optional[aio.Channel] = None
        self.stub: Optional[T] = None
        self.is_connected = False
        
        # Connection configuration
        self.channel_options = [
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000),
        ]
        
        # Metrics
        self.connection_attempts = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.last_health_check: Optional[datetime] = None
        self.last_error: Optional[str] = None
    
    @property
    @abstractmethod
    def stub_class(self) -> type:
        """Return the gRPC stub class for this client"""
        pass
    
    async def connect(self) -> None:
        """Establish gRPC connection to service"""
        try:
            self.connection_attempts += 1
            
            # Create channel
            self.channel = aio.insecure_channel(
                self.service_url,
                options=self.channel_options
            )
            
            # Create stub
            self.stub = self.stub_class(self.channel)
            
            # Test connection
            await self._test_connection()
            
            self.is_connected = True
            self.last_error = None
            
            logger.info(
                f"{self.service_name} gRPC client connected",
                service_url=self.service_url,
                attempts=self.connection_attempts
            )
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(
                f"Failed to connect to {self.service_name}",
                service_url=self.service_url,
                error=str(e),
                attempts=self.connection_attempts
            )
            raise ConnectionError(f"Failed to connect to {self.service_name}: {e}")
    
    async def disconnect(self) -> None:
        """Close gRPC connection"""
        try:
            if self.channel:
                await self.channel.close()
                self.channel = None
                self.stub = None
                self.is_connected = False
                
                logger.info(f"{self.service_name} gRPC client disconnected")
                
        except Exception as e:
            logger.error(
                f"Error disconnecting from {self.service_name}",
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            if not self.is_connected:
                return False
            
            # Perform health check (override in subclasses)
            await self._perform_health_check()
            
            self.last_health_check = datetime.utcnow()
            return True
            
        except Exception as e:
            logger.error(
                f"{self.service_name} health check failed",
                error=str(e)
            )
            return False
    
    async def _make_request(
        self,
        method: Callable,
        request: Any,
        timeout: Optional[int] = None,
        retry_on_failure: bool = True
    ) -> Any:
        """
        Make gRPC request with retry logic and error handling
        
        Args:
            method: gRPC method to call
            request: Request object
            timeout: Request timeout in seconds
            retry_on_failure: Whether to retry on failure
            
        Returns:
            Response object
            
        Raises:
            ClientError: If request fails after retries
        """
        if not self.is_connected or not self.stub:
            raise ConnectionError(f"Not connected to {self.service_name}")
        
        timeout = timeout or self.timeout_seconds
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Make the gRPC call
                response = await method(
                    request,
                    timeout=timeout,
                    metadata=self._get_metadata()
                )
                
                self.successful_requests += 1
                
                logger.debug(
                    f"{self.service_name} request successful",
                    method=method.__name__,
                    attempt=attempt + 1
                )
                
                return response
                
            except grpc.aio.AioRpcError as e:
                last_exception = e
                self.failed_requests += 1
                
                # Check if we should retry
                if not retry_on_failure or not self._should_retry(e):
                    break
                
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"{self.service_name} request failed, retrying",
                        method=method.__name__,
                        attempt=attempt + 1,
                        error=str(e),
                        wait_time=wait_time
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"{self.service_name} request failed after all retries",
                        method=method.__name__,
                        attempts=attempt + 1,
                        error=str(e)
                    )
            
            except Exception as e:
                last_exception = e
                self.failed_requests += 1
                
                logger.error(
                    f"{self.service_name} request error",
                    method=method.__name__,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if not retry_on_failure or attempt >= self.max_retries:
                    break
                
                await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        error_msg = f"{self.service_name} request failed: {str(last_exception)}"
        raise ClientError(error_msg, original_error=last_exception)
    
    def _should_retry(self, error: grpc.aio.AioRpcError) -> bool:
        """Determine if request should be retried based on error"""
        retryable_codes = [
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.ABORTED,
            grpc.StatusCode.INTERNAL
        ]
        
        return error.code() in retryable_codes
    
    def _get_metadata(self) -> list:
        """Get gRPC metadata for requests"""
        metadata = [
            ('client-name', 'chat-service'),
            ('client-version', '2.0.0'),
        ]
        
        return metadata
    
    async def _test_connection(self) -> None:
        """Test connection (override in subclasses)"""
        # Default implementation - just check if channel is ready
        await self.channel.channel_ready()
    
    async def _perform_health_check(self) -> None:
        """Perform service-specific health check (override in subclasses)"""
        # Default implementation
        await self._test_connection()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for monitoring"""
        return {
            "service_name": self.service_name,
            "service_url": self.service_url,
            "is_connected": self.is_connected,
            "connection_attempts": self.connection_attempts,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_error": self.last_error
        }
```

#### `src/clients/grpc/mcp_client.py`
**Purpose:** gRPC client for MCP Engine communication and message processing  
**Usage:** Send messages to MCP Engine for response generation and conversation management

**Classes:**

1. **MCPEngineClient(BaseGRPCClient)**
   - **Purpose:** Communicate with MCP Engine for message processing
   - **Methods:**
     - **process_message(request: ProcessMessageRequest) -> ProcessMessageResponse**: Process message and get response
     - **update_conversation_context(request: UpdateContextRequest) -> UpdateContextResponse**: Update conversation context
     - **get_conversation_flows(tenant_id: str) -> List[ConversationFlow]**: Get available conversation flows

```python
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import grpc
from grpc import aio

from src.clients.base_client import BaseGRPCClient
from src.clients.exceptions import ClientError, ValidationError
from src.config.settings import get_settings

# Import generated gRPC stubs (these would be generated from .proto files)
# from src.clients.protos import mcp_service_pb2, mcp_service_pb2_grpc

@dataclass
class ProcessMessageRequest:
    """Request for processing a message through MCP Engine"""
    tenant_id: str
    conversation_id: str
    message_id: str
    user_id: str
    channel: str
    
    # Message content
    message_content: Dict[str, Any]
    message_type: str
    
    # Context
    conversation_context: Dict[str, Any]
    user_profile: Dict[str, Any]
    session_context: Dict[str, Any]
    
    # Processing options
    processing_hints: Dict[str, Any]
    flow_id: Optional[str] = None
    
    # Metadata
    timestamp: datetime
    request_id: str

@dataclass
class ProcessMessageResponse:
    """Response from MCP Engine message processing"""
    success: bool
    message_id: str
    conversation_id: str
    
    # Generated response
    response_content: Dict[str, Any]
    response_type: str
    confidence_score: float
    
    # Updated context
    updated_context: Dict[str, Any]
    context_changes: List[str]
    
    # Next actions
    next_expected_inputs: List[str]
    suggested_actions: List[Dict[str, Any]]
    
    # Processing metadata
    processing_time_ms: int
    model_used: str
    model_provider: str
    cost_cents: Optional[float]
    
    # Error information
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class MCPEngineClient(BaseGRPCClient):
    """gRPC client for MCP Engine communication"""
    
    def __init__(self, service_url: Optional[str] = None):
        settings = get_settings()
        service_url = service_url or settings.MCP_ENGINE_URL
        
        super().__init__(
            service_url=service_url,
            service_name="MCP Engine",
            timeout_seconds=30,
            max_retries=2  # Lower retries for real-time processing
        )
    
    @property
    def stub_class(self) -> type:
        # In production, this would be the generated gRPC stub
        # return mcp_service_pb2_grpc.MCPEngineServiceStub
        return object  # Placeholder
    
    async def process_message(
        self,
        request: ProcessMessageRequest
    ) -> ProcessMessageResponse:
        """
        Process a message through MCP Engine
        
        Args:
            request: Message processing request
            
        Returns:
            ProcessMessageResponse with generated response and updated context
            
        Raises:
            ClientError: If processing fails
            ValidationError: If request is invalid
        """
        try:
            # Validate request
            self._validate_process_message_request(request)
            
            # Convert to gRPC request format
            grpc_request = self._build_process_message_request(request)
            
            # Make gRPC call
            # In production, this would use the actual gRPC stub method
            # grpc_response = await self._make_request(
            #     self.stub.ProcessMessage,
            #     grpc_request,
            #     timeout=25  # Slightly less than overall timeout
            # )
            
            # For now, simulate the response
            grpc_response = await self._simulate_process_message_response(request)
            
            # Convert response
            response = self._parse_process_message_response(grpc_response)
            
            logger.info(
                "MCP message processing completed",
                message_id=request.message_id,
                conversation_id=request.conversation_id,
                processing_time_ms=response.processing_time_ms,
                confidence_score=response.confidence_score
            )
            
            return response
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "MCP message processing failed",
                message_id=request.message_id,
                error=str(e)
            )
            raise ClientError(f"MCP message processing failed: {e}")
    
    async def update_conversation_context(
        self,
        tenant_id: str,
        conversation_id: str,
        context_updates: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update conversation context in MCP Engine
        
        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            context_updates: Context updates to apply
            user_id: Optional user identifier
            
        Returns:
            True if update successful
        """
        try:
            # Build gRPC request
            # grpc_request = mcp_service_pb2.UpdateContextRequest(
            #     tenant_id=tenant_id,
            #     conversation_id=conversation_id,
            #     context_updates=context_updates,
            #     user_id=user_id or ""
            # )
            
            # Make gRPC call
            # grpc_response = await self._make_request(
            #     self.stub.UpdateConversationContext,
            #     grpc_request
            # )
            
            # Simulate successful update
            grpc_response = {"success": True}
            
            logger.info(
                "MCP conversation context updated",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                updates_count=len(context_updates)
            )
            
            return grpc_response.get("success", False)
            
        except Exception as e:
            logger.error(
                "MCP context update failed",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                error=str(e)
            )
            raise ClientError(f"MCP context update failed: {e}")
    
    async def get_conversation_flows(
        self,
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get available conversation flows for tenant
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of available conversation flows
        """
        try:
            # Build gRPC request
            # grpc_request = mcp_service_pb2.GetFlowsRequest(
            #     tenant_id=tenant_id
            # )
            
            # Make gRPC call
            # grpc_response = await self._make_request(
            #     self.stub.GetConversationFlows,
            #     grpc_request
            # )
            
            # Simulate flows response
            flows = [
                {
                    "flow_id": "default_support",
                    "name": "Customer Support",
                    "description": "General customer support flow",
                    "version": "1.0",
                    "status": "active"
                },
                {
                    "flow_id": "order_inquiry", 
                    "name": "Order Inquiry",
                    "description": "Handle order-related questions",
                    "version": "1.2",
                    "status": "active"
                }
            ]
            
            logger.info(
                "MCP conversation flows retrieved",
                tenant_id=tenant_id,
                flows_count=len(flows)
            )
            
            return flows
            
        except Exception as e:
            logger.error(
                "MCP get flows failed",
                tenant_id=tenant_id,
                error=str(e)
            )
            raise ClientError(f"MCP get flows failed: {e}")
    
    async def _perform_health_check(self) -> None:
        """Perform MCP Engine health check"""
        try:
            # Make health check call
            # grpc_request = mcp_service_pb2.HealthCheckRequest()
            # grpc_response = await self._make_request(
            #     self.stub.HealthCheck,
            #     grpc_request,
            #     timeout=5,
            #     retry_on_failure=False
            # )
            
            # Simulate health check
            await asyncio.sleep(0.1)  # Simulate network delay
            
            logger.debug("MCP Engine health check passed")
            
        except Exception as e:
            logger.error("MCP Engine health check failed", error=str(e))
            raise
    
    def _validate_process_message_request(self, request: ProcessMessageRequest) -> None:
        """Validate process message request"""
        required_fields = [
            'tenant_id', 'conversation_id', 'message_id', 'user_id',
            'channel', 'message_content', 'message_type'
        ]
        
        for field in required_fields:
            if not getattr(request, field, None):
                raise ValidationError(f"Missing required field: {field}")
        
        if not request.conversation_context:
            raise ValidationError("Conversation context is required")
    
    def _build_process_message_request(self, request: ProcessMessageRequest) -> Any:
        """Build gRPC request from internal request"""
        # In production, this would build the actual protobuf message
        # return mcp_service_pb2.ProcessMessageRequest(
        #     tenant_id=request.tenant_id,
        #     conversation_id=request.conversation_id,
        #     message_id=request.message_id,
        #     user_id=request.user_id,
        #     channel=request.channel,
        #     message_content=json.dumps(request.message_content),
        #     message_type=request.message_type,
        #     conversation_context=json.dumps(request.conversation_context),
        #     user_profile=json.dumps(request.user_profile),
        #     session_context=json.dumps(request.session_context),
        #     processing_hints=json.dumps(request.processing_hints),
        #     flow_id=request.flow_id or "",
        #     timestamp=int(request.timestamp.timestamp()),
        #     request_id=request.request_id
        # )
        
        return request  # Placeholder
    
    async def _simulate_process_message_response(
        self, 
        request: ProcessMessageRequest
    ) -> Dict[str, Any]:
        """Simulate MCP Engine response (for development)"""
        import asyncio
        import random
        
        # Simulate processing delay
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Generate simulated response based on request
        message_text = request.message_content.get("text", "").lower()
        
        if "order" in message_text:
            response_text = "I'd be happy to help you with your order. Could you please provide your order number?"
            intent = "order_inquiry"
        elif "help" in message_text or "support" in message_text:
            response_text = "I'm here to help! What can I assist you with today?"
            intent = "general_support"
        else:
            response_text = "Thank you for your message. How can I help you?"
            intent = "general"
        
        return {
            "success": True,
            "message_id": request.message_id,
            "conversation_id": request.conversation_id,
            "response_content": {
                "type": "text",
                "text": response_text,
                "language": "en"
            },
            "response_type": "text",
            "confidence_score": random.uniform(0.8, 0.95),
            "updated_context": {
                "current_intent": intent,
                "entities": {},
                "conversation_stage": "processing"
            },
            "context_changes": ["current_intent", "conversation_stage"],
            "next_expected_inputs": ["order_number"] if intent == "order_inquiry" else [],
            "suggested_actions": [],
            "processing_time_ms": random.randint(150, 400),
            "model_used": "gpt-4-turbo",
            "model_provider": "openai",
            "cost_cents": round(random.uniform(0.5, 2.0), 2)
        }
    
    def _parse_process_message_response(self, grpc_response: Any) -> ProcessMessageResponse:
        """Parse gRPC response to internal response format"""
        # In production, this would parse the actual protobuf response
        return ProcessMessageResponse(
            success=grpc_response.get("success", False),
            message_id=grpc_response.get("message_id", ""),
            conversation_id=grpc_response.get("conversation_id", ""),
            response_content=grpc_response.get("response_content", {}),
            response_type=grpc_response.get("response_type", "text"),
            confidence_score=grpc_response.get("confidence_score", 0.0),
            updated_context=grpc_response.get("updated_context", {}),
            context_changes=grpc_response.get("context_changes", []),
            next_expected_inputs=grpc_response.get("next_expected_inputs", []),
            suggested_actions=grpc_response.get("suggested_actions", []),
            processing_time_ms=grpc_response.get("processing_time_ms", 0),
            model_used=grpc_response.get("model_used", "unknown"),
            model_provider=grpc_response.get("model_provider", "unknown"),
            cost_cents=grpc_response.get("cost_cents"),
            error_code=grpc_response.get("error_code"),
            error_message=grpc_response.get("error_message")
        )

# Global client instance management
_mcp_client: Optional[MCPEngineClient] = None

async def get_mcp_client() -> MCPEngineClient:
    """Get global MCP Engine client instance"""
    global _mcp_client
    
    if _mcp_client is None:
        _mcp_client = MCPEngineClient()
        await _mcp_client.connect()
    
    return _mcp_client

async def process_message_with_mcp(
    tenant_id: str,
    conversation_id: str,
    message_id: str,
    user_id: str,
    channel: str,
    message_content: Dict[str, Any],
    conversation_context: Dict[str, Any]
) -> ProcessMessageResponse:
    """Convenience function to process message with MCP Engine"""
    client = await get_mcp_client()
    
    request = ProcessMessageRequest(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        message_id=message_id,
        user_id=user_id,
        channel=channel,
        message_content=message_content,
        message_type=message_content.get("type", "text"),
        conversation_context=conversation_context,
        user_profile={},
        session_context={},
        processing_hints={},
        timestamp=datetime.utcnow(),
        request_id=f"req_{message_id}"
    )
    
    return await client.process_message(request)
```

---

## 📋 Step 16: Webhook System & External Platform Integration

### What Will Be Implemented
- Webhook receiver endpoints for external platforms
- Webhook signature verification and security
- Platform-specific webhook processors
- Outbound webhook delivery system

### Folders and Files Created

```
src/webhooks/
├── __init__.py
├── base_webhook.py
├── processors/
│   ├── __init__.py
│   ├── whatsapp_webhook.py
│   ├── slack_webhook.py
│   ├── messenger_webhook.py
│   └── teams_webhook.py
├── delivery/
│   ├── __init__.py
│   ├── webhook_sender.py
│   └── delivery_queue.py
├── security/
│   ├── __init__.py
│   ├── signature_validator.py
│   └── rate_limiter.py
└── exceptions.py

src/api/v2/
└── webhook_routes.py
```

### File Documentation

#### `src/webhooks/base_webhook.py`
**Purpose:** Base webhook processing framework with security and validation  
**Usage:** Foundation for all platform-specific webhook implementations

**Classes:**

1. **WebhookEvent(BaseModel)**
   - **Purpose:** Standardized webhook event structure
   - **Fields:** Platform, event type, payload, metadata
   - **Usage:** Consistent webhook data format

2. **BaseWebhookProcessor(ABC)**
   - **Purpose:** Abstract base class for webhook processors
   - **Methods:**
     - **process_webhook(payload: dict, headers: dict) -> WebhookEvent**: Process incoming webhook
     - **verify_signature(payload: bytes, signature: str) -> bool**: Verify webhook signature
     - **extract_events(payload: dict) -> List[WebhookEvent]**: Extract events from payload

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import hashlib
import hmac
import json
import structlog

from src.webhooks.exceptions import (
    WebhookError, SignatureVerificationError, 
    UnsupportedWebhookError, WebhookProcessingError
)

logger = structlog.get_logger()

class WebhookEvent(BaseModel):
    """Standardized webhook event structure"""
    
    # Event identification
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of webhook event")
    platform: str = Field(..., description="Source platform (whatsapp, slack, etc.)")
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    platform_timestamp: Optional[datetime] = None
    
    # Content
    payload: Dict[str, Any] = Field(..., description="Raw webhook payload")
    processed_data: Dict[str, Any] = Field(default_factory=dict, description="Processed event data")
    
    # Message information (if applicable)
    message_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    channel_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing metadata
    processing_status: str = Field(default="pending", description="Processing status")
    processing_errors: List[str] = Field(default_factory=list)
    retry_count: int = Field(default=0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WebhookProcessingResult(BaseModel):
    """Result of webhook processing"""
    success: bool
    events_extracted: int
    events_processed: int
    events_failed: int
    processing_time_ms: int
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class BaseWebhookProcessor(ABC):
    """Abstract base class for webhook processors"""
    
    def __init__(self, platform_name: str, webhook_secret: Optional[str] = None):
        self.platform_name = platform_name
        self.webhook_secret = webhook_secret
        self.logger = structlog.get_logger(f"webhook.{platform_name}")
        
        # Processing statistics
        self.stats = {
            "webhooks_received": 0,
            "webhooks_processed": 0,
            "webhooks_failed": 0,
            "signature_verifications": 0,
            "signature_failures": 0
        }
    
    @property
    @abstractmethod
    def supported_event_types(self) -> List[str]:
        """Return list of supported event types for this platform"""
        pass
    
    @abstractmethod
    async def verify_signature(
        self, 
        payload: bytes, 
        signature: str, 
        headers: Dict[str, str]
    ) -> bool:
        """
        Verify webhook signature
        
        Args:
            payload: Raw webhook payload
            signature: Signature from headers
            headers: All request headers
            
        Returns:
            True if signature is valid
        """
        pass
    
    @abstractmethod
    async def extract_events(self, payload: Dict[str, Any]) -> List[WebhookEvent]:
        """
        Extract events from webhook payload
        
        Args:
            payload: Webhook payload
            
        Returns:
            List of extracted webhook events
        """
        pass
    
    async def process_webhook(
        self,
        payload: Union[Dict[str, Any], bytes],
        headers: Dict[str, str],
        verify_signature: bool = True
    ) -> WebhookProcessingResult:
        """
        Process incoming webhook
        
        Args:
            payload: Webhook payload (dict or raw bytes)
            headers: Request headers
            verify_signature: Whether to verify webhook signature
            
        Returns:
            WebhookProcessingResult with processing details
        """
        start_time = datetime.utcnow()
        
        try:
            self.stats["webhooks_received"] += 1
            
            # Convert payload to bytes and dict
            if isinstance(payload, dict):
                payload_bytes = json.dumps(payload).encode('utf-8')
                payload_dict = payload
            else:
                payload_bytes = payload
                payload_dict = json.loads(payload.decode('utf-8'))
            
            # Verify signature if required
            if verify_signature and self.webhook_secret:
                signature = self._extract_signature(headers)
                if signature:
                    self.stats["signature_verifications"] += 1
                    if not await self.verify_signature(payload_bytes, signature, headers):
                        self.stats["signature_failures"] += 1
                        raise SignatureVerificationError(
                            f"Invalid signature for {self.platform_name} webhook"
                        )
                else:
                    self.logger.warning(
                        "No signature found in webhook headers",
                        platform=self.platform_name
                    )
            
            # Extract events from payload
            events = await self.extract_events(payload_dict)
            
            # Process each event
            processed_events = 0
            failed_events = 0
            errors = []
            
            for event in events:
                try:
                    await self._process_single_event(event)
                    processed_events += 1
                except Exception as e:
                    failed_events += 1
                    errors.append(f"Event {event.event_id}: {str(e)}")
                    self.logger.error(
                        "Failed to process webhook event",
                        platform=self.platform_name,
                        event_id=event.event_id,
                        event_type=event.event_type,
                        error=str(e)
                    )
            
            # Calculate processing time
            processing_time = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            # Update statistics
            if failed_events == 0:
                self.stats["webhooks_processed"] += 1
            else:
                self.stats["webhooks_failed"] += 1
            
            result = WebhookProcessingResult(
                success=failed_events == 0,
                events_extracted=len(events),
                events_processed=processed_events,
                events_failed=failed_events,
                processing_time_ms=processing_time,
                errors=errors
            )
            
            self.logger.info(
                "Webhook processing completed",
                platform=self.platform_name,
                events_extracted=len(events),
                events_processed=processed_events,
                events_failed=failed_events,
                processing_time_ms=processing_time
            )
            
            return result
            
        except SignatureVerificationError:
            raise
        except Exception as e:
            self.stats["webhooks_failed"] += 1
            processing_time = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            self.logger.error(
                "Webhook processing failed",
                platform=self.platform_name,
                error=str(e),
                processing_time_ms=processing_time
            )
            
            return WebhookProcessingResult(
                success=False,
                events_extracted=0,
                events_processed=0,
                events_failed=1,
                processing_time_ms=processing_time,
                errors=[str(e)]
            )
    
    async def _process_single_event(self, event: WebhookEvent) -> None:
        """Process a single webhook event"""
        try:
            # Validate event type
            if event.event_type not in self.supported_event_types:
                raise UnsupportedWebhookError(
                    f"Unsupported event type: {event.event_type}"
                )
            
            # Process based on event type
            if event.event_type == "message":
                await self._process_message_event(event)
            elif event.event_type == "delivery_status":
                await self._process_delivery_event(event)
            elif event.event_type == "read_receipt":
                await self._process_read_receipt_event(event)
            else:
                await self._process_custom_event(event)
            
            event.processing_status = "completed"
            
        except Exception as e:
            event.processing_status = "failed"
            event.processing_errors.append(str(e))
            raise
    
    async def _process_message_event(self, event: WebhookEvent) -> None:
        """Process incoming message event (override in subclasses)"""
        self.logger.debug(
            "Processing message event",
            platform=self.platform_name,
            event_id=event.event_id
        )
    
    async def _process_delivery_event(self, event: WebhookEvent) -> None:
        """Process message delivery event (override in subclasses)"""
        self.logger.debug(
            "Processing delivery event",
            platform=self.platform_name,
            event_id=event.event_id
        )
    
    async def _process_read_receipt_event(self, event: WebhookEvent) -> None:
        """Process read receipt event (override in subclasses)"""
        self.logger.debug(
            "Processing read receipt event",
            platform=self.platform_name,
            event_id=event.event_id
        )
    
    async def _process_custom_event(self, event: WebhookEvent) -> None:
        """Process custom/platform-specific event (override in subclasses)"""
        self.logger.debug(
            "Processing custom event",
            platform=self.platform_name,
            event_id=event.event_id,
            event_type=event.event_type
        )
    
    def _extract_signature(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract signature from headers (override in subclasses)"""
        # Common signature header names
        signature_headers = [
            'x-hub-signature-256',
            'x-signature',
            'signature',
            'x-webhook-signature'
        ]
        
        for header in signature_headers:
            if header in headers:
                return headers[header]
            # Also check case-insensitive
            for key, value in headers.items():
                if key.lower() == header.lower():
                    return value
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "platform": self.platform_name,
            "statistics": self.stats.copy(),
            "success_rate": (
                self.stats["webhooks_processed"] / 
                max(1, self.stats["webhooks_received"])
            ),
            "signature_success_rate": (
                (self.stats["signature_verifications"] - self.stats["signature_failures"]) /
                max(1, self.stats["signature_verifications"])
            ) if self.stats["signature_verifications"] > 0 else 1.0
        }
```

#### `src/webhooks/processors/whatsapp_webhook.py`
**Purpose:** WhatsApp Business API webhook processor implementation  
**Usage:** Handle WhatsApp webhook events and convert to internal message format

**Classes:**

1. **WhatsAppWebhookProcessor(BaseWebhookProcessor)**
   - **Purpose:** Process WhatsApp Business API webhooks
   - **Methods:**
     - **verify_signature(payload: bytes, signature: str, headers: dict) -> bool**: Verify WhatsApp signature
     - **extract_events(payload: dict) -> List[WebhookEvent]**: Extract WhatsApp events
     - **_process_message_event(event: WebhookEvent) -> None**: Handle message events

```python
import hmac
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from src.webhooks.base_webhook import BaseWebhookProcessor, WebhookEvent
from src.webhooks.exceptions import WebhookProcessingError
from src.events.event_schemas import MessageReceivedEvent, WebhookReceivedEvent
from src.events.event_manager import publish_event

class WhatsAppWebhookProcessor(BaseWebhookProcessor):
    """WhatsApp Business API webhook processor"""
    
    def __init__(self, webhook_secret: Optional[str] = None):
        super().__init__("whatsapp", webhook_secret)
    
    @property
    def supported_event_types(self) -> List[str]:
        return [
            "message",
            "delivery_status", 
            "read_receipt",
            "status",
            "errors"
        ]
    
    async def verify_signature(
        self, 
        payload: bytes, 
        signature: str, 
        headers: Dict[str, str]
    ) -> bool:
        """
        Verify WhatsApp webhook signature
        
        WhatsApp uses HMAC-SHA256 with the webhook secret
        """
        try:
            if not self.webhook_secret:
                self.logger.warning("No webhook secret configured for WhatsApp")
                return False
            
            # WhatsApp signature format: "sha256=<hash>"
            if not signature.startswith('sha256='):
                return False
            
            expected_signature = signature[7:]  # Remove 'sha256=' prefix
            
            # Calculate expected signature
            calculated_signature = hmac.new(
                self.webhook_secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures securely
            return hmac.compare_digest(expected_signature, calculated_signature)
            
        except Exception as e:
            self.logger.error(
                "WhatsApp signature verification failed",
                error=str(e)
            )
            return False
    
    async def extract_events(self, payload: Dict[str, Any]) -> List[WebhookEvent]:
        """
        Extract events from WhatsApp webhook payload
        
        WhatsApp webhook structure:
        {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "business_account_id",
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {...},
                                "contacts": [...],
                                "messages": [...],
                                "statuses": [...]
                            },
                            "field": "messages"
                        }
                    ]
                }
            ]
        }
        """
        events = []
        
        try:
            # Validate basic structure
            if payload.get("object") != "whatsapp_business_account":
                raise WebhookProcessingError("Invalid WhatsApp webhook object type")
            
            entries = payload.get("entry", [])
            
            for entry in entries:
                business_account_id = entry.get("id")
                changes = entry.get("changes", [])
                
                for change in changes:
                    if change.get("field") == "messages":
                        value = change.get("value", {})
                        
                        # Process messages
                        messages = value.get("messages", [])
                        for message in messages:
                            event = await self._create_message_event(
                                message, value, business_account_id
                            )
                            events.append(event)
                        
                        # Process status updates
                        statuses = value.get("statuses", [])
                        for status in statuses:
                            event = await self._create_status_event(
                                status, value, business_account_id
                            )
                            events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(
                "Failed to extract WhatsApp events",
                error=str(e),
                payload_keys=list(payload.keys())
            )
            raise WebhookProcessingError(f"Failed to extract WhatsApp events: {e}")
    
    async def _create_message_event(
        self,
        message: Dict[str, Any],
        value: Dict[str, Any],
        business_account_id: str
    ) -> WebhookEvent:
        """Create webhook event from WhatsApp message"""
        try:
            # Extract message information
            message_id = message.get("id")
            from_number = message.get("from")
            timestamp_str = message.get("timestamp")
            message_type = message.get("type", "unknown")
            
            # Convert timestamp
            platform_timestamp = None
            if timestamp_str:
                platform_timestamp = datetime.fromtimestamp(int(timestamp_str))
            
            # Extract message content based on type
            content = {}
            if message_type == "text":
                content = {
                    "type": "text",
                    "text": message.get("text", {}).get("body", "")
                }
            elif message_type in ["image", "video", "audio", "document"]:
                media_data = message.get(message_type, {})
                content = {
                    "type": message_type,
                    "media": {
                        "id": media_data.get("id"),
                        "mime_type": media_data.get("mime_type"),
                        "sha256": media_data.get("sha256"),
                        "caption": media_data.get("caption", "")
                    }
                }
            elif message_type == "location":
                location_data = message.get("location", {})
                content = {
                    "type": "location",
                    "location": {
                        "latitude": location_data.get("latitude"),
                        "longitude": location_data.get("longitude"),
                        "name": location_data.get("name"),
                        "address": location_data.get("address")
                    }
                }
            
            # Create webhook event
            event = WebhookEvent(
                event_id=str(uuid4()),
                event_type="message",
                platform="whatsapp",
                timestamp=datetime.utcnow(),
                platform_timestamp=platform_timestamp,
                payload=message,
                processed_data=content,
                message_id=message_id,
                user_id=from_number,
                channel_info={
                    "business_account_id": business_account_id,
                    "phone_number_id": value.get("metadata", {}).get("phone_number_id"),
                    "display_phone_number": value.get("metadata", {}).get("display_phone_number")
                }
            )
            
            return event
            
        except Exception as e:
            self.logger.error(
                "Failed to create WhatsApp message event",
                message_id=message.get("id"),
                error=str(e)
            )
            raise
    
    async def _create_status_event(
        self,
        status: Dict[str, Any],
        value: Dict[str, Any],
        business_account_id: str
    ) -> WebhookEvent:
        """Create webhook event from WhatsApp status update"""
        try:
            message_id = status.get("id")
            recipient_id = status.get("recipient_id")
            status_type = status.get("status")  # sent, delivered, read, failed
            timestamp_str = status.get("timestamp")
            
            # Convert timestamp
            platform_timestamp = None
            if timestamp_str:
                platform_timestamp = datetime.fromtimestamp(int(timestamp_str))
            
            event = WebhookEvent(
                event_id=str(uuid4()),
                event_type="delivery_status",
                platform="whatsapp",
                timestamp=datetime.utcnow(),
                platform_timestamp=platform_timestamp,
                payload=status,
                processed_data={
                    "status": status_type,
                    "recipient_id": recipient_id,
                    "errors": status.get("errors", [])
                },
                message_id=message_id,
                user_id=recipient_id,
                channel_info={
                    "business_account_id": business_account_id,
                    "phone_number_id": value.get("metadata", {}).get("phone_number_id")
                }
            )
            
            return event
            
        except Exception as e:
            self.logger.error(
                "Failed to create WhatsApp status event",
                status_id=status.get("id"),
                error=str(e)
            )
            raise
    
    async def _process_message_event(self, event: WebhookEvent) -> None:
        """Process incoming WhatsApp message event"""
        try:
            # Publish webhook received event for analytics
            webhook_event = WebhookReceivedEvent(
                webhook_source="whatsapp",
                webhook_type="message",
                webhook_id=event.event_id,
                raw_payload=event.payload,
                processed_payload=event.processed_data,
                validation_status="valid"
            )
            await publish_event(webhook_event)
            
            # Convert to internal message format and publish
            message_event = MessageReceivedEvent(
                message_id=event.message_id,
                conversation_id=None,  # Will be determined by chat service
                user_id=event.user_id,
                channel="whatsapp",
                message_type=event.processed_data.get("type", "text"),
                content=event.processed_data,
                processing_result={}
            )
            
            # Add metadata for tenant resolution
            message_event.metadata.channel_metadata = event.channel_info
            
            await publish_event(message_event)
            
            self.logger.info(
                "WhatsApp message event processed",
                event_id=event.event_id,
                message_id=event.message_id,
                user_id=event.user_id
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to process WhatsApp message event",
                event_id=event.event_id,
                error=str(e)
            )
            raise
    
    async def _process_delivery_event(self, event: WebhookEvent) -> None:
        """Process WhatsApp delivery status event"""
        try:
            # Update message delivery status in database
            # This would typically involve updating the message record
            
            self.logger.info(
                "WhatsApp delivery event processed",
                event_id=event.event_id,
                message_id=event.message_id,
                status=event.processed_data.get("status")
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to process WhatsApp delivery event",
                event_id=event.event_id,
                error=str(e)
            )
            raise

# Global processor instance
_whatsapp_processor: Optional[WhatsAppWebhookProcessor] = None

def get_whatsapp_webhook_processor() -> WhatsAppWebhookProcessor:
    """Get WhatsApp webhook processor instance"""
    global _whatsapp_processor
    
    if _whatsapp_processor is None:
        from src.config.settings import get_settings
        settings = get_settings()
        webhook_secret = settings.WHATSAPP_WEBHOOK_SECRET if hasattr(settings, 'WHATSAPP_WEBHOOK_SECRET') else None
        _whatsapp_processor = WhatsAppWebhookProcessor(webhook_secret)
    
    return _whatsapp_processor
```

---

## 🔧 Technologies Used
- **gRPC**: High-performance RPC framework
- **Protocol Buffers**: Efficient serialization
- **asyncio**: Asynchronous I/O operations
- **HMAC**: Webhook signature verification
- **JSON**: Webhook payload processing

---

## ⚠️ Key Considerations

### Security
- Webhook signature verification for all platforms
- Rate limiting for webhook endpoints
- Input validation and sanitization
- Secure credential management

### Reliability
- Connection retry logic with exponential backoff
- Circuit breaker patterns for external services
- Dead letter queues for failed webhooks
- Health monitoring and alerting

### Performance
- Connection pooling for gRPC clients
- Efficient webhook processing
- Batch processing where applicable
- Async operations throughout

### Scalability
- Horizontal scaling of webhook processors
- Load balancing across instances
- Efficient resource utilization
- Configurable timeout and retry settings

---

## 🎯 Success Criteria
- [ ] gRPC clients connect and communicate with external services
- [ ] Webhook endpoints receive and process platform events correctly
- [ ] Signature verification works for all supported platforms
- [ ] Connection management handles failures gracefully
- [ ] Performance meets latency requirements
- [ ] Security measures are effective

---

## 📋 Next Phase Preview
Phase 9 will focus on testing implementation including unit tests, integration tests, and end-to-end testing scenarios to ensure system reliability and correctness.

