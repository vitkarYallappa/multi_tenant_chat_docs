# Phase 5: Service Layer & Business Logic
**Duration:** Week 7-8  
**Steps:** 9-10 of 18

---

## 🎯 Objectives
- Implement core business logic services
- Create message processing orchestration
- Establish conversation management services
- Build session and delivery management

---

## 📋 Step 9: Core Service Layer Implementation

### What Will Be Implemented
- Message service for orchestrating message processing
- Conversation service for managing conversation lifecycle
- Channel service for channel abstraction
- Session service for user session management

### Folders and Files Created

```
src/services/
├── __init__.py
├── base_service.py
├── message_service.py
├── conversation_service.py
├── channel_service.py
├── session_service.py
├── delivery_service.py
├── audit_service.py
└── exceptions.py
```

### File Documentation

#### `src/services/base_service.py`
**Purpose:** Abstract base service class with common service patterns and utilities  
**Usage:** Foundation for all service implementations with dependency injection and logging

**Classes:**

1. **BaseService(ABC)**
   - **Purpose:** Abstract base class for all services
   - **Methods:**
     - **validate_tenant_access(tenant_id: str, user_context: dict) -> bool**: Validate tenant access
     - **log_operation(operation: str, **kwargs) -> None**: Log service operation
     - **handle_service_error(error: Exception, operation: str) -> ServiceError**: Handle and wrap errors

```python
from abc import ABC
from typing import Dict, Any, Optional
from datetime import datetime
import structlog

from src.models.types import TenantId, UserId
from src.services.exceptions import ServiceError, UnauthorizedError, ValidationError

class BaseService(ABC):
    """Abstract base class for all services"""
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.service_name = self.__class__.__name__
    
    async def validate_tenant_access(
        self, 
        tenant_id: TenantId, 
        user_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate that user has access to tenant resources
        
        Args:
            tenant_id: Tenant ID to validate access for
            user_context: User context with permissions
            
        Returns:
            True if access is allowed
            
        Raises:
            UnauthorizedError: If access is denied
        """
        try:
            if not tenant_id:
                raise ValidationError("Tenant ID is required")
            
            if user_context:
                user_tenant_id = user_context.get("tenant_id")
                if user_tenant_id and user_tenant_id != tenant_id:
                    self.logger.warning(
                        "Tenant access denied",
                        requested_tenant=tenant_id,
                        user_tenant=user_tenant_id,
                        user_id=user_context.get("user_id")
                    )
                    raise UnauthorizedError(f"Access denied to tenant {tenant_id}")
            
            return True
            
        except (UnauthorizedError, ValidationError):
            raise
        except Exception as e:
            self.logger.error(
                "Tenant access validation failed",
                tenant_id=tenant_id,
                error=str(e)
            )
            raise ServiceError(f"Failed to validate tenant access: {e}")
    
    def log_operation(
        self, 
        operation: str,
        tenant_id: Optional[TenantId] = None,
        user_id: Optional[UserId] = None,
        **kwargs
    ) -> None:
        """Log service operation with standard fields"""
        log_data = {
            "service": self.service_name,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        if tenant_id:
            log_data["tenant_id"] = tenant_id
        if user_id:
            log_data["user_id"] = user_id
            
        self.logger.info("Service operation", **log_data)
    
    def handle_service_error(
        self, 
        error: Exception, 
        operation: str,
        tenant_id: Optional[TenantId] = None,
        **context
    ) -> ServiceError:
        """
        Handle and wrap service errors with context
        
        Args:
            error: Original exception
            operation: Operation that failed
            tenant_id: Optional tenant ID
            **context: Additional context
            
        Returns:
            ServiceError with wrapped exception
        """
        error_context = {
            "service": self.service_name,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
        
        if tenant_id:
            error_context["tenant_id"] = tenant_id
        
        self.logger.error("Service operation failed", **error_context)
        
        # Re-raise specific errors
        if isinstance(error, (ValidationError, UnauthorizedError)):
            return error
        
        # Wrap generic errors
        return ServiceError(
            f"{operation} failed: {str(error)}",
            original_error=error
        )
    
    async def _validate_required_fields(
        self, 
        data: Dict[str, Any], 
        required_fields: list
    ) -> None:
        """Validate that required fields are present"""
        missing_fields = [
            field for field in required_fields 
            if field not in data or data[field] is None
        ]
        
        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing_fields)}"
            )
    
    def _sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from log entries"""
        sensitive_fields = [
            "password", "token", "secret", "key", "credential",
            "authorization", "api_key", "private_key"
        ]
        
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_log_data(value)
            else:
                sanitized[key] = value
        
        return sanitized
```

#### `src/services/message_service.py`
**Purpose:** Core message processing service that orchestrates the entire message handling pipeline  
**Usage:** Main service for processing incoming messages and generating responses

**Classes:**

1. **MessageService(BaseService)**
   - **Purpose:** Orchestrate message processing pipeline
   - **Methods:**
     - **process_message(request: SendMessageRequest, user_context: dict) -> MessageResponse**: Process incoming message
     - **generate_response(message_data: dict, conversation_context: dict) -> MessageContent**: Generate bot response
     - **deliver_message(recipient: str, content: MessageContent, channel: str) -> ChannelResponse**: Deliver message via channel
     - **handle_webhook(channel: str, webhook_data: dict) -> dict**: Process incoming webhooks

```python
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4

from src.services.base_service import BaseService
from src.services.exceptions import ServiceError, ValidationError
from src.models.schemas.request_schemas import SendMessageRequest
from src.models.schemas.response_schemas import MessageResponse
from src.models.types import MessageContent, ChannelType, ConversationStatus
from src.models.mongo.conversation_model import ConversationDocument
from src.models.mongo.message_model import MessageDocument
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.repositories.session_repository import SessionRepository
from src.core.channels.channel_factory import ChannelFactory
from src.core.processors.processor_factory import ProcessorFactory
from src.core.processors.base_processor import ProcessingContext

class MessageService(BaseService):
    """Service for message processing and orchestration"""
    
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        session_repo: SessionRepository,
        channel_factory: ChannelFactory,
        processor_factory: ProcessorFactory
    ):
        super().__init__()
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.session_repo = session_repo
        self.channel_factory = channel_factory
        self.processor_factory = processor_factory
    
    async def process_message(
        self, 
        request: SendMessageRequest,
        user_context: Dict[str, Any]
    ) -> MessageResponse:
        """
        Process incoming message and generate response
        
        Args:
            request: Message request data
            user_context: User authentication context
            
        Returns:
            MessageResponse with bot response
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate tenant access
            await self.validate_tenant_access(
                request.tenant_id, 
                user_context
            )
            
            self.log_operation(
                "process_message_start",
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                channel=request.channel,
                message_type=request.content.type
            )
            
            # Get or create conversation
            conversation = await self._get_or_create_conversation(request)
            
            # Process incoming message
            processed_message = await self._process_incoming_message(
                request, conversation, user_context
            )
            
            # Store incoming message
            await self.message_repo.create(processed_message)
            
            # Generate bot response
            response_content = await self._generate_bot_response(
                request, conversation, processed_message
            )
            
            # Create and store response message
            response_message = await self._create_response_message(
                request, conversation, response_content
            )
            await self.message_repo.create(response_message)
            
            # Deliver response via channel
            delivery_result = await self._deliver_response(
                request, response_content
            )
            
            # Update conversation metrics
            await self._update_conversation_metrics(
                conversation, processed_message, response_message
            )
            
            # Calculate processing time
            processing_time = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            self.log_operation(
                "process_message_complete",
                tenant_id=request.tenant_id,
                conversation_id=conversation.conversation_id,
                processing_time_ms=processing_time,
                delivery_success=delivery_result.success
            )
            
            return MessageResponse(
                message_id=processed_message.message_id,
                conversation_id=conversation.conversation_id,
                response=response_content,
                conversation_state=conversation.context.dict(),
                processing_metadata={
                    "processing_time_ms": processing_time,
                    "delivery_status": delivery_result.delivery_status,
                    "channel_response": delivery_result.dict()
                }
            )
            
        except Exception as e:
            error = self.handle_service_error(
                e, "process_message",
                tenant_id=getattr(request, 'tenant_id', None)
            )
            raise error
    
    async def _get_or_create_conversation(
        self, 
        request: SendMessageRequest
    ) -> ConversationDocument:
        """Get existing or create new conversation"""
        try:
            # Try to get existing conversation
            if request.conversation_id:
                conversation = await self.conversation_repo.get_by_id(
                    request.conversation_id
                )
                if conversation:
                    conversation.update_last_activity()
                    return conversation
            
            # Create new conversation
            conversation = ConversationDocument(
                conversation_id=str(uuid4()),
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                session_id=request.session_id,
                channel=request.channel,
                channel_metadata=request.channel_metadata.dict() if request.channel_metadata else {},
                status=ConversationStatus.ACTIVE
            )
            
            return await self.conversation_repo.create(conversation)
            
        except Exception as e:
            raise ServiceError(f"Failed to get or create conversation: {e}")
    
    async def _process_incoming_message(
        self,
        request: SendMessageRequest,
        conversation: ConversationDocument,
        user_context: Dict[str, Any]
    ) -> MessageDocument:
        """Process incoming message through processing pipeline"""
        try:
            # Create processing context
            context = ProcessingContext(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                conversation_id=conversation.conversation_id,
                session_id=request.session_id,
                channel=request.channel.value,
                channel_metadata=request.channel_metadata.dict() if request.channel_metadata else {},
                user_profile=user_context,
                conversation_context=conversation.context.dict(),
                processing_hints=request.processing_hints.dict() if request.processing_hints else {},
                request_id=str(uuid4())
            )
            
            # Get appropriate processor
            processor = await self.processor_factory.get_processor(
                request.content.type
            )
            
            # Process content
            processing_result = await processor.process(request.content, context)
            
            # Create message document
            message_doc = MessageDocument(
                message_id=request.message_id,
                conversation_id=conversation.conversation_id,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                sequence_number=await self._get_next_sequence_number(
                    conversation.conversation_id
                ),
                direction="inbound",
                timestamp=request.timestamp,
                channel=request.channel.value,
                message_type=request.content.type.value,
                content=processing_result.processed_content.dict() if processing_result.processed_content else request.content.dict(),
                ai_analysis={
                    "entities": processing_result.entities,
                    "detected_language": processing_result.detected_language,
                    "language_confidence": processing_result.language_confidence,
                    "content_categories": processing_result.content_categories,
                    "quality_score": processing_result.quality_score,
                    "safety_flags": processing_result.safety_flags
                },
                processing={
                    "pipeline_version": "1.0",
                    "processing_time_ms": processing_result.processing_time_ms,
                    "processor_version": processing_result.processor_version
                }
            )
            
            return message_doc
            
        except Exception as e:
            raise ServiceError(f"Failed to process incoming message: {e}")
    
    async def _generate_bot_response(
        self,
        request: SendMessageRequest,
        conversation: ConversationDocument,
        processed_message: MessageDocument
    ) -> MessageContent:
        """Generate bot response content"""
        try:
            # This is a simplified response generation
            # In production, this would integrate with MCP Engine
            
            # Extract intent from processed message
            intent = processed_message.ai_analysis.get("entities", {}).get("intent")
            
            # Generate appropriate response based on intent and context
            if intent == "greeting":
                response_text = "Hello! How can I help you today?"
            elif intent == "order_inquiry":
                response_text = "I'd be happy to help you with your order. Could you please provide your order number?"
            elif intent == "support":
                response_text = "I'm here to help! Please describe the issue you're experiencing."
            else:
                response_text = "Thank you for your message. How can I assist you?"
            
            # Create response content
            response_content = MessageContent(
                type="text",
                text=response_text,
                language=processed_message.ai_analysis.get("detected_language", "en")
            )
            
            return response_content
            
        except Exception as e:
            raise ServiceError(f"Failed to generate bot response: {e}")
    
    async def _create_response_message(
        self,
        request: SendMessageRequest,
        conversation: ConversationDocument,
        response_content: MessageContent
    ) -> MessageDocument:
        """Create response message document"""
        try:
            return MessageDocument(
                message_id=str(uuid4()),
                conversation_id=conversation.conversation_id,
                tenant_id=request.tenant_id,
                user_id="bot",  # Bot user
                sequence_number=await self._get_next_sequence_number(
                    conversation.conversation_id
                ),
                direction="outbound",
                timestamp=datetime.utcnow(),
                channel=request.channel.value,
                message_type=response_content.type.value,
                content=response_content.dict(),
                generation_metadata={
                    "model_provider": "internal",
                    "model_name": "rule_based",
                    "generation_time_ms": 50,
                    "template_used": "default_response"
                }
            )
            
        except Exception as e:
            raise ServiceError(f"Failed to create response message: {e}")
    
    async def _deliver_response(
        self,
        request: SendMessageRequest,
        response_content: MessageContent
    ) -> Any:  # ChannelResponse
        """Deliver response via appropriate channel"""
        try:
            # Get channel implementation
            channel = await self.channel_factory.get_channel(request.channel)
            
            # Determine recipient from original request
            recipient = request.user_id  # Simplified - should be channel-specific
            
            # Deliver message
            delivery_result = await channel.send_message(
                recipient=recipient,
                content=response_content,
                metadata=request.channel_metadata.dict() if request.channel_metadata else {}
            )
            
            return delivery_result
            
        except Exception as e:
            raise ServiceError(f"Failed to deliver response: {e}")
    
    async def _update_conversation_metrics(
        self,
        conversation: ConversationDocument,
        incoming_message: MessageDocument,
        response_message: MessageDocument
    ) -> None:
        """Update conversation metrics"""
        try:
            conversation.metrics.increment_message_count(True)  # User message
            conversation.metrics.increment_message_count(False)  # Bot message
            
            # Update conversation context if needed
            if incoming_message.ai_analysis.get("entities"):
                conversation.context.entities.update(
                    incoming_message.ai_analysis["entities"]
                )
            
            # Save updated conversation
            await self.conversation_repo.update(conversation)
            
        except Exception as e:
            self.logger.error(
                "Failed to update conversation metrics",
                conversation_id=conversation.conversation_id,
                error=str(e)
            )
            # Don't raise - this is not critical
    
    async def _get_next_sequence_number(self, conversation_id: str) -> int:
        """Get next sequence number for message in conversation"""
        try:
            # Get last message in conversation
            last_message = await self.message_repo.get_last_message_in_conversation(
                conversation_id
            )
            
            if last_message:
                return last_message.sequence_number + 1
            else:
                return 1
                
        except Exception as e:
            self.logger.error(
                "Failed to get next sequence number",
                conversation_id=conversation_id,
                error=str(e)
            )
            return 1  # Default to 1 if we can't determine
    
    async def handle_webhook(
        self, 
        channel_type: ChannelType,
        webhook_data: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process incoming webhook from channel"""
        try:
            self.log_operation(
                "handle_webhook",
                channel=channel_type,
                tenant_id=tenant_id
            )
            
            # Get channel implementation
            channel = await self.channel_factory.get_channel(channel_type)
            
            # Process webhook
            result = await channel.process_webhook(webhook_data)
            
            # Handle any events from webhook
            if "events" in result:
                for event in result["events"]:
                    await self._handle_webhook_event(event, channel_type, tenant_id)
            
            return result
            
        except Exception as e:
            error = self.handle_service_error(
                e, "handle_webhook",
                channel=channel_type,
                tenant_id=tenant_id
            )
            raise error
    
    async def _handle_webhook_event(
        self,
        event: Dict[str, Any],
        channel_type: ChannelType,
        tenant_id: Optional[str]
    ) -> None:
        """Handle individual webhook event"""
        try:
            event_type = event.get("type")
            
            if event_type == "message_received":
                # Convert webhook event to message request and process
                # This would involve mapping webhook data to SendMessageRequest
                pass
            elif event_type == "delivery_status":
                # Update message delivery status
                pass
            elif event_type == "read_receipt":
                # Update message read status
                pass
            
        except Exception as e:
            self.logger.error(
                "Failed to handle webhook event",
                event_type=event.get("type"),
                channel=channel_type,
                error=str(e)
            )
```

---

## 📋 Step 10: Supporting Services Implementation

### What Will Be Implemented
- Session service for user session management
- Delivery service for message delivery tracking
- Audit service for compliance and monitoring
- Service orchestration and dependency injection

### Folders and Files Created

```
src/services/
├── session_service.py
├── delivery_service.py
├── audit_service.py
└── service_container.py

src/
└── dependencies.py
```

### File Documentation

#### `src/services/session_service.py`
**Purpose:** Manage user sessions, conversation context, and state persistence  
**Usage:** Handle session lifecycle, context updates, and session-based routing

**Classes:**

1. **SessionService(BaseService)**
   - **Purpose:** Manage user sessions and conversation state
   - **Methods:**
     - **create_session(user_id: str, tenant_id: str, channel: ChannelType) -> SessionData**: Create new session
     - **get_session(tenant_id: str, session_id: str) -> Optional[SessionData]**: Retrieve session
     - **update_session_context(session_id: str, context_updates: dict) -> bool**: Update session context
     - **cleanup_expired_sessions() -> int**: Clean up expired sessions

```python
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import uuid4

from src.services.base_service import BaseService
from src.services.exceptions import ServiceError, ValidationError, NotFoundError
from src.models.redis.session_cache import SessionData
from src.models.types import TenantId, UserId, SessionId, ChannelType
from src.repositories.session_repository import SessionRepository

class SessionService(BaseService):
    """Service for managing user sessions and conversation state"""
    
    def __init__(self, session_repo: SessionRepository):
        super().__init__()
        self.session_repo = session_repo
        self.default_session_duration_hours = 2
        self.max_session_duration_hours = 24
    
    async def create_session(
        self,
        user_id: UserId,
        tenant_id: TenantId,
        channel: ChannelType,
        conversation_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """
        Create a new user session
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            channel: Communication channel
            conversation_id: Optional existing conversation ID
            initial_context: Optional initial session context
            
        Returns:
            Created SessionData
        """
        try:
            await self.validate_tenant_access(tenant_id)
            
            # Generate session ID
            session_id = str(uuid4())
            
            # Create session data
            session_data = SessionData(
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                channel=channel,
                context=initial_context or {},
                expires_at=datetime.utcnow() + timedelta(
                    hours=self.default_session_duration_hours
                )
            )
            
            # Store in repository
            success = await self.session_repo.create_session(session_data)
            
            if not success:
                raise ServiceError("Failed to create session in repository")
            
            self.log_operation(
                "create_session",
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                channel=channel.value
            )
            
            return session_data
            
        except Exception as e:
            error = self.handle_service_error(
                e, "create_session",
                tenant_id=tenant_id,
                user_id=user_id
            )
            raise error
    
    async def get_session(
        self,
        tenant_id: TenantId,
        session_id: SessionId
    ) -> Optional[SessionData]:
        """
        Retrieve session by ID
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            
        Returns:
            SessionData if found, None otherwise
        """
        try:
            await self.validate_tenant_access(tenant_id)
            
            session_data = await self.session_repo.get_session(tenant_id, session_id)
            
            if session_data:
                # Check if session is expired
                if session_data.is_expired():
                    await self.session_repo.delete_session(tenant_id, session_id)
                    return None
                
                self.log_operation(
                    "get_session",
                    tenant_id=tenant_id,
                    session_id=session_id,
                    user_id=session_data.user_id
                )
            
            return session_data
            
        except Exception as e:
            error = self.handle_service_error(
                e, "get_session",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error
    
    async def update_session_context(
        self,
        tenant_id: TenantId,
        session_id: SessionId,
        context_updates: Dict[str, Any]
    ) -> bool:
        """
        Update session context
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            context_updates: Context updates to apply
            
        Returns:
            True if successful
        """
        try:
            await self.validate_tenant_access(tenant_id)
            
            # Get existing session
            session_data = await self.get_session(tenant_id, session_id)
            
            if not session_data:
                raise NotFoundError(f"Session {session_id} not found")
            
            # Update context
            session_data.context.update(context_updates)
            
            # Save updated session
            success = await self.session_repo.update_session(session_data)
            
            if success:
                self.log_operation(
                    "update_session_context",
                    tenant_id=tenant_id,
                    session_id=session_id,
                    updates_count=len(context_updates)
                )
            
            return success
            
        except NotFoundError:
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "update_session_context",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error
    
    async def extend_session(
        self,
        tenant_id: TenantId,
        session_id: SessionId,
        hours: int = None
    ) -> bool:
        """
        Extend session expiration
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            hours: Hours to extend (default: default_session_duration_hours)
            
        Returns:
            True if successful
        """
        try:
            await self.validate_tenant_access(tenant_id)
            
            hours = hours or self.default_session_duration_hours
            
            if hours > self.max_session_duration_hours:
                raise ValidationError(
                    f"Cannot extend session beyond {self.max_session_duration_hours} hours"
                )
            
            success = await self.session_repo.extend_session(
                tenant_id, session_id, hours
            )
            
            if success:
                self.log_operation(
                    "extend_session",
                    tenant_id=tenant_id,
                    session_id=session_id,
                    extended_hours=hours
                )
            
            return success
            
        except ValidationError:
            raise
        except Exception as e:
            error = self.handle_service_error(
                e, "extend_session",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error
    
    async def delete_session(
        self,
        tenant_id: TenantId,
        session_id: SessionId
    ) -> bool:
        """
        Delete session
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        try:
            await self.validate_tenant_access(tenant_id)
            
            success = await self.session_repo.delete_session(tenant_id, session_id)
            
            if success:
                self.log_operation(
                    "delete_session",
                    tenant_id=tenant_id,
                    session_id=session_id
                )
            
            return success
            
        except Exception as e:
            error = self.handle_service_error(
                e, "delete_session",
                tenant_id=tenant_id,
                session_id=session_id
            )
            raise error
    
    async def get_user_sessions(
        self,
        tenant_id: TenantId,
        user_id: UserId
    ) -> List[SessionData]:
        """
        Get all active sessions for a user
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            
        Returns:
            List of active sessions
        """
        try:
            await self.validate_tenant_access(tenant_id)
            
            sessions = await self.session_repo.get_user_sessions(tenant_id, user_id)
            
            # Filter out expired sessions
            active_sessions = []
            for session in sessions:
                if not session.is_expired():
                    active_sessions.append(session)
                else:
                    # Clean up expired session
                    await self.session_repo.delete_session(
                        tenant_id, session.session_id
                    )
            
            self.log_operation(
                "get_user_sessions",
                tenant_id=tenant_id,
                user_id=user_id,
                sessions_count=len(active_sessions)
            )
            
            return active_sessions
            
        except Exception as e:
            error = self.handle_service_error(
                e, "get_user_sessions",
                tenant_id=tenant_id,
                user_id=user_id
            )
            raise error
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions (background task)
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            cleaned_count = await self.session_repo.cleanup_expired_sessions()
            
            self.log_operation(
                "cleanup_expired_sessions",
                cleaned_count=cleaned_count
            )
            
            return cleaned_count
            
        except Exception as e:
            error = self.handle_service_error(e, "cleanup_expired_sessions")
            raise error
```

#### `src/services/exceptions.py`
**Purpose:** Service-specific exception definitions for proper error handling  
**Usage:** Provide specific exceptions for different service error scenarios

**Classes:**

1. **ServiceError(Exception)**
   - **Purpose:** Base exception for all service layer errors
   - **Usage:** Generic service operation failures

2. **ValidationError(ServiceError)**
   - **Purpose:** Input validation failures
   - **Usage:** Invalid request data or parameters

3. **UnauthorizedError(ServiceError)**
   - **Purpose:** Authorization failures
   - **Usage:** Access denied scenarios

4. **NotFoundError(ServiceError)**
   - **Purpose:** Resource not found errors
   - **Usage:** When requested resources don't exist

5. **ConflictError(ServiceError)**
   - **Purpose:** Resource conflict errors
   - **Usage:** Duplicate creation attempts or conflicting operations

```python
"""Service layer exceptions"""

class ServiceError(Exception):
    """Base exception for service layer errors"""
    
    def __init__(self, message: str, original_error: Exception = None, error_code: str = None):
        super().__init__(message)
        self.original_error = original_error
        self.error_code = error_code or "SERVICE_ERROR"
        self.timestamp = None
        
        # Import here to avoid circular imports
        from datetime import datetime
        self.timestamp = datetime.utcnow()

class ValidationError(ServiceError):
    """Exception for input validation failures"""
    
    def __init__(self, message: str, field: str = None, value: str = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field
        self.value = value

class UnauthorizedError(ServiceError):
    """Exception for authorization failures"""
    
    def __init__(self, message: str, user_id: str = None, resource: str = None):
        super().__init__(message, error_code="UNAUTHORIZED")
        self.user_id = user_id
        self.resource = resource

class NotFoundError(ServiceError):
    """Exception for resource not found errors"""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None):
        super().__init__(message, error_code="NOT_FOUND")
        self.resource_type = resource_type
        self.resource_id = resource_id

class ConflictError(ServiceError):
    """Exception for resource conflict errors"""
    
    def __init__(self, message: str, resource_type: str = None, conflict_field: str = None):
        super().__init__(message, error_code="CONFLICT")
        self.resource_type = resource_type
        self.conflict_field = conflict_field

class RateLimitError(ServiceError):
    """Exception for rate limit exceeded errors"""
    
    def __init__(self, message: str, limit: int = None, reset_time: int = None):
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")
        self.limit = limit
        self.reset_time = reset_time

class ExternalServiceError(ServiceError):
    """Exception for external service failures"""
    
    def __init__(self, message: str, service_name: str = None, status_code: int = None):
        super().__init__(message, error_code="EXTERNAL_SERVICE_ERROR")
        self.service_name = service_name
        self.status_code = status_code

class ConfigurationError(ServiceError):
    """Exception for configuration errors"""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, error_code="CONFIGURATION_ERROR")
        self.config_key = config_key

class ProcessingError(ServiceError):
    """Exception for message processing errors"""
    
    def __init__(self, message: str, stage: str = None, processor: str = None):
        super().__init__(message, error_code="PROCESSING_ERROR")
        self.stage = stage
        self.processor = processor

class DeliveryError(ServiceError):
    """Exception for message delivery errors"""
    
    def __init__(self, message: str, channel: str = None, recipient: str = None):
        super().__init__(message, error_code="DELIVERY_ERROR")
        self.channel = channel
        self.recipient = recipient
```

#### `src/dependencies.py`
**Purpose:** Dependency injection container and FastAPI dependency providers  
**Usage:** Centralized dependency management for services and repositories

**Functions:**

1. **get_message_service() -> MessageService**
   - **Purpose:** Provide MessageService instance with all dependencies
   - **Parameters:** None (uses FastAPI dependency injection)
   - **Return:** Configured MessageService instance
   - **Usage:** FastAPI route dependency

2. **get_session_service() -> SessionService**
   - **Purpose:** Provide SessionService instance
   - **Parameters:** None
   - **Return:** Configured SessionService instance
   - **Usage:** FastAPI route dependency

```python
"""Dependency injection for services and repositories"""

from functools import lru_cache
from typing import Annotated
from fastapi import Depends

# Database dependencies
from src.database.mongodb import get_mongodb
from src.database.redis_client import get_redis

# Repository dependencies
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.repositories.session_repository import SessionRepository
from src.repositories.rate_limit_repository import RateLimitRepository

# Core component dependencies
from src.core.channels.channel_factory import ChannelFactory
from src.core.processors.processor_factory import ProcessorFactory

# Service dependencies
from src.services.message_service import MessageService
from src.services.session_service import SessionService
from src.services.conversation_service import ConversationService
from src.services.audit_service import AuditService

# Repository dependency providers
async def get_conversation_repository() -> ConversationRepository:
    """Get conversation repository instance"""
    database = await get_mongodb()
    return ConversationRepository(database)

async def get_message_repository() -> MessageRepository:
    """Get message repository instance"""
    database = await get_mongodb()
    return MessageRepository(database)

async def get_session_repository() -> SessionRepository:
    """Get session repository instance"""
    redis_client = await get_redis()
    return SessionRepository(redis_client)

async def get_rate_limit_repository() -> RateLimitRepository:
    """Get rate limit repository instance"""
    redis_client = await get_redis()
    return RateLimitRepository(redis_client)

# Core component dependency providers
@lru_cache()
def get_channel_factory() -> ChannelFactory:
    """Get channel factory instance (cached)"""
    return ChannelFactory()

@lru_cache()
def get_processor_factory() -> ProcessorFactory:
    """Get processor factory instance (cached)"""
    return ProcessorFactory()

# Service dependency providers
async def get_message_service(
    conversation_repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    message_repo: Annotated[MessageRepository, Depends(get_message_repository)],
    session_repo: Annotated[SessionRepository, Depends(get_session_repository)],
    channel_factory: Annotated[ChannelFactory, Depends(get_channel_factory)],
    processor_factory: Annotated[ProcessorFactory, Depends(get_processor_factory)]
) -> MessageService:
    """Get message service instance with all dependencies"""
    return MessageService(
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        session_repo=session_repo,
        channel_factory=channel_factory,
        processor_factory=processor_factory
    )

async def get_session_service(
    session_repo: Annotated[SessionRepository, Depends(get_session_repository)]
) -> SessionService:
    """Get session service instance"""
    return SessionService(session_repo)

async def get_conversation_service(
    conversation_repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    message_repo: Annotated[MessageRepository, Depends(get_message_repository)]
) -> ConversationService:
    """Get conversation service instance"""
    return ConversationService(conversation_repo, message_repo)

async def get_audit_service(
    # Add audit repository when implemented
) -> AuditService:
    """Get audit service instance"""
    return AuditService()

# Health check dependencies
async def get_health_checkers():
    """Get all health check dependencies"""
    return {
        "mongodb": await get_mongodb(),
        "redis": await get_redis(),
    }
```

---

## 🔧 Technologies Used
- **FastAPI Dependency Injection**: Service and repository management
- **Async/await**: Asynchronous service operations
- **structlog**: Structured logging throughout services
- **Pydantic**: Data validation in service layer
- **UUID**: Unique identifier generation

---

## ⚠️ Key Considerations

### Error Handling
- Comprehensive exception hierarchy
- Graceful error propagation
- Context preservation in error handling
- Detailed error logging

### Performance
- Async operations throughout
- Efficient dependency injection
- Connection reuse across services
- Optimal data access patterns

### Scalability
- Stateless service design
- Horizontal scaling support
- Efficient resource utilization
- Connection pooling

### Security
- Tenant isolation enforcement
- Input validation at service boundaries
- Secure session management
- Audit trail generation

---

## 🎯 Success Criteria
- [ ] All core services are implemented and functional
- [ ] Message processing pipeline works end-to-end
- [ ] Session management is operational
- [ ] Dependency injection is properly configured
- [ ] Error handling is comprehensive
- [ ] Service orchestration works correctly

---

## 📋 Next Phase Preview
Phase 6 will focus on implementing the API layer with endpoints, middleware, and request/response handling, building upon the solid service foundation established in this phase.

