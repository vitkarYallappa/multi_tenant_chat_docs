# Phase 7: Event Handling & Real-time Integration
**Duration:** Week 11-12  
**Steps:** 13-14 of 18

---

## 🎯 Objectives
- Implement event-driven architecture with Kafka
- Create event publishers and subscribers for analytics
- Build real-time webhook delivery system
- Establish cross-service communication patterns

---

## 📋 Step 13: Event Architecture & Kafka Integration

### What Will Be Implemented
- Event publisher and subscriber base classes
- Kafka producer and consumer implementations
- Event schema definitions and validation
- Event routing and topic management

### Folders and Files Created

```
src/events/
├── __init__.py
├── base_event.py
├── event_publisher.py
├── event_subscriber.py
├── kafka_producer.py
├── kafka_consumer.py
├── event_schemas.py
├── event_router.py
└── handlers/
    ├── __init__.py
    ├── analytics_handler.py
    ├── notification_handler.py
    └── audit_handler.py
```

### File Documentation

#### `src/events/base_event.py`
**Purpose:** Base event classes and interfaces for event-driven architecture  
**Usage:** Foundation for all event types with consistent structure and validation

**Classes:**

1. **BaseEvent(BaseModel)**
   - **Purpose:** Base class for all event types
   - **Fields:** Event ID, timestamp, version, metadata
   - **Usage:** Ensure consistent event structure across system

2. **EventMetadata(BaseModel)**
   - **Purpose:** Common metadata for all events
   - **Fields:** Source service, correlation ID, user context
   - **Usage:** Track event provenance and context

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Type, TypeVar, Generic
from uuid import uuid4
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()

T = TypeVar('T')

class EventMetadata(BaseModel):
    """Common metadata for all events"""
    source_service: str = "chat-service"
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None  # ID of event that caused this event
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Context information
    channel: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Processing metadata
    retry_count: int = 0
    processing_attempts: int = 0
    
    class Config:
        extra = "allow"  # Allow additional metadata

class BaseEvent(BaseModel, ABC):
    """Base class for all events"""
    
    # Event identification
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    event_version: str = "1.0"
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: EventMetadata = Field(default_factory=EventMetadata)
    
    # Event data (to be defined in subclasses)
    data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @property
    @abstractmethod
    def topic_name(self) -> str:
        """Return the Kafka topic name for this event type"""
        pass
    
    @property
    def partition_key(self) -> Optional[str]:
        """Return partition key for Kafka (defaults to tenant_id)"""
        return self.metadata.tenant_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return self.dict(by_alias=True)
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create event from dictionary"""
        return cls(**data)
    
    def add_metadata(self, **kwargs) -> None:
        """Add additional metadata to the event"""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
            else:
                # Add to extra fields
                if not hasattr(self.metadata, '__dict__'):
                    self.metadata.__dict__ = {}
                self.metadata.__dict__[key] = value
    
    def increment_retry_count(self) -> None:
        """Increment retry count for failed processing"""
        self.metadata.retry_count += 1
    
    def increment_processing_attempts(self) -> None:
        """Increment processing attempts counter"""
        self.metadata.processing_attempts += 1

class DomainEvent(BaseEvent):
    """Base class for domain events (business events)"""
    
    def __init__(self, **data):
        if "event_type" not in data:
            data["event_type"] = self.__class__.__name__
        super().__init__(**data)

class SystemEvent(BaseEvent):
    """Base class for system events (technical events)"""
    
    def __init__(self, **data):
        if "event_type" not in data:
            data["event_type"] = self.__class__.__name__
        super().__init__(**data)

class IntegrationEvent(BaseEvent):
    """Base class for integration events (cross-service events)"""
    
    def __init__(self, **data):
        if "event_type" not in data:
            data["event_type"] = self.__class__.__name__
        super().__init__(**data)

# Event handler interface
class EventHandler(ABC, Generic[T]):
    """Abstract base class for event handlers"""
    
    @property
    @abstractmethod
    def event_type(self) -> Type[T]:
        """Return the event type this handler processes"""
        pass
    
    @abstractmethod
    async def handle(self, event: T) -> None:
        """
        Handle the event
        
        Args:
            event: Event to process
            
        Raises:
            Exception: If event processing fails
        """
        pass
    
    async def can_handle(self, event: BaseEvent) -> bool:
        """
        Check if this handler can process the event
        
        Args:
            event: Event to check
            
        Returns:
            True if handler can process the event
        """
        return isinstance(event, self.event_type)
    
    def get_handler_name(self) -> str:
        """Get handler name for logging"""
        return self.__class__.__name__

# Event processing result
class EventProcessingResult(BaseModel):
    """Result of event processing"""
    success: bool
    event_id: str
    handler_name: str
    processing_time_ms: int
    error_message: Optional[str] = None
    retry_required: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

#### `src/events/event_schemas.py`
**Purpose:** Specific event type definitions for different business scenarios  
**Usage:** Define structured events for messages, conversations, and system operations

**Classes:**

1. **MessageReceivedEvent(DomainEvent)**
   - **Purpose:** Event for incoming message processing
   - **Fields:** Message data, conversation context, processing metadata
   - **Usage:** Trigger analytics and workflow processing

2. **MessageSentEvent(DomainEvent)**
   - **Purpose:** Event for outgoing message delivery
   - **Fields:** Response data, delivery status, channel info
   - **Usage:** Track message delivery and analytics

3. **ConversationStartedEvent(DomainEvent)**
   - **Purpose:** Event for new conversation initiation
   - **Fields:** Conversation metadata, user info, channel
   - **Usage:** Initialize conversation tracking and analytics

```python
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import Field

from src.events.base_event import DomainEvent, SystemEvent, IntegrationEvent
from src.models.types import ChannelType, MessageType, ConversationStatus

# Message Events

class MessageReceivedEvent(DomainEvent):
    """Event published when a message is received"""
    
    @property
    def topic_name(self) -> str:
        return "chat.message.received.v1"
    
    def __init__(self, **data):
        event_data = {
            "message_id": data.get("message_id"),
            "conversation_id": data.get("conversation_id"),
            "user_id": data.get("user_id"),
            "channel": data.get("channel"),
            "message_type": data.get("message_type"),
            "content": data.get("content", {}),
            "processing_result": data.get("processing_result", {}),
            "timestamp": data.get("timestamp", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

class MessageSentEvent(DomainEvent):
    """Event published when a message is sent"""
    
    @property
    def topic_name(self) -> str:
        return "chat.message.sent.v1"
    
    def __init__(self, **data):
        event_data = {
            "message_id": data.get("message_id"),
            "conversation_id": data.get("conversation_id"),
            "recipient": data.get("recipient"),
            "channel": data.get("channel"),
            "message_type": data.get("message_type"),
            "content": data.get("content", {}),
            "delivery_status": data.get("delivery_status"),
            "delivery_metadata": data.get("delivery_metadata", {}),
            "timestamp": data.get("timestamp", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

class MessageDeliveredEvent(DomainEvent):
    """Event published when message delivery is confirmed"""
    
    @property
    def topic_name(self) -> str:
        return "chat.message.delivered.v1"
    
    def __init__(self, **data):
        event_data = {
            "message_id": data.get("message_id"),
            "conversation_id": data.get("conversation_id"),
            "channel": data.get("channel"),
            "platform_message_id": data.get("platform_message_id"),
            "delivery_timestamp": data.get("delivery_timestamp", datetime.utcnow()),
            "delivery_metadata": data.get("delivery_metadata", {})
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

# Conversation Events

class ConversationStartedEvent(DomainEvent):
    """Event published when a conversation is started"""
    
    @property
    def topic_name(self) -> str:
        return "chat.conversation.started.v1"
    
    def __init__(self, **data):
        event_data = {
            "conversation_id": data.get("conversation_id"),
            "user_id": data.get("user_id"),
            "channel": data.get("channel"),
            "flow_id": data.get("flow_id"),
            "initial_context": data.get("initial_context", {}),
            "user_profile": data.get("user_profile", {}),
            "started_at": data.get("started_at", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

class ConversationEndedEvent(DomainEvent):
    """Event published when a conversation ends"""
    
    @property
    def topic_name(self) -> str:
        return "chat.conversation.ended.v1"
    
    def __init__(self, **data):
        event_data = {
            "conversation_id": data.get("conversation_id"),
            "user_id": data.get("user_id"),
            "channel": data.get("channel"),
            "end_reason": data.get("end_reason"),  # completed, abandoned, escalated, timeout
            "final_status": data.get("final_status"),
            "duration_seconds": data.get("duration_seconds"),
            "message_count": data.get("message_count"),
            "satisfaction_score": data.get("satisfaction_score"),
            "ended_at": data.get("ended_at", datetime.utcnow()),
            "conversation_summary": data.get("conversation_summary", {})
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

class ConversationContextUpdatedEvent(DomainEvent):
    """Event published when conversation context is updated"""
    
    @property
    def topic_name(self) -> str:
        return "chat.conversation.context_updated.v1"
    
    def __init__(self, **data):
        event_data = {
            "conversation_id": data.get("conversation_id"),
            "user_id": data.get("user_id"),
            "context_changes": data.get("context_changes", {}),
            "previous_context": data.get("previous_context", {}),
            "new_context": data.get("new_context", {}),
            "trigger": data.get("trigger"),  # message_received, intent_detected, etc.
            "updated_at": data.get("updated_at", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

# User Events

class UserSessionStartedEvent(DomainEvent):
    """Event published when a user session starts"""
    
    @property
    def topic_name(self) -> str:
        return "chat.user.session_started.v1"
    
    def __init__(self, **data):
        event_data = {
            "session_id": data.get("session_id"),
            "user_id": data.get("user_id"),
            "channel": data.get("channel"),
            "device_info": data.get("device_info", {}),
            "location_info": data.get("location_info", {}),
            "started_at": data.get("started_at", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

class UserSessionEndedEvent(DomainEvent):
    """Event published when a user session ends"""
    
    @property
    def topic_name(self) -> str:
        return "chat.user.session_ended.v1"
    
    def __init__(self, **data):
        event_data = {
            "session_id": data.get("session_id"),
            "user_id": data.get("user_id"),
            "channel": data.get("channel"),
            "duration_seconds": data.get("duration_seconds"),
            "end_reason": data.get("end_reason"),  # timeout, explicit_logout, etc.
            "ended_at": data.get("ended_at", datetime.utcnow()),
            "session_summary": data.get("session_summary", {})
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

# System Events

class ServiceHealthCheckEvent(SystemEvent):
    """Event published for service health monitoring"""
    
    @property
    def topic_name(self) -> str:
        return "system.health_check.v1"
    
    def __init__(self, **data):
        event_data = {
            "service_name": data.get("service_name", "chat-service"),
            "health_status": data.get("health_status"),  # healthy, degraded, unhealthy
            "check_details": data.get("check_details", {}),
            "dependencies": data.get("dependencies", {}),
            "metrics": data.get("metrics", {}),
            "checked_at": data.get("checked_at", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

class ErrorOccurredEvent(SystemEvent):
    """Event published when system errors occur"""
    
    @property
    def topic_name(self) -> str:
        return "system.error.v1"
    
    def __init__(self, **data):
        event_data = {
            "error_type": data.get("error_type"),
            "error_message": data.get("error_message"),
            "error_code": data.get("error_code"),
            "component": data.get("component"),
            "operation": data.get("operation"),
            "stack_trace": data.get("stack_trace"),
            "context": data.get("context", {}),
            "severity": data.get("severity", "error"),  # info, warning, error, critical
            "occurred_at": data.get("occurred_at", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

# Integration Events

class WebhookReceivedEvent(IntegrationEvent):
    """Event published when webhook is received from external service"""
    
    @property
    def topic_name(self) -> str:
        return "integration.webhook.received.v1"
    
    def __init__(self, **data):
        event_data = {
            "webhook_source": data.get("webhook_source"),  # whatsapp, slack, etc.
            "webhook_type": data.get("webhook_type"),
            "webhook_id": data.get("webhook_id"),
            "raw_payload": data.get("raw_payload", {}),
            "processed_payload": data.get("processed_payload", {}),
            "validation_status": data.get("validation_status"),
            "received_at": data.get("received_at", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

class ExternalAPICallEvent(IntegrationEvent):
    """Event published when external API is called"""
    
    @property
    def topic_name(self) -> str:
        return "integration.api_call.v1"
    
    def __init__(self, **data):
        event_data = {
            "api_name": data.get("api_name"),
            "endpoint": data.get("endpoint"),
            "method": data.get("method"),
            "request_id": data.get("request_id"),
            "response_status": data.get("response_status"),
            "response_time_ms": data.get("response_time_ms"),
            "success": data.get("success"),
            "error_details": data.get("error_details"),
            "called_at": data.get("called_at", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

# Analytics Events

class ConversationAnalyticsEvent(DomainEvent):
    """Event for conversation analytics aggregation"""
    
    @property
    def topic_name(self) -> str:
        return "analytics.conversation.v1"
    
    def __init__(self, **data):
        event_data = {
            "conversation_id": data.get("conversation_id"),
            "metrics": data.get("metrics", {}),
            "performance": data.get("performance", {}),
            "quality_scores": data.get("quality_scores", {}),
            "business_metrics": data.get("business_metrics", {}),
            "aggregation_timestamp": data.get("aggregation_timestamp", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})

class UserBehaviorEvent(DomainEvent):
    """Event for user behavior analytics"""
    
    @property
    def topic_name(self) -> str:
        return "analytics.user_behavior.v1"
    
    def __init__(self, **data):
        event_data = {
            "behavior_type": data.get("behavior_type"),  # click, scroll, dwell, etc.
            "element": data.get("element"),
            "page": data.get("page"),
            "value": data.get("value"),
            "context": data.get("context", {}),
            "recorded_at": data.get("recorded_at", datetime.utcnow())
        }
        
        super().__init__(data=event_data, **{k: v for k, v in data.items() if k != "data"})
```

---

## 📋 Step 14: Event Publishing & Subscription

### What Will Be Implemented
- Kafka producer for reliable event publishing
- Kafka consumer for event subscription and processing
- Event routing and handler registration
- Dead letter queue handling for failed events

### Folders and Files Created

```
src/events/
├── kafka_producer.py
├── kafka_consumer.py
├── event_router.py
├── dead_letter_queue.py
└── event_manager.py
```

### File Documentation

#### `src/events/kafka_producer.py`
**Purpose:** Kafka producer implementation for reliable event publishing  
**Usage:** Publish events to Kafka topics with partitioning and error handling

**Classes:**

1. **KafkaEventProducer**
   - **Purpose:** Publish events to Kafka topics
   - **Methods:**
     - **publish_event(event: BaseEvent) -> bool**: Publish single event
     - **publish_batch(events: List[BaseEvent]) -> List[bool]**: Publish multiple events
     - **start() -> None**: Initialize producer connection
     - **stop() -> None**: Close producer connection

```python
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError
import structlog

from src.events.base_event import BaseEvent
from src.config.settings import get_settings
from src.events.event_schemas import ErrorOccurredEvent

logger = structlog.get_logger()

class KafkaEventProducer:
    """Kafka producer for publishing events"""
    
    def __init__(self, 
                 bootstrap_servers: Optional[List[str]] = None,
                 topic_prefix: Optional[str] = None):
        self.settings = get_settings()
        self.bootstrap_servers = bootstrap_servers or self.settings.KAFKA_BROKERS
        self.topic_prefix = topic_prefix or self.settings.KAFKA_TOPIC_PREFIX
        
        self.producer: Optional[KafkaProducer] = None
        self.is_connected = False
        
        # Configuration
        self.producer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'value_serializer': self._serialize_event,
            'key_serializer': lambda k: k.encode('utf-8') if k else None,
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
            'retry_backoff_ms': 1000,
            'request_timeout_ms': 30000,
            'compression_type': 'gzip',
            'batch_size': 16384,
            'linger_ms': 100,  # Wait up to 100ms to batch messages
            'buffer_memory': 33554432,  # 32MB buffer
        }
        
        # Callbacks
        self.success_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
    
    async def start(self) -> None:
        """Initialize Kafka producer connection"""
        try:
            self.producer = KafkaProducer(**self.producer_config)
            self.is_connected = True
            
            logger.info(
                "Kafka producer started",
                bootstrap_servers=self.bootstrap_servers,
                topic_prefix=self.topic_prefix
            )
            
        except Exception as e:
            logger.error("Failed to start Kafka producer", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Close Kafka producer connection"""
        if self.producer:
            try:
                self.producer.flush()  # Ensure all messages are sent
                self.producer.close()
                self.is_connected = False
                
                logger.info("Kafka producer stopped")
                
            except Exception as e:
                logger.error("Error stopping Kafka producer", error=str(e))
    
    async def publish_event(self, event: BaseEvent) -> bool:
        """
        Publish a single event to Kafka
        
        Args:
            event: Event to publish
            
        Returns:
            True if published successfully
        """
        if not self.is_connected or not self.producer:
            logger.error("Kafka producer not connected")
            return False
        
        try:
            # Build topic name
            topic = self._build_topic_name(event.topic_name)
            
            # Get partition key
            partition_key = event.partition_key
            
            # Add publishing metadata
            event.add_metadata(
                published_at=datetime.utcnow(),
                publisher="chat-service"
            )
            
            # Send message
            future = self.producer.send(
                topic=topic,
                value=event,
                key=partition_key
            )
            
            # Add callbacks
            future.add_callback(self._on_send_success, event)
            future.add_errback(self._on_send_error, event)
            
            # Wait for result (with timeout)
            record_metadata = future.get(timeout=10)
            
            logger.info(
                "Event published successfully",
                event_id=event.event_id,
                event_type=event.event_type,
                topic=topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset
            )
            
            return True
            
        except KafkaTimeoutError:
            logger.error(
                "Event publish timeout",
                event_id=event.event_id,
                event_type=event.event_type
            )
            return False
            
        except KafkaError as e:
            logger.error(
                "Kafka error publishing event",
                event_id=event.event_id,
                event_type=event.event_type,
                error=str(e)
            )
            return False
            
        except Exception as e:
            logger.error(
                "Unexpected error publishing event",
                event_id=event.event_id,
                event_type=event.event_type,
                error=str(e)
            )
            return False
    
    async def publish_batch(self, events: List[BaseEvent]) -> List[bool]:
        """
        Publish multiple events as a batch
        
        Args:
            events: List of events to publish
            
        Returns:
            List of success/failure results for each event
        """
        if not events:
            return []
        
        results = []
        
        try:
            # Group events by topic for better batching
            events_by_topic = {}
            for event in events:
                topic = self._build_topic_name(event.topic_name)
                if topic not in events_by_topic:
                    events_by_topic[topic] = []
                events_by_topic[topic].append(event)
            
            # Send all events
            futures = []
            for topic, topic_events in events_by_topic.items():
                for event in topic_events:
                    try:
                        # Add publishing metadata
                        event.add_metadata(
                            published_at=datetime.utcnow(),
                            publisher="chat-service"
                        )
                        
                        future = self.producer.send(
                            topic=topic,
                            value=event,
                            key=event.partition_key
                        )
                        futures.append((future, event))
                        
                    except Exception as e:
                        logger.error(
                            "Failed to send event in batch",
                            event_id=event.event_id,
                            error=str(e)
                        )
                        results.append(False)
            
            # Wait for all futures
            for future, event in futures:
                try:
                    future.get(timeout=10)
                    results.append(True)
                    
                    logger.debug(
                        "Batch event published",
                        event_id=event.event_id,
                        event_type=event.event_type
                    )
                    
                except Exception as e:
                    logger.error(
                        "Batch event publish failed",
                        event_id=event.event_id,
                        error=str(e)
                    )
                    results.append(False)
            
            successful_count = sum(results)
            logger.info(
                "Batch publish completed",
                total_events=len(events),
                successful=successful_count,
                failed=len(events) - successful_count
            )
            
            return results
            
        except Exception as e:
            logger.error("Batch publish failed", error=str(e))
            return [False] * len(events)
    
    def _serialize_event(self, event: BaseEvent) -> bytes:
        """Serialize event to JSON bytes"""
        try:
            event_dict = event.to_dict()
            json_str = json.dumps(event_dict, default=str)
            return json_str.encode('utf-8')
        except Exception as e:
            logger.error("Event serialization failed", error=str(e))
            raise
    
    def _build_topic_name(self, base_topic: str) -> str:
        """Build full topic name with prefix"""
        if self.topic_prefix:
            return f"{self.topic_prefix}.{base_topic}"
        return base_topic
    
    def _on_send_success(self, record_metadata, event: BaseEvent):
        """Callback for successful message send"""
        logger.debug(
            "Event send callback - success",
            event_id=event.event_id,
            topic=record_metadata.topic,
            partition=record_metadata.partition,
            offset=record_metadata.offset
        )
        
        if self.success_callback:
            try:
                self.success_callback(event, record_metadata)
            except Exception as e:
                logger.error("Success callback failed", error=str(e))
    
    def _on_send_error(self, exception, event: BaseEvent):
        """Callback for failed message send"""
        logger.error(
            "Event send callback - error",
            event_id=event.event_id,
            error=str(exception)
        )
        
        if self.error_callback:
            try:
                self.error_callback(event, exception)
            except Exception as e:
                logger.error("Error callback failed", error=str(e))
    
    async def publish_error_event(self, error: Exception, context: Dict[str, Any]):
        """Publish system error event"""
        try:
            error_event = ErrorOccurredEvent(
                error_type=type(error).__name__,
                error_message=str(error),
                component="kafka_producer",
                operation="publish_event",
                context=context,
                severity="error"
            )
            
            # Try to publish error event (but don't fail if this fails)
            await self.publish_event(error_event)
            
        except Exception as e:
            logger.error("Failed to publish error event", error=str(e))

# Global producer instance
_event_producer: Optional[KafkaEventProducer] = None

async def get_event_producer() -> KafkaEventProducer:
    """Get global event producer instance"""
    global _event_producer
    
    if _event_producer is None:
        _event_producer = KafkaEventProducer()
        await _event_producer.start()
    
    return _event_producer

async def publish_event(event: BaseEvent) -> bool:
    """Convenience function to publish an event"""
    producer = await get_event_producer()
    return await producer.publish_event(event)

async def publish_events(events: List[BaseEvent]) -> List[bool]:
    """Convenience function to publish multiple events"""
    producer = await get_event_producer()
    return await producer.publish_batch(events)
```

#### `src/events/event_manager.py`
**Purpose:** Central event management for registration, publishing, and processing  
**Usage:** Orchestrate event handling across the application

**Classes:**

1. **EventManager**
   - **Purpose:** Central coordinator for all event operations
   - **Methods:**
     - **register_handler(handler: EventHandler) -> None**: Register event handler
     - **publish(event: BaseEvent) -> bool**: Publish event
     - **start_consumers() -> None**: Start event consumers
     - **stop_consumers() -> None**: Stop event consumers

```python
import asyncio
from datetime import datetime
from typing import Dict, List, Type, Optional, Set
from collections import defaultdict
import structlog

from src.events.base_event import BaseEvent, EventHandler, EventProcessingResult
from src.events.kafka_producer import KafkaEventProducer, get_event_producer
from src.events.kafka_consumer import KafkaEventConsumer
from src.events.event_schemas import ErrorOccurredEvent, ServiceHealthCheckEvent

logger = structlog.get_logger()

class EventManager:
    """Central event management system"""
    
    def __init__(self):
        self.handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.producer: Optional[KafkaEventProducer] = None
        self.consumers: List[KafkaEventConsumer] = []
        self.is_running = False
        
        # Event processing statistics
        self.stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "handlers_registered": 0
        }
        
        # Consumer configuration
        self.consumer_config = {
            "group_id": "chat-service-consumers",
            "auto_offset_reset": "latest",
            "enable_auto_commit": False,  # Manual commit for reliability
            "max_poll_records": 500,
            "session_timeout_ms": 30000,
            "heartbeat_interval_ms": 10000
        }
    
    async def initialize(self) -> None:
        """Initialize event manager"""
        try:
            # Initialize producer
            self.producer = await get_event_producer()
            
            logger.info("Event manager initialized")
            
        except Exception as e:
            logger.error("Failed to initialize event manager", error=str(e))
            raise
    
    def register_handler(self, handler: EventHandler) -> None:
        """
        Register an event handler
        
        Args:
            handler: Event handler to register
        """
        try:
            event_type_name = handler.event_type.__name__
            self.handlers[event_type_name].append(handler)
            self.stats["handlers_registered"] += 1
            
            logger.info(
                "Event handler registered",
                handler=handler.get_handler_name(),
                event_type=event_type_name,
                total_handlers=len(self.handlers[event_type_name])
            )
            
        except Exception as e:
            logger.error(
                "Failed to register event handler",
                handler=handler.get_handler_name() if handler else "unknown",
                error=str(e)
            )
            raise
    
    def get_handlers_for_event(self, event: BaseEvent) -> List[EventHandler]:
        """Get all handlers that can process the given event"""
        event_type_name = event.event_type
        return self.handlers.get(event_type_name, [])
    
    async def publish(self, event: BaseEvent) -> bool:
        """
        Publish an event
        
        Args:
            event: Event to publish
            
        Returns:
            True if published successfully
        """
        try:
            if not self.producer:
                logger.error("Event producer not initialized")
                return False
            
            # Add event manager metadata
            event.add_metadata(
                event_manager_version="1.0",
                published_by="event_manager"
            )
            
            # Publish event
            success = await self.producer.publish_event(event)
            
            if success:
                self.stats["events_published"] += 1
                logger.debug(
                    "Event published",
                    event_id=event.event_id,
                    event_type=event.event_type
                )
            else:
                self.stats["events_failed"] += 1
                logger.error(
                    "Event publish failed",
                    event_id=event.event_id,
                    event_type=event.event_type
                )
            
            return success
            
        except Exception as e:
            self.stats["events_failed"] += 1
            logger.error(
                "Event publish error",
                event_id=event.event_id if event else "unknown",
                error=str(e)
            )
            return False
    
    async def publish_batch(self, events: List[BaseEvent]) -> List[bool]:
        """
        Publish multiple events
        
        Args:
            events: List of events to publish
            
        Returns:
            List of success/failure results
        """
        try:
            if not self.producer:
                logger.error("Event producer not initialized")
                return [False] * len(events)
            
            # Add metadata to all events
            for event in events:
                event.add_metadata(
                    event_manager_version="1.0",
                    published_by="event_manager"
                )
            
            # Publish batch
            results = await self.producer.publish_batch(events)
            
            # Update statistics
            successful = sum(results)
            failed = len(results) - successful
            
            self.stats["events_published"] += successful
            self.stats["events_failed"] += failed
            
            logger.info(
                "Batch events published",
                total=len(events),
                successful=successful,
                failed=failed
            )
            
            return results
            
        except Exception as e:
            self.stats["events_failed"] += len(events)
            logger.error("Batch event publish error", error=str(e))
            return [False] * len(events)
    
    async def process_event(self, event: BaseEvent) -> List[EventProcessingResult]:
        """
        Process an event through all registered handlers
        
        Args:
            event: Event to process
            
        Returns:
            List of processing results from all handlers
        """
        results = []
        handlers = self.get_handlers_for_event(event)
        
        if not handlers:
            logger.debug(
                "No handlers found for event",
                event_type=event.event_type,
                event_id=event.event_id
            )
            return results
        
        logger.debug(
            "Processing event",
            event_type=event.event_type,
            event_id=event.event_id,
            handler_count=len(handlers)
        )
        
        # Process event with each handler
        for handler in handlers:
            start_time = datetime.utcnow()
            
            try:
                # Check if handler can process this event
                if not await handler.can_handle(event):
                    continue
                
                # Process event
                await handler.handle(event)
                
                # Calculate processing time
                processing_time = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )
                
                result = EventProcessingResult(
                    success=True,
                    event_id=event.event_id,
                    handler_name=handler.get_handler_name(),
                    processing_time_ms=processing_time
                )
                
                results.append(result)
                self.stats["events_processed"] += 1
                
                logger.debug(
                    "Event processed successfully",
                    event_id=event.event_id,
                    handler=handler.get_handler_name(),
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                processing_time = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )
                
                result = EventProcessingResult(
                    success=False,
                    event_id=event.event_id,
                    handler_name=handler.get_handler_name(),
                    processing_time_ms=processing_time,
                    error_message=str(e),
                    retry_required=self._should_retry_event(e)
                )
                
                results.append(result)
                self.stats["events_failed"] += 1
                
                logger.error(
                    "Event processing failed",
                    event_id=event.event_id,
                    handler=handler.get_handler_name(),
                    error=str(e),
                    processing_time_ms=processing_time
                )
                
                # Publish error event
                await self._publish_error_event(event, handler, e)
        
        return results
    
    async def start_processing(self) -> None:
        """Start event processing (consumers)"""
        try:
            if self.is_running:
                logger.warning("Event processing already running")
                return
            
            # Create consumers for different topic patterns
            consumer_topics = [
                "chat.message.*",
                "chat.conversation.*", 
                "chat.user.*",
                "system.*",
                "integration.*",
                "analytics.*"
            ]
            
            for topic_pattern in consumer_topics:
                consumer = KafkaEventConsumer(
                    topic_pattern=topic_pattern,
                    group_id=f"{self.consumer_config['group_id']}-{topic_pattern.replace('*', 'all')}",
                    event_processor=self.process_event
                )
                
                self.consumers.append(consumer)
                await consumer.start()
            
            self.is_running = True
            
            logger.info(
                "Event processing started",
                consumer_count=len(self.consumers)
            )
            
        except Exception as e:
            logger.error("Failed to start event processing", error=str(e))
            raise
    
    async def stop_processing(self) -> None:
        """Stop event processing (consumers)"""
        try:
            if not self.is_running:
                return
            
            # Stop all consumers
            for consumer in self.consumers:
                await consumer.stop()
            
            self.consumers.clear()
            self.is_running = False
            
            logger.info("Event processing stopped")
            
        except Exception as e:
            logger.error("Failed to stop event processing", error=str(e))
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of event system"""
        try:
            return {
                "status": "healthy" if self.is_running else "stopped",
                "producer_connected": self.producer.is_connected if self.producer else False,
                "consumer_count": len(self.consumers),
                "handlers_registered": self.stats["handlers_registered"],
                "statistics": self.stats.copy(),
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get event system health", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    def _should_retry_event(self, error: Exception) -> bool:
        """Determine if event should be retried based on error type"""
        # Retry for transient errors
        transient_errors = [
            "ConnectionError",
            "TimeoutError", 
            "TemporaryError",
            "ServiceUnavailable"
        ]
        
        error_type = type(error).__name__
        return error_type in transient_errors
    
    async def _publish_error_event(
        self, 
        original_event: BaseEvent, 
        handler: EventHandler, 
        error: Exception
    ) -> None:
        """Publish error event for failed event processing"""
        try:
            error_event = ErrorOccurredEvent(
                error_type=type(error).__name__,
                error_message=str(error),
                component="event_manager",
                operation=f"process_event:{handler.get_handler_name()}",
                context={
                    "original_event_id": original_event.event_id,
                    "original_event_type": original_event.event_type,
                    "handler_name": handler.get_handler_name()
                },
                severity="error"
            )
            
            # Publish error event (but don't fail if this fails)
            await self.publish(error_event)
            
        except Exception as e:
            logger.error("Failed to publish error event", error=str(e))

# Global event manager instance
_event_manager: Optional[EventManager] = None

async def get_event_manager() -> EventManager:
    """Get global event manager instance"""
    global _event_manager
    
    if _event_manager is None:
        _event_manager = EventManager()
        await _event_manager.initialize()
    
    return _event_manager

# Convenience functions
async def publish_event(event: BaseEvent) -> bool:
    """Publish an event through the global event manager"""
    manager = await get_event_manager()
    return await manager.publish(event)

async def register_event_handler(handler: EventHandler) -> None:
    """Register an event handler with the global event manager"""
    manager = await get_event_manager()
    manager.register_handler(handler)
```

---

## 🔧 Technologies Used
- **Apache Kafka**: Event streaming platform
- **kafka-python**: Python Kafka client
- **JSON**: Event serialization format
- **asyncio**: Asynchronous event processing
- **structlog**: Structured logging for event tracking

---

## ⚠️ Key Considerations

### Reliability
- At-least-once delivery semantics
- Dead letter queue for failed events
- Retry mechanisms with exponential backoff
- Event ordering within partitions

### Scalability
- Partitioned topics for parallel processing
- Consumer groups for load distribution
- Batch processing for high throughput
- Configurable consumer instances

### Monitoring
- Event processing metrics and statistics
- Error tracking and alerting
- Performance monitoring
- Health check endpoints

### Data Consistency
- Event schema versioning
- Backward compatibility
- Event ordering guarantees
- Transactional event publishing

---

## 🎯 Success Criteria
- [ ] Event publishing to Kafka is working reliably
- [ ] Event consumers are processing messages correctly
- [ ] Event handlers are registered and functioning
- [ ] Dead letter queue handles failed events
- [ ] Event schema validation is enforced
- [ ] Performance meets throughput requirements

---

## 📋 Next Phase Preview
Phase 8 will focus on implementing external integrations and webhook handling, including gRPC clients for MCP Engine and Security Hub communication.

