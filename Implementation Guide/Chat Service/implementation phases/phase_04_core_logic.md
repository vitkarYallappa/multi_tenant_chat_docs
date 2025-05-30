# Phase 4: Core Business Logic (Channels & Processors)
**Duration:** Week 5-6  
**Steps:** 7-8 of 18

---

## 🎯 Objectives
- Implement channel abstraction and specific channel handlers
- Create message processing pipeline with processors and normalizers
- Establish content validation and transformation logic
- Build factory patterns for dynamic component creation

---

## 📋 Step 7: Channel Architecture & Implementations

### What Will Be Implemented
- Abstract base channel with common interface
- Channel-specific implementations (Web, WhatsApp, Slack, etc.)
- Channel factory for dynamic instantiation
- Message validation and formatting per channel

### Folders and Files Created

```
src/core/
├── __init__.py
├── channels/
│   ├── __init__.py
│   ├── base_channel.py
│   ├── web_channel.py
│   ├── whatsapp_channel.py
│   ├── messenger_channel.py
│   ├── slack_channel.py
│   ├── teams_channel.py
│   └── channel_factory.py
└── exceptions.py
```

### File Documentation

#### `src/core/channels/base_channel.py`
**Purpose:** Abstract base class defining the channel interface and common functionality  
**Usage:** Foundation for all channel implementations with standardized message handling

**Classes:**

1. **ChannelConfig(BaseModel)**
   - **Purpose:** Configuration model for channel settings
   - **Fields:** Authentication, rate limits, formatting options
   - **Usage:** Store channel-specific configuration

2. **ChannelResponse(BaseModel)**
   - **Purpose:** Standardized response format for channel operations
   - **Fields:** Success status, delivery info, metadata
   - **Usage:** Consistent response format across channels

3. **BaseChannel(ABC)**
   - **Purpose:** Abstract base class for all channel implementations
   - **Methods:**
     - **send_message(recipient: str, content: MessageContent, metadata: Dict) -> ChannelResponse**: Send message via channel
     - **validate_recipient(recipient: str) -> bool**: Validate recipient format
     - **validate_content(content: MessageContent) -> bool**: Validate content for channel
     - **format_message(content: MessageContent) -> Dict**: Format message for channel API

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import structlog

from src.models.types import MessageContent, ChannelType, DeliveryStatus

logger = structlog.get_logger()

class ChannelConfig(BaseModel):
    """Configuration for channel implementations"""
    channel_type: ChannelType
    enabled: bool = True
    
    # Authentication
    api_token: Optional[str] = None
    api_secret: Optional[str] = None
    webhook_secret: Optional[str] = None
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_day: int = 10000
    
    # Message formatting
    max_message_length: int = 4096
    supported_message_types: List[str] = Field(default_factory=lambda: ["text"])
    supports_rich_media: bool = False
    supports_buttons: bool = False
    supports_quick_replies: bool = False
    
    # Delivery settings
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    timeout_seconds: int = 30
    
    # Feature flags
    features: Dict[str, bool] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True

class ChannelResponse(BaseModel):
    """Standardized response from channel operations"""
    success: bool
    channel_type: ChannelType
    message_id: Optional[str] = None
    platform_message_id: Optional[str] = None
    delivery_status: DeliveryStatus = DeliveryStatus.SENT
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Error information
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Delivery metadata
    recipient: Optional[str] = None
    retry_count: int = 0
    delivery_attempt_at: Optional[datetime] = None
    
    # Performance metrics
    processing_time_ms: Optional[int] = None
    
    # Channel-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True

class BaseChannel(ABC):
    """Abstract base class for all channel implementations"""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._validate_config()
    
    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Return the channel type"""
        pass
    
    @abstractmethod
    async def send_message(
        self,
        recipient: str,
        content: MessageContent,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChannelResponse:
        """
        Send a message through this channel
        
        Args:
            recipient: Channel-specific recipient identifier
            content: Message content to send
            metadata: Optional channel-specific metadata
            
        Returns:
            ChannelResponse with delivery information
        """
        pass
    
    @abstractmethod
    async def validate_recipient(self, recipient: str) -> bool:
        """
        Validate recipient format for this channel
        
        Args:
            recipient: Recipient identifier to validate
            
        Returns:
            True if recipient format is valid
        """
        pass
    
    @abstractmethod
    async def format_message(
        self, 
        content: MessageContent
    ) -> Dict[str, Any]:
        """
        Format message content for channel API
        
        Args:
            content: Message content to format
            
        Returns:
            Formatted message for channel API
        """
        pass
    
    async def validate_content(self, content: MessageContent) -> bool:
        """
        Validate message content for this channel
        
        Args:
            content: Message content to validate
            
        Returns:
            True if content is valid for this channel
        """
        try:
            # Check message type support
            if content.type.value not in self.config.supported_message_types:
                self.logger.warning(
                    "Unsupported message type",
                    channel=self.channel_type,
                    message_type=content.type,
                    supported_types=self.config.supported_message_types
                )
                return False
            
            # Check text length
            if content.text and len(content.text) > self.config.max_message_length:
                self.logger.warning(
                    "Message text too long",
                    channel=self.channel_type,
                    length=len(content.text),
                    max_length=self.config.max_message_length
                )
                return False
            
            # Check rich media support
            if content.media and not self.config.supports_rich_media:
                self.logger.warning(
                    "Rich media not supported",
                    channel=self.channel_type
                )
                return False
            
            # Check buttons support
            if content.buttons and not self.config.supports_buttons:
                self.logger.warning(
                    "Buttons not supported",
                    channel=self.channel_type
                )
                return False
            
            # Check quick replies support
            if content.quick_replies and not self.config.supports_quick_replies:
                self.logger.warning(
                    "Quick replies not supported",
                    channel=self.channel_type
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Content validation failed",
                channel=self.channel_type,
                error=str(e)
            )
            return False
    
    async def health_check(self) -> bool:
        """
        Perform health check for this channel
        
        Returns:
            True if channel is healthy
        """
        try:
            # Default implementation - can be overridden
            return self.config.enabled
        except Exception as e:
            self.logger.error(
                "Health check failed",
                channel=self.channel_type,
                error=str(e)
            )
            return False
    
    def _validate_config(self) -> None:
        """Validate channel configuration"""
        if not self.config.enabled:
            self.logger.warning(
                "Channel is disabled",
                channel=self.channel_type
            )
        
        if self.config.requests_per_minute <= 0:
            raise ValueError("Requests per minute must be positive")
        
        if self.config.max_message_length <= 0:
            raise ValueError("Max message length must be positive")
    
    def _create_error_response(
        self,
        error_code: str,
        error_message: str,
        recipient: Optional[str] = None
    ) -> ChannelResponse:
        """Create standardized error response"""
        return ChannelResponse(
            success=False,
            channel_type=self.channel_type,
            delivery_status=DeliveryStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
            recipient=recipient
        )
    
    def _create_success_response(
        self,
        message_id: str,
        platform_message_id: Optional[str] = None,
        recipient: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChannelResponse:
        """Create standardized success response"""
        return ChannelResponse(
            success=True,
            channel_type=self.channel_type,
            message_id=message_id,
            platform_message_id=platform_message_id,
            delivery_status=DeliveryStatus.SENT,
            recipient=recipient,
            metadata=metadata or {}
        )
```

#### `src/core/channels/whatsapp_channel.py`
**Purpose:** WhatsApp Business API channel implementation  
**Usage:** Handle WhatsApp-specific message sending and webhook processing

**Classes:**

1. **WhatsAppChannel(BaseChannel)**
   - **Purpose:** WhatsApp Business API integration
   - **Methods:**
     - **send_message(recipient: str, content: MessageContent, metadata: Dict) -> ChannelResponse**: Send WhatsApp message
     - **validate_recipient(recipient: str) -> bool**: Validate phone number format
     - **format_message(content: MessageContent) -> Dict**: Format for WhatsApp API
     - **upload_media(media_content: MediaContent) -> str**: Upload media to WhatsApp
     - **process_webhook(webhook_data: Dict) -> Dict**: Process incoming webhook

```python
import re
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.channels.base_channel import BaseChannel, ChannelConfig, ChannelResponse
from src.models.types import MessageContent, MessageType, ChannelType, DeliveryStatus, MediaContent
from src.core.exceptions import ChannelError, ValidationError

class WhatsAppChannel(BaseChannel):
    """WhatsApp Business API channel implementation"""
    
    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self.api_base_url = "https://graph.facebook.com/v18.0"
        self.phone_number_id = config.features.get("phone_number_id")
        self.business_account_id = config.features.get("business_account_id")
        
        if not self.phone_number_id:
            raise ValueError("WhatsApp phone_number_id is required")
        
        # Configure HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=config.timeout_seconds,
            headers={
                "Authorization": f"Bearer {config.api_token}",
                "Content-Type": "application/json"
            }
        )
    
    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WHATSAPP
    
    async def send_message(
        self,
        recipient: str,
        content: MessageContent,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChannelResponse:
        """Send message via WhatsApp Business API"""
        start_time = datetime.utcnow()
        
        try:
            # Validate recipient
            if not await self.validate_recipient(recipient):
                return self._create_error_response(
                    "INVALID_RECIPIENT",
                    f"Invalid WhatsApp phone number: {recipient}",
                    recipient
                )
            
            # Validate content
            if not await self.validate_content(content):
                return self._create_error_response(
                    "INVALID_CONTENT",
                    "Message content is not valid for WhatsApp",
                    recipient
                )
            
            # Format message for WhatsApp API
            message_payload = await self.format_message(content)
            message_payload["to"] = recipient
            
            # Send message
            url = f"{self.api_base_url}/{self.phone_number_id}/messages"
            
            response = await self.http_client.post(url, json=message_payload)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract message ID
            platform_message_id = None
            if "messages" in response_data and response_data["messages"]:
                platform_message_id = response_data["messages"][0].get("id")
            
            # Calculate processing time
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            self.logger.info(
                "WhatsApp message sent successfully",
                recipient=recipient,
                platform_message_id=platform_message_id,
                processing_time_ms=processing_time
            )
            
            return ChannelResponse(
                success=True,
                channel_type=self.channel_type,
                platform_message_id=platform_message_id,
                delivery_status=DeliveryStatus.SENT,
                recipient=recipient,
                processing_time_ms=processing_time,
                metadata={"api_response": response_data}
            )
            
        except httpx.HTTPStatusError as e:
            error_message = f"WhatsApp API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_message = error_data["error"].get("message", error_message)
            except:
                pass
            
            self.logger.error(
                "WhatsApp API request failed",
                recipient=recipient,
                status_code=e.response.status_code,
                error=error_message
            )
            
            return self._create_error_response(
                "API_ERROR",
                error_message,
                recipient
            )
            
        except Exception as e:
            self.logger.error(
                "WhatsApp message send failed",
                recipient=recipient,
                error=str(e)
            )
            
            return self._create_error_response(
                "SEND_FAILED",
                f"Failed to send WhatsApp message: {str(e)}",
                recipient
            )
    
    async def validate_recipient(self, recipient: str) -> bool:
        """Validate WhatsApp phone number (E.164 format)"""
        # E.164 format: +[country][number] (max 15 digits total)
        pattern = r'^\+[1-9]\d{1,14}$'
        return bool(re.match(pattern, recipient))
    
    async def format_message(self, content: MessageContent) -> Dict[str, Any]:
        """Format message content for WhatsApp API"""
        try:
            if content.type == MessageType.TEXT:
                return await self._format_text_message(content)
            elif content.type == MessageType.IMAGE:
                return await self._format_media_message(content, "image")
            elif content.type == MessageType.AUDIO:
                return await self._format_media_message(content, "audio")
            elif content.type == MessageType.VIDEO:
                return await self._format_media_message(content, "video")
            elif content.type == MessageType.FILE:
                return await self._format_media_message(content, "document")
            elif content.type == MessageType.LOCATION:
                return await self._format_location_message(content)
            else:
                raise ValidationError(f"Unsupported message type: {content.type}")
                
        except Exception as e:
            self.logger.error(
                "Message formatting failed",
                message_type=content.type,
                error=str(e)
            )
            raise ChannelError(f"Failed to format WhatsApp message: {e}")
    
    async def _format_text_message(self, content: MessageContent) -> Dict[str, Any]:
        """Format text message for WhatsApp"""
        message = {
            "type": "text",
            "text": {"body": content.text}
        }
        
        # Add interactive elements if supported
        if content.buttons and len(content.buttons) <= 3:
            message["type"] = "interactive"
            message["interactive"] = {
                "type": "button",
                "body": {"text": content.text},
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {
                                "id": button.payload,
                                "title": button.title
                            }
                        }
                        for button in content.buttons
                    ]
                }
            }
            message.pop("text")
        
        return message
    
    async def _format_media_message(
        self, 
        content: MessageContent, 
        media_type: str
    ) -> Dict[str, Any]:
        """Format media message for WhatsApp"""
        if not content.media:
            raise ValidationError("Media content is required for media messages")
        
        message = {
            "type": media_type,
            media_type: {
                "link": content.media.url
            }
        }
        
        # Add caption if text is provided
        if content.text:
            message[media_type]["caption"] = content.text
        
        return message
    
    async def _format_location_message(self, content: MessageContent) -> Dict[str, Any]:
        """Format location message for WhatsApp"""
        if not content.location:
            raise ValidationError("Location content is required for location messages")
        
        message = {
            "type": "location",
            "location": {
                "latitude": content.location.latitude,
                "longitude": content.location.longitude
            }
        }
        
        if content.location.address:
            message["location"]["address"] = content.location.address
        
        return message
    
    async def process_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process WhatsApp webhook data"""
        try:
            # Verify webhook signature (implementation depends on webhook setup)
            # if not self._verify_webhook_signature(webhook_data):
            #     raise SecurityError("Invalid webhook signature")
            
            processed_events = []
            
            if "entry" in webhook_data:
                for entry in webhook_data["entry"]:
                    if "changes" in entry:
                        for change in entry["changes"]:
                            if change.get("field") == "messages":
                                event = await self._process_message_event(change["value"])
                                if event:
                                    processed_events.append(event)
            
            return {
                "status": "processed",
                "events_count": len(processed_events),
                "events": processed_events
            }
            
        except Exception as e:
            self.logger.error(
                "Webhook processing failed",
                error=str(e),
                webhook_data=webhook_data
            )
            raise ChannelError(f"Failed to process WhatsApp webhook: {e}")
    
    async def _process_message_event(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process individual message event from webhook"""
        try:
            if "messages" not in message_data:
                return None
            
            for message in message_data["messages"]:
                return {
                    "type": "message_received",
                    "platform_message_id": message.get("id"),
                    "from": message.get("from"),
                    "timestamp": message.get("timestamp"),
                    "message_type": message.get("type"),
                    "content": message
                }
            
        except Exception as e:
            self.logger.error(
                "Message event processing failed",
                error=str(e),
                message_data=message_data
            )
            return None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
```

---

## 📋 Step 8: Message Processing Pipeline

### What Will Be Implemented
- Message processor abstractions and implementations
- Content normalizers for consistent data format
- Processing pipeline with validation and transformation
- Processor factory for dynamic component selection

### Folders and Files Created

```
src/core/
├── processors/
│   ├── __init__.py
│   ├── base_processor.py
│   ├── text_processor.py
│   ├── media_processor.py
│   ├── location_processor.py
│   └── processor_factory.py
├── normalizers/
│   ├── __init__.py
│   ├── message_normalizer.py
│   ├── content_normalizer.py
│   └── metadata_normalizer.py
└── pipeline/
    ├── __init__.py
    ├── processing_pipeline.py
    └── pipeline_stage.py
```

### File Documentation

#### `src/core/processors/base_processor.py`
**Purpose:** Abstract base class for message processors with common processing interface  
**Usage:** Foundation for all message type processors

**Classes:**

1. **ProcessingContext(BaseModel)**
   - **Purpose:** Context data passed through processing pipeline
   - **Fields:** Tenant info, user info, conversation context, processing hints
   - **Usage:** Maintain state and metadata during processing

2. **ProcessingResult(BaseModel)**
   - **Purpose:** Result of message processing operation
   - **Fields:** Processed content, extracted entities, confidence scores, metadata
   - **Usage:** Standardized processing output

3. **BaseProcessor(ABC)**
   - **Purpose:** Abstract base class for all message processors
   - **Methods:**
     - **process(content: MessageContent, context: ProcessingContext) -> ProcessingResult**: Process message content
     - **validate_input(content: MessageContent) -> bool**: Validate input content
     - **extract_entities(content: MessageContent) -> Dict**: Extract entities from content

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import structlog

from src.models.types import MessageContent, MessageType, TenantId, UserId

logger = structlog.get_logger()

class ProcessingContext(BaseModel):
    """Context data for message processing"""
    tenant_id: TenantId
    user_id: UserId
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Channel information
    channel: str
    channel_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # User and conversation context
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Processing hints
    processing_hints: Dict[str, Any] = Field(default_factory=dict)
    language: str = "en"
    timezone: str = "UTC"
    
    # Request metadata
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ProcessingResult(BaseModel):
    """Result of message processing"""
    success: bool = True
    
    # Processed content
    original_content: MessageContent
    processed_content: Optional[MessageContent] = None
    normalized_content: Optional[Dict[str, Any]] = None
    
    # Analysis results
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    
    # Entity extraction
    entities: Dict[str, Any] = Field(default_factory=dict)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Sentiment and intent (placeholder for future AI integration)
    sentiment: Optional[Dict[str, Any]] = None
    intent: Optional[Dict[str, Any]] = None
    
    # Content analysis
    content_categories: List[str] = Field(default_factory=list)
    content_tags: List[str] = Field(default_factory=list)
    
    # Quality and safety
    quality_score: Optional[float] = None
    safety_flags: List[str] = Field(default_factory=list)
    moderation_required: bool = False
    
    # Processing metadata
    processing_time_ms: Optional[int] = None
    processor_version: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseProcessor(ABC):
    """Abstract base class for message processors"""
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.processor_version = "1.0.0"
    
    @property
    @abstractmethod
    def supported_message_types(self) -> List[MessageType]:
        """Return list of supported message types"""
        pass
    
    @property
    @abstractmethod
    def processor_name(self) -> str:
        """Return processor name"""
        pass
    
    @abstractmethod
    async def process(
        self, 
        content: MessageContent, 
        context: ProcessingContext
    ) -> ProcessingResult:
        """
        Process message content
        
        Args:
            content: Message content to process
            context: Processing context
            
        Returns:
            ProcessingResult with analysis and processed content
        """
        pass
    
    async def validate_input(
        self, 
        content: MessageContent, 
        context: ProcessingContext
    ) -> bool:
        """
        Validate input content for processing
        
        Args:
            content: Message content to validate
            context: Processing context
            
        Returns:
            True if content is valid for this processor
        """
        try:
            # Check if message type is supported
            if content.type not in self.supported_message_types:
                self.logger.warning(
                    "Unsupported message type",
                    processor=self.processor_name,
                    message_type=content.type,
                    supported_types=[t.value for t in self.supported_message_types]
                )
                return False
            
            # Perform type-specific validation
            return await self._validate_type_specific(content, context)
            
        except Exception as e:
            self.logger.error(
                "Input validation failed",
                processor=self.processor_name,
                error=str(e)
            )
            return False
    
    async def _validate_type_specific(
        self, 
        content: MessageContent, 
        context: ProcessingContext
    ) -> bool:
        """Override in subclasses for type-specific validation"""
        return True
    
    async def extract_entities(
        self, 
        content: MessageContent, 
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """
        Extract entities from message content
        
        Args:
            content: Message content
            context: Processing context
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        try:
            # Basic entity extraction (override in subclasses)
            if content.text:
                entities.update(await self._extract_text_entities(content.text, context))
            
            if content.location:
                entities["location"] = {
                    "latitude": content.location.latitude,
                    "longitude": content.location.longitude,
                    "address": content.location.address
                }
            
            return entities
            
        except Exception as e:
            self.logger.error(
                "Entity extraction failed",
                processor=self.processor_name,
                error=str(e)
            )
            return {}
    
    async def _extract_text_entities(
        self, 
        text: str, 
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Extract entities from text (basic implementation)"""
        entities = {}
        
        # Basic patterns (extend with NER models in production)
        import re
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            entities["emails"] = emails
        
        # Phone number extraction (simple)
        phone_pattern = r'\+?[\d\s\-\(\)]{10,}'
        phones = re.findall(phone_pattern, text)
        if phones:
            entities["phones"] = phones
        
        # URL extraction
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        if urls:
            entities["urls"] = urls
        
        return entities
    
    def _create_result(
        self, 
        content: MessageContent, 
        processing_time_ms: int,
        **kwargs
    ) -> ProcessingResult:
        """Create processing result with common fields"""
        return ProcessingResult(
            original_content=content,
            processing_time_ms=processing_time_ms,
            processor_version=self.processor_version,
            **kwargs
        )
    
    def _measure_processing_time(self, start_time: datetime) -> int:
        """Calculate processing time in milliseconds"""
        return int((datetime.utcnow() - start_time).total_seconds() * 1000)
```

#### `src/core/processors/text_processor.py`
**Purpose:** Specialized processor for text message content analysis and transformation  
**Usage:** Handle text-specific processing like language detection, content analysis, and normalization

**Classes:**

1. **TextProcessor(BaseProcessor)**
   - **Purpose:** Process text message content
   - **Methods:**
     - **process(content: MessageContent, context: ProcessingContext) -> ProcessingResult**: Process text content
     - **detect_language(text: str) -> tuple**: Detect text language
     - **analyze_sentiment(text: str) -> Dict**: Analyze text sentiment
     - **extract_keywords(text: str) -> List[str]**: Extract keywords from text
     - **normalize_text(text: str) -> str**: Normalize text formatting

```python
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from langdetect import detect, LangDetectError

from src.core.processors.base_processor import (
    BaseProcessor, ProcessingContext, ProcessingResult
)
from src.models.types import MessageContent, MessageType

class TextProcessor(BaseProcessor):
    """Processor for text message content"""
    
    def __init__(self):
        super().__init__()
        self.max_text_length = 10000  # Safety limit
        self.supported_languages = ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"]
    
    @property
    def supported_message_types(self) -> List[MessageType]:
        return [MessageType.TEXT]
    
    @property
    def processor_name(self) -> str:
        return "TextProcessor"
    
    async def process(
        self, 
        content: MessageContent, 
        context: ProcessingContext
    ) -> ProcessingResult:
        """Process text message content"""
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not await self.validate_input(content, context):
                return self._create_result(
                    content,
                    self._measure_processing_time(start_time),
                    success=False,
                    errors=["Input validation failed"]
                )
            
            text = content.text
            if not text:
                return self._create_result(
                    content,
                    self._measure_processing_time(start_time),
                    success=False,
                    errors=["No text content to process"]
                )
            
            # Normalize text
            normalized_text = await self.normalize_text(text)
            
            # Detect language
            detected_language, language_confidence = await self.detect_language(text)
            
            # Extract entities
            entities = await self.extract_entities(content, context)
            
            # Extract keywords
            keywords = await self.extract_keywords(text)
            
            # Analyze content
            content_analysis = await self.analyze_content(text)
            
            # Perform safety checks
            safety_flags = await self.check_content_safety(text)
            
            # Create processed content
            processed_content = MessageContent(
                type=content.type,
                text=normalized_text,
                language=detected_language or context.language,
                media=content.media,
                location=content.location,
                quick_replies=content.quick_replies,
                buttons=content.buttons
            )
            
            # Calculate processing time
            processing_time = self._measure_processing_time(start_time)
            
            self.logger.info(
                "Text processing completed",
                text_length=len(text),
                detected_language=detected_language,
                entities_count=len(entities),
                keywords_count=len(keywords),
                processing_time_ms=processing_time
            )
            
            return ProcessingResult(
                success=True,
                original_content=content,
                processed_content=processed_content,
                detected_language=detected_language,
                language_confidence=language_confidence,
                entities=entities,
                extracted_data={
                    "keywords": keywords,
                    "normalized_text": normalized_text,
                    "content_analysis": content_analysis
                },
                content_categories=content_analysis.get("categories", []),
                content_tags=keywords[:10],  # Limit tags
                quality_score=content_analysis.get("quality_score"),
                safety_flags=safety_flags,
                moderation_required=len(safety_flags) > 0,
                processing_time_ms=processing_time,
                processor_version=self.processor_version
            )
            
        except Exception as e:
            self.logger.error(
                "Text processing failed",
                error=str(e),
                text_length=len(content.text) if content.text else 0
            )
            
            return self._create_result(
                content,
                self._measure_processing_time(start_time),
                success=False,
                errors=[f"Processing failed: {str(e)}"]
            )
    
    async def _validate_type_specific(
        self, 
        content: MessageContent, 
        context: ProcessingContext
    ) -> bool:
        """Validate text-specific content"""
        if not content.text:
            return False
        
        if len(content.text) > self.max_text_length:
            self.logger.warning(
                "Text too long for processing",
                text_length=len(content.text),
                max_length=self.max_text_length
            )
            return False
        
        return True
    
    async def detect_language(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """Detect language of text"""
        try:
            # Clean text for better detection
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) < 3:
                return None, None
            
            detected_lang = detect(clean_text)
            
            # Simple confidence estimation based on text length
            confidence = min(0.95, 0.5 + (len(clean_text) / 200))
            
            if detected_lang in self.supported_languages:
                return detected_lang, confidence
            else:
                return "en", 0.3  # Default to English with low confidence
                
        except LangDetectError:
            return None, None
        except Exception as e:
            self.logger.error("Language detection failed", error=str(e))
            return None, None
    
    async def normalize_text(self, text: str) -> str:
        """Normalize text formatting"""
        try:
            # Remove excessive whitespace
            normalized = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            normalized = normalized.strip()
            
            # Convert to lowercase for consistency (optional)
            # normalized = normalized.lower()
            
            # Remove or replace special characters if needed
            # This is conservative - adjust based on requirements
            
            return normalized
            
        except Exception as e:
            self.logger.error("Text normalization failed", error=str(e))
            return text
    
    async def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        try:
            # Simple keyword extraction (use more sophisticated NLP in production)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Remove common stop words
            stop_words = {
                "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
                "by", "from", "up", "about", "into", "through", "during", "before",
                "after", "above", "below", "out", "off", "down", "under", "again",
                "further", "then", "once", "here", "there", "when", "where", "why",
                "how", "all", "any", "both", "each", "few", "more", "most", "other",
                "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                "than", "too", "very", "can", "will", "just", "should", "could",
                "would", "may", "might", "must", "shall", "ought", "need", "dare",
                "used", "able", "like", "well", "also", "back", "even", "still",
                "way", "take", "come", "good", "new", "first", "last", "long",
                "great", "little", "own", "other", "old", "right", "big", "high",
                "different", "small", "large", "next", "early", "young", "important",
                "few", "public", "bad", "same", "able"
            }
            
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count frequency and return top keywords
            from collections import Counter
            word_counts = Counter(keywords)
            top_keywords = [word for word, count in word_counts.most_common(20)]
            
            return top_keywords
            
        except Exception as e:
            self.logger.error("Keyword extraction failed", error=str(e))
            return []
    
    async def analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze content characteristics"""
        try:
            analysis = {
                "word_count": len(text.split()),
                "character_count": len(text),
                "sentence_count": len(re.split(r'[.!?]+', text)),
                "categories": [],
                "quality_score": 0.5,  # Default score
                "readability": "medium"
            }
            
            # Basic categorization
            categories = []
            
            # Business/commerce indicators
            business_keywords = ["order", "purchase", "buy", "sell", "price", "cost", "payment", "invoice"]
            if any(keyword in text.lower() for keyword in business_keywords):
                categories.append("business")
            
            # Support/help indicators
            support_keywords = ["help", "support", "problem", "issue", "error", "bug", "question"]
            if any(keyword in text.lower() for keyword in support_keywords):
                categories.append("support")
            
            # Technical indicators
            tech_keywords = ["api", "code", "software", "application", "system", "database"]
            if any(keyword in text.lower() for keyword in tech_keywords):
                categories.append("technical")
            
            analysis["categories"] = categories
            
            # Simple quality scoring
            quality_score = 0.5
            if len(text) > 10:
                quality_score += 0.2
            if len(text.split()) > 5:
                quality_score += 0.2
            if not re.search(r'[!@#$%^&*()]{3,}', text):  # Not too many special chars
                quality_score += 0.1
            
            analysis["quality_score"] = min(1.0, quality_score)
            
            return analysis
            
        except Exception as e:
            self.logger.error("Content analysis failed", error=str(e))
            return {"categories": [], "quality_score": 0.5}
    
    async def check_content_safety(self, text: str) -> List[str]:
        """Check content for safety issues"""
        safety_flags = []
        
        try:
            text_lower = text.lower()
            
            # Basic safety checks (extend with proper content moderation)
            if any(word in text_lower for word in ["spam", "scam", "fraud"]):
                safety_flags.append("potential_spam")
            
            if len(re.findall(r'[A-Z]{5,}', text)) > 3:
                safety_flags.append("excessive_caps")
            
            if len(re.findall(r'[!]{3,}', text)) > 0:
                safety_flags.append("excessive_punctuation")
            
            # Check for PII (basic patterns)
            if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):  # SSN pattern
                safety_flags.append("potential_pii")
            
            if re.search(r'\b\d{16}\b', text):  # Credit card pattern
                safety_flags.append("potential_pii")
            
            return safety_flags
            
        except Exception as e:
            self.logger.error("Safety check failed", error=str(e))
            return []
```

---

## 🔧 Technologies Used
- **httpx**: HTTP client for external API calls
- **langdetect**: Language detection for text content
- **Python regex**: Pattern matching and text processing
- **Abstract Base Classes**: Interface definitions
- **Pydantic**: Data validation and modeling
- **structlog**: Structured logging

---

## ⚠️ Key Considerations

### Error Handling
- Graceful degradation for channel failures
- Comprehensive validation at each processing step
- Retry mechanisms for transient failures
- Detailed error logging and monitoring

### Performance
- Async processing throughout
- Efficient text processing algorithms
- Caching of processed results
- Connection pooling for external APIs

### Security
- Input validation and sanitization
- Content safety checks
- PII detection and handling
- Webhook signature verification

### Extensibility
- Factory patterns for dynamic component creation
- Plugin-like architecture for new channels
- Configurable processing pipelines
- Standardized interfaces for easy extension

---

## 🎯 Success Criteria
- [ ] All channel implementations are working
- [ ] Message processing pipeline is functional
- [ ] Content validation and normalization work correctly
- [ ] Factory patterns enable dynamic component creation
- [ ] Error handling is comprehensive
- [ ] Performance benchmarks are met

---

## 📋 Next Phase Preview
Phase 5 will focus on implementing the service layer that orchestrates the channels and processors, building business logic services that tie everything together.