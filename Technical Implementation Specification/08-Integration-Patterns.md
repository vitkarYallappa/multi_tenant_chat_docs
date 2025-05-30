# Integration Patterns
## Multi-Tenant AI Chatbot Platform

**Document:** 08-Integration-Patterns.md  
**Version:** 2.0  
**Last Updated:** May 30, 2025

---

## Table of Contents

1. [Integration Architecture Overview](#integration-architecture-overview)
2. [Core Integration Patterns](#core-integration-patterns)
3. [Channel Integration Patterns](#channel-integration-patterns)
4. [Business System Integration](#business-system-integration)
5. [AI/ML Service Integration](#aiml-service-integration)
6. [Integration Security Patterns](#integration-security-patterns)
7. [Error Handling and Resilience](#error-handling-and-resilience)
8. [Integration Marketplace](#integration-marketplace)
9. [Testing and Validation](#testing-and-validation)

---

## Integration Architecture Overview

### Integration Philosophy

1. **Standardized Interface:** Common integration framework for all external systems
2. **Configuration-Driven:** No-code/low-code integration configuration
3. **Resilient by Design:** Built-in error handling, retries, and circuit breakers
4. **Security First:** Authentication, authorization, and data protection
5. **Observable:** Comprehensive monitoring and debugging capabilities
6. **Tenant Isolated:** Complete separation of tenant data and configurations

### Integration Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────┘

Application Layer:
├── MCP Engine (Integration Orchestrator)
├── Chat Service (Channel Integrations)
├── Model Orchestrator (AI Service Integrations)
└── Analytics Engine (Data Export Integrations)

Integration Layer:
├── Adaptor Service (Core Integration Engine)
├── Authentication Manager
├── Request/Response Transformers
├── Error Handler and Circuit Breakers
├── Rate Limiter and Queue Manager
└── Monitoring and Logging

Protocol Layer:
├── REST API Client
├── GraphQL Client
├── WebSocket Manager
├── gRPC Client
├── Database Connectors
├── Message Queue Clients
└── File Transfer Protocols

External Systems:
├── CRM Systems (Salesforce, HubSpot)
├── E-commerce (Shopify, WooCommerce)
├── Support Systems (Zendesk, Intercom)
├── Payment Gateways (Stripe, PayPal)
├── Communication Channels (WhatsApp, Slack)
├── AI/ML Services (OpenAI, Anthropic)
└── Custom APIs and Databases
```

---

## Core Integration Patterns

### Universal Integration Framework

#### Integration Definition Schema

```json
{
  "integration_schema": {
    "metadata": {
      "name": "string",
      "version": "string",
      "description": "string",
      "category": "crm|ecommerce|support|payment|communication|ai|custom",
      "provider": "string",
      "documentation_url": "string",
      "support_contact": "string"
    },
    
    "authentication": {
      "type": "oauth2|api_key|basic_auth|jwt|custom",
      "configuration": {
        "oauth2": {
          "authorization_url": "string",
          "token_url": "string",
          "scopes": ["array"],
          "client_id_required": true,
          "client_secret_required": true
        },
        "api_key": {
          "header_name": "string",
          "query_param_name": "string",
          "location": "header|query|body"
        }
      },
      "test_endpoint": "string"
    },
    
    "endpoints": {
      "endpoint_name": {
        "url": "string",
        "method": "GET|POST|PUT|DELETE|PATCH",
        "description": "string",
        "timeout_ms": 30000,
        "headers": {},
        "query_params": {},
        "body_schema": {},
        "response_schema": {},
        "rate_limit": {
          "requests_per_minute": 60,
          "burst_limit": 10
        },
        "retry_config": {
          "max_retries": 3,
          "backoff_strategy": "exponential|linear|fixed",
          "initial_delay_ms": 1000,
          "max_delay_ms": 30000,
          "retry_on_status_codes": [500, 502, 503, 504]
        }
      }
    },
    
    "data_mapping": {
      "request_templates": {},
      "response_transformations": {},
      "field_mappings": {},
      "validation_rules": {}
    },
    
    "webhook_support": {
      "enabled": true,
      "endpoint_path": "/webhooks/{integration_id}",
      "verification": {
        "signature_header": "string",
        "secret_key": "string",
        "algorithm": "sha256"
      },
      "events": ["array_of_supported_events"]
    },
    
    "testing": {
      "sandbox_mode": true,
      "test_credentials": {},
      "sample_requests": [],
      "sample_responses": []
    }
  }
}
```

#### Dynamic Integration Engine

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class AuthenticationType(Enum):
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    BASIC_AUTH = "basic_auth"
    JWT = "jwt"
    CUSTOM = "custom"

class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"

@dataclass
class IntegrationConfig:
    """Configuration for a specific integration"""
    integration_id: str
    name: str
    provider: str
    category: str
    base_url: str
    authentication: Dict[str, Any]
    endpoints: Dict[str, Any]
    default_timeout_ms: int = 30000
    max_retries: int = 3
    rate_limit_per_minute: int = 60

@dataclass
class IntegrationRequest:
    """Request to execute an integration"""
    endpoint_name: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    tenant_id: str
    user_id: Optional[str] = None
    timeout_override_ms: Optional[int] = None

@dataclass
class IntegrationResponse:
    """Response from integration execution"""
    success: bool
    status_code: Optional[int]
    data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    execution_time_ms: int
    retry_count: int
    cached: bool = False

class IntegrationAuthenticator(ABC):
    """Base class for authentication handlers"""
    
    @abstractmethod
    async def authenticate(self, config: IntegrationConfig) -> Dict[str, str]:
        """Return authentication headers"""
        pass
    
    @abstractmethod
    async def is_valid(self, config: IntegrationConfig) -> bool:
        """Check if authentication is still valid"""
        pass

class OAuth2Authenticator(IntegrationAuthenticator):
    """OAuth 2.0 authentication handler"""
    
    def __init__(self, token_storage, encryption_service):
        self.token_storage = token_storage
        self.encryption_service = encryption_service
    
    async def authenticate(self, config: IntegrationConfig) -> Dict[str, str]:
        """Get or refresh OAuth 2.0 access token"""
        auth_config = config.authentication
        
        # Check for existing valid token
        stored_token = await self.token_storage.get_token(
            config.integration_id, 
            config.tenant_id
        )
        
        if stored_token and not self._is_token_expired(stored_token):
            return {"Authorization": f"Bearer {stored_token['access_token']}"}
        
        # Refresh token if available
        if stored_token and stored_token.get('refresh_token'):
            try:
                new_token = await self._refresh_token(config, stored_token['refresh_token'])
                await self.token_storage.store_token(
                    config.integration_id,
                    config.tenant_id,
                    new_token
                )
                return {"Authorization": f"Bearer {new_token['access_token']}"}
            except Exception as e:
                logging.error(f"Token refresh failed: {e}")
        
        # If no valid token, need to re-authenticate
        raise AuthenticationError("OAuth token invalid and refresh failed")
    
    async def _refresh_token(self, config: IntegrationConfig, refresh_token: str):
        """Refresh OAuth 2.0 access token"""
        auth_config = config.authentication
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'client_id': auth_config['client_id'],
            'client_secret': auth_config['client_secret']
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_config['token_url'], data=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise AuthenticationError(f"Token refresh failed: {response.status}")
    
    def _is_token_expired(self, token_data: Dict) -> bool:
        """Check if token is expired"""
        if 'expires_at' not in token_data:
            return True
        
        expires_at = datetime.fromisoformat(token_data['expires_at'])
        return datetime.utcnow() >= expires_at - timedelta(minutes=5)  # 5-minute buffer

class APIKeyAuthenticator(IntegrationAuthenticator):
    """API Key authentication handler"""
    
    def __init__(self, encryption_service):
        self.encryption_service = encryption_service
    
    async def authenticate(self, config: IntegrationConfig) -> Dict[str, str]:
        """Return API key authentication headers"""
        auth_config = config.authentication
        
        # Decrypt API key
        encrypted_key = auth_config.get('api_key')
        if not encrypted_key:
            raise AuthenticationError("API key not configured")
        
        api_key = await self.encryption_service.decrypt(encrypted_key)
        
        # Return appropriate headers based on configuration
        if auth_config.get('header_name'):
            return {auth_config['header_name']: api_key}
        elif auth_config.get('query_param_name'):
            # For query param auth, return in special format for processing
            return {"_query_param": f"{auth_config['query_param_name']}={api_key}"}
        else:
            return {"Authorization": f"ApiKey {api_key}"}
    
    async def is_valid(self, config: IntegrationConfig) -> bool:
        """API keys don't expire, so always valid if configured"""
        return config.authentication.get('api_key') is not None

class IntegrationExecutor:
    """Core integration execution engine"""
    
    def __init__(self, 
                 authenticators: Dict[str, IntegrationAuthenticator],
                 cache_service,
                 rate_limiter,
                 circuit_breaker):
        self.authenticators = authenticators
        self.cache_service = cache_service
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20)
        )
    
    async def execute(self, config: IntegrationConfig, 
                     request: IntegrationRequest) -> IntegrationResponse:
        """Execute an integration request"""
        start_time = datetime.utcnow()
        
        try:
            # Check rate limiting
            await self._check_rate_limit(config, request.tenant_id)
            
            # Check circuit breaker
            if not self.circuit_breaker.can_execute(config.integration_id):
                raise IntegrationError("Circuit breaker open")
            
            # Check cache first
            cached_response = await self._check_cache(config, request)
            if cached_response:
                return cached_response
            
            # Execute request with retries
            response = await self._execute_with_retries(config, request)
            
            # Cache successful responses
            if response.success:
                await self._cache_response(config, request, response)
                self.circuit_breaker.record_success(config.integration_id)
            else:
                self.circuit_breaker.record_failure(config.integration_id)
            
            return response
            
        except Exception as e:
            self.circuit_breaker.record_failure(config.integration_id)
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return IntegrationResponse(
                success=False,
                status_code=None,
                data=None,
                error_message=str(e),
                execution_time_ms=execution_time,
                retry_count=0
            )
    
    async def _execute_with_retries(self, config: IntegrationConfig, 
                                   request: IntegrationRequest) -> IntegrationResponse:
        """Execute request with retry logic"""
        endpoint_config = config.endpoints.get(request.endpoint_name)
        if not endpoint_config:
            raise IntegrationError(f"Endpoint {request.endpoint_name} not found")
        
        retry_config = endpoint_config.get('retry_config', {})
        max_retries = retry_config.get('max_retries', config.max_retries)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self._execute_single_request(config, request, endpoint_config)
                
                # Check if response indicates we should retry
                if (response.status_code in retry_config.get('retry_on_status_codes', [500, 502, 503, 504]) 
                    and attempt < max_retries):
                    await self._wait_before_retry(retry_config, attempt)
                    continue
                
                return response
                
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < max_retries:
                    await self._wait_before_retry(retry_config, attempt)
                    continue
                else:
                    raise IntegrationError(f"Request failed after {max_retries} retries: {str(e)}")
        
        raise IntegrationError(f"Request failed after {max_retries} retries: {str(last_exception)}")
    
    async def _execute_single_request(self, config: IntegrationConfig,
                                     request: IntegrationRequest,
                                     endpoint_config: Dict) -> IntegrationResponse:
        """Execute a single HTTP request"""
        start_time = datetime.utcnow()
        
        # Get authentication headers
        auth_type = config.authentication.get('type')
        authenticator = self.authenticators.get(auth_type)
        if not authenticator:
            raise IntegrationError(f"Unsupported authentication type: {auth_type}")
        
        auth_headers = await authenticator.authenticate(config)
        
        # Build request
        url = self._build_url(config.base_url, endpoint_config['url'], request.parameters)
        headers = {**endpoint_config.get('headers', {}), **auth_headers}
        
        # Handle query param authentication
        query_params = endpoint_config.get('query_params', {}).copy()
        if '_query_param' in auth_headers:
            param_str = auth_headers['_query_param']
            key, value = param_str.split('=', 1)
            query_params[key] = value
            del headers['_query_param']
        
        # Prepare request body
        request_body = None
        if endpoint_config['method'] in ['POST', 'PUT', 'PATCH']:
            request_body = self._build_request_body(endpoint_config, request.parameters)
        
        # Set timeout
        timeout = request.timeout_override_ms or endpoint_config.get('timeout_ms', config.default_timeout_ms)
        timeout_seconds = timeout / 1000
        
        # Execute request
        try:
            async with self.session.request(
                method=endpoint_config['method'],
                url=url,
                headers=headers,
                params=query_params,
                json=request_body if request_body else None,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as response:
                
                response_data = None
                try:
                    response_data = await response.json()
                except:
                    response_data = {"raw_response": await response.text()}
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                # Transform response data
                transformed_data = self._transform_response(endpoint_config, response_data)
                
                return IntegrationResponse(
                    success=200 <= response.status < 300,
                    status_code=response.status,
                    data=transformed_data,
                    error_message=None if 200 <= response.status < 300 else f"HTTP {response.status}",
                    execution_time_ms=execution_time,
                    retry_count=0
                )
                
        except asyncio.TimeoutError:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return IntegrationResponse(
                success=False,
                status_code=None,
                data=None,
                error_message="Request timeout",
                execution_time_ms=execution_time,
                retry_count=0
            )
    
    def _build_url(self, base_url: str, endpoint_url: str, parameters: Dict[str, Any]) -> str:
        """Build full URL with parameter substitution"""
        url = base_url.rstrip('/') + '/' + endpoint_url.lstrip('/')
        
        # Replace path parameters
        for key, value in parameters.items():
            placeholder = f"{{{key}}}"
            if placeholder in url:
                url = url.replace(placeholder, str(value))
        
        return url
    
    def _build_request_body(self, endpoint_config: Dict, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Build request body from parameters and template"""
        body_schema = endpoint_config.get('body_schema', {})
        
        if not body_schema:
            return parameters
        
        # Apply body template transformation
        # This is a simplified version - production would use JSONPath or similar
        result = {}
        for field, config in body_schema.items():
            if isinstance(config, dict) and 'source' in config:
                source_field = config['source']
                if source_field in parameters:
                    result[field] = parameters[source_field]
            elif field in parameters:
                result[field] = parameters[field]
        
        return result
    
    def _transform_response(self, endpoint_config: Dict, response_data: Any) -> Any:
        """Transform response data according to mapping rules"""
        response_mapping = endpoint_config.get('response_mapping', {})
        
        if not response_mapping:
            return response_data
        
        # Apply response transformations
        # This is simplified - production would use JSONPath expressions
        transformed = {}
        for output_field, source_path in response_mapping.items():
            try:
                # Simple dot notation parsing
                value = response_data
                for part in source_path.split('.'):
                    if part.startswith('$'):
                        continue  # Skip JSONPath root indicator
                    value = value[part]
                transformed[output_field] = value
            except (KeyError, TypeError):
                # Field not found or type error
                transformed[output_field] = None
        
        return transformed
    
    async def _wait_before_retry(self, retry_config: Dict, attempt: int):
        """Wait before retry based on backoff strategy"""
        strategy = retry_config.get('backoff_strategy', 'exponential')
        initial_delay = retry_config.get('initial_delay_ms', 1000) / 1000
        max_delay = retry_config.get('max_delay_ms', 30000) / 1000
        
        if strategy == 'exponential':
            delay = min(initial_delay * (2 ** attempt), max_delay)
        elif strategy == 'linear':
            delay = min(initial_delay * (attempt + 1), max_delay)
        else:  # fixed
            delay = initial_delay
        
        await asyncio.sleep(delay)
    
    async def _check_rate_limit(self, config: IntegrationConfig, tenant_id: str):
        """Check rate limiting for integration"""
        key = f"integration_rate_limit:{config.integration_id}:{tenant_id}"
        allowed = await self.rate_limiter.check_rate_limit(
            key, 
            config.rate_limit_per_minute,
            60
        )
        if not allowed:
            raise RateLimitExceededError("Integration rate limit exceeded")
    
    async def _check_cache(self, config: IntegrationConfig, 
                          request: IntegrationRequest) -> Optional[IntegrationResponse]:
        """Check if response is cached"""
        # Only cache GET requests
        endpoint_config = config.endpoints.get(request.endpoint_name, {})
        if endpoint_config.get('method') != 'GET':
            return None
        
        cache_key = f"integration_cache:{config.integration_id}:{request.endpoint_name}:{hash(str(request.parameters))}"
        cached_data = await self.cache_service.get(cache_key)
        
        if cached_data:
            cached_data['cached'] = True
            return IntegrationResponse(**cached_data)
        
        return None
    
    async def _cache_response(self, config: IntegrationConfig,
                             request: IntegrationRequest,
                             response: IntegrationResponse):
        """Cache successful response"""
        if not response.success:
            return
        
        endpoint_config = config.endpoints.get(request.endpoint_name, {})
        cache_ttl = endpoint_config.get('cache_ttl_seconds', 300)  # 5 minutes default
        
        if cache_ttl > 0:
            cache_key = f"integration_cache:{config.integration_id}:{request.endpoint_name}:{hash(str(request.parameters))}"
            cache_data = {
                'success': response.success,
                'status_code': response.status_code,
                'data': response.data,
                'error_message': response.error_message,
                'execution_time_ms': response.execution_time_ms,
                'retry_count': response.retry_count,
                'cached': False
            }
            await self.cache_service.set(cache_key, cache_data, cache_ttl)

class IntegrationError(Exception):
    """Base exception for integration errors"""
    pass

class AuthenticationError(IntegrationError):
    """Authentication-related errors"""
    pass

class RateLimitExceededError(IntegrationError):
    """Rate limit exceeded errors"""
    pass
```

---

## Channel Integration Patterns

### Multi-Channel Architecture

#### Channel Abstraction Layer

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    LOCATION = "location"
    QUICK_REPLY = "quick_reply"
    CAROUSEL = "carousel"
    FORM = "form"

class MessageDirection(Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"

@dataclass
class ChannelMessage:
    """Standardized message format across all channels"""
    message_id: str
    conversation_id: str
    user_id: str
    channel: str
    direction: MessageDirection
    message_type: MessageType
    timestamp: str
    
    # Content
    text: Optional[str] = None
    media_url: Optional[str] = None
    media_type: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    quick_replies: Optional[List[Dict[str, Any]]] = None
    carousel_items: Optional[List[Dict[str, Any]]] = None
    
    # Channel-specific metadata
    channel_metadata: Optional[Dict[str, Any]] = None
    
    # Platform-specific identifiers
    platform_message_id: Optional[str] = None
    platform_user_id: Optional[str] = None
    platform_thread_id: Optional[str] = None

class ChannelAdapter(ABC):
    """Base class for all channel adapters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.channel_name = config['channel_name']
    
    @abstractmethod
    async def send_message(self, message: ChannelMessage) -> bool:
        """Send message through the channel"""
        pass
    
    @abstractmethod
    async def process_webhook(self, webhook_data: Dict[str, Any]) -> List[ChannelMessage]:
        """Process incoming webhook and convert to standard format"""
        pass
    
    @abstractmethod
    def get_webhook_url(self) -> str:
        """Get webhook URL for this channel"""
        pass
    
    @abstractmethod
    async def verify_webhook(self, headers: Dict[str, str], body: bytes) -> bool:
        """Verify webhook authenticity"""
        pass
    
    def supports_message_type(self, message_type: MessageType) -> bool:
        """Check if channel supports specific message type"""
        return message_type in self.config.get('supported_message_types', [])

class WhatsAppAdapter(ChannelAdapter):
    """WhatsApp Business API integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.access_token = config['access_token']
        self.phone_number_id = config['phone_number_id']
        self.webhook_verify_token = config['webhook_verify_token']
        self.base_url = "https://graph.facebook.com/v18.0"
    
    async def send_message(self, message: ChannelMessage) -> bool:
        """Send message via WhatsApp Business API"""
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # Convert standard message to WhatsApp format
        whatsapp_message = self._convert_to_whatsapp_format(message)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=whatsapp_message) as response:
                return response.status == 200
    
    async def process_webhook(self, webhook_data: Dict[str, Any]) -> List[ChannelMessage]:
        """Process WhatsApp webhook data"""
        messages = []
        
        for entry in webhook_data.get('entry', []):
            for change in entry.get('changes', []):
                if change.get('field') == 'messages':
                    value = change.get('value', {})
                    
                    # Process incoming messages
                    for msg in value.get('messages', []):
                        standard_message = self._convert_from_whatsapp_format(msg, value)
                        if standard_message:
                            messages.append(standard_message)
        
        return messages
    
    def _convert_to_whatsapp_format(self, message: ChannelMessage) -> Dict[str, Any]:
        """Convert standard message to WhatsApp API format"""
        whatsapp_msg = {
            "messaging_product": "whatsapp",
            "to": message.platform_user_id,
            "type": "text"
        }
        
        if message.message_type == MessageType.TEXT:
            whatsapp_msg["text"] = {"body": message.text}
        
        elif message.message_type == MessageType.IMAGE:
            whatsapp_msg["type"] = "image"
            whatsapp_msg["image"] = {
                "link": message.media_url,
                "caption": message.text
            }
        
        elif message.message_type == MessageType.QUICK_REPLY:
            whatsapp_msg["type"] = "interactive"
            whatsapp_msg["interactive"] = {
                "type": "button",
                "body": {"text": message.text},
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {
                                "id": reply["payload"],
                                "title": reply["title"]
                            }
                        }
                        for reply in message.quick_replies[:3]  # WhatsApp supports max 3 buttons
                    ]
                }
            }
        
        elif message.message_type == MessageType.LOCATION:
            whatsapp_msg["type"] = "location"
            whatsapp_msg["location"] = {
                "latitude": message.location["latitude"],
                "longitude": message.location["longitude"],
                "name": message.location.get("name", ""),
                "address": message.location.get("address", "")
            }
        
        return whatsapp_msg
    
    def _convert_from_whatsapp_format(self, whatsapp_msg: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Optional[ChannelMessage]:
        """Convert WhatsApp message to standard format"""
        msg_type = whatsapp_msg.get('type')
        
        # Extract basic information
        from_number = whatsapp_msg.get('from')
        msg_id = whatsapp_msg.get('id')
        timestamp = whatsapp_msg.get('timestamp')
        
        if msg_type == 'text':
            return ChannelMessage(
                message_id=f"whatsapp_{msg_id}",
                conversation_id=f"whatsapp_{from_number}",
                user_id=from_number,
                channel="whatsapp",
                direction=MessageDirection.INBOUND,
                message_type=MessageType.TEXT,
                timestamp=timestamp,
                text=whatsapp_msg['text']['body'],
                platform_message_id=msg_id,
                platform_user_id=from_number,
                channel_metadata={
                    "whatsapp_message_type": msg_type,
                    "context": context
                }
            )
        
        elif msg_type == 'image':
            image_data = whatsapp_msg.get('image', {})
            return ChannelMessage(
                message_id=f"whatsapp_{msg_id}",
                conversation_id=f"whatsapp_{from_number}",
                user_id=from_number,
                channel="whatsapp",
                direction=MessageDirection.INBOUND,
                message_type=MessageType.IMAGE,
                timestamp=timestamp,
                text=image_data.get('caption'),
                media_url=image_data.get('link'),
                media_type="image",
                platform_message_id=msg_id,
                platform_user_id=from_number,
                channel_metadata={
                    "whatsapp_message_type": msg_type,
                    "media_id": image_data.get('id'),
                    "mime_type": image_data.get('mime_type')
                }
            )
        
        elif msg_type == 'interactive':
            interactive_data = whatsapp_msg.get('interactive', {})
            if interactive_data.get('type') == 'button_reply':
                button_reply = interactive_data.get('button_reply', {})
                return ChannelMessage(
                    message_id=f"whatsapp_{msg_id}",
                    conversation_id=f"whatsapp_{from_number}",
                    user_id=from_number,
                    channel="whatsapp",
                    direction=MessageDirection.INBOUND,
                    message_type=MessageType.QUICK_REPLY,
                    timestamp=timestamp,
                    text=button_reply.get('title'),
                    platform_message_id=msg_id,
                    platform_user_id=from_number,
                    channel_metadata={
                        "whatsapp_message_type": msg_type,
                        "button_payload": button_reply.get('id'),
                        "button_title": button_reply.get('title')
                    }
                )
        
        # Add handlers for other message types...
        
        return None
    
    async def verify_webhook(self, headers: Dict[str, str], body: bytes) -> bool:
        """Verify WhatsApp webhook signature"""
        import hmac
        import hashlib
        
        signature = headers.get('X-Hub-Signature-256', '')
        if not signature.startswith('sha256='):
            return False
        
        expected_signature = hmac.new(
            self.webhook_verify_token.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(
            signature[7:],  # Remove 'sha256=' prefix
            expected_signature
        )

class SlackAdapter(ChannelAdapter):
    """Slack integration adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bot_token = config['bot_token']
        self.signing_secret = config['signing_secret']
        self.base_url = "https://slack.com/api"
    
    async def send_message(self, message: ChannelMessage) -> bool:
        """Send message via Slack API"""
        url = f"{self.base_url}/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json"
        }
        
        slack_message = self._convert_to_slack_format(message)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=slack_message) as response:
                result = await response.json()
                return result.get('ok', False)
    
    async def process_webhook(self, webhook_data: Dict[str, Any]) -> List[ChannelMessage]:
        """Process Slack webhook/event data"""
        messages = []
        
        # Handle different Slack event types
        if webhook_data.get('type') == 'event_callback':
            event = webhook_data.get('event', {})
            
            if event.get('type') == 'message' and not event.get('bot_id'):
                # Convert Slack message to standard format
                standard_message = self._convert_from_slack_format(event)
                if standard_message:
                    messages.append(standard_message)
        
        return messages
    
    def _convert_to_slack_format(self, message: ChannelMessage) -> Dict[str, Any]:
        """Convert standard message to Slack format"""
        slack_msg = {
            "channel": message.platform_thread_id or message.platform_user_id,
            "text": message.text or ""
        }
        
        if message.message_type == MessageType.QUICK_REPLY and message.quick_replies:
            # Use Slack interactive components
            slack_msg["attachments"] = [{
                "text": "",
                "actions": [
                    {
                        "type": "button",
                        "text": reply["title"],
                        "value": reply["payload"]
                    }
                    for reply in message.quick_replies[:5]  # Slack supports up to 5 buttons
                ]
            }]
        
        elif message.message_type == MessageType.IMAGE:
            slack_msg["attachments"] = [{
                "title": message.text or "Image",
                "image_url": message.media_url
            }]
        
        return slack_msg
    
    def _convert_from_slack_format(self, slack_event: Dict[str, Any]) -> Optional[ChannelMessage]:
        """Convert Slack event to standard format"""
        user_id = slack_event.get('user')
        channel_id = slack_event.get('channel')
        timestamp = slack_event.get('ts')
        text = slack_event.get('text', '')
        
        # Check for file attachments
        files = slack_event.get('files', [])
        if files:
            file_info = files[0]
            return ChannelMessage(
                message_id=f"slack_{timestamp}_{user_id}",
                conversation_id=f"slack_{channel_id}",
                user_id=user_id,
                channel="slack",
                direction=MessageDirection.INBOUND,
                message_type=MessageType.FILE,
                timestamp=timestamp,
                text=text,
                media_url=file_info.get('url_private'),
                media_type=file_info.get('mimetype'),
                platform_message_id=timestamp,
                platform_user_id=user_id,
                platform_thread_id=channel_id,
                channel_metadata={
                    "file_name": file_info.get('name'),
                    "file_size": file_info.get('size'),
                    "channel_type": slack_event.get('channel_type')
                }
            )
        else:
            return ChannelMessage(
                message_id=f"slack_{timestamp}_{user_id}",
                conversation_id=f"slack_{channel_id}",
                user_id=user_id,
                channel="slack",
                direction=MessageDirection.INBOUND,
                message_type=MessageType.TEXT,
                timestamp=timestamp,
                text=text,
                platform_message_id=timestamp,
                platform_user_id=user_id,
                platform_thread_id=channel_id,
                channel_metadata={
                    "channel_type": slack_event.get('channel_type'),
                    "thread_ts": slack_event.get('thread_ts')
                }
            )

class ChannelManager:
    """Manages multiple channel adapters"""
    
    def __init__(self):
        self.adapters: Dict[str, ChannelAdapter] = {}
        self.channel_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_adapter(self, channel_name: str, adapter: ChannelAdapter):
        """Register a channel adapter"""
        self.adapters[channel_name] = adapter
    
    async def send_message(self, channel: str, message: ChannelMessage) -> bool:
        """Send message through specified channel"""
        adapter = self.adapters.get(channel)
        if not adapter:
            raise ValueError(f"Channel {channel} not configured")
        
        return await adapter.send_message(message)
    
    async def process_webhook(self, channel: str, webhook_data: Dict[str, Any]) -> List[ChannelMessage]:
        """Process webhook for specified channel"""
        adapter = self.adapters.get(channel)
        if not adapter:
            raise ValueError(f"Channel {channel} not configured")
        
        return await adapter.process_webhook(webhook_data)
    
    def get_supported_channels(self) -> List[str]:
        """Get list of configured channels"""
        return list(self.adapters.keys())
    
    def channel_supports_message_type(self, channel: str, message_type: MessageType) -> bool:
        """Check if channel supports message type"""
        adapter = self.adapters.get(channel)
        if not adapter:
            return False
        
        return adapter.supports_message_type(message_type)
```

### Web Widget Integration

```html
<!-- Web Widget Implementation -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot Widget</title>
    <style>
        /* Widget Styles */
        .chatbot-widget {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .widget-button {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        }
        
        .widget-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 25px rgba(0,0,0,0.2);
        }
        
        .chat-window {
            position: absolute;
            bottom: 80px;
            right: 0;
            width: 350px;
            height: 500px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            display: none;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.user {
            align-self: flex-end;
            background: #007AFF;
            color: white;
        }
        
        .message.bot {
            align-self: flex-start;
            background: #F2F2F7;
            color: #1C1C1E;
        }
        
        .chat-input {
            border-top: 1px solid #E5E5EA;
            padding: 16px;
            display: flex;
            gap: 8px;
        }
        
        .input-field {
            flex: 1;
            border: 1px solid #E5E5EA;
            border-radius: 20px;
            padding: 8px 16px;
            outline: none;
            font-size: 14px;
        }
        
        .send-button {
            background: #007AFF;
            color: white;
            border: none;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .quick-replies {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }
        
        .quick-reply {
            background: white;
            border: 1px solid #007AFF;
            color: #007AFF;
            padding: 6px 12px;
            border-radius: 16px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s ease;
        }
        
        .quick-reply:hover {
            background: #007AFF;
            color: white;
        }
        
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
            color: #8E8E93;
            font-style: italic;
        }
        
        .typing-dots {
            display: flex;
            gap: 2px;
        }
        
        .typing-dot {
            width: 4px;
            height: 4px;
            background: #8E8E93;
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="chatbot-widget">
        <button class="widget-button" onclick="toggleChat()">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
                <path d="M20 2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h4l4 4 4-4h4c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
            </svg>
        </button>
        
        <div class="chat-window" id="chatWindow">
            <div class="chat-header">
                <div>
                    <div style="font-weight: 600;">Support Assistant</div>
                    <div style="font-size: 12px; opacity: 0.8;">We're here to help!</div>
                </div>
                <button onclick="toggleChat()" style="background: none; border: none; color: white; font-size: 20px; cursor: pointer;">×</button>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message bot">
                    Hi! I'm here to help you. What can I assist you with today?
                </div>
            </div>
            
            <div class="chat-input">
                <input type="text" class="input-field" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                <button class="send-button" onclick="sendMessage()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                    </svg>
                </button>
            </div>
        </div>
    </div>

    <script>
        class ChatbotWidget {
            constructor(config) {
                this.config = config;
                this.isOpen = false;
                this.conversationId = null;
                this.userId = this.generateUserId();
                this.sessionId = this.generateSessionId();
                this.isTyping = false;
                
                this.apiEndpoint = config.apiEndpoint || '/api/v2/chat/message';
                this.tenantId = config.tenantId;
                this.apiKey = config.apiKey;
                
                this.init();
            }
            
            init() {
                // Initialize widget
                this.setupEventListeners();
                this.loadConversationHistory();
            }
            
            generateUserId() {
                // Generate or retrieve persistent user ID
                let userId = localStorage.getItem('chatbot_user_id');
                if (!userId) {
                    userId = 'user_' + Math.random().toString(36).substr(2, 9);
                    localStorage.setItem('chatbot_user_id', userId);
                }
                return userId;
            }
            
            generateSessionId() {
                return 'session_' + Math.random().toString(36).substr(2, 9);
            }
            
            setupEventListeners() {
                // Set up any additional event listeners
                window.addEventListener('beforeunload', () => {
                    this.saveConversationHistory();
                });
            }
            
            async sendMessage(text) {
                if (!text.trim()) return;
                
                // Add user message to UI
                this.addMessage(text, 'user');
                
                // Show typing indicator
                this.showTyping();
                
                try {
                    const response = await this.callChatAPI(text);
                    this.hideTyping();
                    
                    if (response.success) {
                        const botResponse = response.data.response;
                        this.addMessage(botResponse.text, 'bot');
                        
                        // Handle quick replies
                        if (botResponse.quick_replies) {
                            this.addQuickReplies(botResponse.quick_replies);
                        }
                        
                        // Update conversation ID
                        this.conversationId = response.data.conversation_id;
                    } else {
                        this.addMessage('Sorry, I encountered an error. Please try again.', 'bot');
                    }
                } catch (error) {
                    this.hideTyping();
                    this.addMessage('Sorry, I\'m having trouble connecting. Please try again later.', 'bot');
                    console.error('Chat API error:', error);
                }
            }
            
            async callChatAPI(text) {
                const payload = {
                    message_id: this.generateMessageId(),
                    conversation_id: this.conversationId,
                    user_id: this.userId,
                    session_id: this.sessionId,
                    channel: 'web',
                    timestamp: new Date().toISOString(),
                    content: {
                        type: 'text',
                        text: text,
                        language: navigator.language.split('-')[0] || 'en'
                    },
                    channel_metadata: {
                        user_agent: navigator.userAgent,
                        url: window.location.href,
                        referrer: document.referrer
                    }
                };
                
                const response = await fetch(this.apiEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Tenant-ID': this.tenantId,
                        'Authorization': this.apiKey ? `ApiKey ${this.apiKey}` : undefined
                    },
                    body: JSON.stringify(payload)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                return {
                    success: data.status === 'success',
                    data: data.data,
                    error: data.error
                };
            }
            
            generateMessageId() {
                return 'msg_' + Math.random().toString(36).substr(2, 9);
            }
            
            addMessage(text, sender) {
                const messagesContainer = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                messageDiv.textContent = text;
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            addQuickReplies(quickReplies) {
                const messagesContainer = document.getElementById('chatMessages');
                const quickRepliesDiv = document.createElement('div');
                quickRepliesDiv.className = 'quick-replies';
                
                quickReplies.forEach(reply => {
                    const button = document.createElement('button');
                    button.className = 'quick-reply';
                    button.textContent = reply.title;
                    button.onclick = () => {
                        this.sendMessage(reply.title);
                        quickRepliesDiv.remove();
                    };
                    quickRepliesDiv.appendChild(button);
                });
                
                messagesContainer.appendChild(quickRepliesDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            showTyping() {
                if (this.isTyping) return;
                
                this.isTyping = true;
                const messagesContainer = document.getElementById('chatMessages');
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message bot typing-indicator';
                typingDiv.id = 'typingIndicator';
                typingDiv.innerHTML = `
                    <span>Assistant is typing</span>
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                `;
                
                messagesContainer.appendChild(typingDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            hideTyping() {
                this.isTyping = false;
                const typingIndicator = document.getElementById('typingIndicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }
            
            saveConversationHistory() {
                const messages = Array.from(document.querySelectorAll('.message:not(.typing-indicator)')).map(msg => ({
                    text: msg.textContent,
                    sender: msg.classList.contains('user') ? 'user' : 'bot'
                }));
                
                localStorage.setItem('chatbot_conversation', JSON.stringify(messages));
                if (this.conversationId) {
                    localStorage.setItem('chatbot_conversation_id', this.conversationId);
                }
            }
            
            loadConversationHistory() {
                const savedMessages = localStorage.getItem('chatbot_conversation');
                const savedConversationId = localStorage.getItem('chatbot_conversation_id');
                
                if (savedMessages) {
                    const messages = JSON.parse(savedMessages);
                    messages.forEach(msg => {
                        this.addMessage(msg.text, msg.sender);
                    });
                }
                
                if (savedConversationId) {
                    this.conversationId = savedConversationId;
                }
            }
        }
        
        // Global functions for HTML integration
        let chatbot;
        
        function toggleChat() {
            const chatWindow = document.getElementById('chatWindow');
            const isCurrentlyOpen = chatWindow.style.display === 'flex';
            
            if (isCurrentlyOpen) {
                chatWindow.style.display = 'none';
            } else {
                chatWindow.style.display = 'flex';
                document.getElementById('messageInput').focus();
            }
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const text = input.value.trim();
            
            if (text) {
                chatbot.sendMessage(text);
                input.value = '';
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Initialize chatbot when page loads
        document.addEventListener('DOMContentLoaded', () => {
            // Configuration would typically come from the embedding website
            const config = {
                tenantId: window.CHATBOT_TENANT_ID || 'demo-tenant',
                apiKey: window.CHATBOT_API_KEY,
                apiEndpoint: window.CHATBOT_API_ENDPOINT || '/api/v2/chat/message'
            };
            
            chatbot = new ChatbotWidget(config);
        });
    </script>
</body>
</html>
```

---

## Business System Integration

### CRM Integration Patterns

#### Salesforce Integration

```python
class SalesforceIntegration:
    """Salesforce CRM integration with comprehensive functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.instance_url = config['instance_url']
        self.consumer_key = config['consumer_key']
        self.consumer_secret = config['consumer_secret']
        self.username = config['username']
        self.password = config['password']
        self.security_token = config['security_token']
        self.access_token = None
        self.token_expires_at = None
    
    async def authenticate(self) -> bool:
        """Authenticate with Salesforce using OAuth 2.0"""
        token_url = f"{self.instance_url}/services/oauth2/token"
        
        data = {
            'grant_type': 'password',
            'client_id': self.consumer_key,
            'client_secret': self.consumer_secret,
            'username': self.username,
            'password': self.password + self.security_token
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data['access_token']
                    self.instance_url = token_data['instance_url']
                    # Calculate expiration (Salesforce doesn't provide expires_in for username/password flow)
                    self.token_expires_at = datetime.utcnow() + timedelta(hours=2)
                    return True
                else:
                    raise AuthenticationError(f"Salesforce authentication failed: {response.status}")
    
    async def get_contact_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve contact information by email"""
        await self._ensure_authenticated()
        
        query = f"SELECT Id, FirstName, LastName, Email, Phone, Account.Name FROM Contact WHERE Email = '{email}' LIMIT 1"
        return await self._execute_soql_query(query)
    
    async def get_contact_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """Retrieve contact information by phone number"""
        await self._ensure_authenticated()
        
        # Clean phone number for search
        clean_phone = ''.join(filter(str.isdigit, phone))
        
        query = f"""
        SELECT Id, FirstName, LastName, Email, Phone, Account.Name 
        FROM Contact 
        WHERE Phone LIKE '%{clean_phone}%' OR MobilePhone LIKE '%{clean_phone}%'
        LIMIT 1
        """
        return await self._execute_soql_query(query)
    
    async def create_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new case in Salesforce"""
        await self._ensure_authenticated()
        
        url = f"{self.instance_url}/services/data/v58.0/sobjects/Case"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        # Map conversation data to Salesforce case fields
        salesforce_case = {
            'Subject': case_data.get('subject', 'Chatbot Inquiry'),
            'Description': case_data.get('description'),
            'Origin': 'Chat',
            'Priority': case_data.get('priority', 'Medium'),
            'Status': 'New',
            'ContactId': case_data.get('contact_id')
        }
        
        # Add custom fields if configured
        custom_fields = self.config.get('case_custom_fields', {})
        for field, value in custom_fields.items():
            salesforce_case[field] = value
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=salesforce_case) as response:
                if response.status == 201:
                    result = await response.json()
                    return {
                        'success': True,
                        'case_id': result['id'],
                        'case_number': await self._get_case_number(result['id'])
                    }
                else:
                    error_data = await response.json()
                    return {
                        'success': False,
                        'error': error_data
                    }
    
    async def update_case(self, case_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing case"""
        await self._ensure_authenticated()
        
        url = f"{self.instance_url}/services/data/v58.0/sobjects/Case/{case_id}"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.patch(url, headers=headers, json=updates) as response:
                return response.status == 204
    
    async def add_case_comment(self, case_id: str, comment: str, is_public: bool = True) -> bool:
        """Add a comment to a case"""
        await self._ensure_authenticated()
        
        url = f"{self.instance_url}/services/data/v58.0/sobjects/CaseComment"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        comment_data = {
            'ParentId': case_id,
            'CommentBody': comment,
            'IsPublished': is_public
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=comment_data) as response:
                return response.status == 201
    
    async def get_account_info(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get account information"""
        await self._ensure_authenticated()
        
        query = f"""
        SELECT Id, Name, Industry, Phone, Website, BillingAddress, 
               NumberOfEmployees, AnnualRevenue, Type
        FROM Account 
        WHERE Id = '{account_id}'
        """
        return await self._execute_soql_query(query)
    
    async def search_knowledge_base(self, search_term: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search Salesforce Knowledge articles"""
        await self._ensure_authenticated()
        
        # Use SOSL (Salesforce Object Search Language) for full-text search
        search_query = f"FIND '{search_term}' IN ALL FIELDS RETURNING KnowledgeArticleVersion(Id, Title, Summary, UrlName) LIMIT {limit}"
        
        url = f"{self.instance_url}/services/data/v58.0/search"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        params = {'q': search_query}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('searchRecords', [])
                else:
                    return []
    
    async def _execute_soql_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute SOQL query and return first result"""
        url = f"{self.instance_url}/services/data/v58.0/query"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        params = {'q': query}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    records = result.get('records', [])
                    return records[0] if records else None
                else:
                    return None
    
    async def _get_case_number(self, case_id: str) -> Optional[str]:
        """Get case number for a case ID"""
        query = f"SELECT CaseNumber FROM Case WHERE Id = '{case_id}'"
        result = await self._execute_soql_query(query)
        return result.get('CaseNumber') if result else None
    
    async def _ensure_authenticated(self):
        """Ensure we have a valid access token"""
        if not self.access_token or (self.token_expires_at and datetime.utcnow() >= self.token_expires_at):
            await self.authenticate()

# Usage example for conversation flow integration
async def handle_customer_inquiry(conversation_context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle customer inquiry with Salesforce integration"""
    
    # Extract customer information from conversation
    user_email = conversation_context.get('user_email')
    user_phone = conversation_context.get('user_phone')
    inquiry_type = conversation_context.get('intent')
    message_text = conversation_context.get('message_text')
    
    salesforce = SalesforceIntegration(SALESFORCE_CONFIG)
    
    # Try to find existing contact
    contact = None
    if user_email:
        contact = await salesforce.get_contact_by_email(user_email)
    elif user_phone:
        contact = await salesforce.get_contact_by_phone(user_phone)
    
    if contact:
        # Existing customer
        response_context = {
            'customer_found': True,
            'customer_name': f"{contact.get('FirstName', '')} {contact.get('LastName', '')}".strip(),
            'account_name': contact.get('Account', {}).get('Name'),
            'contact_id': contact['Id']
        }
        
        # Create case for the inquiry
        case_data = {
            'subject': f"Chatbot Inquiry - {inquiry_type}",
            'description': message_text,
            'contact_id': contact['Id'],
            'priority': 'Medium'
        }
        
        case_result = await salesforce.create_case(case_data)
        if case_result['success']:
            response_context['case_created'] = True
            response_context['case_number'] = case_result['case_number']
            response_context['case_id'] = case_result['case_id']
        
        return response_context
    else:
        # New customer or no contact info provided
        return {
            'customer_found': False,
            'needs_contact_info': not (user_email or user_phone)
        }
```

### E-commerce Integration Patterns

#### Shopify Integration

```python
class ShopifyIntegration:
    """Shopify e-commerce platform integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shop_domain = config['shop_domain']
        self.access_token = config['access_token']
        self.api_version = config.get('api_version', '2023-10')
        self.base_url = f"https://{self.shop_domain}/admin/api/{self.api_version}"
    
    async def get_order_by_number(self, order_number: str) -> Optional[Dict[str, Any]]:
        """Get order details by order number"""
        url = f"{self.base_url}/orders.json"
        headers = {
            'X-Shopify-Access-Token': self.access_token,
            'Content-Type': 'application/json'
        }
        
        params = {
            'name': order_number,
            'status': 'any'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    orders = data.get('orders', [])
                    return orders[0] if orders else None
                else:
                    return None
    
    async def get_order_by_email(self, email: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent orders for a customer by email"""
        url = f"{self.base_url}/orders.json"
        headers = {
            'X-Shopify-Access-Token': self.access_token,
            'Content-Type': 'application/json'
        }
        
        params = {
            'email': email,
            'limit': limit,
            'status': 'any'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('orders', [])
                else:
                    return []
    
    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product details by product ID"""
        url = f"{self.base_url}/products/{product_id}.json"
        headers = {
            'X-Shopify-Access-Token': self.access_token,
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('product')
                else:
                    return None
    
    async def search_products(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search products by title or tags"""
        url = f"{self.base_url}/products.json"
        headers = {
            'X-Shopify-Access-Token': self.access_token,
            'Content-Type': 'application/json'
        }
        
        params = {
            'title': query,
            'limit': limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('products', [])
                else:
                    return []
    
    async def get_customer_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get customer information by email"""
        url = f"{self.base_url}/customers/search.json"
        headers = {
            'X-Shopify-Access-Token': self.access_token,
            'Content-Type': 'application/json'
        }
        
        params = {
            'query': f'email:{email}'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    customers = data.get('customers', [])
                    return customers[0] if customers else None
                else:
                    return None
    
    async def create_draft_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a draft order for assisted purchasing"""
        url = f"{self.base_url}/draft_orders.json"
        headers = {
            'X-Shopify-Access-Token': self.access_token,
            'Content-Type': 'application/json'
        }
        
        draft_order = {
            'draft_order': {
                'line_items': order_data.get('line_items', []),
                'customer': order_data.get('customer'),
                'shipping_address': order_data.get('shipping_address'),
                'billing_address': order_data.get('billing_address'),
                'note': order_data.get('note', 'Created via chatbot'),
                'tags': 'chatbot-assisted'
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=draft_order) as response:
                if response.status == 201:
                    data = await response.json()
                    return {
                        'success': True,
                        'draft_order': data['draft_order'],
                        'invoice_url': data['draft_order']['invoice_url']
                    }
                else:
                    error_data = await response.json()
                    return {
                        'success': False,
                        'error': error_data
                    }
    
    async def get_shipping_rates(self, address: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get available shipping rates for address and items"""
        # This would integrate with Shopify's shipping API
        # Implementation depends on specific shipping configuration
        return []
    
    def format_order_info(self, order: Dict[str, Any]) -> str:
        """Format order information for display in chat"""
        if not order:
            return "Order not found."
        
        order_number = order.get('name', 'Unknown')
        status = order.get('fulfillment_status', 'unfulfilled')
        financial_status = order.get('financial_status', 'pending')
        total_price = order.get('total_price', '0.00')
        currency = order.get('currency', 'USD')
        
        # Format order items
        line_items = order.get('line_items', [])
        items_text = "\n".join([
            f"• {item['title']} (Qty: {item['quantity']}) - {currency} {item['price']}"
            for item in line_items[:3]  # Show first 3 items
        ])
        
        if len(line_items) > 3:
            items_text += f"\n... and {len(line_items) - 3} more items"
        
        order_info = f"""
**Order {order_number}**
Status: {status.title()}
Payment: {financial_status.title()}
Total: {currency} {total_price}

Items:
{items_text}
        """.strip()
        
        return order_info
    
    def format_product_info(self, product: Dict[str, Any]) -> str:
        """Format product information for display in chat"""
        if not product:
            return "Product not found."
        
        title = product.get('title', 'Unknown Product')
        description = product.get('body_html', '')
        # Strip HTML tags from description
        import re
        description = re.sub('<[^<]+?>', '', description)[:200] + "..." if len(description) > 200 else description
        
        # Get price from first variant
        variants = product.get('variants', [])
        price_info = ""
        if variants:
            first_variant = variants[0]
            price = first_variant.get('price', '0.00')
            currency = 'USD'  # This would come from shop settings
            price_info = f"Price: {currency} {price}"
        
        product_info = f"""
**{title}**
{price_info}

{description}
        """.strip()
        
        return product_info

# Integration usage in conversation flow
async def handle_order_inquiry(conversation_context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle order status inquiry with Shopify integration"""
    
    shopify = ShopifyIntegration(SHOPIFY_CONFIG)
    
    # Try to extract order number from message
    message_text = conversation_context.get('message_text', '')
    order_number = extract_order_number(message_text)
    
    if order_number:
        order = await shopify.get_order_by_number(order_number)
        if order:
            return {
                'order_found': True,
                'order_info': shopify.format_order_info(order),
                'order_status': order.get('fulfillment_status'),
                'tracking_number': order.get('tracking_number')
            }
    
    # If no order number or not found, try email
    user_email = conversation_context.get('user_email')
    if user_email:
        orders = await shopify.get_order_by_email(user_email)
        if orders:
            return {
                'customer_orders_found': True,
                'recent_orders': [shopify.format_order_info(order) for order in orders[:3]],
                'order_count': len(orders)
            }
    
    return {
        'order_found': False,
        'needs_order_number': not order_number,
        'needs_email': not user_email
    }

def extract_order_number(text: str) -> Optional[str]:
    """Extract order number from text using regex patterns"""
    import re
    
    # Common order number patterns
    patterns = [
        r'#(\d+)',  # #12345
        r'order\s*#?(\d+)',  # order 12345 or order #12345
        r'(\d{4,8})',  # 4-8 digit number
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    return None
```

---

## AI/ML Service Integration

### Multi-Provider LLM Integration

#### Model Provider Abstraction

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
import json

class ModelCapability(Enum):
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"

@dataclass
class ModelRequest:
    """Standardized model request"""
    messages: List[Dict[str, str]]
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    
    # Request metadata
    request_id: str = ""
    user_id: str = ""
    tenant_id: str = ""

@dataclass
class ModelResponse:
    """Standardized model response"""
    content: str
    finish_reason: str
    model_used: str
    provider: str
    
    # Usage statistics
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Cost and performance
    cost_cents: float
    latency_ms: int
    
    # Quality metrics
    confidence_score: Optional[float] = None
    safety_score: Optional[float] = None
    
    # Provider-specific metadata
    provider_metadata: Dict[str, Any] = None

class ModelProvider(ABC):
    """Base class for AI model providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = config['provider_name']
        self.api_key = config['api_key']
        self.base_url = config.get('base_url')
        self.rate_limiter = config.get('rate_limiter')
    
    @abstractmethod
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate a response from the model"""
        pass
    
    @abstractmethod
    async def stream_response(self, request: ModelRequest) -> AsyncIterator[str]:
        """Stream response chunks from the model"""
        pass
    
    @abstractmethod
    def get_supported_capabilities(self) -> List[ModelCapability]:
        """Get list of capabilities supported by this provider"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
        """Calculate cost in cents for token usage"""
        pass
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if provider supports a specific capability"""
        return capability in self.get_supported_capabilities()

class OpenAIProvider(ModelProvider):
    """OpenAI API integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.organization = config.get('organization')
        
        # Token pricing per 1K tokens (in cents)
        self.pricing = {
            'gpt-4-turbo': {'input': 1.0, 'output': 3.0},
            'gpt-4': {'input': 3.0, 'output': 6.0},
            'gpt-3.5-turbo': {'input': 0.05, 'output': 0.15},
            'gpt-3.5-turbo-16k': {'input': 0.3, 'output': 0.4}
        }
    
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using OpenAI API"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        if self.organization:
            headers['OpenAI-Organization'] = self.organization
        
        payload = {
            'model': request.model_name,
            'messages': request.messages,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'frequency_penalty': request.frequency_penalty,
            'presence_penalty': request.presence_penalty,
            'stream': False
        }
        
        if request.max_tokens:
            payload['max_tokens'] = request.max_tokens
        
        if request.stop_sequences:
            payload['stop'] = request.stop_sequences
        
        if request.functions:
            payload['functions'] = request.functions
            payload['function_call'] = 'auto'
        
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    end_time = asyncio.get_event_loop().time()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    choice = data['choices'][0]
                    usage = data['usage']
                    
                    cost_cents = self.calculate_cost(
                        usage['prompt_tokens'],
                        usage['completion_tokens'],
                        request.model_name
                    )
                    
                    return ModelResponse(
                        content=choice['message']['content'],
                        finish_reason=choice['finish_reason'],
                        model_used=data['model'],
                        provider='openai',
                        prompt_tokens=usage['prompt_tokens'],
                        completion_tokens=usage['completion_tokens'],
                        total_tokens=usage['total_tokens'],
                        cost_cents=cost_cents,
                        latency_ms=latency_ms,
                        provider_metadata={
                            'openai_response_id': data.get('id'),
                            'created': data.get('created')
                        }
                    )
                else:
                    error_data = await response.json()
                    raise ModelProviderError(f"OpenAI API error: {error_data}")
    
    async def stream_response(self, request: ModelRequest) -> AsyncIterator[str]:
        """Stream response chunks from OpenAI"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': request.model_name,
            'messages': request.messages,
            'temperature': request.temperature,
            'stream': True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                choice = data['choices'][0]
                                delta = choice.get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
                else:
                    error_data = await response.json()
                    raise ModelProviderError(f"OpenAI streaming error: {error_data}")
    
    def get_supported_capabilities(self) -> List[ModelCapability]:
        return [
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STREAMING
        ]
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'gpt-4-turbo',
                'description': 'Most capable GPT-4 model',
                'max_tokens': 4096,
                'context_length': 128000,
                'capabilities': ['chat', 'function_calling']
            },
            {
                'name': 'gpt-4',
                'description': 'Standard GPT-4 model',
                'max_tokens': 4096,
                'context_length': 8192,
                'capabilities': ['chat', 'function_calling']
            },
            {
                'name': 'gpt-3.5-turbo',
                'description': 'Fast and cost-effective model',
                'max_tokens': 4096,
                'context_length': 16384,
                'capabilities': ['chat', 'function_calling']
            }
        ]
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
        """Calculate cost in cents"""
        if model_name not in self.pricing:
            return 0.0
        
        pricing = self.pricing[model_name]
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost

class AnthropicProvider(ModelProvider):
    """Anthropic Claude API integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://api.anthropic.com/v1')
        
        # Token pricing per 1K tokens (in cents)
        self.pricing = {
            'claude-3-5-sonnet-20241022': {'input': 0.3, 'output': 1.5},
            'claude-3-haiku-20240307': {'input': 0.025, 'output': 0.125},
            'claude-3-opus-20240229': {'input': 1.5, 'output': 7.5}
        }
    
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using Anthropic API"""
        url = f"{self.base_url}/messages"
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        # Convert OpenAI-style messages to Anthropic format
        system_message = ""
        user_messages = []
        
        for msg in request.messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                user_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        payload = {
            'model': request.model_name,
            'messages': user_messages,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'stream': False
        }
        
        if system_message:
            payload['system'] = system_message
        
        if request.max_tokens:
            payload['max_tokens'] = request.max_tokens
        else:
            payload['max_tokens'] = 4096  # Required for Anthropic
        
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    end_time = asyncio.get_event_loop().time()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    content = data['content'][0]['text']
                    usage = data['usage']
                    
                    cost_cents = self.calculate_cost(
                        usage['input_tokens'],
                        usage['output_tokens'],
                        request.model_name
                    )
                    
                    return ModelResponse(
                        content=content,
                        finish_reason=data['stop_reason'],
                        model_used=data['model'],
                        provider='anthropic',
                        prompt_tokens=usage['input_tokens'],
                        completion_tokens=usage['output_tokens'],
                        total_tokens=usage['input_tokens'] + usage['output_tokens'],
                        cost_cents=cost_cents,
                        latency_ms=latency_ms,
                        provider_metadata={
                            'anthropic_response_id': data.get('id'),
                            'anthropic_type': data.get('type')
                        }
                    )
                else:
                    error_data = await response.json()
                    raise ModelProviderError(f"Anthropic API error: {error_data}")
    
    async def stream_response(self, request: ModelRequest) -> AsyncIterator[str]:
        """Stream response chunks from Anthropic"""
        # Similar implementation to OpenAI but with Anthropic's streaming format
        url = f"{self.base_url}/messages"
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        # Convert messages and create payload
        payload = {
            'model': request.model_name,
            'messages': self._convert_messages(request.messages),
            'temperature': request.temperature,
            'max_tokens': request.max_tokens or 4096,
            'stream': True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                if data.get('type') == 'content_block_delta':
                                    text = data.get('delta', {}).get('text', '')
                                    if text:
                                        yield text
                            except json.JSONDecodeError:
                                continue
                else:
                    error_data = await response.json()
                    raise ModelProviderError(f"Anthropic streaming error: {error_data}")
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert OpenAI-style messages to Anthropic format"""
        converted = []
        for msg in messages:
            if msg['role'] != 'system':  # System messages handled separately
                converted.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        return converted
    
    def get_supported_capabilities(self) -> List[ModelCapability]:
        return [
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.STREAMING
        ]
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'claude-3-5-sonnet-20241022',
                'description': 'Most intelligent model',
                'max_tokens': 8192,
                'context_length': 200000,
                'capabilities': ['chat', 'analysis']
            },
            {
                'name': 'claude-3-haiku-20240307',
                'description': 'Fast and efficient model',
                'max_tokens': 4096,
                'context_length': 200000,
                'capabilities': ['chat']
            },
            {
                'name': 'claude-3-opus-20240229',
                'description': 'Most powerful model for complex tasks',
                'max_tokens': 4096,
                'context_length': 200000,
                'capabilities': ['chat', 'analysis', 'reasoning']
            }
        ]
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
        """Calculate cost in cents"""
        if model_name not in self.pricing:
            return 0.0
        
        pricing = self.pricing[model_name]
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost

class ModelOrchestrator:
    """Intelligent model orchestration and routing"""
    
    def __init__(self, providers: Dict[str, ModelProvider], config: Dict[str, Any]):
        self.providers = providers
        self.config = config
        self.fallback_chains = config.get('fallback_chains', {})
        self.cost_optimizer = config.get('cost_optimizer', True)
        self.performance_tracker = config.get('performance_tracker')
    
    async def process_request(self, request: ModelRequest, routing_config: Dict[str, Any] = None) -> ModelResponse:
        """Process request with intelligent routing"""
        routing_config = routing_config or {}
        
        # Determine optimal provider and model
        provider_chain = self._build_provider_chain(request, routing_config)
        
        last_error = None
        for provider_name, model_name in provider_chain:
            try:
                provider = self.providers[provider_name]
                
                # Update request with specific model
                request.model_name = model_name
                
                # Check rate limits and availability
                if not await self._check_provider_availability(provider_name):
                    continue
                
                # Execute request
                response = await provider.generate_response(request)
                
                # Track success
                await self._track_request_success(provider_name, model_name, response)
                
                return response
                
            except Exception as e:
                last_error = e
                await self._track_request_failure(provider_name, model_name, str(e))
                continue
        
        # All providers failed
        raise ModelProviderError(f"All providers failed. Last error: {last_error}")
    
    def _build_provider_chain(self, request: ModelRequest, routing_config: Dict[str, Any]) -> List[tuple]:
        """Build ordered list of (provider, model) to try"""
        chain = []
        
        # Primary choice based on routing configuration
        primary_provider = routing_config.get('primary_provider')
        primary_model = routing_config.get('primary_model')
        
        if primary_provider and primary_model:
            chain.append((primary_provider, primary_model))
        
        # Add fallback options
        fallback_providers = routing_config.get('fallback_providers', [])
        for fallback in fallback_providers:
            if isinstance(fallback, str):
                # Provider name only - use default model
                default_model = self._get_default_model(fallback)
                if default_model:
                    chain.append((fallback, default_model))
            elif isinstance(fallback, dict):
                # Provider and model specified
                provider = fallback.get('provider')
                model = fallback.get('model')
                if provider and model:
                    chain.append((provider, model))
        
        # Add cost-optimized options if enabled
        if self.cost_optimizer and routing_config.get('cost_optimization'):
            cost_effective_options = self._get_cost_effective_options(request)
            chain.extend(cost_effective_options)
        
        return chain
    
    def _get_default_model(self, provider_name: str) -> Optional[str]:
        """Get default model for provider"""
        provider = self.providers.get(provider_name)
        if not provider:
            return None
        
        models = provider.get_available_models()
        return models[0]['name'] if models else None
    
    def _get_cost_effective_options(self, request: ModelRequest) -> List[tuple]:
        """Get cost-effective provider/model combinations"""
        options = []
        
        # Estimate costs for different providers
        for provider_name, provider in self.providers.items():
            models = provider.get_available_models()
            for model in models:
                # Estimate cost based on average token usage
                estimated_prompt_tokens = len(str(request.messages)) // 4  # Rough estimate
                estimated_completion_tokens = request.max_tokens or 500
                
                cost = provider.calculate_cost(
                    estimated_prompt_tokens,
                    estimated_completion_tokens,
                    model['name']
                )
                
                options.append((cost, provider_name, model['name']))
        
        # Sort by cost and return provider/model pairs
        options.sort(key=lambda x: x[0])
        return [(provider, model) for _, provider, model in options[:3]]
    
    async def _check_provider_availability(self, provider_name: str) -> bool:
        """Check if provider is available (not rate limited, not in circuit breaker open state)"""
        # This would check circuit breaker state, rate limits, etc.
        return True
    
    async def _track_request_success(self, provider: str, model: str, response: ModelResponse):
        """Track successful request for performance monitoring"""
        if self.performance_tracker:
            await self.performance_tracker.record_success(provider, model, response)
    
    async def _track_request_failure(self, provider: str, model: str, error: str):
        """Track failed request for performance monitoring"""
        if self.performance_tracker:
            await self.performance_tracker.record_failure(provider, model, error)

class ModelProviderError(Exception):
    """Exception raised by model providers"""
    pass

---

## Integration Security Patterns

### Secure Configuration Management

#### Credential Encryption and Storage

```python
import base64
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecureCredentialManager:
    """Secure management of integration credentials"""
    
    def __init__(self, master_key: bytes):
        self.fernet = Fernet(master_key)
        self.credential_store = {}  # In production, this would be a secure database
    
    async def store_credentials(self, integration_id: str, tenant_id: str, 
                               credentials: Dict[str, Any]) -> str:
        """Store encrypted credentials for an integration"""
        
        # Generate unique credential ID
        credential_id = f"{integration_id}:{tenant_id}:{secrets.token_hex(8)}"
        
        # Encrypt sensitive fields
        encrypted_credentials = {}
        sensitive_fields = ['api_key', 'client_secret', 'password', 'private_key', 'access_token']
        
        for key, value in credentials.items():
            if key in sensitive_fields and value:
                encrypted_credentials[key] = self.fernet.encrypt(str(value).encode()).decode()
            else:
                encrypted_credentials[key] = value
        
        # Store with metadata
        self.credential_store[credential_id] = {
            'credentials': encrypted_credentials,
            'integration_id': integration_id,
            'tenant_id': tenant_id,
            'created_at': datetime.utcnow().isoformat(),
            'last_accessed': None,
            'access_count': 0
        }
        
        return credential_id
    
    async def retrieve_credentials(self, credential_id: str) -> Dict[str, Any]:
        """Retrieve and decrypt credentials"""
        
        if credential_id not in self.credential_store:
            raise ValueError("Credentials not found")
        
        stored_data = self.credential_store[credential_id]
        encrypted_credentials = stored_data['credentials']
        
        # Decrypt sensitive fields
        decrypted_credentials = {}
        sensitive_fields = ['api_key', 'client_secret', 'password', 'private_key', 'access_token']
        
        for key, value in encrypted_credentials.items():
            if key in sensitive_fields and value:
                try:
                    decrypted_credentials[key] = self.fernet.decrypt(value.encode()).decode()
                except Exception:
                    # If decryption fails, return None
                    decrypted_credentials[key] = None
            else:
                decrypted_credentials[key] = value
        
        # Update access tracking
        stored_data['last_accessed'] = datetime.utcnow().isoformat()
        stored_data['access_count'] += 1
        
        return decrypted_credentials
    
    async def rotate_credentials(self, credential_id: str, new_credentials: Dict[str, Any]) -> bool:
        """Rotate credentials for an integration"""
        
        if credential_id not in self.credential_store:
            return False
        
        stored_data = self.credential_store[credential_id]
        
        # Store old credentials for rollback
        old_credentials = stored_data['credentials'].copy()
        
        # Encrypt and store new credentials
        encrypted_credentials = {}
        sensitive_fields = ['api_key', 'client_secret', 'password', 'private_key', 'access_token']
        
        for key, value in new_credentials.items():
            if key in sensitive_fields and value:
                encrypted_credentials[key] = self.fernet.encrypt(str(value).encode()).decode()
            else:
                encrypted_credentials[key] = value
        
        stored_data['credentials'] = encrypted_credentials
        stored_data['previous_credentials'] = old_credentials
        stored_data['rotated_at'] = datetime.utcnow().isoformat()
        
        return True
    
    async def revoke_credentials(self, credential_id: str) -> bool:
        """Revoke credentials (mark as inactive)"""
        
        if credential_id not in self.credential_store:
            return False
        
        stored_data = self.credential_store[credential_id]
        stored_data['revoked'] = True
        stored_data['revoked_at'] = datetime.utcnow().isoformat()
        
        return True

class IntegrationSecurityManager:
    """Comprehensive security management for integrations"""
    
    def __init__(self, credential_manager: SecureCredentialManager):
        self.credential_manager = credential_manager
        self.access_logs = []
        self.security_policies = {}
    
    def validate_integration_request(self, request: IntegrationRequest, 
                                   integration_config: IntegrationConfig) -> bool:
        """Validate integration request against security policies"""
        
        # Check tenant access permissions
        if not self._check_tenant_permissions(request.tenant_id, integration_config.integration_id):
            return False
        
        # Validate request parameters
        if not self._validate_request_parameters(request, integration_config):
            return False
        
        # Check rate limiting
        if not self._check_rate_limits(request.tenant_id, integration_config.integration_id):
            return False
        
        # Validate data sensitivity
        if not self._validate_data_sensitivity(request, integration_config):
            return False
        
        return True
    
    def _check_tenant_permissions(self, tenant_id: str, integration_id: str) -> bool:
        """Check if tenant has permission to use integration"""
        # Implementation would check tenant's subscription level,
        # integration marketplace permissions, etc.
        return True
    
    def _validate_request_parameters(self, request: IntegrationRequest, 
                                   config: IntegrationConfig) -> bool:
        """Validate request parameters against integration schema"""
        endpoint_config = config.endpoints.get(request.endpoint_name)
        if not endpoint_config:
            return False
        
        # Validate required parameters
        required_params = endpoint_config.get('required_parameters', [])
        for param in required_params:
            if param not in request.parameters:
                return False
        
        # Validate parameter types and formats
        param_schemas = endpoint_config.get('parameter_schemas', {})
        for param, value in request.parameters.items():
            if param in param_schemas:
                schema = param_schemas[param]
                if not self._validate_parameter_value(value, schema):
                    return False
        
        return True
    
    def _validate_parameter_value(self, value: Any, schema: Dict[str, Any]) -> bool:
        """Validate parameter value against schema"""
        param_type = schema.get('type')
        
        if param_type == 'string':
            if not isinstance(value, str):
                return False
            
            min_length = schema.get('min_length', 0)
            max_length = schema.get('max_length', float('inf'))
            if not (min_length <= len(value) <= max_length):
                return False
            
            pattern = schema.get('pattern')
            if pattern:
                import re
                if not re.match(pattern, value):
                    return False
        
        elif param_type == 'integer':
            if not isinstance(value, int):
                return False
            
            minimum = schema.get('minimum', float('-inf'))
            maximum = schema.get('maximum', float('inf'))
            if not (minimum <= value <= maximum):
                return False
        
        elif param_type == 'email':
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
            if not re.match(email_pattern, str(value)):
                return False
        
        return True
    
    def _check_rate_limits(self, tenant_id: str, integration_id: str) -> bool:
        """Check integration-specific rate limits"""
        # Implementation would check Redis-based rate limiting
        return True
    
    def _validate_data_sensitivity(self, request: IntegrationRequest, 
                                 config: IntegrationConfig) -> bool:
        """Validate that sensitive data is properly handled"""
        
        # Check for PII in request parameters
        pii_detector = PIIDetector()
        for param, value in request.parameters.items():
            if isinstance(value, str):
                detected_pii = pii_detector.detect_pii(value)
                if detected_pii:
                    # Log PII detection
                    self._log_pii_detection(request.tenant_id, integration_id, param, detected_pii)
                    
                    # Check if integration allows PII
                    if not config.configuration.get('allows_pii', False):
                        return False
        
        return True
    
    def _log_pii_detection(self, tenant_id: str, integration_id: str, 
                          parameter: str, pii_types: List[str]):
        """Log PII detection for compliance"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'tenant_id': tenant_id,
            'integration_id': integration_id,
            'parameter': parameter,
            'pii_types': pii_types,
            'action': 'detected'
        }
        self.access_logs.append(log_entry)
    
    async def audit_integration_access(self, tenant_id: str, 
                                     time_range: tuple) -> List[Dict[str, Any]]:
        """Generate audit report for integration access"""
        start_time, end_time = time_range
        
        audit_entries = []
        for log_entry in self.access_logs:
            log_time = datetime.fromisoformat(log_entry['timestamp'])
            if start_time <= log_time <= end_time and log_entry['tenant_id'] == tenant_id:
                audit_entries.append(log_entry)
        
        return audit_entries

class PIIDetector:
    """Detect personally identifiable information in text"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
    
    def detect_pii(self, text: str) -> List[str]:
        """Detect PII types in text"""
        import re
        detected_types = []
        
        for pii_type, pattern in self.patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_types.append(pii_type)
        
        return detected_types
```

### API Security Patterns

#### Request Signing and Verification

```python
import hmac
import hashlib
import time
from typing import Dict, Optional

class RequestSigner:
    """Sign and verify API requests for enhanced security"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
    
    def sign_request(self, method: str, url: str, body: str, 
                    timestamp: Optional[int] = None) -> Dict[str, str]:
        """Generate signature for API request"""
        
        if timestamp is None:
            timestamp = int(time.time())
        
        # Create string to sign
        string_to_sign = f"{method}\n{url}\n{body}\n{timestamp}"
        
        # Generate signature
        signature = hmac.new(
            self.secret_key,
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'X-Signature': signature,
            'X-Timestamp': str(timestamp)
        }
    
    def verify_signature(self, method: str, url: str, body: str,
                        signature: str, timestamp: str,
                        max_age_seconds: int = 300) -> bool:
        """Verify request signature"""
        
        # Check timestamp to prevent replay attacks
        current_time = int(time.time())
        request_time = int(timestamp)
        
        if abs(current_time - request_time) > max_age_seconds:
            return False
        
        # Generate expected signature
        string_to_sign = f"{method}\n{url}\n{body}\n{timestamp}"
        expected_signature = hmac.new(
            self.secret_key,
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures securely
        return hmac.compare_digest(signature, expected_signature)

class IntegrationFirewall:
    """Firewall rules for integration requests"""
    
    def __init__(self):
        self.rules = []
        self.blocked_ips = set()
        self.rate_limits = {}
    
    def add_rule(self, rule: Dict[str, Any]):
        """Add firewall rule"""
        self.rules.append(rule)
    
    def check_request(self, request_info: Dict[str, Any]) -> bool:
        """Check if request is allowed by firewall rules"""
        
        # Check IP blocking
        client_ip = request_info.get('client_ip')
        if client_ip in self.blocked_ips:
            return False
        
        # Check custom rules
        for rule in self.rules:
            if not self._evaluate_rule(rule, request_info):
                return False
        
        return True
    
    def _evaluate_rule(self, rule: Dict[str, Any], request_info: Dict[str, Any]) -> bool:
        """Evaluate a single firewall rule"""
        
        rule_type = rule.get('type')
        
        if rule_type == 'ip_whitelist':
            allowed_ips = rule.get('allowed_ips', [])
            return request_info.get('client_ip') in allowed_ips
        
        elif rule_type == 'user_agent_filter':
            blocked_agents = rule.get('blocked_user_agents', [])
            user_agent = request_info.get('user_agent', '')
            return not any(blocked in user_agent for blocked in blocked_agents)
        
        elif rule_type == 'geographic_filter':
            blocked_countries = rule.get('blocked_countries', [])
            country = request_info.get('country')
            return country not in blocked_countries
        
        elif rule_type == 'time_based':
            allowed_hours = rule.get('allowed_hours', list(range(24)))
            current_hour = datetime.utcnow().hour
            return current_hour in allowed_hours
        
        return True
    
    def block_ip(self, ip_address: str, duration_seconds: Optional[int] = None):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        
        if duration_seconds:
            # Schedule unblocking (in production, use a proper scheduler)
            import threading
            def unblock():
                time.sleep(duration_seconds)
                self.blocked_ips.discard(ip_address)
            
            threading.Thread(target=unblock, daemon=True).start()
```

---

## Error Handling and Resilience

### Circuit Breaker Pattern

```python
import asyncio
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    timeout_duration: int = 60
    success_threshold: int = 3
    monitor_window: int = 300

class CircuitBreaker:
    """Circuit breaker implementation for integration resilience"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        self.failure_times = []
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        
        if not self.can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure()
            raise e
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            if self.next_attempt_time and current_time >= self.next_attempt_time:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    async def record_success(self):
        """Record successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.failure_times = []
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on successful execution
            self.failure_count = max(0, self.failure_count - 1)
    
    async def record_failure(self):
        """Record failed execution"""
        current_time = time.time()
        self.last_failure_time = current_time
        self.failure_times.append(current_time)
        
        # Clean old failure times outside monitor window
        cutoff_time = current_time - self.config.monitor_window
        self.failure_times = [t for t in self.failure_times if t > cutoff_time]
        
        self.failure_count = len(self.failure_times)
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.next_attempt_time = current_time + self.config.timeout_duration
        
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.next_attempt_time = current_time + self.config.timeout_duration
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'next_attempt_time': self.next_attempt_time
        }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerManager:
    """Manage multiple circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            circuit_config = config or self.default_config
            self.circuit_breakers[name] = CircuitBreaker(name, circuit_config)
        
        return self.circuit_breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {name: cb.get_status() for name, cb in self.circuit_breakers.items()}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all circuit breakers"""
        total_breakers = len(self.circuit_breakers)
        open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN)
        half_open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.HALF_OPEN)
        
        return {
            'total_circuit_breakers': total_breakers,
            'open_circuit_breakers': open_breakers,
            'half_open_circuit_breakers': half_open_breakers,
            'overall_health': 'healthy' if open_breakers == 0 else 'degraded' if open_breakers < total_breakers / 2 else 'unhealthy'
        }
```

### Retry Strategies

```python
import asyncio
import random
from typing import Callable, Any, List, Optional
from dataclasses import dataclass

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[type] = None

class RetryStrategy:
    """Advanced retry strategies for integration calls"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.retryable_exceptions = config.retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            aiohttp.ClientError
        ]
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs)
                return result
            
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    raise e
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry"""
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add jitter to prevent thundering herd
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)

class BulkheadPattern:
    """Bulkhead pattern for resource isolation"""
    
    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_calls = 0
        self.total_calls = 0
        self.rejected_calls = 0
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead protection"""
        self.total_calls += 1
        
        if self.semaphore.locked():
            self.rejected_calls += 1
            raise BulkheadRejectError("Resource pool exhausted")
        
        async with self.semaphore:
            self.active_calls += 1
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                self.active_calls -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics"""
        return {
            'active_calls': self.active_calls,
            'total_calls': self.total_calls,
            'rejected_calls': self.rejected_calls,
            'rejection_rate': self.rejected_calls / self.total_calls if self.total_calls > 0 else 0
        }

class BulkheadRejectError(Exception):
    """Exception raised when bulkhead rejects request"""
    pass

class IntegrationResilience:
    """Comprehensive resilience patterns for integrations"""
    
    def __init__(self):
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.bulkheads = {}
        self.retry_configs = {}
    
    def configure_integration(self, integration_id: str, 
                            circuit_config: Optional[CircuitBreakerConfig] = None,
                            retry_config: Optional[RetryConfig] = None,
                            bulkhead_size: Optional[int] = None):
        """Configure resilience patterns for an integration"""
        
        # Configure circuit breaker
        if circuit_config:
            self.circuit_breaker_manager.get_circuit_breaker(integration_id, circuit_config)
        
        # Configure retry strategy
        if retry_config:
            self.retry_configs[integration_id] = RetryStrategy(retry_config)
        
        # Configure bulkhead
        if bulkhead_size:
            self.bulkheads[integration_id] = BulkheadPattern(bulkhead_size)
    
    async def execute_integration_call(self, integration_id: str, func: Callable, 
                                     *args, **kwargs) -> Any:
        """Execute integration call with all resilience patterns"""
        
        # Get components
        circuit_breaker = self.circuit_breaker_manager.get_circuit_breaker(integration_id)
        retry_strategy = self.retry_configs.get(integration_id)
        bulkhead = self.bulkheads.get(integration_id)
        
        # Define execution function
        async def execute():
            if bulkhead:
                return await bulkhead.execute(func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        # Execute with circuit breaker
        async def circuit_breaker_execute():
            return await circuit_breaker.call(execute)
        
        # Execute with retry if configured
        if retry_strategy:
            return await retry_strategy.execute_with_retry(circuit_breaker_execute)
        else:
            return await circuit_breaker_execute()
    
    def get_resilience_status(self, integration_id: str) -> Dict[str, Any]:
        """Get resilience status for integration"""
        status = {}
        
        # Circuit breaker status
        if integration_id in self.circuit_breaker_manager.circuit_breakers:
            status['circuit_breaker'] = self.circuit_breaker_manager.circuit_breakers[integration_id].get_status()
        
        # Bulkhead status
        if integration_id in self.bulkheads:
            status['bulkhead'] = self.bulkheads[integration_id].get_stats()
        
        # Retry configuration
        if integration_id in self.retry_configs:
            retry_strategy = self.retry_configs[integration_id]
            status['retry_config'] = {
                'max_attempts': retry_strategy.config.max_attempts,
                'base_delay': retry_strategy.config.base_delay,
                'max_delay': retry_strategy.config.max_delay
            }
        
        return status
```

---

## Integration Marketplace

### Marketplace Architecture

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class IntegrationCategory(Enum):
    CRM = "crm"
    ECOMMERCE = "ecommerce"
    SUPPORT = "support"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    PAYMENT = "payment"
    PRODUCTIVITY = "productivity"
    MARKETING = "marketing"
    SOCIAL_MEDIA = "social_media"
    CUSTOM = "custom"

class IntegrationStatus(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    BETA = "beta"
    COMING_SOON = "coming_soon"

@dataclass
class IntegrationTemplate:
    """Template for marketplace integrations"""
    template_id: str
    name: str
    description: str
    category: IntegrationCategory
    provider: str
    version: str
    status: IntegrationStatus
    
    # Configuration
    configuration_schema: Dict[str, Any]
    default_configuration: Dict[str, Any]
    required_credentials: List[str]
    
    # Metadata
    logo_url: str
    documentation_url: str
    support_url: str
    pricing_model: str
    installation_complexity: str  # easy, medium, advanced
    
    # Features
    supported_operations: List[str]
    webhook_support: bool
    real_time_sync: bool
    batch_operations: bool
    
    # Ratings and usage
    average_rating: float
    total_installations: int
    total_reviews: int
    
    # Requirements
    minimum_plan_required: str
    technical_requirements: List[str]

class IntegrationMarketplace:
    """Marketplace for integration templates"""
    
    def __init__(self):
        self.templates: Dict[str, IntegrationTemplate] = {}
        self.categories: Dict[IntegrationCategory, List[str]] = {}
        self.featured_integrations: List[str] = []
        self.reviews: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_integration(self, template: IntegrationTemplate):
        """Register a new integration template"""
        self.templates[template.template_id] = template
        
        # Add to category index
        if template.category not in self.categories:
            self.categories[template.category] = []
        self.categories[template.category].append(template.template_id)
    
    def search_integrations(self, query: str = "", 
                          category: Optional[IntegrationCategory] = None,
                          status: Optional[IntegrationStatus] = None,
                          limit: int = 20) -> List[IntegrationTemplate]:
        """Search integrations in marketplace"""
        results = []
        
        for template in self.templates.values():
            # Filter by category
            if category and template.category != category:
                continue
            
            # Filter by status
            if status and template.status != status:
                continue
            
            # Filter by search query
            if query:
                search_text = f"{template.name} {template.description} {template.provider}".lower()
                if query.lower() not in search_text:
                    continue
            
            results.append(template)
        
        # Sort by relevance (rating and installations)
        results.sort(key=lambda t: (t.average_rating, t.total_installations), reverse=True)
        
        return results[:limit]
    
    def get_featured_integrations(self) -> List[IntegrationTemplate]:
        """Get featured integrations"""
        return [self.templates[tid] for tid in self.featured_integrations if tid in self.templates]
    
    def get_popular_integrations(self, category: Optional[IntegrationCategory] = None, 
                               limit: int = 10) -> List[IntegrationTemplate]:
        """Get popular integrations by installation count"""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        templates.sort(key=lambda t: t.total_installations, reverse=True)
        return templates[:limit]
    
    def get_integration_details(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an integration"""
        if template_id not in self.templates:
            return None
        
        template = self.templates[template_id]
        reviews = self.reviews.get(template_id, [])
        
        return {
            'template': template,
            'reviews': reviews[-10:],  # Last 10 reviews
            'installation_instructions': self._get_installation_instructions(template_id),
            'sample_configuration': self._generate_sample_configuration(template),
            'compatibility_info': self._get_compatibility_info(template)
        }
    
    def install_integration(self, template_id: str, tenant_id: str, 
                          configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Install integration for a tenant"""
        if template_id not in self.templates:
            raise ValueError("Integration template not found")
        
        template = self.templates[template_id]
        
        # Validate configuration
        validation_result = self._validate_configuration(template, configuration)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': 'Configuration validation failed',
                'details': validation_result['errors']
            }
        
        # Create integration instance
        integration_id = f"{tenant_id}_{template_id}_{int(time.time())}"
        
        # Generate final configuration
        final_config = {**template.default_configuration, **configuration}
        
        # Store integration (in production, this would be in database)
        integration_instance = {
            'integration_id': integration_id,
            'template_id': template_id,
            'tenant_id': tenant_id,
            'configuration': final_config,
            'status': 'installed',
            'installed_at': datetime.utcnow().isoformat(),
            'last_used': None
        }
        
        # Update installation count
        template.total_installations += 1
        
        return {
            'success': True,
            'integration_id': integration_id,
            'configuration': final_config,
            'next_steps': self._get_post_installation_steps(template)
        }
    
    def add_review(self, template_id: str, tenant_id: str, review: Dict[str, Any]) -> bool:
        """Add review for an integration"""
        if template_id not in self.templates:
            return False
        
        if template_id not in self.reviews:
            self.reviews[template_id] = []
        
        review_entry = {
            'tenant_id': tenant_id,
            'rating': review['rating'],
            'comment': review.get('comment', ''),
            'version': review.get('version', ''),
            'created_at': datetime.utcnow().isoformat(),
            'helpful_votes': 0
        }
        
        self.reviews[template_id].append(review_entry)
        
        # Update average rating
        template = self.templates[template_id]
        all_ratings = [r['rating'] for r in self.reviews[template_id]]
        template.average_rating = sum(all_ratings) / len(all_ratings)
        template.total_reviews = len(all_ratings)
        
        return True
    
    def _validate_configuration(self, template: IntegrationTemplate, 
                              configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Validate integration configuration"""
        errors = []
        
        # Check required credentials
        for cred in template.required_credentials:
            if cred not in configuration:
                errors.append(f"Missing required credential: {cred}")
        
        # Validate against schema
        schema = template.configuration_schema
        for field, field_schema in schema.items():
            if field_schema.get('required', False) and field not in configuration:
                errors.append(f"Missing required field: {field}")
            
            if field in configuration:
                value = configuration[field]
                field_type = field_schema.get('type')
                
                if field_type == 'url' and not self._is_valid_url(value):
                    errors.append(f"Invalid URL format for field: {field}")
                elif field_type == 'email' and not self._is_valid_email(value):
                    errors.append(f"Invalid email format for field: {field}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+), re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})
        return email_pattern.match(email) is not None
    
    def _get_installation_instructions(self, template_id: str) -> List[Dict[str, Any]]:
        """Get step-by-step installation instructions"""
        # This would return detailed installation steps
        return [
            {
                'step': 1,
                'title': 'Configure Authentication',
                'description': 'Set up API credentials in your account',
                'type': 'configuration'
            },
            {
                'step': 2,
                'title': 'Test Connection',
                'description': 'Verify the integration is working correctly',
                'type': 'test'
            },
            {
                'step': 3,
                'title': 'Configure Conversation Flows',
                'description': 'Set up conversation flows to use the integration',
                'type': 'flow_setup'
            }
        ]
    
    def _generate_sample_configuration(self, template: IntegrationTemplate) -> Dict[str, Any]:
        """Generate sample configuration with placeholders"""
        sample = template.default_configuration.copy()
        
        for field, schema in template.configuration_schema.items():
            if field not in sample:
                field_type = schema.get('type', 'string')
                if field_type == 'url':
                    sample[field] = 'https://api.example.com'
                elif field_type == 'email':
                    sample[field] = 'user@example.com'
                else:
                    sample[field] = f'<{field}>'
        
        return sample
    
    def _get_compatibility_info(self, template: IntegrationTemplate) -> Dict[str, Any]:
        """Get compatibility information"""
        return {
            'minimum_plan': template.minimum_plan_required,
            'technical_requirements': template.technical_requirements,
            'supported_regions': ['US', 'EU', 'APAC'],  # Would be template-specific
            'compliance_certifications': ['SOC2', 'GDPR']  # Would be template-specific
        }
    
    def _get_post_installation_steps(self, template: IntegrationTemplate) -> List[str]:
        """Get recommended steps after installation"""
        return [
            "Test the integration with a sample request",
            "Set up webhook endpoints if supported",
            "Configure conversation flows to use the integration",
            "Set up monitoring and alerts",
            "Review security settings and permissions"
        ]

# Example marketplace integrations
def initialize_marketplace() -> IntegrationMarketplace:
    """Initialize marketplace with common integrations"""
    marketplace = IntegrationMarketplace()
    
    # Salesforce CRM
    salesforce_template = IntegrationTemplate(
        template_id="salesforce_crm",
        name="Salesforce CRM",
        description="Connect to Salesforce for customer data and case management",
        category=IntegrationCategory.CRM,
        provider="Salesforce",
        version="1.2.0",
        status=IntegrationStatus.ACTIVE,
        configuration_schema={
            "instance_url": {"type": "url", "required": True},
            "username": {"type": "string", "required": True},
            "password": {"type": "password", "required": True},
            "security_token": {"type": "password", "required": True},
            "sandbox": {"type": "boolean", "default": False}
        },
        default_configuration={
            "sandbox": False,
            "api_version": "58.0"
        },
        required_credentials=["username", "password", "security_token"],
        logo_url="https://marketplace.example.com/logos/salesforce.png",
        documentation_url="https://docs.example.com/integrations/salesforce",
        support_url="https://support.example.com/salesforce",
        pricing_model="included",
        installation_complexity="medium",
        supported_operations=["get_contact", "create_case", "update_case", "search_accounts"],
        webhook_support=True,
        real_time_sync=True,
        batch_operations=True,
        average_rating=4.8,
        total_installations=15420,
        total_reviews=324,
        minimum_plan_required="professional",
        technical_requirements=["API access enabled in Salesforce org"]
    )
    
    marketplace.register_integration(salesforce_template)
    
    # Shopify E-commerce
    shopify_template = IntegrationTemplate(
        template_id="shopify_ecommerce",
        name="Shopify",
        description="Connect to Shopify for order management and product information",
        category=IntegrationCategory.ECOMMERCE,
        provider="Shopify",
        version="2.1.0",
        status=IntegrationStatus.ACTIVE,
        configuration_schema={
            "shop_domain": {"type": "string", "required": True},
            "access_token": {"type": "password", "required": True},
            "api_version": {"type": "string", "default": "2023-10"}
        },
        default_configuration={
            "api_version": "2023-10"
        },
        required_credentials=["access_token"],
        logo_url="https://marketplace.example.com/logos/shopify.png",
        documentation_url="https://docs.example.com/integrations/shopify",
        support_url="https://support.example.com/shopify",
        pricing_model="included",
        installation_complexity="easy",
        supported_operations=["get_order", "search_products", "get_customer", "create_draft_order"],
        webhook_support=True,
        real_time_sync=True,
        batch_operations=False,
        average_rating=4.6,
        total_installations=8932,
        total_reviews=187,
        minimum_plan_required="starter",
        technical_requirements=["Shopify store with API access"]
    )
    
    marketplace.register_integration(shopify_template)
    
    return marketplace
```

---

## Testing and Validation

### Integration Testing Framework

```python
import pytest
import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class IntegrationTestCase:
    """Test case for integration validation"""
    name: str
    description: str
    endpoint: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    expected_status_code: int = 200
    timeout_seconds: int = 30
    retry_on_failure: bool = False

class IntegrationTester:
    """Comprehensive integration testing framework"""
    
    def __init__(self, integration_executor: IntegrationExecutor):
        self.executor = integration_executor
        self.test_results = []
    
    async def run_test_suite(self, integration_config: IntegrationConfig,
                           test_cases: List[IntegrationTestCase]) -> Dict[str, Any]:
        """Run complete test suite for an integration"""
        
        results = {
            'integration_id': integration_config.integration_id,
            'total_tests': len(test_cases),
            'passed_tests': 0,
            'failed_tests': 0,
            'test_results': [],
            'overall_status': 'unknown',
            'execution_time_ms': 0
        }
        
        start_time = asyncio.get_event_loop().time()
        
        for test_case in test_cases:
            test_result = await self._run_single_test(integration_config, test_case)
            results['test_results'].append(test_result)
            
            if test_result['passed']:
                results['passed_tests'] += 1
            else:
                results['failed_tests'] += 1
        
        end_time = asyncio.get_event_loop().time()
        results['execution_time_ms'] = int((end_time - start_time) * 1000)
        
        # Determine overall status
        if results['failed_tests'] == 0:
            results['overall_status'] = 'passed'
        elif results['passed_tests'] > results['failed_tests']:
            results['overall_status'] = 'mostly_passed'
        else:
            results['overall_status'] = 'failed'
        
        return results
    
    async def _run_single_test(self, integration_config: IntegrationConfig,
                             test_case: IntegrationTestCase) -> Dict[str, Any]:
        """Run a single test case"""
        
        test_result = {
            'test_name': test_case.name,
            'description': test_case.description,
            'passed': False,
            'execution_time_ms': 0,
            'error_message': None,
            'response_data': None,
            'assertions': []
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create integration request
            request = IntegrationRequest(
                endpoint_name=test_case.endpoint,
                parameters=test_case.input_data,
                context={},
                tenant_id="test_tenant",
                timeout_override_ms=test_case.timeout_seconds * 1000
            )
            
            # Execute integration
            response = await self.executor.execute(integration_config, request)
            
            end_time = asyncio.get_event_loop().time()
            test_result['execution_time_ms'] = int((end_time - start_time) * 1000)
            test_result['response_data'] = response.data
            
            # Validate response
            assertions = self._validate_response(response, test_case)
            test_result['assertions'] = assertions
            test_result['passed'] = all(assertion['passed'] for assertion in assertions)
            
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            test_result['execution_time_ms'] = int((end_time - start_time) * 1000)
            test_result['error_message'] = str(e)
            test_result['passed'] = False
        
        return test_result
    
    def _validate_response(self, response: IntegrationResponse, 
                         test_case: IntegrationTestCase) -> List[Dict[str, Any]]:
        """Validate integration response against test expectations"""
        assertions = []
        
        # Check response success
        assertions.append({
            'assertion': 'response_success',
            'expected': True,
            'actual': response.success,
            'passed': response.success,
            'message': 'Response should be successful'
        })
        
        # Check status code if provided
        if hasattr(response, 'status_code') and response.status_code:
            assertions.append({
                'assertion': 'status_code',
                'expected': test_case.expected_status_code,
                'actual': response.status_code,
                'passed': response.status_code == test_case.expected_status_code,
                'message': f'Status code should be {test_case.expected_status_code}'
            })
        
        # Check response data structure
        if response.data and test_case.expected_output:
            data_assertions = self._validate_data_structure(
                response.data, 
                test_case.expected_output
            )
            assertions.extend(data_assertions)
        
        # Check response time
        reasonable_response_time = test_case.timeout_seconds * 1000  # Convert to ms
        assertions.append({
            'assertion': 'response_time',
            'expected': f'< {reasonable_response_time}ms',
            'actual': f'{response.execution_time_ms}ms',
            'passed': response.execution_time_ms < reasonable_response_time,
            'message': f'Response time should be under {reasonable_response_time}ms'
        })
        
        return assertions
    
    def _validate_data_structure(self, actual_data: Any, expected_structure: Any,
                               path: str = "root") -> List[Dict[str, Any]]:
        """Recursively validate data structure"""
        assertions = []
        
        if isinstance(expected_structure, dict):
            if not isinstance(actual_data, dict):
                assertions.append({
                    'assertion': 'data_type',
                    'expected': 'dict',
                    'actual': type(actual_data).__name__,
                    'passed': False,
                    'message': f'Expected dict at {path}, got {type(actual_data).__name__}',
                    'path': path
                })
                return assertions
            
            for key, expected_value in expected_structure.items():
                current_path = f"{path}.{key}"
                
                if key not in actual_data:
                    assertions.append({
                        'assertion': 'required_field',
                        'expected': f'field "{key}" present',
                        'actual': 'field missing',
                        'passed': False,
                        'message': f'Required field "{key}" missing at {path}',
                        'path': current_path
                    })
                else:
                    # Recursively validate nested structure
                    nested_assertions = self._validate_data_structure(
                        actual_data[key], 
                        expected_value, 
                        current_path
                    )
                    assertions.extend(nested_assertions)
        
        elif isinstance(expected_structure, list):
            if not isinstance(actual_data, list):
                assertions.append({
                    'assertion': 'data_type',
                    'expected': 'list',
                    'actual': type(actual_data).__name__,
                    'passed': False,
                    'message': f'Expected list at {path}, got {type(actual_data).__name__}',
                    'path': path
                })
            elif len(expected_structure) > 0 and len(actual_data) > 0:
                # Validate first item structure (assuming homogeneous list)
                item_assertions = self._validate_data_structure(
                    actual_data[0], 
                    expected_structure[0], 
                    f"{path}[0]"
                )
                assertions.extend(item_assertions)
        
        elif isinstance(expected_structure, str) and expected_structure.startswith('):
            # Special validation rules
            rule = expected_structure[1:]  # Remove ' prefix
            
            if rule == 'string':
                assertions.append({
                    'assertion': 'data_type',
                    'expected': 'string',
                    'actual': type(actual_data).__name__,
                    'passed': isinstance(actual_data, str),
                    'message': f'Expected string at {path}',
                    'path': path
                })
            elif rule == 'number':
                assertions.append({
                    'assertion': 'data_type',
                    'expected': 'number',
                    'actual': type(actual_data).__name__,
                    'passed': isinstance(actual_data, (int, float)),
                    'message': f'Expected number at {path}',
                    'path': path
                })
            elif rule == 'boolean':
                assertions.append({
                    'assertion': 'data_type',
                    'expected': 'boolean',
                    'actual': type(actual_data).__name__,
                    'passed': isinstance(actual_data, bool),
                    'message': f'Expected boolean at {path}',
                    'path': path
                })
            elif rule == 'not_empty':
                is_not_empty = actual_data is not None and len(str(actual_data)) > 0
                assertions.append({
                    'assertion': 'not_empty',
                    'expected': 'non-empty value',
                    'actual': actual_data,
                    'passed': is_not_empty,
                    'message': f'Expected non-empty value at {path}',
                    'path': path
                })
        
        else:
            # Direct value comparison
            assertions.append({
                'assertion': 'value_match',
                'expected': expected_structure,
                'actual': actual_data,
                'passed': actual_data == expected_structure,
                'message': f'Expected {expected_structure} at {path}',
                'path': path
            })
        
        return assertions
    
    async def run_load_test(self, integration_config: IntegrationConfig,
                          test_case: IntegrationTestCase,
                          concurrent_requests: int = 10,
                          duration_seconds: int = 60) -> Dict[str, Any]:
        """Run load test on integration"""
        
        results = {
            'integration_id': integration_config.integration_id,
            'test_name': test_case.name,
            'concurrent_requests': concurrent_requests,
            'duration_seconds': duration_seconds,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time_ms': 0,
            'min_response_time_ms': float('inf'),
            'max_response_time_ms': 0,
            'requests_per_second': 0,
            'error_rate': 0
        }
        
        response_times = []
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + duration_seconds
        
        # Create tasks for concurrent requests
        async def make_request():
            while asyncio.get_event_loop().time() < end_time:
                request_start = asyncio.get_event_loop().time()
                try:
                    request = IntegrationRequest(
                        endpoint_name=test_case.endpoint,
                        parameters=test_case.input_data,
                        context={},
                        tenant_id="load_test_tenant"
                    )
                    
                    response = await self.executor.execute(integration_config, request)
                    
                    request_end = asyncio.get_event_loop().time()
                    response_time = (request_end - request_start) * 1000
                    
                    response_times.append(response_time)
                    results['total_requests'] += 1
                    
                    if response.success:
                        results['successful_requests'] += 1
                    else:
                        results['failed_requests'] += 1
                        
                except Exception:
                    results['total_requests'] += 1
                    results['failed_requests'] += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
        
        # Run concurrent requests
        tasks = [asyncio.create_task(make_request()) for _ in range(concurrent_requests)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        if response_times:
            results['avg_response_time_ms'] = sum(response_times) / len(response_times)
            results['min_response_time_ms'] = min(response_times)
            results['max_response_time_ms'] = max(response_times)
        
        actual_duration = asyncio.get_event_loop().time() - start_time
        if actual_duration > 0:
            results['requests_per_second'] = results['total_requests'] / actual_duration
        
        if results['total_requests'] > 0:
            results['error_rate'] = results['failed_requests'] / results['total_requests']
        
        return results

# Example test cases
def create_salesforce_test_cases() -> List[IntegrationTestCase]:
    """Create test cases for Salesforce integration"""
    return [
        IntegrationTestCase(
            name="Get Contact by Email",
            description="Test retrieving contact information by email address",
            endpoint="get_contact_by_email",
            input_data={"email": "test@example.com"},
            expected_output={
                "Id": "$string",
                "FirstName": "$string",
                "LastName": "$string",
                "Email": "test@example.com",
                "Account": {
                    "Name": "$string"
                }
            }
        ),
        IntegrationTestCase(
            name="Create Case",
            description="Test creating a new support case",
            endpoint="create_case",
            input_data={
                "subject": "Test Case",
                "description": "This is a test case created by automation",
                "priority": "Medium"
            },
            expected_output={
                "success": True,
                "case_id": "$string",
                "case_number": "$string"
            }
        ),
        IntegrationTestCase(
            name="Search Knowledge Base",
            description="Test searching knowledge base articles",
            endpoint="search_knowledge_base",
            input_data={
                "search_term": "password reset",
                "limit": 5
            },
            expected_output=[
                {
                    "Id": "$string",
                    "Title": "$string",
                    "Summary": "$string"
                }
            ]
        )
    ]

def create_shopify_test_cases() -> List[IntegrationTestCase]:
    """Create test cases for Shopify integration"""
    return [
        IntegrationTestCase(
            name="Get Order by Number",
            description="Test retrieving order information by order number",
            endpoint="get_order_by_number",
            input_data={"order_number": "#1001"},
            expected_output={
                "name": "#1001",
                "total_price": "$string",
                "financial_status": "$string",
                "fulfillment_status": "$string",
                "line_items": [
                    {
                        "title": "$string",
                        "quantity": "$number",
                        "price": "$string"
                    }
                ]
            }
        ),
        IntegrationTestCase(
            name="Search Products",
            description="Test searching for products by title",
            endpoint="search_products",
            input_data={
                "query": "laptop",
                "limit": 10
            },
            expected_output=[
                {
                    "title": "$string",
                    "handle": "$string",
                    "variants": [
                        {
                            "price": "$string",
                            "inventory_quantity": "$number"
                        }
                    ]
                }
            ]
        )
    ]
```


**Document Maintainer:** Integration Engineering Team  
**Review Schedule:** Bi-weekly during development, monthly in production  
**Related Documents:** System Architecture, API Specifications, Security Implementation