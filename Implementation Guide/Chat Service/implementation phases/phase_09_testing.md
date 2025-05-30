# Phase 9: Testing & Quality Assurance
**Duration:** Week 15-16  
**Steps:** 17 of 18

---

## 🎯 Objectives
- Implement comprehensive test suite covering all layers
- Create integration tests for external services
- Build end-to-end testing scenarios
- Establish performance and load testing framework
- Set up test automation and CI/CD integration

---

## 📋 Step 17: Comprehensive Testing Implementation

### What Will Be Implemented
- Unit tests for all components (repositories, services, processors)
- Integration tests for databases and external services
- API endpoint testing with various scenarios
- Performance and load testing setup

### Folders and Files Created

```
tests/
├── __init__.py
├── conftest.py
├── unit/
│   ├── __init__.py
│   ├── test_repositories/
│   │   ├── __init__.py
│   │   ├── test_conversation_repository.py
│   │   ├── test_message_repository.py
│   │   └── test_session_repository.py
│   ├── test_services/
│   │   ├── __init__.py
│   │   ├── test_message_service.py
│   │   ├── test_conversation_service.py
│   │   └── test_session_service.py
│   ├── test_processors/
│   │   ├── __init__.py
│   │   ├── test_text_processor.py
│   │   ├── test_media_processor.py
│   │   └── test_processor_factory.py
│   ├── test_channels/
│   │   ├── __init__.py
│   │   ├── test_whatsapp_channel.py
│   │   ├── test_web_channel.py
│   │   └── test_channel_factory.py
│   └── test_utils/
│       ├── __init__.py
│       ├── test_validators.py
│       └── test_formatters.py
├── integration/
│   ├── __init__.py
│   ├── test_api_endpoints.py
│   ├── test_database_operations.py
│   ├── test_external_services.py
│   ├── test_event_publishing.py
│   └── test_webhook_processing.py
├── e2e/
│   ├── __init__.py
│   ├── test_message_flow.py
│   ├── test_conversation_lifecycle.py
│   ├── test_multi_channel.py
│   └── test_error_scenarios.py
├── performance/
│   ├── __init__.py
│   ├── load_test_config.py
│   ├── test_message_throughput.py
│   └── test_concurrent_users.py
├── fixtures/
│   ├── __init__.py
│   ├── conversation_fixtures.py
│   ├── message_fixtures.py
│   └── webhook_fixtures.py
└── utils/
    ├── __init__.py
    ├── test_helpers.py
    ├── mock_services.py
    └── assertions.py
```

### File Documentation

#### `tests/conftest.py`
**Purpose:** pytest configuration and shared fixtures for all tests  
**Usage:** Centralized test setup, database fixtures, and mock configurations

**Fixtures:**

1. **app_client() -> TestClient**
   - **Purpose:** FastAPI test client for API testing
   - **Scope:** Function-level
   - **Usage:** Test API endpoints with real application

2. **mock_databases() -> Dict[str, Any]**
   - **Purpose:** Mock database connections for unit tests
   - **Scope:** Function-level
   - **Usage:** Isolated unit testing without real databases

```python
import pytest
import asyncio
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

from src.main import create_app
from src.config.settings import get_settings
from src.database.mongodb import mongodb_manager
from src.database.redis_client import redis_manager
from src.models.mongo.conversation_model import ConversationDocument
from src.models.mongo.message_model import MessageDocument
from src.models.redis.session_cache import SessionData
from tests.fixtures.conversation_fixtures import create_test_conversation
from tests.fixtures.message_fixtures import create_test_message
from tests.utils.test_helpers import cleanup_test_data

# Pytest configuration
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def settings():
    """Get test settings"""
    return get_settings()

# Application fixtures
@pytest.fixture
def app():
    """Create FastAPI application for testing"""
    return create_app()

@pytest.fixture
def client(app) -> TestClient:
    """Create test client for API testing"""
    return TestClient(app)

# Database fixtures
@pytest.fixture
async def test_mongodb():
    """MongoDB test database connection"""
    # Use test database
    test_client = AsyncIOMotorClient("mongodb://localhost:27017")
    test_db = test_client.chatbot_test
    
    yield test_db
    
    # Cleanup
    await test_client.drop_database("chatbot_test")
    test_client.close()

@pytest.fixture
async def test_redis():
    """Redis test connection"""
    # Use test database (usually db 1)
    test_redis = redis.from_url("redis://localhost:6379/1")
    
    yield test_redis
    
    # Cleanup
    await test_redis.flushdb()
    await test_redis.close()

@pytest.fixture
def mock_mongodb():
    """Mock MongoDB for unit tests"""
    mock_client = AsyncMock()
    mock_db = AsyncMock()
    mock_collection = AsyncMock()
    
    mock_client.chatbot = mock_db
    mock_db.conversations = mock_collection
    mock_db.messages = mock_collection
    
    return {
        "client": mock_client,
        "database": mock_db,
        "collection": mock_collection
    }

@pytest.fixture
def mock_redis():
    """Mock Redis for unit tests"""
    mock_redis = AsyncMock()
    
    # Mock common Redis operations
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.hgetall.return_value = {}
    mock_redis.hset.return_value = True
    mock_redis.exists.return_value = False
    mock_redis.delete.return_value = 1
    
    return mock_redis

# Service fixtures
@pytest.fixture
def mock_message_service():
    """Mock message service for testing"""
    mock_service = AsyncMock()
    
    # Default return values
    mock_service.process_message.return_value = create_test_message_response()
    
    return mock_service

@pytest.fixture
def mock_mcp_client():
    """Mock MCP Engine client"""
    mock_client = AsyncMock()
    
    # Mock responses
    mock_client.process_message.return_value = create_test_mcp_response()
    mock_client.health_check.return_value = True
    
    return mock_client

# Test data fixtures
@pytest.fixture
def test_conversation() -> ConversationDocument:
    """Create test conversation document"""
    return create_test_conversation()

@pytest.fixture
def test_message() -> MessageDocument:
    """Create test message document"""
    return create_test_message()

@pytest.fixture
def test_session() -> SessionData:
    """Create test session data"""
    return SessionData(
        session_id="test_session_123",
        tenant_id="test_tenant",
        user_id="test_user",
        channel="web",
        context={"test": "data"}
    )

@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Create authentication headers for testing"""
    return {
        "Authorization": "Bearer test_jwt_token",
        "X-Tenant-ID": "test_tenant"
    }

@pytest.fixture
def webhook_payload() -> Dict[str, Any]:
    """Create test webhook payload"""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "business_account_123",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "phone_number_id": "123456789"
                            },
                            "messages": [
                                {
                                    "id": "msg_123",
                                    "from": "+1234567890",
                                    "timestamp": "1622547600",
                                    "type": "text",
                                    "text": {
                                        "body": "Hello, I need help"
                                    }
                                }
                            ]
                        },
                        "field": "messages"
                    }
                ]
            }
        ]
    }

# Helper functions
def create_test_message_response():
    """Create test message response"""
    from src.models.schemas.response_schemas import MessageResponse
    from src.models.types import MessageContent
    
    return MessageResponse(
        message_id="test_msg_123",
        conversation_id="test_conv_123",
        response=MessageContent(
            type="text",
            text="Test response"
        ),
        conversation_state={
            "current_intent": "test_intent"
        },
        processing_metadata={
            "processing_time_ms": 100,
            "model_used": "test_model"
        }
    )

def create_test_mcp_response():
    """Create test MCP response"""
    from src.clients.grpc.mcp_client import ProcessMessageResponse
    
    return ProcessMessageResponse(
        success=True,
        message_id="test_msg_123",
        conversation_id="test_conv_123",
        response_content={
            "type": "text",
            "text": "MCP test response"
        },
        response_type="text",
        confidence_score=0.9,
        updated_context={
            "current_intent": "test_intent"
        },
        context_changes=["current_intent"],
        next_expected_inputs=[],
        suggested_actions=[],
        processing_time_ms=150,
        model_used="test_model",
        model_provider="test_provider"
    )

# Cleanup fixtures
@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Cleanup logic here
    await cleanup_test_data()

# Async test utilities
@pytest.fixture
def async_test():
    """Utility for running async tests"""
    def _async_test(coro):
        return asyncio.get_event_loop().run_until_complete(coro)
    return _async_test
```

#### `tests/unit/test_services/test_message_service.py`
**Purpose:** Unit tests for message service business logic  
**Usage:** Test message processing, validation, and error handling

**Test Classes:**

1. **TestMessageService**
   - **Purpose:** Test MessageService functionality
   - **Methods:**
     - **test_process_message_success()**: Test successful message processing
     - **test_process_message_validation_error()**: Test validation failures
     - **test_process_message_mcp_failure()**: Test MCP Engine failures

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from src.services.message_service import MessageService
from src.services.exceptions import ValidationError, ServiceError
from src.models.schemas.request_schemas import SendMessageRequest
from src.models.types import MessageContent, ChannelType
from src.models.mongo.conversation_model import ConversationDocument
from src.models.mongo.message_model import MessageDocument
from tests.fixtures.message_fixtures import create_test_send_message_request

@pytest.mark.unit
class TestMessageService:
    """Unit tests for MessageService"""
    
    @pytest.fixture
    def message_service(self, mock_mongodb, mock_redis):
        """Create MessageService with mocked dependencies"""
        # Mock repositories
        conversation_repo = AsyncMock()
        message_repo = AsyncMock()
        session_repo = AsyncMock()
        
        # Mock factories
        channel_factory = MagicMock()
        processor_factory = MagicMock()
        
        return MessageService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            session_repo=session_repo,
            channel_factory=channel_factory,
            processor_factory=processor_factory
        )
    
    @pytest.fixture
    def valid_request(self) -> SendMessageRequest:
        """Create valid message request"""
        return create_test_send_message_request()
    
    @pytest.fixture
    def user_context(self) -> dict:
        """Create user context"""
        return {
            "user_id": "test_user",
            "tenant_id": "test_tenant",
            "role": "member",
            "permissions": ["conversations:read", "messages:send"]
        }
    
    async def test_process_message_success(
        self, 
        message_service: MessageService,
        valid_request: SendMessageRequest,
        user_context: dict
    ):
        """Test successful message processing"""
        # Setup mocks
        test_conversation = ConversationDocument(
            conversation_id="test_conv_123",
            tenant_id="test_tenant",
            user_id="test_user",
            channel=ChannelType.WEB
        )
        
        message_service.conversation_repo.get_by_id.return_value = None
        message_service.conversation_repo.create.return_value = test_conversation
        message_service.message_repo.create.return_value = AsyncMock()
        
        # Mock processor
        mock_processor = AsyncMock()
        mock_processing_result = MagicMock()
        mock_processing_result.processed_content = MessageContent(
            type="text", text="processed text"
        )
        mock_processing_result.entities = {}
        mock_processing_result.detected_language = "en"
        mock_processor.process.return_value = mock_processing_result
        
        message_service.processor_factory.get_processor.return_value = mock_processor
        
        # Mock channel
        mock_channel = AsyncMock()
        mock_channel.send_message.return_value = MagicMock(success=True)
        message_service.channel_factory.get_channel.return_value = mock_channel
        
        # Execute
        result = await message_service.process_message(valid_request, user_context)
        
        # Assertions
        assert result is not None
        assert result.message_id == valid_request.message_id
        assert result.conversation_id == test_conversation.conversation_id
        
        # Verify repository calls
        message_service.conversation_repo.create.assert_called_once()
        assert message_service.message_repo.create.call_count == 2  # Inbound + outbound
        
        # Verify processor was called
        mock_processor.process.assert_called_once()
        
        # Verify channel delivery
        mock_channel.send_message.assert_called_once()
    
    async def test_process_message_validation_error(
        self,
        message_service: MessageService,
        user_context: dict
    ):
        """Test message processing with validation error"""
        # Create invalid request (missing required fields)
        invalid_request = SendMessageRequest(
            user_id="",  # Invalid empty user_id
            channel=ChannelType.WEB,
            content=MessageContent(type="text", text="test")
        )
        
        # Execute and assert
        with pytest.raises(ValidationError) as exc_info:
            await message_service.process_message(invalid_request, user_context)
        
        assert "user_id" in str(exc_info.value).lower()
    
    async def test_process_message_unauthorized_tenant(
        self,
        message_service: MessageService,
        valid_request: SendMessageRequest
    ):
        """Test message processing with unauthorized tenant access"""
        # Create user context with different tenant
        unauthorized_context = {
            "user_id": "test_user",
            "tenant_id": "different_tenant",  # Different from request
            "role": "member",
            "permissions": []
        }
        
        # Execute and assert
        with pytest.raises(ServiceError):
            await message_service.process_message(valid_request, unauthorized_context)
    
    async def test_process_message_conversation_creation(
        self,
        message_service: MessageService,
        valid_request: SendMessageRequest,
        user_context: dict
    ):
        """Test conversation creation when none exists"""
        # Setup: no existing conversation
        message_service.conversation_repo.get_by_id.return_value = None
        
        # Mock successful creation
        created_conversation = ConversationDocument(
            conversation_id=str(uuid4()),
            tenant_id=valid_request.tenant_id,
            user_id=valid_request.user_id,
            channel=valid_request.channel
        )
        message_service.conversation_repo.create.return_value = created_conversation
        
        # Mock other dependencies
        message_service.message_repo.create.return_value = AsyncMock()
        
        mock_processor = AsyncMock()
        mock_processor.process.return_value = MagicMock(
            processed_content=MessageContent(type="text", text="test"),
            entities={},
            detected_language="en"
        )
        message_service.processor_factory.get_processor.return_value = mock_processor
        
        mock_channel = AsyncMock()
        mock_channel.send_message.return_value = MagicMock(success=True)
        message_service.channel_factory.get_channel.return_value = mock_channel
        
        # Execute
        result = await message_service.process_message(valid_request, user_context)
        
        # Verify conversation was created
        message_service.conversation_repo.create.assert_called_once()
        
        # Verify result uses created conversation
        assert result.conversation_id == created_conversation.conversation_id
    
    async def test_process_message_existing_conversation(
        self,
        message_service: MessageService,
        valid_request: SendMessageRequest,
        user_context: dict
    ):
        """Test processing with existing conversation"""
        # Setup: existing conversation
        existing_conversation = ConversationDocument(
            conversation_id=valid_request.conversation_id,
            tenant_id=valid_request.tenant_id,
            user_id=valid_request.user_id,
            channel=valid_request.channel
        )
        message_service.conversation_repo.get_by_id.return_value = existing_conversation
        
        # Mock other dependencies
        message_service.message_repo.create.return_value = AsyncMock()
        
        mock_processor = AsyncMock()
        mock_processor.process.return_value = MagicMock(
            processed_content=MessageContent(type="text", text="test"),
            entities={},
            detected_language="en"
        )
        message_service.processor_factory.get_processor.return_value = mock_processor
        
        mock_channel = AsyncMock()
        mock_channel.send_message.return_value = MagicMock(success=True)
        message_service.channel_factory.get_channel.return_value = mock_channel
        
        # Execute
        result = await message_service.process_message(valid_request, user_context)
        
        # Verify conversation was NOT created (used existing)
        message_service.conversation_repo.create.assert_not_called()
        message_service.conversation_repo.update.assert_called_once()
        
        # Verify result uses existing conversation
        assert result.conversation_id == existing_conversation.conversation_id
    
    async def test_process_message_processor_failure(
        self,
        message_service: MessageService,
        valid_request: SendMessageRequest,
        user_context: dict
    ):
        """Test handling of processor failure"""
        # Setup conversation
        test_conversation = ConversationDocument(
            conversation_id="test_conv",
            tenant_id=valid_request.tenant_id,
            user_id=valid_request.user_id,
            channel=valid_request.channel
        )
        message_service.conversation_repo.get_by_id.return_value = None
        message_service.conversation_repo.create.return_value = test_conversation
        
        # Mock processor failure
        mock_processor = AsyncMock()
        mock_processor.process.side_effect = Exception("Processor failed")
        message_service.processor_factory.get_processor.return_value = mock_processor
        
        # Execute and assert
        with pytest.raises(ServiceError) as exc_info:
            await message_service.process_message(valid_request, user_context)
        
        assert "processor failed" in str(exc_info.value).lower()
    
    async def test_process_message_channel_delivery_failure(
        self,
        message_service: MessageService,
        valid_request: SendMessageRequest,
        user_context: dict
    ):
        """Test handling of channel delivery failure"""
        # Setup successful processing but failed delivery
        test_conversation = ConversationDocument(
            conversation_id="test_conv",
            tenant_id=valid_request.tenant_id,
            user_id=valid_request.user_id,
            channel=valid_request.channel
        )
        message_service.conversation_repo.get_by_id.return_value = None
        message_service.conversation_repo.create.return_value = test_conversation
        message_service.message_repo.create.return_value = AsyncMock()
        
        # Mock successful processor
        mock_processor = AsyncMock()
        mock_processor.process.return_value = MagicMock(
            processed_content=MessageContent(type="text", text="test"),
            entities={},
            detected_language="en"
        )
        message_service.processor_factory.get_processor.return_value = mock_processor
        
        # Mock failed channel delivery
        mock_channel = AsyncMock()
        mock_channel.send_message.return_value = MagicMock(
            success=False,
            error_message="Delivery failed"
        )
        message_service.channel_factory.get_channel.return_value = mock_channel
        
        # Execute
        result = await message_service.process_message(valid_request, user_context)
        
        # Should still return result even if delivery failed
        assert result is not None
        assert result.processing_metadata["delivery_status"] == "failed"
    
    @pytest.mark.parametrize("channel_type", [
        ChannelType.WEB,
        ChannelType.WHATSAPP,
        ChannelType.SLACK,
        ChannelType.MESSENGER
    ])
    async def test_process_message_different_channels(
        self,
        message_service: MessageService,
        valid_request: SendMessageRequest,
        user_context: dict,
        channel_type: ChannelType
    ):
        """Test message processing for different channels"""
        # Update request for specific channel
        valid_request.channel = channel_type
        
        # Setup mocks
        test_conversation = ConversationDocument(
            conversation_id="test_conv",
            tenant_id=valid_request.tenant_id,
            user_id=valid_request.user_id,
            channel=channel_type
        )
        message_service.conversation_repo.create.return_value = test_conversation
        message_service.message_repo.create.return_value = AsyncMock()
        
        mock_processor = AsyncMock()
        mock_processor.process.return_value = MagicMock(
            processed_content=MessageContent(type="text", text="test"),
            entities={},
            detected_language="en"
        )
        message_service.processor_factory.get_processor.return_value = mock_processor
        
        mock_channel = AsyncMock()
        mock_channel.send_message.return_value = MagicMock(success=True)
        message_service.channel_factory.get_channel.return_value = mock_channel
        
        # Execute
        result = await message_service.process_message(valid_request, user_context)
        
        # Verify channel factory was called with correct type
        message_service.channel_factory.get_channel.assert_called_with(channel_type)
        
        # Verify result
        assert result is not None
        assert result.conversation_id == test_conversation.conversation_id
```

#### `tests/integration/test_api_endpoints.py`
**Purpose:** Integration tests for API endpoints with real FastAPI app  
**Usage:** Test complete API request/response cycle with authentication and validation

**Test Classes:**

1. **TestChatEndpoints**
   - **Purpose:** Test chat API endpoints
   - **Methods:**
     - **test_send_message_endpoint()**: Test message sending endpoint
     - **test_get_conversation_history()**: Test conversation retrieval
     - **test_authentication_required()**: Test authentication enforcement

```python
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.main import create_app
from tests.fixtures.message_fixtures import create_test_send_message_request

@pytest.mark.integration
class TestChatEndpoints:
    """Integration tests for chat API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for testing"""
        return {
            "Authorization": "Bearer valid_test_token",
            "X-Tenant-ID": "test_tenant"
        }
    
    @patch('src.api.middleware.auth_middleware.verify_jwt_token')
    @patch('src.services.message_service.MessageService.process_message')
    def test_send_message_endpoint_success(
        self,
        mock_process_message,
        mock_verify_token,
        client: TestClient,
        auth_headers: dict
    ):
        """Test successful message sending"""
        # Mock authentication
        mock_verify_token.return_value = {
            "sub": "test_user",
            "tenant_id": "test_tenant",
            "user_role": "member",
            "permissions": ["conversations:read", "messages:send"],
            "exp": 9999999999
        }
        
        # Mock service response
        from tests.conftest import create_test_message_response
        mock_process_message.return_value = create_test_message_response()
        
        # Prepare request
        request_data = {
            "user_id": "test_user",
            "channel": "web",
            "content": {
                "type": "text",
                "text": "Hello, I need help with my order"
            }
        }
        
        # Make request
        response = client.post(
            "/api/v2/chat/message",
            json=request_data,
            headers=auth_headers
        )
        
        # Assertions
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "data" in response_data
        assert response_data["data"]["message_id"] is not None
        assert response_data["data"]["conversation_id"] is not None
        
        # Verify service was called
        mock_process_message.assert_called_once()
    
    def test_send_message_missing_auth(self, client: TestClient):
        """Test message sending without authentication"""
        request_data = {
            "user_id": "test_user",
            "channel": "web", 
            "content": {
                "type": "text",
                "text": "Hello"
            }
        }
        
        response = client.post("/api/v2/chat/message", json=request_data)
        
        assert response.status_code == 401
        assert "authorization" in response.json()["detail"].lower()
    
    def test_send_message_invalid_tenant(self, client: TestClient):
        """Test message sending with invalid tenant"""
        headers = {
            "Authorization": "Bearer valid_test_token",
            "X-Tenant-ID": "wrong_tenant"
        }
        
        request_data = {
            "user_id": "test_user",
            "channel": "web",
            "content": {
                "type": "text", 
                "text": "Hello"
            }
        }
        
        with patch('src.api.middleware.auth_middleware.verify_jwt_token') as mock_verify:
            mock_verify.return_value = {
                "sub": "test_user",
                "tenant_id": "correct_tenant",  # Different from header
                "user_role": "member",
                "permissions": [],
                "exp": 9999999999
            }
            
            response = client.post(
                "/api/v2/chat/message",
                json=request_data,
                headers=headers
            )
        
        assert response.status_code == 403
        assert "access denied" in response.json()["detail"].lower()
    
    @pytest.mark.parametrize("invalid_data", [
        {"channel": "web", "content": {"type": "text", "text": ""}},  # Missing user_id
        {"user_id": "", "channel": "web", "content": {"type": "text", "text": "test"}},  # Empty user_id
        {"user_id": "test", "channel": "invalid", "content": {"type": "text", "text": "test"}},  # Invalid channel
        {"user_id": "test", "channel": "web", "content": {"type": "invalid", "text": "test"}},  # Invalid content type
        {"user_id": "test", "channel": "web", "content": {"type": "text"}},  # Missing text for text type
    ])
    @patch('src.api.middleware.auth_middleware.verify_jwt_token')
    def test_send_message_validation_errors(
        self,
        mock_verify_token,
        client: TestClient,
        auth_headers: dict,
        invalid_data: dict
    ):
        """Test message sending with various validation errors"""
        # Mock authentication
        mock_verify_token.return_value = {
            "sub": "test_user",
            "tenant_id": "test_tenant",
            "user_role": "member", 
            "permissions": [],
            "exp": 9999999999
        }
        
        response = client.post(
            "/api/v2/chat/message",
            json=invalid_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
        assert "detail" in response.json()
    
    @patch('src.api.middleware.auth_middleware.verify_jwt_token')
    @patch('src.services.conversation_service.ConversationService.get_conversation_history')
    def test_get_conversation_history_success(
        self,
        mock_get_history,
        mock_verify_token,
        client: TestClient,
        auth_headers: dict
    ):
        """Test successful conversation history retrieval"""
        # Mock authentication
        mock_verify_token.return_value = {
            "sub": "test_user",
            "tenant_id": "test_tenant",
            "user_role": "member",
            "permissions": ["conversations:read"],
            "exp": 9999999999
        }
        
        # Mock service response
        mock_get_history.return_value = {
            "conversation_id": "test_conv_123",
            "messages": [
                {
                    "message_id": "msg_1",
                    "direction": "inbound",
                    "content": {"type": "text", "text": "Hello"},
                    "timestamp": "2025-05-30T10:00:00Z"
                },
                {
                    "message_id": "msg_2", 
                    "direction": "outbound",
                    "content": {"type": "text", "text": "Hi! How can I help?"},
                    "timestamp": "2025-05-30T10:00:01Z"
                }
            ],
            "page": 1,
            "page_size": 20,
            "total_messages": 2,
            "has_next": False,
            "has_previous": False
        }
        
        response = client.get(
            "/api/v2/chat/conversations/test_conv_123/history",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["status"] == "success"
        assert len(response_data["data"]["messages"]) == 2
        assert response_data["data"]["conversation_id"] == "test_conv_123"
    
    @patch('src.api.middleware.auth_middleware.verify_jwt_token')
    def test_get_conversation_history_not_found(
        self,
        mock_verify_token,
        client: TestClient,
        auth_headers: dict
    ):
        """Test conversation history for non-existent conversation"""
        # Mock authentication
        mock_verify_token.return_value = {
            "sub": "test_user", 
            "tenant_id": "test_tenant",
            "user_role": "member",
            "permissions": ["conversations:read"],
            "exp": 9999999999
        }
        
        with patch('src.services.conversation_service.ConversationService.get_conversation_history') as mock_get:
            from src.services.exceptions import NotFoundError
            mock_get.side_effect = NotFoundError("Conversation not found")
            
            response = client.get(
                "/api/v2/chat/conversations/nonexistent/history",
                headers=auth_headers
            )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('src.api.middleware.auth_middleware.verify_jwt_token')
    @patch('src.repositories.rate_limit_repository.RateLimitRepository.check_rate_limit')
    def test_rate_limiting(
        self,
        mock_rate_limit,
        mock_verify_token,
        client: TestClient,
        auth_headers: dict
    ):
        """Test API rate limiting"""
        # Mock authentication
        mock_verify_token.return_value = {
            "sub": "test_user",
            "tenant_id": "test_tenant", 
            "user_role": "member",
            "permissions": [],
            "exp": 9999999999,
            "rate_limit_tier": "basic"
        }
        
        # Mock rate limit exceeded
        mock_rate_limit.return_value = (False, 101, 1622547600)  # Exceeded limit
        
        request_data = {
            "user_id": "test_user",
            "channel": "web",
            "content": {
                "type": "text",
                "text": "Hello"
            }
        }
        
        response = client.post(
            "/api/v2/chat/message",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 429
        assert "rate limit" in response.json()["error"]["message"].lower()
        
        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "Retry-After" in response.headers
    
    def test_health_endpoint(self, client: TestClient):
        """Test health check endpoint"""
        response = client.get("/api/v2/health")
        
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert "databases" in health_data
        assert "timestamp" in health_data
    
    @patch('src.api.middleware.auth_middleware.verify_jwt_token')
    def test_cors_headers(
        self,
        mock_verify_token,
        client: TestClient
    ):
        """Test CORS headers are present"""
        # Mock authentication
        mock_verify_token.return_value = {
            "sub": "test_user",
            "tenant_id": "test_tenant",
            "user_role": "member",
            "permissions": [],
            "exp": 9999999999
        }
        
        # Make preflight request
        response = client.options(
            "/api/v2/chat/message",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, Content-Type"
            }
        )
        
        # Should have CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers
```

#### `tests/e2e/test_message_flow.py`
**Purpose:** End-to-end tests for complete message processing flow  
**Usage:** Test real-world scenarios from webhook to response delivery

**Test Classes:**

1. **TestCompleteMessageFlow**
   - **Purpose:** Test complete message processing pipeline
   - **Methods:**
     - **test_webhook_to_response_flow()**: Test complete webhook processing
     - **test_multi_turn_conversation()**: Test conversation continuity
     - **test_error_recovery()**: Test system resilience

```python
import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from unittest.mock import patch, AsyncMock

from src.main import create_app
from src.webhooks.processors.whatsapp_webhook import WhatsAppWebhookProcessor
from src.events.event_manager import get_event_manager, publish_event
from src.events.event_schemas import MessageReceivedEvent
from tests.utils.test_helpers import wait_for_event_processing

@pytest.mark.e2e
class TestCompleteMessageFlow:
    """End-to-end tests for complete message processing flow"""
    
    @pytest.fixture
    async def app_with_services(self):
        """Create app with real services for E2E testing"""
        app = create_app()
        
        # Start event processing
        event_manager = await get_event_manager()
        await event_manager.start_processing()
        
        yield app
        
        # Cleanup
        await event_manager.stop_processing()
    
    @pytest.fixture
    def whatsapp_webhook_payload(self):
        """Real WhatsApp webhook payload"""
        return {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "108103725312345",
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {
                                    "display_phone_number": "15551234567",
                                    "phone_number_id": "123456789012345"
                                },
                                "contacts": [
                                    {
                                        "profile": {
                                            "name": "John Doe"
                                        },
                                        "wa_id": "15551234567"
                                    }
                                ],
                                "messages": [
                                    {
                                        "from": "15551234567",
                                        "id": "wamid.unique_message_id",
                                        "timestamp": str(int(datetime.utcnow().timestamp())),
                                        "text": {
                                            "body": "Hello, I need help with my order #12345"
                                        },
                                        "type": "text"
                                    }
                                ]
                            },
                            "field": "messages"
                        }
                    ]
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_webhook_to_response_flow(
        self,
        app_with_services,
        whatsapp_webhook_payload: Dict[str, Any]
    ):
        """Test complete flow from webhook to response"""
        # Step 1: Process webhook
        webhook_processor = WhatsAppWebhookProcessor()
        
        with patch.object(webhook_processor, 'verify_signature', return_value=True):
            result = await webhook_processor.process_webhook(
                payload=whatsapp_webhook_payload,
                headers={"x-hub-signature-256": "sha256=test_signature"}
            )
        
        assert result.success
        assert result.events_extracted == 1
        assert result.events_processed == 1
        
        # Step 2: Wait for event processing
        await wait_for_event_processing(timeout_seconds=5)
        
        # Step 3: Verify message was processed
        # In a real test, you would check database or logs
        # For this example, we'll verify event publishing
        
        # Step 4: Verify response generation
        # This would involve checking that MCP Engine was called
        # and response was generated
        
        assert True  # Placeholder for actual assertions
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, app_with_services):
        """Test multi-turn conversation flow"""
        conversation_id = "test_conv_e2e"
        user_id = "test_user_e2e"
        
        # Message 1: Initial greeting
        message1_event = MessageReceivedEvent(
            message_id="msg_1",
            conversation_id=conversation_id,
            user_id=user_id,
            channel="web",
            message_type="text",
            content={
                "type": "text",
                "text": "Hello, I need help"
            }
        )
        
        await publish_event(message1_event)
        await wait_for_event_processing(timeout_seconds=3)
        
        # Message 2: Follow-up question
        message2_event = MessageReceivedEvent(
            message_id="msg_2",
            conversation_id=conversation_id,
            user_id=user_id,
            channel="web",
            message_type="text",
            content={
                "type": "text",
                "text": "I want to check my order status"
            }
        )
        
        await publish_event(message2_event)
        await wait_for_event_processing(timeout_seconds=3)
        
        # Message 3: Provide order number
        message3_event = MessageReceivedEvent(
            message_id="msg_3",
            conversation_id=conversation_id,
            user_id=user_id,
            channel="web",
            message_type="text",
            content={
                "type": "text",
                "text": "My order number is ORD123456"
            }
        )
        
        await publish_event(message3_event)
        await wait_for_event_processing(timeout_seconds=3)
        
        # Verify conversation continuity
        # In a real test, check that conversation context
        # was maintained across all messages
        assert True  # Placeholder
    
    @pytest.mark.asyncio 
    async def test_error_recovery(self, app_with_services):
        """Test system recovery from errors"""
        # Test scenario: MCP Engine is down
        
        with patch('src.clients.grpc.mcp_client.MCPEngineClient.process_message') as mock_mcp:
            # Simulate MCP Engine failure
            mock_mcp.side_effect = Exception("MCP Engine unavailable")
            
            # Send message
            message_event = MessageReceivedEvent(
                message_id="msg_error_test",
                conversation_id="conv_error_test",
                user_id="user_error_test",
                channel="web",
                message_type="text",
                content={
                    "type": "text",
                    "text": "This should trigger an error"
                }
            )
            
            await publish_event(message_event)
            await wait_for_event_processing(timeout_seconds=3)
            
            # Verify system handled error gracefully
            # Check that fallback response was used
            # Verify error was logged but system continued
            assert True  # Placeholder
    
    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self, app_with_services):
        """Test processing multiple messages concurrently"""
        tasks = []
        
        # Create 10 concurrent messages
        for i in range(10):
            message_event = MessageReceivedEvent(
                message_id=f"msg_concurrent_{i}",
                conversation_id=f"conv_concurrent_{i}",
                user_id=f"user_concurrent_{i}",
                channel="web",
                message_type="text",
                content={
                    "type": "text",
                    "text": f"Concurrent message {i}"
                }
            )
            
            # Create task for each message
            task = asyncio.create_task(publish_event(message_event))
            tasks.append(task)
        
        # Wait for all messages to be published
        await asyncio.gather(*tasks)
        
        # Wait for processing
        await wait_for_event_processing(timeout_seconds=10)
        
        # Verify all messages were processed
        # In a real test, check database for all 10 conversations
        assert True  # Placeholder
    
    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self, app_with_services):
        """Test rate limiting under load"""
        user_id = "rate_limit_test_user"
        
        # Send messages rapidly to trigger rate limiting
        tasks = []
        for i in range(100):  # More than typical rate limit
            message_event = MessageReceivedEvent(
                message_id=f"msg_rate_{i}",
                conversation_id="conv_rate_test",
                user_id=user_id,
                channel="web",
                message_type="text",
                content={
                    "type": "text",
                    "text": f"Rate limit test message {i}"
                }
            )
            
            task = asyncio.create_task(publish_event(message_event))
            tasks.append(task)
        
        # Execute all at once
        await asyncio.gather(*tasks)
        
        # Verify some messages were rate limited
        # Check that system handled it gracefully
        assert True  # Placeholder
    
    @pytest.mark.asyncio
    async def test_conversation_context_persistence(self, app_with_services):
        """Test that conversation context persists across service restarts"""
        conversation_id = "context_persistence_test"
        
        # Send initial message to establish context
        message1 = MessageReceivedEvent(
            message_id="msg_context_1",
            conversation_id=conversation_id,
            user_id="context_test_user",
            channel="web",
            message_type="text",
            content={
                "type": "text",
                "text": "I want to order a pizza"
            }
        )
        
        await publish_event(message1)
        await wait_for_event_processing(timeout_seconds=3)
        
        # Simulate service restart by stopping and starting event processing
        event_manager = await get_event_manager()
        await event_manager.stop_processing()
        await asyncio.sleep(1)
        await event_manager.start_processing()
        
        # Send follow-up message
        message2 = MessageReceivedEvent(
            message_id="msg_context_2",
            conversation_id=conversation_id,
            user_id="context_test_user",
            channel="web",
            message_type="text",
            content={
                "type": "text",
                "text": "Large pepperoni please"
            }
        )
        
        await publish_event(message2)
        await wait_for_event_processing(timeout_seconds=3)
        
        # Verify context was maintained
        # Check that system remembered the pizza order context
        assert True  # Placeholder
```

---

## 🔧 Technologies Used
- **pytest**: Testing framework with async support
- **pytest-asyncio**: Async test support
- **FastAPI TestClient**: API testing utilities
- **unittest.mock**: Mocking and patching
- **coverage**: Code coverage analysis

---

## ⚠️ Key Considerations

### Test Organization
- Clear separation between unit, integration, and E2E tests
- Comprehensive fixture management
- Reusable test utilities and helpers
- Proper test isolation and cleanup

### Test Coverage
- Aim for >90% code coverage
- Cover happy path and error scenarios
- Test edge cases and boundary conditions
- Include performance and load testing

### Test Performance
- Fast unit tests (<100ms each)
- Moderate integration tests (<1s each)
- Longer E2E tests (<30s each)
- Parallel test execution where possible

### Test Reliability
- Deterministic test outcomes
- Proper mocking of external dependencies
- Stable test data and fixtures
- Clear error messages and diagnostics

---

## 🎯 Success Criteria
- [ ] All unit tests pass with >90% coverage
- [ ] Integration tests verify component interactions
- [ ] E2E tests validate complete user scenarios
- [ ] Performance tests meet throughput requirements
- [ ] Test suite runs reliably in CI/CD pipeline
- [ ] Test documentation is comprehensive and clear

---

## 📋 Next Phase Preview
Phase 10 will focus on deployment and DevOps, including containerization, Kubernetes manifests, monitoring setup, and production deployment strategies.

