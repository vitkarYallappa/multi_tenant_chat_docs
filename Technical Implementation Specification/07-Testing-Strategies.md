# Testing Strategies
## Multi-Tenant AI Chatbot Platform

**Document:** 07-Testing-Strategies.md  
**Version:** 2.0  
**Last Updated:** May 30, 2025

---

## Table of Contents

1. [Testing Strategy Overview](#testing-strategy-overview)
2. [Unit Testing Framework](#unit-testing-framework)
3. [Integration Testing](#integration-testing)
4. [End-to-End Testing](#end-to-end-testing)
5. [Performance Testing](#performance-testing)
6. [Security Testing](#security-testing)
7. [Chaos Engineering](#chaos-engineering)
8. [Test Data Management](#test-data-management)
9. [Quality Gates and Automation](#quality-gates-and-automation)

---

## Testing Strategy Overview

### Testing Philosophy

1. **Shift-Left Testing:** Identify and fix issues early in the development cycle
2. **Test Pyramid:** Majority unit tests, fewer integration tests, minimal UI tests
3. **Risk-Based Testing:** Focus testing efforts on high-risk, high-impact areas
4. **Continuous Testing:** Automated testing integrated into CI/CD pipeline
5. **Production Testing:** Safe testing in production environments

### Testing Pyramid

```
                    ┌─────────────────┐
                    │   Manual/E2E    │  <- 5% (Exploratory, User Journey)
                    │     Testing     │
                ┌───┴─────────────────┴───┐
                │   Integration Testing   │  <- 15% (API, Service Integration)
                │     (Component)         │
            ┌───┴─────────────────────────┴───┐
            │        Unit Testing             │  <- 80% (Functions, Classes)
            │     (Fast, Isolated)            │
            └─────────────────────────────────┘

Testing Scope Distribution:
├── Unit Tests: 80% (Fast execution, high coverage)
├── Integration Tests: 15% (Service boundaries, API contracts)
├── End-to-End Tests: 4% (Critical user workflows)
└── Manual Testing: 1% (Exploratory, usability)
```

### Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Code Coverage** | >90% | Unit + Integration tests |
| **Test Execution Time** | <10 minutes | Full test suite |
| **Defect Escape Rate** | <2% | Production defects vs total |
| **Test Flakiness** | <1% | Failed tests that pass on retry |
| **Mean Time to Detection** | <5 minutes | Time to detect failures |
| **Mean Time to Recovery** | <15 minutes | Time to fix critical issues |

---

## Unit Testing Framework

### Python Testing Stack

```python
# conftest.py - Shared test configuration
import pytest
import asyncio
import asyncpg
import motor.motor_asyncio
import redis.asyncio as redis
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Test fixtures for database connections
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def postgres_pool():
    """PostgreSQL connection pool for testing."""
    pool = await asyncpg.create_pool(
        "postgresql://test:test@localhost:5432/test_db",
        min_size=1,
        max_size=5
    )
    yield pool
    await pool.close()

@pytest.fixture(scope="session")
async def mongodb_client():
    """MongoDB client for testing."""
    client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
    yield client
    client.close()

@pytest.fixture(scope="session")
async def redis_client():
    """Redis client for testing."""
    client = redis.Redis(host="localhost", port=6379, db=1)
    yield client
    await client.close()

@pytest.fixture
async def clean_databases(postgres_pool, mongodb_client, redis_client):
    """Clean databases before each test."""
    # Clean PostgreSQL
    async with postgres_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE conversations, messages CASCADE")
    
    # Clean MongoDB
    db = mongodb_client.test_db
    await db.conversations.drop()
    await db.messages.drop()
    
    # Clean Redis
    await redis_client.flushdb()
    
    yield
    
    # Cleanup after test
    await redis_client.flushdb()

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock = AsyncMock()
    mock.generate_response.return_value = {
        "response": "Test response",
        "confidence": 0.95,
        "tokens_used": 50,
        "cost_cents": 0.1
    }
    return mock

@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return {
        "conversation_id": "conv-123",
        "tenant_id": "tenant-456",
        "user_id": "user-789",
        "channel": "web",
        "status": "active",
        "context": {
            "intent_history": ["greeting"],
            "entities": {},
            "slots": {}
        }
    }

@pytest.fixture
def sample_message_data():
    """Sample message data for testing."""
    return {
        "message_id": "msg-123",
        "conversation_id": "conv-123",
        "content": {
            "type": "text",
            "text": "Hello, I need help"
        },
        "direction": "inbound",
        "timestamp": "2025-05-30T10:00:00Z"
    }
```

### Service-Level Unit Tests

```python
# tests/unit/test_chat_service.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from chat_service.message_processor import MessageProcessor
from chat_service.models import Message, Conversation

class TestMessageProcessor:
    """Test suite for message processing logic."""
    
    @pytest.fixture
    def message_processor(self, mock_llm_client, redis_client, mongodb_client):
        """Create message processor with mocked dependencies."""
        return MessageProcessor(
            llm_client=mock_llm_client,
            redis_client=redis_client,
            mongodb_client=mongodb_client
        )
    
    @pytest.mark.asyncio
    async def test_process_text_message_success(
        self, 
        message_processor, 
        sample_message_data,
        sample_conversation_data
    ):
        """Test successful text message processing."""
        # Arrange
        message = Message(**sample_message_data)
        
        # Mock conversation retrieval
        with patch.object(
            message_processor, 
            'get_conversation',
            return_value=Conversation(**sample_conversation_data)
        ):
            # Act
            result = await message_processor.process_message(message)
            
            # Assert
            assert result.status == "success"
            assert result.response_content is not None
            assert result.response_content.text is not None
            assert result.processing_metadata.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_process_message_with_intent_detection(
        self,
        message_processor,
        sample_message_data,
        mock_llm_client
    ):
        """Test message processing with intent detection."""
        # Arrange
        message_data = sample_message_data.copy()
        message_data["content"]["text"] = "I want to check my order status"
        message = Message(**message_data)
        
        # Mock intent detection response
        mock_llm_client.detect_intent.return_value = {
            "intent": "order_inquiry",
            "confidence": 0.92,
            "entities": {"product_type": "order"}
        }
        
        # Act
        result = await message_processor.process_message(message)
        
        # Assert
        assert result.status == "success"
        assert result.detected_intent == "order_inquiry"
        assert result.confidence_score >= 0.9
        mock_llm_client.detect_intent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_message_with_integration_call(
        self,
        message_processor,
        sample_message_data
    ):
        """Test message processing that triggers integration call."""
        # Arrange
        message_data = sample_message_data.copy()
        message_data["content"]["text"] = "Show me product ABC123"
        message = Message(**message_data)
        
        # Mock integration service
        mock_integration = AsyncMock()
        mock_integration.call_integration.return_value = {
            "status": "success",
            "data": {"product_name": "Test Product", "price": "$99.99"}
        }
        
        with patch.object(message_processor, 'integration_service', mock_integration):
            # Act
            result = await message_processor.process_message(message)
            
            # Assert
            assert result.status == "success"
            assert "Test Product" in result.response_content.text
            mock_integration.call_integration.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_message_handles_errors_gracefully(
        self,
        message_processor,
        sample_message_data,
        mock_llm_client
    ):
        """Test error handling in message processing."""
        # Arrange
        message = Message(**sample_message_data)
        mock_llm_client.generate_response.side_effect = Exception("LLM API Error")
        
        # Act
        result = await message_processor.process_message(message)
        
        # Assert
        assert result.status == "error"
        assert result.error_code == "LLM_API_ERROR"
        assert result.fallback_response is not None
    
    @pytest.mark.asyncio
    async def test_conversation_state_management(
        self,
        message_processor,
        sample_message_data,
        redis_client
    ):
        """Test conversation state persistence and retrieval."""
        # Arrange
        conversation_id = "conv-state-test"
        initial_state = {
            "current_intent": "greeting",
            "entities": {},
            "message_count": 0
        }
        
        # Store initial state
        await message_processor.save_conversation_state(conversation_id, initial_state)
        
        # Act
        retrieved_state = await message_processor.get_conversation_state(conversation_id)
        
        # Assert
        assert retrieved_state == initial_state
        
        # Update state
        updated_state = initial_state.copy()
        updated_state["message_count"] = 1
        updated_state["current_intent"] = "product_inquiry"
        
        await message_processor.save_conversation_state(conversation_id, updated_state)
        
        # Verify update
        final_state = await message_processor.get_conversation_state(conversation_id)
        assert final_state["message_count"] == 1
        assert final_state["current_intent"] == "product_inquiry"

    @pytest.mark.parametrize("message_type,expected_processing", [
        ("text", "text_processor"),
        ("image", "image_processor"), 
        ("file", "file_processor"),
        ("location", "location_processor")
    ])
    @pytest.mark.asyncio
    async def test_message_type_routing(
        self,
        message_processor,
        sample_message_data,
        message_type,
        expected_processing
    ):
        """Test that different message types are routed to correct processors."""
        # Arrange
        message_data = sample_message_data.copy()
        message_data["content"]["type"] = message_type
        message = Message(**message_data)
        
        # Mock processors
        with patch.object(message_processor, expected_processing) as mock_processor:
            mock_processor.return_value = {"status": "success", "response": "processed"}
            
            # Act
            await message_processor.process_message(message)
            
            # Assert
            mock_processor.assert_called_once()

# tests/unit/test_model_orchestrator.py
import pytest
from unittest.mock import AsyncMock, patch
from model_orchestrator.orchestrator import ModelOrchestrator
from model_orchestrator.providers import OpenAIProvider, AnthropicProvider

class TestModelOrchestrator:
    """Test suite for model orchestration logic."""
    
    @pytest.fixture
    def model_orchestrator(self):
        """Create model orchestrator with mocked providers."""
        openai_provider = AsyncMock(spec=OpenAIProvider)
        anthropic_provider = AsyncMock(spec=AnthropicProvider)
        
        return ModelOrchestrator(
            providers={
                "openai": openai_provider,
                "anthropic": anthropic_provider
            }
        )
    
    @pytest.mark.asyncio
    async def test_route_to_primary_provider(self, model_orchestrator):
        """Test routing to primary provider when available."""
        # Arrange
        request = {
            "text": "What is AI?",
            "provider_preference": "openai",
            "max_cost_cents": 10
        }
        
        model_orchestrator.providers["openai"].generate_response.return_value = {
            "response": "AI is artificial intelligence",
            "cost_cents": 5,
            "tokens_used": 20
        }
        
        # Act
        result = await model_orchestrator.process_request(request)
        
        # Assert
        assert result["status"] == "success"
        assert result["provider_used"] == "openai"
        assert result["cost_cents"] == 5
        model_orchestrator.providers["openai"].generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_to_secondary_provider(self, model_orchestrator):
        """Test fallback when primary provider fails."""
        # Arrange
        request = {
            "text": "What is AI?",
            "provider_preference": "openai",
            "fallback_providers": ["anthropic"]
        }
        
        # Primary provider fails
        model_orchestrator.providers["openai"].generate_response.side_effect = Exception("API Error")
        
        # Fallback provider succeeds
        model_orchestrator.providers["anthropic"].generate_response.return_value = {
            "response": "AI stands for artificial intelligence",
            "cost_cents": 8,
            "tokens_used": 25
        }
        
        # Act
        result = await model_orchestrator.process_request(request)
        
        # Assert
        assert result["status"] == "success"
        assert result["provider_used"] == "anthropic"
        assert result["fallback_applied"] == True
        model_orchestrator.providers["anthropic"].generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cost_based_provider_selection(self, model_orchestrator):
        """Test provider selection based on cost optimization."""
        # Arrange
        request = {
            "text": "Simple question",
            "cost_optimization": True,
            "max_cost_cents": 5
        }
        
        # Configure provider costs
        model_orchestrator.get_estimated_cost = AsyncMock(side_effect=lambda provider, text: {
            "openai": 8,  # Too expensive
            "anthropic": 3  # Within budget
        }[provider])
        
        model_orchestrator.providers["anthropic"].generate_response.return_value = {
            "response": "Simple answer",
            "cost_cents": 3,
            "tokens_used": 15
        }
        
        # Act
        result = await model_orchestrator.process_request(request)
        
        # Assert
        assert result["provider_used"] == "anthropic"
        assert result["cost_cents"] <= 5

# Performance-focused unit tests
class TestPerformanceRequirements:
    """Test suite focused on performance requirements."""
    
    @pytest.mark.asyncio
    async def test_message_processing_performance(self, message_processor, sample_message_data):
        """Test that message processing meets performance requirements."""
        import time
        
        # Arrange
        message = Message(**sample_message_data)
        
        # Act
        start_time = time.time()
        result = await message_processor.process_message(message)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Assert performance requirement: <300ms for simple messages
        assert processing_time_ms < 300
        assert result.processing_metadata.processing_time_ms < 300
    
    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self, message_processor, sample_message_data):
        """Test concurrent message processing performance."""
        import asyncio
        import time
        
        # Arrange
        messages = [
            Message(**{**sample_message_data, "message_id": f"msg-{i}"})
            for i in range(10)
        ]
        
        # Act
        start_time = time.time()
        results = await asyncio.gather(*[
            message_processor.process_message(msg) for msg in messages
        ])
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        
        # Assert
        assert len(results) == 10
        assert all(r.status == "success" for r in results)
        # Concurrent processing should be significantly faster than sequential
        assert total_time_ms < 1000  # 10 messages in under 1 second
```

### Test Utilities and Helpers

```python
# tests/utils/test_helpers.py
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class TestConversation:
    """Helper class for creating test conversations."""
    conversation_id: str
    tenant_id: str
    user_id: str
    channel: str = "web"
    status: str = "active"
    message_count: int = 0
    
    @classmethod
    def create_sample(cls, **overrides) -> 'TestConversation':
        """Create a sample conversation with optional overrides."""
        defaults = {
            "conversation_id": f"conv-{uuid.uuid4().hex[:8]}",
            "tenant_id": f"tenant-{uuid.uuid4().hex[:8]}",
            "user_id": f"user-{uuid.uuid4().hex[:8]}"
        }
        defaults.update(overrides)
        return cls(**defaults)

class AsyncTestClient:
    """Enhanced async test client with convenience methods."""
    
    def __init__(self, app, base_url: str = "http://testserver"):
        from httpx import AsyncClient
        self.client = AsyncClient(app=app, base_url=base_url)
        self.auth_token = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def authenticate(self, user_data: Dict[str, Any]) -> str:
        """Authenticate user and store token."""
        response = await self.client.post("/api/v2/auth/login", json=user_data)
        assert response.status_code == 200
        
        token_data = response.json()
        self.auth_token = token_data["access_token"]
        return self.auth_token
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if not self.auth_token:
            raise ValueError("Not authenticated. Call authenticate() first.")
        return {"Authorization": f"Bearer {self.auth_token}"}
    
    async def post_authenticated(self, url: str, **kwargs) -> Any:
        """Make authenticated POST request."""
        headers = kwargs.get("headers", {})
        headers.update(self.get_auth_headers())
        kwargs["headers"] = headers
        return await self.client.post(url, **kwargs)
    
    async def get_authenticated(self, url: str, **kwargs) -> Any:
        """Make authenticated GET request."""
        headers = kwargs.get("headers", {})
        headers.update(self.get_auth_headers())
        kwargs["headers"] = headers
        return await self.client.get(url, **kwargs)

class DatabaseFixtures:
    """Helper class for managing test database fixtures."""
    
    @staticmethod
    async def create_tenant(postgres_pool, tenant_data: Dict[str, Any] = None) -> str:
        """Create a test tenant and return tenant ID."""
        tenant_id = str(uuid.uuid4())
        data = {
            "tenant_id": tenant_id,
            "name": "Test Tenant",
            "status": "active",
            "plan_type": "starter",
            **(tenant_data or {})
        }
        
        async with postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenants (tenant_id, name, status, plan_type, created_at)
                VALUES ($1, $2, $3, $4, $5)
            """, data["tenant_id"], data["name"], data["status"], 
                data["plan_type"], datetime.utcnow())
        
        return tenant_id
    
    @staticmethod
    async def create_user(postgres_pool, tenant_id: str, user_data: Dict[str, Any] = None) -> str:
        """Create a test user and return user ID."""
        user_id = str(uuid.uuid4())
        data = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "email": "test@example.com",
            "role": "admin",
            "status": "active",
            **(user_data or {})
        }
        
        async with postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_users (user_id, tenant_id, email, role, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, data["user_id"], data["tenant_id"], data["email"],
                data["role"], data["status"], datetime.utcnow())
        
        return user_id
    
    @staticmethod
    async def create_conversation(mongodb_client, conversation_data: Dict[str, Any] = None) -> str:
        """Create a test conversation and return conversation ID."""
        conversation_id = str(uuid.uuid4())
        data = {
            "conversation_id": conversation_id,
            "tenant_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "channel": "web",
            "status": "active",
            "started_at": datetime.utcnow(),
            "context": {},
            **(conversation_data or {})
        }
        
        db = mongodb_client.test_db
        await db.conversations.insert_one(data)
        return conversation_id

class TestDataGenerator:
    """Generate realistic test data for various scenarios."""
    
    @staticmethod
    def generate_messages(count: int, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Generate a list of test messages."""
        conversation_id = conversation_id or str(uuid.uuid4())
        messages = []
        
        for i in range(count):
            message = {
                "message_id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "sequence_number": i + 1,
                "direction": "inbound" if i % 2 == 0 else "outbound",
                "timestamp": datetime.utcnow() - timedelta(minutes=count - i),
                "content": {
                    "type": "text",
                    "text": f"Test message {i + 1}"
                }
            }
            messages.append(message)
        
        return messages
    
    @staticmethod
    def generate_load_test_scenarios() -> List[Dict[str, Any]]:
        """Generate scenarios for load testing."""
        return [
            {
                "name": "simple_qa",
                "weight": 60,
                "messages": [
                    {"text": "Hello", "expected_intent": "greeting"},
                    {"text": "What are your hours?", "expected_intent": "business_hours"}
                ]
            },
            {
                "name": "product_search",
                "weight": 25,
                "messages": [
                    {"text": "I'm looking for a laptop", "expected_intent": "product_search"},
                    {"text": "Under $1000", "expected_intent": "price_filter"},
                    {"text": "Show me options", "expected_intent": "show_results"}
                ]
            },
            {
                "name": "support_escalation",
                "weight": 15,
                "messages": [
                    {"text": "I have a problem", "expected_intent": "support_request"},
                    {"text": "My order is missing", "expected_intent": "order_issue"},
                    {"text": "I want to speak to someone", "expected_intent": "human_escalation"}
                ]
            }
        ]

# Test decorators for common patterns
def require_database(func):
    """Decorator to ensure test has database dependencies."""
    func._requires_database = True
    return func

def require_external_services(services: List[str]):
    """Decorator to mark tests that require external services."""
    def decorator(func):
        func._requires_external_services = services
        return func
    return decorator

def performance_test(max_duration_ms: int):
    """Decorator to mark performance tests with time limits."""
    def decorator(func):
        func._performance_test = True
        func._max_duration_ms = max_duration_ms
        return func
    return decorator

# Example usage of decorators
@require_database
@performance_test(max_duration_ms=500)
async def test_fast_database_query():
    """Test that database queries complete within performance limits."""
    pass

@require_external_services(["openai", "anthropic"])
async def test_llm_integration():
    """Test that requires external LLM services."""
    pass
```

---

## Integration Testing

### API Contract Testing

```python
# tests/integration/test_api_contracts.py
import pytest
from httpx import AsyncClient
from tests.utils.test_helpers import AsyncTestClient, DatabaseFixtures

class TestChatServiceAPIContract:
    """Test API contracts for Chat Service."""
    
    @pytest.mark.asyncio
    async def test_send_message_api_contract(self, app, clean_databases, postgres_pool):
        """Test send message API contract compliance."""
        # Arrange
        tenant_id = await DatabaseFixtures.create_tenant(postgres_pool)
        user_id = await DatabaseFixtures.create_user(postgres_pool, tenant_id)
        
        async with AsyncTestClient(app) as client:
            await client.authenticate({
                "email": "test@example.com",
                "password": "test123",
                "tenant_id": tenant_id
            })
            
            # Valid request payload
            message_payload = {
                "message_id": "msg-123",
                "conversation_id": "conv-456",
                "user_id": "user-789",
                "channel": "web",
                "timestamp": "2025-05-30T10:00:00Z",
                "content": {
                    "type": "text",
                    "text": "Hello, I need help"
                }
            }
            
            # Act
            response = await client.post_authenticated(
                "/api/v2/chat/message",
                json=message_payload,
                headers={"X-Tenant-ID": tenant_id}
            )
            
            # Assert response structure
            assert response.status_code == 200
            
            response_data = response.json()
            assert "status" in response_data
            assert "data" in response_data
            assert "meta" in response_data
            
            # Validate data structure
            data = response_data["data"]
            assert "message_id" in data
            assert "conversation_id" in data
            assert "response" in data
            assert "processing_metadata" in data
            
            # Validate response content
            response_content = data["response"]
            assert "type" in response_content
            assert "text" in response_content
            assert "confidence_score" in response_content
            
            # Validate metadata
            metadata = response_data["meta"]
            assert "request_id" in metadata
            assert "timestamp" in metadata
            assert "processing_time_ms" in metadata
    
    @pytest.mark.asyncio
    async def test_send_message_validation_errors(self, app):
        """Test API validation error responses."""
        async with AsyncTestClient(app) as client:
            # Test missing required fields
            invalid_payloads = [
                {},  # Empty payload
                {"message_id": "msg-123"},  # Missing required fields
                {
                    "message_id": "msg-123",
                    "content": {"type": "invalid_type"}  # Invalid content type
                }
            ]
            
            for payload in invalid_payloads:
                response = await client.client.post("/api/v2/chat/message", json=payload)
                
                assert response.status_code == 400
                error_response = response.json()
                assert "error" in error_response
                assert error_response["error"]["code"] == "VALIDATION_ERROR"
    
    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self, app, postgres_pool):
        """Test API rate limiting behavior."""
        # Arrange
        tenant_id = await DatabaseFixtures.create_tenant(postgres_pool)
        
        async with AsyncTestClient(app) as client:
            await client.authenticate({
                "email": "test@example.com", 
                "password": "test123",
                "tenant_id": tenant_id
            })
            
            # Make requests up to rate limit
            for i in range(100):  # Assuming 100/minute limit for test
                response = await client.post_authenticated(
                    "/api/v2/chat/message",
                    json={
                        "message_id": f"msg-{i}",
                        "user_id": "user-test",
                        "channel": "web",
                        "content": {"type": "text", "text": f"Message {i}"}
                    },
                    headers={"X-Tenant-ID": tenant_id}
                )
                
                if response.status_code == 429:
                    # Rate limit hit
                    assert "X-RateLimit-Limit" in response.headers
                    assert "X-RateLimit-Remaining" in response.headers
                    assert "Retry-After" in response.headers
                    break
            else:
                pytest.fail("Rate limit was not enforced")

class TestServiceIntegration:
    """Test integration between internal services."""
    
    @pytest.mark.asyncio
    async def test_chat_service_to_mcp_integration(self, app, clean_databases):
        """Test Chat Service to MCP Engine integration."""
        async with AsyncTestClient(app) as client:
            # Simulate message that should trigger MCP processing
            message_payload = {
                "message_id": "msg-mcp-test",
                "user_id": "user-test",
                "channel": "web",
                "content": {
                    "type": "text", 
                    "text": "I want to check my order status"
                }
            }
            
            response = await client.client.post("/api/v2/chat/message", json=message_payload)
            
            assert response.status_code == 200
            response_data = response.json()
            
            # Verify MCP processing occurred
            assert "conversation_state" in response_data["data"]
            state = response_data["data"]["conversation_state"]
            assert state["current_intent"] == "order_inquiry"
    
    @pytest.mark.asyncio
    async def test_model_orchestrator_fallback_chain(self, app):
        """Test model orchestrator fallback behavior."""
        # This test would mock primary provider failure
        # and verify fallback to secondary provider
        
        with patch('model_orchestrator.providers.OpenAIProvider.generate_response') as mock_openai:
            mock_openai.side_effect = Exception("Primary provider failed")
            
            with patch('model_orchestrator.providers.AnthropicProvider.generate_response') as mock_anthropic:
                mock_anthropic.return_value = {
                    "response": "Fallback response",
                    "cost_cents": 5,
                    "tokens_used": 20
                }
                
                async with AsyncTestClient(app) as client:
                    response = await client.client.post(
                        "/api/v2/model/process",
                        json={
                            "text": "Test message",
                            "provider_preference": "openai",
                            "fallback_providers": ["anthropic"]
                        }
                    )
                    
                    assert response.status_code == 200
                    result = response.json()
                    
                    assert result["data"]["provider_used"] == "anthropic"
                    assert result["data"]["fallback_applied"] == True
```

### Database Integration Testing

```python
# tests/integration/test_database_integration.py
import pytest
import asyncio
from datetime import datetime, timedelta
from tests.utils.test_helpers import DatabaseFixtures

class TestPostgreSQLIntegration:
    """Test PostgreSQL database operations."""
    
    @pytest.mark.asyncio
    async def test_tenant_crud_operations(self, postgres_pool):
        """Test tenant CRUD operations."""
        # Create
        tenant_id = await DatabaseFixtures.create_tenant(postgres_pool, {
            "name": "Integration Test Tenant",
            "plan_type": "professional"
        })
        
        # Read
        async with postgres_pool.acquire() as conn:
            tenant = await conn.fetchrow(
                "SELECT * FROM tenants WHERE tenant_id = $1", tenant_id
            )
            
            assert tenant is not None
            assert tenant["name"] == "Integration Test Tenant"
            assert tenant["plan_type"] == "professional"
            assert tenant["status"] == "active"
        
        # Update
        async with postgres_pool.acquire() as conn:
            await conn.execute(
                "UPDATE tenants SET name = $1 WHERE tenant_id = $2",
                "Updated Tenant Name", tenant_id
            )
            
            updated_tenant = await conn.fetchrow(
                "SELECT * FROM tenants WHERE tenant_id = $1", tenant_id
            )
            assert updated_tenant["name"] == "Updated Tenant Name"
        
        # Delete
        async with postgres_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM tenants WHERE tenant_id = $1", tenant_id
            )
            
            deleted_tenant = await conn.fetchrow(
                "SELECT * FROM tenants WHERE tenant_id = $1", tenant_id
            )
            assert deleted_tenant is None
    
    @pytest.mark.asyncio
    async def test_user_permissions_and_roles(self, postgres_pool):
        """Test user role and permission management."""
        tenant_id = await DatabaseFixtures.create_tenant(postgres_pool)
        
        # Create users with different roles
        admin_user_id = await DatabaseFixtures.create_user(postgres_pool, tenant_id, {
            "email": "admin@test.com",
            "role": "admin"
        })
        
        member_user_id = await DatabaseFixtures.create_user(postgres_pool, tenant_id, {
            "email": "member@test.com", 
            "role": "member"
        })
        
        # Verify role-based queries work correctly
        async with postgres_pool.acquire() as conn:
            admin_users = await conn.fetch(
                "SELECT * FROM tenant_users WHERE tenant_id = $1 AND role = $2",
                tenant_id, "admin"
            )
            assert len(admin_users) == 1
            assert admin_users[0]["user_id"] == admin_user_id
            
            all_users = await conn.fetch(
                "SELECT * FROM tenant_users WHERE tenant_id = $1",
                tenant_id
            )
            assert len(all_users) == 2

class TestMongoDBIntegration:
    """Test MongoDB operations."""
    
    @pytest.mark.asyncio
    async def test_conversation_document_operations(self, mongodb_client):
        """Test conversation document CRUD operations."""
        db = mongodb_client.test_db
        
        # Create conversation
        conversation_data = {
            "conversation_id": "conv-integration-test",
            "tenant_id": "tenant-123",
            "user_id": "user-456",
            "channel": "web",
            "status": "active",
            "started_at": datetime.utcnow(),
            "context": {
                "intent_history": ["greeting"],
                "entities": {},
                "slots": {"user_name": "John"}
            }
        }
        
        # Insert
        result = await db.conversations.insert_one(conversation_data)
        assert result.inserted_id is not None
        
        # Read
        conversation = await db.conversations.find_one(
            {"conversation_id": "conv-integration-test"}
        )
        assert conversation is not None
        assert conversation["tenant_id"] == "tenant-123"
        assert conversation["context"]["slots"]["user_name"] == "John"
        
        # Update
        await db.conversations.update_one(
            {"conversation_id": "conv-integration-test"},
            {"$set": {"status": "completed", "completed_at": datetime.utcnow()}}
        )
        
        updated_conversation = await db.conversations.find_one(
            {"conversation_id": "conv-integration-test"}
        )
        assert updated_conversation["status"] == "completed"
        assert "completed_at" in updated_conversation
        
        # Delete
        delete_result = await db.conversations.delete_one(
            {"conversation_id": "conv-integration-test"}
        )
        assert delete_result.deleted_count == 1
    
    @pytest.mark.asyncio
    async def test_message_collection_indexing(self, mongodb_client):
        """Test message collection queries and indexing performance."""
        db = mongodb_client.test_db
        
        # Insert test messages
        messages = []
        conversation_id = "conv-perf-test"
        
        for i in range(1000):
            message = {
                "message_id": f"msg-{i}",
                "conversation_id": conversation_id,
                "sequence_number": i + 1,
                "timestamp": datetime.utcnow() - timedelta(minutes=1000 - i),
                "direction": "inbound" if i % 2 == 0 else "outbound",
                "content": {"type": "text", "text": f"Message {i}"}
            }
            messages.append(message)
        
        await db.messages.insert_many(messages)
        
        # Test query performance
        import time
        
        # Query by conversation_id (should be fast with index)
        start_time = time.time()
        conversation_messages = await db.messages.find(
            {"conversation_id": conversation_id}
        ).to_list(length=None)
        query_time = (time.time() - start_time) * 1000
        
        assert len(conversation_messages) == 1000
        assert query_time < 100  # Should complete in <100ms with proper indexing
        
        # Query with sorting (should be fast with compound index)
        start_time = time.time()
        sorted_messages = await db.messages.find(
            {"conversation_id": conversation_id}
        ).sort("sequence_number", 1).limit(10).to_list(length=10)
        sort_query_time = (time.time() - start_time) * 1000
        
        assert len(sorted_messages) == 10
        assert sorted_messages[0]["sequence_number"] == 1
        assert sort_query_time < 50  # Should be very fast

class TestRedisIntegration:
    """Test Redis operations."""
    
    @pytest.mark.asyncio
    async def test_session_management(self, redis_client):
        """Test Redis session management operations."""
        session_id = "session-integration-test"
        session_data = {
            "user_id": "user-123",
            "tenant_id": "tenant-456", 
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
        
        # Set session with TTL
        await redis_client.setex(
            f"session:{session_id}",
            3600,  # 1 hour TTL
            json.dumps(session_data)
        )
        
        # Retrieve session
        stored_session = await redis_client.get(f"session:{session_id}")
        assert stored_session is not None
        
        retrieved_data = json.loads(stored_session)
        assert retrieved_data["user_id"] == "user-123"
        assert retrieved_data["tenant_id"] == "tenant-456"
        
        # Test TTL
        ttl = await redis_client.ttl(f"session:{session_id}")
        assert ttl > 3500  # Should be close to 3600 seconds
    
    @pytest.mark.asyncio
    async def test_rate_limiting_with_redis(self, redis_client):
        """Test rate limiting implementation with Redis."""
        user_id = "user-rate-limit-test"
        rate_limit_key = f"rate_limit:{user_id}:minute"
        
        # Simulate rate limiting with sliding window
        import time
        current_minute = int(time.time() // 60)
        
        # Add requests to current minute window
        for i in range(10):
            await redis_client.zadd(
                rate_limit_key,
                {f"request-{i}": time.time()}
            )
        
        # Set expiration
        await redis_client.expire(rate_limit_key, 60)
        
        # Check current count
        count = await redis_client.zcard(rate_limit_key)
        assert count == 10
        
        # Remove old entries (simulate sliding window)
        cutoff_time = time.time() - 60
        removed = await redis_client.zremrangebyscore(rate_limit_key, 0, cutoff_time)
        
        # Current requests should still be there
        remaining_count = await redis_client.zcard(rate_limit_key)
        assert remaining_count == 10
```

---

## End-to-End Testing

### User Journey Testing

```python
# tests/e2e/test_user_journeys.py
import pytest
from playwright.async_api import async_playwright
from tests.utils.test_helpers import DatabaseFixtures

class TestCompleteUserJourneys:
    """Test complete user journeys across the platform."""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_tenant_onboarding_journey(self, postgres_pool, mongodb_client):
        """Test complete tenant onboarding process."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                # Step 1: Navigate to registration page
                await page.goto("https://app.chatbot-platform.com/register")
                
                # Step 2: Fill registration form
                await page.fill("[data-testid=company-name]", "Test Company")
                await page.fill("[data-testid=email]", "admin@testcompany.com")
                await page.fill("[data-testid=password]", "SecurePassword123!")
                await page.fill("[data-testid=confirm-password]", "SecurePassword123!")
                await page.click("[data-testid=submit-registration]")
                
                # Step 3: Verify email confirmation page
                await page.wait_for_selector("[data-testid=email-confirmation]")
                confirmation_text = await page.text_content("[data-testid=confirmation-message]")
                assert "verify your email" in confirmation_text.lower()
                
                # Step 4: Simulate email verification (in real test, would check email)
                # For testing, directly verify in database
                async with postgres_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE tenant_users SET email_verified = true WHERE email = $1",
                        "admin@testcompany.com"
                    )
                
                # Step 5: Login with verified account
                await page.goto("https://app.chatbot-platform.com/login")
                await page.fill("[data-testid=email]", "admin@testcompany.com")
                await page.fill("[data-testid=password]", "SecurePassword123!")
                await page.click("[data-testid=login-button]")
                
                # Step 6: Verify successful login and dashboard access
                await page.wait_for_selector("[data-testid=dashboard]")
                
                # Step 7: Check onboarding checklist
                checklist_items = await page.query_selector_all("[data-testid=checklist-item]")
                assert len(checklist_items) >= 3  # Should have setup steps
                
                # Step 8: Complete basic setup
                await page.click("[data-testid=setup-channels]")
                await page.wait_for_selector("[data-testid=channel-setup]")
                
                # Enable web widget
                await page.click("[data-testid=enable-web-widget]")
                await page.wait_for_selector("[data-testid=widget-config]")
                
                # Configure basic settings
                await page.fill("[data-testid=widget-title]", "Help Assistant")
                await page.fill("[data-testid=widget-welcome]", "How can I help you today?")
                await page.click("[data-testid=save-widget-config]")
                
                # Step 9: Verify configuration saved
                await page.wait_for_selector("[data-testid=config-saved]")
                
                # Step 10: Test the chat widget
                await page.click("[data-testid=test-widget]")
                await page.wait_for_selector("[data-testid=chat-widget]")
                
                # Send test message
                await page.fill("[data-testid=chat-input]", "Hello, this is a test")
                await page.click("[data-testid=send-message]")
                
                # Verify response
                await page.wait_for_selector("[data-testid=bot-response]")
                response_text = await page.text_content("[data-testid=bot-response]")
                assert len(response_text) > 0
                
            finally:
                await browser.close()
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_conversation_flow_journey(self, app):
        """Test complete conversation flow from start to finish."""
        async with AsyncTestClient(app) as client:
            # Setup test tenant and user
            tenant_id = await DatabaseFixtures.create_tenant(postgres_pool, {
                "name": "E2E Test Tenant",
                "plan_type": "professional"
            })
            
            # Authenticate
            await client.authenticate({
                "email": "test@e2e-tenant.com",
                "password": "test123",
                "tenant_id": tenant_id
            })
            
            conversation_id = None
            
            # Step 1: Start conversation with greeting
            response = await client.post_authenticated(
                "/api/v2/chat/message",
                json={
                    "message_id": "msg-greeting",
                    "user_id": "e2e-user",
                    "channel": "web",
                    "content": {"type": "text", "text": "Hello"}
                },
                headers={"X-Tenant-ID": tenant_id}
            )
            
            assert response.status_code == 200
            data = response.json()["data"]
            conversation_id = data["conversation_id"]
            
            # Verify greeting response
            assert "hello" in data["response"]["text"].lower()
            assert data["conversation_state"]["current_intent"] == "greeting"
            
            # Step 2: Express product interest
            response = await client.post_authenticated(
                "/api/v2/chat/message",
                json={
                    "message_id": "msg-product-interest",
                    "conversation_id": conversation_id,
                    "user_id": "e2e-user",
                    "channel": "web",
                    "content": {"type": "text", "text": "I'm looking for a laptop"}
                },
                headers={"X-Tenant-ID": tenant_id}
            )
            
            assert response.status_code == 200
            data = response.json()["data"]
            
            # Verify intent detection
            assert data["conversation_state"]["current_intent"] == "product_search"
            assert "laptop" in str(data["conversation_state"]["entities"])
            
            # Step 3: Provide specifications
            response = await client.post_authenticated(
                "/api/v2/chat/message", 
                json={
                    "message_id": "msg-specs",
                    "conversation_id": conversation_id,
                    "user_id": "e2e-user",
                    "channel": "web",
                    "content": {"type": "text", "text": "Under $1000, for programming"}
                },
                headers={"X-Tenant-ID": tenant_id}
            )
            
            assert response.status_code == 200
            data = response.json()["data"]
            
            # Verify slot filling
            slots = data["conversation_state"]["slots"]
            assert "budget" in slots
            assert "use_case" in slots
            
            # Step 4: Request product recommendations
            response = await client.post_authenticated(
                "/api/v2/chat/message",
                json={
                    "message_id": "msg-recommendations",
                    "conversation_id": conversation_id,
                    "user_id": "e2e-user", 
                    "channel": "web",
                    "content": {"type": "text", "text": "Show me some options"}
                },
                headers={"X-Tenant-ID": tenant_id}
            )
            
            assert response.status_code == 200
            data = response.json()["data"]
            
            # Verify product recommendations (would trigger integration)
            assert data["response"]["type"] in ["carousel", "text"]
            
            # Step 5: Get conversation summary
            summary_response = await client.get_authenticated(
                f"/api/v2/chat/conversations/{conversation_id}",
                headers={"X-Tenant-ID": tenant_id}
            )
            
            assert summary_response.status_code == 200
            conversation_data = summary_response.json()["data"]
            
            # Verify conversation progression
            assert conversation_data["message_count"] >= 4
            assert conversation_data["summary"]["primary_intent"] == "product_search"
            assert conversation_data["summary"]["resolution_status"] in ["in_progress", "resolved"]

class TestCrossChannelJourneys:
    """Test user journeys across multiple channels."""
    
    @pytest.mark.asyncio 
    @pytest.mark.e2e
    async def test_cross_channel_conversation_continuity(self, app):
        """Test conversation continuity across channels."""
        async with AsyncTestClient(app) as client:
            tenant_id = await DatabaseFixtures.create_tenant(postgres_pool)
            
            await client.authenticate({
                "email": "crosschannel@test.com",
                "password": "test123", 
                "tenant_id": tenant_id
            })
            
            user_id = "cross-channel-user"
            session_id = "cross-channel-session"
            
            # Step 1: Start conversation on web
            web_response = await client.post_authenticated(
                "/api/v2/chat/message",
                json={
                    "message_id": "msg-web-start",
                    "user_id": user_id,
                    "session_id": session_id,
                    "channel": "web",
                    "content": {"type": "text", "text": "I need help with my order"}
                },
                headers={"X-Tenant-ID": tenant_id}
            )
            
            assert web_response.status_code == 200
            web_data = web_response.json()["data"]
            conversation_id = web_data["conversation_id"]
            
            # Verify initial state
            assert web_data["conversation_state"]["current_intent"] == "order_inquiry"
            
            # Step 2: Continue conversation on WhatsApp
            whatsapp_response = await client.post_authenticated(
                "/api/v2/chat/message",
                json={
                    "message_id": "msg-whatsapp-continue",
                    "conversation_id": conversation_id,  # Same conversation
                    "user_id": user_id,
                    "session_id": session_id,
                    "channel": "whatsapp",
                    "content": {"type": "text", "text": "Order number ORD123456"},
                    "channel_metadata": {
                        "platform_user_id": "+1234567890",
                        "platform_message_id": "whatsapp_msg_123"
                    }
                },
                headers={"X-Tenant-ID": tenant_id}
            )
            
            assert whatsapp_response.status_code == 200
            whatsapp_data = whatsapp_response.json()["data"]
            
            # Verify conversation continuity
            assert whatsapp_data["conversation_id"] == conversation_id
            assert "order_number" in whatsapp_data["conversation_state"]["slots"]
            
            # Step 3: Get conversation from either channel
            conversation_response = await client.get_authenticated(
                f"/api/v2/chat/conversations/{conversation_id}",
                headers={"X-Tenant-ID": tenant_id}
            )
            
            assert conversation_response.status_code == 200
            conversation = conversation_response.json()["data"]
            
            # Verify messages from both channels are included
            channels_used = set(msg["channel"] for msg in conversation["messages"])
            assert "web" in channels_used
            assert "whatsapp" in channels_used
            
            # Verify context is maintained
            assert conversation["context"]["slots"]["order_number"] == "ORD123456"
```

---

## Performance Testing

### Load Testing Framework

```python
# tests/performance/test_load_scenarios.py
import pytest
import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import List, Dict, Any
import statistics

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    name: str
    duration_seconds: int
    concurrent_users: int
    ramp_up_seconds: int
    target_rps: int
    success_rate_threshold: float = 0.95
    response_time_p95_threshold_ms: int = 1000

class LoadTestRunner:
    """Framework for running load tests."""
    
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url
        self.auth_token = auth_token
        self.results = []
    
    async def run_load_test(self, config: LoadTestConfig, scenario_func):
        """Run a load test scenario."""
        print(f"Starting load test: {config.name}")
        print(f"Duration: {config.duration_seconds}s, Users: {config.concurrent_users}")
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=config.concurrent_users * 2,
            limit_per_host=config.concurrent_users * 2
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Calculate user spawn rate
            spawn_rate = config.concurrent_users / config.ramp_up_seconds
            
            # Start users gradually
            tasks = []
            start_time = time.time()
            
            for user_id in range(config.concurrent_users):
                # Calculate when this user should start
                user_start_delay = user_id / spawn_rate
                
                task = asyncio.create_task(
                    self._run_user_scenario(
                        session, user_id, config, scenario_func, 
                        start_time + user_start_delay
                    )
                )
                tasks.append(task)
            
            # Wait for all users to complete
            await asyncio.gather(*tasks)
            
            # Analyze results
            return self._analyze_results(config)
    
    async def _run_user_scenario(self, session, user_id: int, config: LoadTestConfig, 
                                scenario_func, start_time: float):
        """Run scenario for a single user."""
        # Wait for user start time
        delay = start_time - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        
        scenario_start_time = time.time()
        end_time = scenario_start_time + config.duration_seconds
        
        while time.time() < end_time:
            try:
                # Run scenario
                start_request_time = time.time()
                success = await scenario_func(session, user_id)
                response_time = (time.time() - start_request_time) * 1000
                
                # Record result
                self.results.append({
                    "user_id": user_id,
                    "timestamp": start_request_time,
                    "response_time_ms": response_time,
                    "success": success
                })
                
                # Calculate delay to maintain target RPS
                target_interval = config.concurrent_users / config.target_rps
                elapsed = time.time() - start_request_time
                sleep_time = max(0, target_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                print(f"User {user_id} error: {e}")
                self.results.append({
                    "user_id": user_id,
                    "timestamp": time.time(),
                    "response_time_ms": 0,
                    "success": False
                })
    
    def _analyze_results(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Analyze load test results."""
        if not self.results:
            return {"error": "No results collected"}
        
        # Calculate metrics
        successful_requests = [r for r in self.results if r["success"]]
        total_requests = len(self.results)
        successful_count = len(successful_requests)
        
        success_rate = successful_count / total_requests if total_requests > 0 else 0
        
        if successful_requests:
            response_times = [r["response_time_ms"] for r in successful_requests]
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
            min_response_time = 0
            max_response_time = 0
        
        # Calculate throughput
        if self.results:
            test_duration = max(r["timestamp"] for r in self.results) - min(r["timestamp"] for r in self.results)
            throughput_rps = successful_count / test_duration if test_duration > 0 else 0
        else:
            throughput_rps = 0
        
        results = {
            "config": config,
            "total_requests": total_requests,
            "successful_requests": successful_count,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time,
            "p95_response_time_ms": p95_response_time,
            "p99_response_time_ms": p99_response_time,
            "min_response_time_ms": min_response_time,
            "max_response_time_ms": max_response_time,
            "throughput_rps": throughput_rps
        }
        
        # Check if test passed thresholds
        results["passed"] = (
            success_rate >= config.success_rate_threshold and
            p95_response_time <= config.response_time_p95_threshold_ms
        )
        
        return results

class ChatServiceLoadTests:
    """Load tests for Chat Service."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_message_processing_load(self):
        """Test message processing under load."""
        config = LoadTestConfig(
            name="Message Processing Load Test",
            duration_seconds=300,  # 5 minutes
            concurrent_users=100,
            ramp_up_seconds=60,
            target_rps=500,
            response_time_p95_threshold_ms=500
        )
        
        async def message_scenario(session: aiohttp.ClientSession, user_id: int) -> bool:
            """Single message scenario for load testing."""
            try:
                message_payload = {
                    "message_id": f"load-test-{user_id}-{int(time.time())}",
                    "user_id": f"load-user-{user_id}",
                    "channel": "web",
                    "content": {
                        "type": "text",
                        "text": f"Load test message from user {user_id}"
                    }
                }
                
                headers = {"Content-Type": "application/json"}
                if self.auth_token:
                    headers["Authorization"] = f"Bearer {self.auth_token}"
                
                async with session.post(
                    f"{self.base_url}/api/v2/chat/message",
                    json=message_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
                    
            except Exception as e:
                print(f"Request failed: {e}")
                return False
        
        runner = LoadTestRunner("https://api.chatbot-platform.com")
        results = await runner.run_load_test(config, message_scenario)
        
        # Assert performance requirements
        assert results["passed"], f"Load test failed: {results}"
        assert results["success_rate"] >= 0.95
        assert results["p95_response_time_ms"] <= 500
        assert results["throughput_rps"] >= 400  # Should achieve 80% of target
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_conversation_flow_load(self):
        """Test complete conversation flows under load."""
        config = LoadTestConfig(
            name="Conversation Flow Load Test",
            duration_seconds=600,  # 10 minutes
            concurrent_users=50,
            ramp_up_seconds=120,
            target_rps=100
        )
        
        async def conversation_scenario(session: aiohttp.ClientSession, user_id: int) -> bool:
            """Multi-message conversation scenario."""
            try:
                conversation_messages = [
                    "Hello, I need help",
                    "I'm looking for a laptop", 
                    "Under $1000",
                    "For programming work",
                    "Show me some options"
                ]
                
                conversation_id = None
                
                for i, message_text in enumerate(conversation_messages):
                    message_payload = {
                        "message_id": f"conv-{user_id}-{i}",
                        "user_id": f"conv-user-{user_id}",
                        "channel": "web",
                        "content": {"type": "text", "text": message_text}
                    }
                    
                    if conversation_id:
                        message_payload["conversation_id"] = conversation_id
                    
                    async with session.post(
                        f"{self.base_url}/api/v2/chat/message",
                        json=message_payload,
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as response:
                        if response.status != 200:
                            return False
                        
                        if not conversation_id:
                            response_data = await response.json()
                            conversation_id = response_data["data"]["conversation_id"]
                    
                    # Small delay between messages in conversation
                    await asyncio.sleep(0.5)
                
                return True
                
            except Exception as e:
                print(f"Conversation scenario failed: {e}")
                return False
        
        runner = LoadTestRunner("https://api.chatbot-platform.com")
        results = await runner.run_load_test(config, conversation_scenario)
        
        # Assert conversation flow performance
        assert results["passed"], f"Conversation load test failed: {results}"
        assert results["success_rate"] >= 0.90  # Slightly lower threshold for complex flows
        assert results["p95_response_time_ms"] <= 1000

# Stress Testing
class StressTestRunner(LoadTestRunner):
    """Extended runner for stress testing."""
    
    async def run_stress_test(self, max_users: int, step_size: int, step_duration: int):
        """Run stress test with gradually increasing load."""
        stress_results = []
        
        for user_count in range(step_size, max_users + 1, step_size):
            config = LoadTestConfig(
                name=f"Stress Test - {user_count} users",
                duration_seconds=step_duration,
                concurrent_users=user_count,
                ramp_up_seconds=30,
                target_rps=user_count * 2
            )
            
            print(f"Testing with {user_count} concurrent users...")
            
            async def simple_scenario(session, user_id):
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        return response.status == 200
                except:
                    return False
            
            results = await self.run_load_test(config, simple_scenario)
            stress_results.append(results)
            
            # Stop if success rate drops below threshold
            if results["success_rate"] < 0.8:
                print(f"System breaking point reached at {user_count} users")
                break
        
        return stress_results

@pytest.mark.asyncio
@pytest.mark.stress
async def test_system_breaking_point():
    """Find system breaking point through stress testing."""
    runner = StressTestRunner("https://api.chatbot-platform.com")
    
    stress_results = await runner.run_stress_test(
        max_users=1000,
        step_size=50,
        step_duration=120
    )
    
    # Analyze breaking point
    for result in stress_results:
        print(f"Users: {result['config'].concurrent_users}, "
              f"Success Rate: {result['success_rate']:.2f}, "
              f"P95 Response Time: {result['p95_response_time_ms']:.0f}ms")
    
    # System should handle at least 500 concurrent users
    assert len(stress_results) >= 10, "System broke too early in stress test"
```


**Document Maintainer:** QA Engineering Team  
**Review Schedule:** Weekly during development, monthly in production  
**Related Documents:** System Architecture, API Specifications, Performance Monitoring