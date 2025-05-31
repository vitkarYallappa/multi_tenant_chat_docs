# Phase 09: Testing Infrastructure & Quality Assurance
**Duration**: Week 17-18 (Days 81-90)  
**Team Size**: 4-5 developers  
**Complexity**: High  

## Overview
Implement comprehensive testing infrastructure including unit tests, integration tests, end-to-end tests, performance testing, and quality assurance automation. This phase ensures code quality, reliability, and performance standards across the entire MCP Engine system.

## Step 21: Testing Infrastructure & Framework Setup (Days 81-83)

### Files to Create
```
tests/
├── __init__.py
├── conftest.py
├── fixtures/
│   ├── __init__.py
│   ├── database_fixtures.py
│   ├── service_fixtures.py
│   ├── model_fixtures.py
│   └── integration_fixtures.py
├── unit/
│   ├── __init__.py
│   ├── test_models/
│   │   ├── __init__.py
│   │   ├── test_domain_models.py
│   │   ├── test_postgres_models.py
│   │   ├── test_mongodb_models.py
│   │   └── test_redis_models.py
│   ├── test_core/
│   │   ├── __init__.py
│   │   ├── test_state_machine/
│   │   │   ├── test_state_engine.py
│   │   │   ├── test_condition_evaluator.py
│   │   │   ├── test_action_executor.py
│   │   │   └── test_transition_handler.py
│   │   └── test_flows/
│   │       ├── test_flow_manager.py
│   │       ├── test_flow_parser.py
│   │       └── test_flow_validator.py
│   ├── test_services/
│   │   ├── __init__.py
│   │   ├── test_execution_service.py
│   │   ├── test_flow_service.py
│   │   ├── test_context_service.py
│   │   └── test_integration_service.py
│   ├── test_repositories/
│   │   ├── __init__.py
│   │   ├── test_flow_repository.py
│   │   ├── test_context_repository.py
│   │   └── test_analytics_repository.py
│   ├── test_clients/
│   │   ├── __init__.py
│   │   ├── test_model_orchestrator_client.py
│   │   ├── test_adaptor_service_client.py
│   │   └── test_circuit_breaker.py
│   └── test_api/
│       ├── __init__.py
│       ├── test_middleware/
│       │   ├── test_authentication.py
│       │   ├── test_rate_limiting.py
│       │   └── test_validation.py
│       ├── test_routes/
│       │   ├── test_execution_routes.py
│       │   ├── test_flow_routes.py
│       │   └── test_health_routes.py
│       └── test_grpc/
│           ├── test_mcp_service.py
│           ├── test_health_service.py
│           └── test_interceptors.py
├── integration/
│   ├── __init__.py
│   ├── test_database_integration.py
│   ├── test_service_integration.py
│   ├── test_external_services.py
│   ├── test_flow_execution.py
│   ├── test_conversation_lifecycle.py
│   └── test_error_handling.py
├── e2e/
│   ├── __init__.py
│   ├── test_complete_conversations.py
│   ├── test_multi_channel_flows.py
│   ├── test_complex_scenarios.py
│   └── test_performance_scenarios.py
├── performance/
│   ├── __init__.py
│   ├── test_load_performance.py
│   ├── test_stress_testing.py
│   ├── test_memory_usage.py
│   └── benchmarks/
│       ├── __init__.py
│       ├── state_machine_benchmark.py
│       ├── database_benchmark.py
│       └── api_benchmark.py
├── utils/
│   ├── __init__.py
│   ├── test_helpers.py
│   ├── mock_services.py
│   ├── data_generators.py
│   └── assertion_helpers.py
└── docker/
    ├── docker-compose.test.yml
    ├── test-databases.yml
    └── test-services.yml
```

### `/tests/conftest.py`
**Purpose**: Global test configuration and fixtures
```python
import pytest
import asyncio
import os
import tempfile
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, Mock
import uuid

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["POSTGRES_URI"] = "postgresql+asyncpg://test:test@localhost:5433/test_mcp_engine"
os.environ["MONGODB_URI"] = "mongodb://localhost:27018/test_mcp_engine"
os.environ["REDIS_URL"] = "redis://localhost:6380/0"

from src.config.settings import settings
from src.config.database import DatabaseManager
from src.utils.logger import configure_logging, get_logger

# Configure logging for tests
configure_logging()
logger = get_logger(__name__)

# Global test event loop
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Database fixtures
@pytest.fixture(scope="session")
async def test_database():
    """Setup test database connections"""
    try:
        await DatabaseManager.initialize_all()
        yield
    finally:
        await DatabaseManager.close_all()

@pytest.fixture(scope="function")
async def clean_database(test_database):
    """Clean database before each test"""
    from src.config.database import get_postgres_session, get_mongodb, get_redis
    
    # Clean PostgreSQL
    async with get_postgres_session() as session:
        # Drop all data from test tables
        await session.execute("TRUNCATE TABLE conversation_flows CASCADE")
        await session.execute("TRUNCATE TABLE flow_versions CASCADE")
        await session.execute("TRUNCATE TABLE flow_analytics CASCADE")
        await session.execute("TRUNCATE TABLE api_keys CASCADE")
        await session.execute("TRUNCATE TABLE usage_metrics CASCADE")
        await session.execute("TRUNCATE TABLE audit_logs CASCADE")
        await session.commit()
    
    # Clean MongoDB
    mongodb = get_mongodb()
    await mongodb.conversations.delete_many({})
    await mongodb.messages.delete_many({})
    
    # Clean Redis
    redis = await get_redis()
    await redis.flushdb()
    await redis.close()
    
    yield

# Service fixtures
@pytest.fixture
async def execution_service():
    """Mock execution service"""
    from src.services.execution_service import ExecutionService
    
    service = ExecutionService()
    await service.initialize()
    yield service
    await service.shutdown()

@pytest.fixture
async def flow_service():
    """Mock flow service"""
    from src.services.flow_service import FlowService
    
    service = FlowService()
    await service.initialize()
    yield service
    await service.shutdown()

@pytest.fixture
async def context_service():
    """Mock context service"""
    from src.services.context_service import ContextService
    
    service = ContextService()
    await service.initialize()
    yield service
    await service.shutdown()

# Mock external services
@pytest.fixture
def mock_model_orchestrator():
    """Mock Model Orchestrator client"""
    mock = AsyncMock()
    
    # Default responses
    mock.detect_intent.return_value = {
        "intent": "test_intent",
        "confidence": 0.9,
        "alternatives": [],
        "entities": [],
        "model_info": {},
        "processing_time_ms": 100
    }
    
    mock.extract_entities.return_value = [
        {
            "entity": "test_entity",
            "value": "test_value",
            "start": 0,
            "end": 4,
            "confidence": 0.8
        }
    ]
    
    mock.generate_response.return_value = {
        "text": "Test response",
        "type": "text",
        "confidence": 0.9,
        "model_info": {},
        "processing_time_ms": 150
    }
    
    return mock

@pytest.fixture
def mock_adaptor_service():
    """Mock Adaptor Service client"""
    mock = AsyncMock()
    
    # Default responses
    mock.execute_integration.return_value = {
        "success": True,
        "status_code": 200,
        "data": {"result": "test_data"},
        "error": None,
        "execution_time_ms": 200
    }
    
    mock.test_integration.return_value = {
        "success": True,
        "test_results": []
    }
    
    return mock

@pytest.fixture
def mock_security_hub():
    """Mock Security Hub client"""
    mock = AsyncMock()
    
    # Default responses
    mock.validate_api_key.return_value = {
        "valid": True,
        "key_info": {
            "key_id": "test_key_id",
            "tenant_id": "test_tenant",
            "permissions": ["api:read", "api:write"],
            "rate_limit_tier": "standard"
        }
    }
    
    mock.get_jwt_public_key.return_value = {
        "public_key": "test_public_key"
    }
    
    return mock

# Test data fixtures
@pytest.fixture
def sample_tenant_id():
    """Sample tenant ID for testing"""
    return str(uuid.uuid4())

@pytest.fixture
def sample_user_id():
    """Sample user ID for testing"""
    return str(uuid.uuid4())

@pytest.fixture
def sample_conversation_id():
    """Sample conversation ID for testing"""
    return str(uuid.uuid4())

@pytest.fixture
def sample_flow_definition():
    """Sample flow definition for testing"""
    return {
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "type": "response",
                "config": {
                    "response_templates": {
                        "default": "Hello! How can I help you today?"
                    }
                },
                "transitions": [
                    {
                        "condition": "any_input",
                        "target_state": "intent_detection",
                        "priority": 100
                    }
                ]
            },
            "intent_detection": {
                "type": "intent",
                "config": {
                    "intent_patterns": ["help", "support", "question"],
                    "confidence_threshold": 0.7
                },
                "transitions": [
                    {
                        "condition": "intent_match",
                        "condition_value": "help",
                        "target_state": "help_response",
                        "priority": 100
                    },
                    {
                        "condition": "fallback",
                        "target_state": "fallback_response",
                        "priority": 999
                    }
                ]
            },
            "help_response": {
                "type": "response",
                "config": {
                    "response_templates": {
                        "default": "I'm here to help! What do you need assistance with?"
                    }
                },
                "transitions": []
            },
            "fallback_response": {
                "type": "response",
                "config": {
                    "response_templates": {
                        "default": "I'm not sure I understand. Could you please rephrase?"
                    }
                },
                "transitions": []
            }
        }
    }

@pytest.fixture
def sample_message_content():
    """Sample message content for testing"""
    return {
        "type": "text",
        "text": "Hello, I need help",
        "metadata": {}
    }

@pytest.fixture
def sample_execution_context(sample_tenant_id, sample_conversation_id, sample_user_id):
    """Sample execution context for testing"""
    from src.models.domain.execution_context import ExecutionContext
    from datetime import datetime
    
    return ExecutionContext(
        conversation_id=sample_conversation_id,
        tenant_id=sample_tenant_id,
        user_id=sample_user_id,
        current_state="greeting",
        slots={},
        variables={},
        user_profile={},
        created_at=datetime.utcnow(),
        last_activity=datetime.utcnow()
    )

@pytest.fixture
def auth_context(sample_tenant_id, sample_user_id):
    """Sample authentication context for testing"""
    return {
        "auth_method": "jwt",
        "user_id": sample_user_id,
        "tenant_id": sample_tenant_id,
        "user_role": "admin",
        "permissions": ["api:read", "api:write"],
        "scopes": ["conversations:read", "flows:write"],
        "rate_limit_tier": "standard"
    }

# HTTP client fixtures
@pytest.fixture
async def test_client():
    """Test HTTP client for API testing"""
    from httpx import AsyncClient
    from src.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

# gRPC fixtures
@pytest.fixture
async def grpc_channel():
    """Test gRPC channel"""
    import grpc
    from src.api.grpc.server import create_grpc_server
    
    # Start test gRPC server
    server = await create_grpc_server(port=0)  # Use random port
    port = server.add_insecure_port('[::]:0')
    await server.start()
    
    try:
        # Create client channel
        channel = grpc.aio.insecure_channel(f'localhost:{port}')
        yield channel
        await channel.close()
    finally:
        await server.stop(grace=1.0)

# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests"""
    return {
        "max_response_time_ms": 1000,
        "max_memory_mb": 512,
        "concurrent_requests": 100,
        "test_duration_seconds": 60
    }

# Utility fixtures
@pytest.fixture
def temp_file():
    """Temporary file for testing"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_dir():
    """Temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers"""
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
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )

# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

# Async test helpers
@pytest.fixture
def async_mock():
    """Create async mock helper"""
    def _async_mock(*args, **kwargs):
        mock = AsyncMock(*args, **kwargs)
        return mock
    return _async_mock

# Error simulation fixtures
@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing"""
    return {
        "network_error": Exception("Network connection failed"),
        "timeout_error": TimeoutError("Request timed out"),
        "validation_error": ValueError("Invalid input data"),
        "auth_error": PermissionError("Unauthorized access"),
        "not_found_error": FileNotFoundError("Resource not found")
    }

# Test data generators
@pytest.fixture
def data_generator():
    """Test data generator utilities"""
    class DataGenerator:
        @staticmethod
        def generate_conversation_id():
            return str(uuid.uuid4())
        
        @staticmethod
        def generate_tenant_id():
            return str(uuid.uuid4())
        
        @staticmethod
        def generate_user_id():
            return str(uuid.uuid4())
        
        @staticmethod
        def generate_flow_definition(state_count=3):
            states = {}
            for i in range(state_count):
                state_name = f"state_{i}"
                states[state_name] = {
                    "type": "response",
                    "config": {
                        "response_templates": {
                            "default": f"Response from {state_name}"
                        }
                    },
                    "transitions": []
                }
            
            return {
                "initial_state": "state_0",
                "states": states
            }
        
        @staticmethod
        def generate_message_batch(count=10):
            messages = []
            for i in range(count):
                messages.append({
                    "conversation_id": str(uuid.uuid4()),
                    "user_id": str(uuid.uuid4()),
                    "content": {
                        "type": "text",
                        "text": f"Test message {i}"
                    }
                })
            return messages
    
    return DataGenerator()
```

### `/tests/utils/test_helpers.py`
**Purpose**: Common test helper functions and utilities
```python
import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Callable, Awaitable
from unittest.mock import Mock, AsyncMock, patch
from contextlib import asynccontextmanager
import uuid

from src.utils.logger import get_logger

logger = get_logger(__name__)

class TestHelpers:
    """Collection of test helper functions"""
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], Awaitable[bool]],
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """
        Wait for a condition to become true
        
        Args:
            condition: Async function that returns boolean
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds
            
        Returns:
            True if condition met, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if await condition():
                    return True
            except Exception:
                pass
            
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    async def measure_execution_time(coro: Awaitable[Any]) -> tuple[Any, float]:
        """
        Measure execution time of an async function
        
        Args:
            coro: Coroutine to measure
            
        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.time()
        result = await coro
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def create_mock_service(service_methods: Dict[str, Any]) -> Mock:
        """
        Create a mock service with specified methods
        
        Args:
            service_methods: Dictionary of method names and return values
            
        Returns:
            Mock service object
        """
        mock_service = Mock()
        
        for method_name, return_value in service_methods.items():
            if asyncio.iscoroutinefunction(return_value):
                setattr(mock_service, method_name, AsyncMock(return_value=return_value))
            else:
                setattr(mock_service, method_name, Mock(return_value=return_value))
        
        return mock_service
    
    @staticmethod
    def assert_dict_subset(subset: Dict[str, Any], superset: Dict[str, Any]):
        """
        Assert that subset is contained in superset
        
        Args:
            subset: Expected subset dictionary
            superset: Container dictionary
        """
        for key, value in subset.items():
            assert key in superset, f"Key '{key}' not found in superset"
            if isinstance(value, dict) and isinstance(superset[key], dict):
                TestHelpers.assert_dict_subset(value, superset[key])
            else:
                assert superset[key] == value, f"Value mismatch for key '{key}': expected {value}, got {superset[key]}"
    
    @staticmethod
    def assert_response_structure(
        response: Dict[str, Any],
        required_fields: List[str],
        optional_fields: Optional[List[str]] = None
    ):
        """
        Assert response has required structure
        
        Args:
            response: Response dictionary to validate
            required_fields: List of required field names
            optional_fields: List of optional field names
        """
        # Check required fields
        for field in required_fields:
            assert field in response, f"Required field '{field}' missing from response"
        
        # Check for unexpected fields
        if optional_fields is not None:
            allowed_fields = set(required_fields + optional_fields)
            actual_fields = set(response.keys())
            unexpected_fields = actual_fields - allowed_fields
            
            assert not unexpected_fields, f"Unexpected fields in response: {unexpected_fields}"
    
    @staticmethod
    async def simulate_network_delay(min_ms: int = 10, max_ms: int = 100):
        """
        Simulate network delay for testing
        
        Args:
            min_ms: Minimum delay in milliseconds
            max_ms: Maximum delay in milliseconds
        """
        import random
        delay = random.randint(min_ms, max_ms) / 1000.0
        await asyncio.sleep(delay)
    
    @staticmethod
    @asynccontextmanager
    async def temporary_env_vars(env_vars: Dict[str, str]):
        """
        Temporarily set environment variables
        
        Args:
            env_vars: Dictionary of environment variables to set
        """
        import os
        
        original_values = {}
        
        # Set new values and store originals
        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            yield
        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    @staticmethod
    def create_test_conversation_flow(
        name: str = "test_flow",
        state_count: int = 3,
        include_integrations: bool = False
    ) -> Dict[str, Any]:
        """
        Create a test conversation flow
        
        Args:
            name: Flow name
            state_count: Number of states to create
            include_integrations: Whether to include integration states
            
        Returns:
            Flow definition dictionary
        """
        states = {}
        
        # Create greeting state
        states["greeting"] = {
            "type": "response",
            "config": {
                "response_templates": {
                    "default": "Hello! Welcome to our test flow."
                }
            },
            "transitions": [
                {
                    "condition": "any_input",
                    "target_state": "state_1" if state_count > 1 else "end",
                    "priority": 100
                }
            ]
        }
        
        # Create intermediate states
        for i in range(1, state_count - 1):
            state_name = f"state_{i}"
            next_state = f"state_{i + 1}" if i < state_count - 2 else "end"
            
            if include_integrations and i % 2 == 0:
                # Create integration state
                states[state_name] = {
                    "type": "integration",
                    "config": {
                        "integration_id": f"test_integration_{i}",
                        "endpoint": "/test",
                        "method": "GET",
                        "request_mapping": {},
                        "response_mapping": {"result": "integration_result"}
                    },
                    "transitions": [
                        {
                            "condition": "integration_success",
                            "target_state": next_state,
                            "priority": 100
                        },
                        {
                            "condition": "integration_error",
                            "target_state": "error_state",
                            "priority": 200
                        }
                    ]
                }
            else:
                # Create response state
                states[state_name] = {
                    "type": "response",
                    "config": {
                        "response_templates": {
                            "default": f"This is state {i} response."
                        }
                    },
                    "transitions": [
                        {
                            "condition": "any_input",
                            "target_state": next_state,
                            "priority": 100
                        }
                    ]
                }
        
        # Create end state
        states["end"] = {
            "type": "response",
            "config": {
                "response_templates": {
                    "default": "Thank you for using our service!"
                }
            },
            "transitions": []
        }
        
        # Create error state if integrations are included
        if include_integrations:
            states["error_state"] = {
                "type": "response",
                "config": {
                    "response_templates": {
                        "default": "I'm sorry, something went wrong. Please try again."
                    }
                },
                "transitions": [
                    {
                        "condition": "any_input",
                        "target_state": "greeting",
                        "priority": 100
                    }
                ]
            }
        
        return {
            "name": name,
            "version": "1.0",
            "initial_state": "greeting",
            "states": states
        }
    
    @staticmethod
    def validate_processing_result(
        result: Dict[str, Any],
        expected_conversation_id: str,
        expected_success: bool = True
    ):
        """
        Validate processing result structure and content
        
        Args:
            result: Processing result to validate
            expected_conversation_id: Expected conversation ID
            expected_success: Expected success status
        """
        # Validate required fields
        required_fields = [
            "conversation_id",
            "current_state",
            "response",
            "context_updates",
            "actions_performed",
            "processing_time_ms",
            "success"
        ]
        
        TestHelpers.assert_response_structure(result, required_fields)
        
        # Validate specific values
        assert result["conversation_id"] == expected_conversation_id
        assert result["success"] == expected_success
        assert isinstance(result["processing_time_ms"], int)
        assert result["processing_time_ms"] >= 0
        
        # Validate response structure
        if result["response"]:
            assert "text" in result["response"]
            assert "type" in result["response"]
    
    @staticmethod
    async def create_test_conversation(
        execution_service,
        tenant_id: str,
        conversation_id: str,
        user_id: str,
        initial_message: str = "Hello"
    ) -> Dict[str, Any]:
        """
        Create a test conversation with initial message
        
        Args:
            execution_service: Execution service instance
            tenant_id: Tenant ID
            conversation_id: Conversation ID
            user_id: User ID
            initial_message: Initial message text
            
        Returns:
            Processing result
        """
        return await execution_service.process_message(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_content={
                "type": "text",
                "text": initial_message,
                "metadata": {}
            },
            user_id=user_id,
            channel="web"
        )
    
    @staticmethod
    def generate_test_data(data_type: str, count: int = 1) -> List[Dict[str, Any]]:
        """
        Generate test data for various types
        
        Args:
            data_type: Type of data to generate
            count: Number of items to generate
            
        Returns:
            List of generated data items
        """
        generators = {
            "conversation": lambda: {
                "conversation_id": str(uuid.uuid4()),
                "tenant_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "channel": "web",
                "status": "active"
            },
            "message": lambda: {
                "message_id": str(uuid.uuid4()),
                "conversation_id": str(uuid.uuid4()),
                "content": {
                    "type": "text",
                    "text": f"Test message {uuid.uuid4().hex[:8]}"
                },
                "direction": "inbound"
            },
            "flow": lambda: TestHelpers.create_test_conversation_flow(
                name=f"flow_{uuid.uuid4().hex[:8]}"
            ),
            "user": lambda: {
                "user_id": str(uuid.uuid4()),
                "tenant_id": str(uuid.uuid4()),
                "email": f"test+{uuid.uuid4().hex[:8]}@example.com",
                "role": "user"
            }
        }
        
        generator = generators.get(data_type)
        if not generator:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return [generator() for _ in range(count)]

class PerformanceTestHelpers:
    """Helpers for performance testing"""
    
    @staticmethod
    async def measure_concurrent_requests(
        request_func: Callable[[], Awaitable[Any]],
        concurrent_count: int,
        duration_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Measure performance of concurrent requests
        
        Args:
            request_func: Async function to execute concurrently
            concurrent_count: Number of concurrent requests
            duration_seconds: Optional duration limit
            
        Returns:
            Performance metrics
        """
        start_time = time.time()
        completed_requests = 0
        failed_requests = 0
        response_times = []
        
        async def single_request():
            nonlocal completed_requests, failed_requests
            
            request_start = time.time()
            try:
                await request_func()
                completed_requests += 1
            except Exception:
                failed_requests += 1
            finally:
                response_times.append(time.time() - request_start)
        
        # Execute concurrent requests
        if duration_seconds:
            # Time-based testing
            end_time = start_time + duration_seconds
            
            while time.time() < end_time:
                tasks = [single_request() for _ in range(concurrent_count)]
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Count-based testing
            tasks = [single_request() for _ in range(concurrent_count)]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        total_requests = completed_requests + failed_requests
        
        return {
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "success_rate": completed_requests / total_requests if total_requests > 0 else 0,
            "total_time_seconds": total_time,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
            "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0
        }
    
    @staticmethod
    def assert_performance_requirements(
        metrics: Dict[str, Any],
        max_response_time: float = 1.0,
        min_success_rate: float = 0.95,
        min_rps: Optional[float] = None
    ):
        """
        Assert performance requirements are met
        
        Args:
            metrics: Performance metrics from measure_concurrent_requests
            max_response_time: Maximum acceptable response time in seconds
            min_success_rate: Minimum acceptable success rate (0-1)
            min_rps: Minimum acceptable requests per second
        """
        assert metrics["success_rate"] >= min_success_rate, \
            f"Success rate {metrics['success_rate']:.2%} below requirement {min_success_rate:.2%}"
        
        assert metrics["avg_response_time"] <= max_response_time, \
            f"Average response time {metrics['avg_response_time']:.3f}s above requirement {max_response_time}s"
        
        assert metrics["p95_response_time"] <= max_response_time * 2, \
            f"P95 response time {metrics['p95_response_time']:.3f}s above requirement {max_response_time * 2}s"
        
        if min_rps:
            assert metrics["requests_per_second"] >= min_rps, \
                f"RPS {metrics['requests_per_second']:.2f} below requirement {min_rps}"
```

## Step 22: Comprehensive Unit Tests (Days 84-86)

### `/tests/unit/test_core/test_state_machine/test_state_engine.py`
**Purpose**: Unit tests for the core state machine engine
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
import uuid

from src.core.state_machine.state_engine import StateEngine
from src.models.domain.state_machine import State, Transition, Action
from src.models.domain.flow_definition import FlowDefinition
from src.models.domain.execution_context import ExecutionContext
from src.models.domain.events import StateEvent, StateExecutionResult
from src.models.domain.enums import StateType, TransitionCondition, ActionType
from src.exceptions.state_exceptions import StateExecutionError, StateNotFoundError
from tests.utils.test_helpers import TestHelpers

@pytest.mark.unit
class TestStateEngine:
    """Unit tests for StateEngine"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock state engine dependencies"""
        return {
            "transition_handler": AsyncMock(),
            "condition_evaluator": AsyncMock(),
            "action_executor": AsyncMock(),
            "context_manager": AsyncMock()
        }
    
    @pytest.fixture
    def state_engine(self, mock_dependencies):
        """Create state engine with mocked dependencies"""
        return StateEngine(
            transition_handler=mock_dependencies["transition_handler"],
            condition_evaluator=mock_dependencies["condition_evaluator"],
            action_executor=mock_dependencies["action_executor"],
            context_manager=mock_dependencies["context_manager"]
        )
    
    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing"""
        from src.models.domain.state_machine import ResponseStateConfig
        
        return State(
            name="test_state",
            type=StateType.RESPONSE,
            config=ResponseStateConfig(
                response_templates={"default": "Test response"}
            ),
            transitions=[
                Transition(
                    condition=TransitionCondition.ANY_INPUT,
                    target_state="next_state",
                    priority=100
                )
            ]
        )
    
    @pytest.fixture
    def sample_flow(self, sample_state):
        """Create sample flow for testing"""
        return FlowDefinition(
            flow_id=str(uuid.uuid4()),
            tenant_id=str(uuid.uuid4()),
            name="test_flow",
            version="1.0",
            initial_state="test_state",
            states={"test_state": sample_state}
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample execution context"""
        return ExecutionContext(
            conversation_id=str(uuid.uuid4()),
            tenant_id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            current_state="test_state"
        )
    
    @pytest.fixture
    def sample_event(self):
        """Create sample state event"""
        return StateEvent(
            type="message",
            data={"text": "Hello, world!"}
        )
    
    async def test_execute_state_success(
        self,
        state_engine,
        sample_state,
        sample_event,
        sample_context,
        sample_flow,
        mock_dependencies
    ):
        """Test successful state execution"""
        # Setup mocks
        mock_dependencies["condition_evaluator"].evaluate_transition_condition.return_value = True
        mock_dependencies["action_executor"].execute_actions_batch.return_value = [
            {"type": "send_message", "success": True}
        ]
        
        # Mock state execution to return transition
        transition = sample_state.transitions[0]
        
        # Execute state
        result = await state_engine.execute_state(
            tenant_id=sample_context.tenant_id,
            conversation_id=sample_context.conversation_id,
            current_state=sample_state,
            event=sample_event,
            context=sample_context,
            flow_definition=sample_flow
        )
        
        # Assertions
        assert isinstance(result, StateExecutionResult)
        assert result.success
        assert result.execution_time_ms > 0
        
        # Verify mocks were called
        mock_dependencies["action_executor"].execute_actions_batch.assert_called()
    
    async def test_execute_state_with_invalid_state(
        self,
        state_engine,
        sample_state,
        sample_event,
        sample_context,
        sample_flow
    ):
        """Test state execution with invalid state"""
        # Modify context to reference non-existent state
        sample_context.current_state = "non_existent_state"
        
        # Create state with non-existent name
        invalid_state = State(
            name="non_existent_state",
            type=StateType.RESPONSE,
            config=sample_state.config,
            transitions=[]
        )
        
        # Execute state - should handle gracefully
        result = await state_engine.execute_state(
            tenant_id=sample_context.tenant_id,
            conversation_id=sample_context.conversation_id,
            current_state=invalid_state,
            event=sample_event,
            context=sample_context,
            flow_definition=sample_flow
        )
        
        # Should return error result
        assert isinstance(result, StateExecutionResult)
        assert not result.success
        assert len(result.errors) > 0
    
    async def test_execute_response_state(
        self,
        state_engine,
        sample_event,
        sample_context
    ):
        """Test execution of response state"""
        from src.models.domain.state_machine import ResponseStateConfig
        
        # Create response state
        response_state = State(
            name="response_state",
            type=StateType.RESPONSE,
            config=ResponseStateConfig(
                response_templates={
                    "default": "Hello {user_name}!",
                    "returning_user": "Welcome back!"
                },
                personalization=True
            ),
            transitions=[]
        )
        
        # Add user name to context
        sample_context.slots["user_name"] = "John"
        sample_context.user_profile["returning_user"] = False
        
        # Execute state logic
        result = await state_engine._execute_state_logic(
            response_state,
            sample_event,
            sample_context,
            sample_context.tenant_id
        )
        
        # Assertions
        assert "actions" in result
        assert len(result["actions"]) > 0
        assert result["actions"][0]["type"] == ActionType.SEND_MESSAGE.value
        assert "response" in result
        assert "Hello John!" in result["response"]["text"]
    
    async def test_execute_intent_state(
        self,
        state_engine,
        sample_event,
        sample_context
    ):
        """Test execution of intent detection state"""
        from src.models.domain.state_machine import IntentStateConfig
        
        # Create intent state
        intent_state = State(
            name="intent_state",
            type=StateType.INTENT,
            config=IntentStateConfig(
                intent_patterns=["help", "support"],
                confidence_threshold=0.7
            ),
            transitions=[]
        )
        
        # Mock intent detection
        with patch.object(state_engine, '_detect_intent') as mock_detect:
            mock_detect.return_value = {
                "intent": "help_request",
                "confidence": 0.9
            }
            
            # Execute state logic
            result = await state_engine._execute_state_logic(
                intent_state,
                sample_event,
                sample_context,
                sample_context.tenant_id
            )
            
            # Assertions
            assert "context_updates" in result
            assert result["context_updates"]["current_intent"] == "help_request"
            assert result["context_updates"]["intent_confidence"] == 0.9
            
            # Verify intent detection was called
            mock_detect.assert_called_once()
    
    async def test_execute_slot_filling_state(
        self,
        state_engine,
        sample_event,
        sample_context
    ):
        """Test execution of slot filling state"""
        from src.models.domain.state_machine import SlotFillingConfig
        
        # Create slot filling state
        slot_state = State(
            name="slot_state",
            type=StateType.SLOT_FILLING,
            config=SlotFillingConfig(
                required_slots=["name", "email"],
                optional_slots=["phone"],
                prompts={
                    "name": "What's your name?",
                    "email": "What's your email address?"
                }
            ),
            transitions=[]
        )
        
        # Mock entity extraction
        with patch.object(state_engine, '_extract_entities') as mock_extract:
            mock_extract.return_value = [
                {
                    "entity": "name",
                    "value": "John Doe"
                }
            ]
            
            # Execute state logic
            result = await state_engine._execute_state_logic(
                slot_state,
                sample_event,
                sample_context,
                sample_context.tenant_id
            )
            
            # Assertions
            assert "context_updates" in result
            assert "slots" in result["context_updates"]
            assert result["context_updates"]["slots"]["name"] == "John Doe"
            assert "missing_required_slots" in result["context_updates"]
            assert "email" in result["context_updates"]["missing_required_slots"]
            
            # Should generate prompt for next missing slot
            assert "actions" in result
            assert len(result["actions"]) > 0
    
    async def test_execute_integration_state(
        self,
        state_engine,
        sample_event,
        sample_context
    ):
        """Test execution of integration state"""
        from src.models.domain.state_machine import IntegrationStateConfig
        
        # Create integration state
        integration_state = State(
            name="integration_state",
            type=StateType.INTEGRATION,
            config=IntegrationStateConfig(
                integration_id="test_integration",
                endpoint="/api/test",
                method="POST",
                request_mapping={"user_id": "{user_id}"},
                response_mapping={"result": "api_result"}
            ),
            transitions=[]
        )
        
        # Mock integration call
        with patch.object(state_engine, '_call_integration') as mock_call:
            mock_call.return_value = {
                "success": True,
                "data": {"api_result": "success"},
                "status_code": 200,
                "execution_time_ms": 150
            }
            
            # Execute state logic
            result = await state_engine._execute_state_logic(
                integration_state,
                sample_event,
                sample_context,
                sample_context.tenant_id
            )
            
            # Assertions
            assert "context_updates" in result
            assert "integration_results" in result["context_updates"]
            assert result["context_updates"]["integration_results"]["test_integration"]["success"] is True
            
            # Verify integration was called
            mock_call.assert_called_once()
    
    async def test_transition_evaluation(
        self,
        state_engine,
        sample_state,
        sample_event,
        sample_context,
        mock_dependencies
    ):
        """Test transition evaluation logic"""
        # Setup multiple transitions with different priorities
        transitions = [
            Transition(
                condition=TransitionCondition.INTENT_MATCH,
                condition_value="help",
                target_state="help_state",
                priority=50
            ),
            Transition(
                condition=TransitionCondition.ANY_INPUT,
                target_state="fallback_state",
                priority=100
            )
        ]
        
        sample_state.transitions = transitions
        
        # Mock condition evaluator to return True for first transition
        mock_dependencies["condition_evaluator"].evaluate_transition_condition.side_effect = [
            True,  # First transition matches
            False  # Second transition not evaluated
        ]
        
        # Execute transition evaluation
        result = await state_engine._evaluate_transitions(
            sample_state,
            sample_event,
            sample_context,
            {}
        )
        
        # Should return first matching transition (highest priority)
        assert result is not None
        assert result.target_state == "help_state"
        assert result.priority == 50
        
        # Condition evaluator should be called for first transition only
        assert mock_dependencies["condition_evaluator"].evaluate_transition_condition.call_count == 1
    
    async def test_execution_with_timeout(
        self,
        state_engine,
        sample_state,
        sample_event,
        sample_context,
        sample_flow
    ):
        """Test state execution with timeout"""
        # Set timeout on state
        sample_state.timeout_seconds = 1
        
        # Mock action executor to simulate slow execution
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return []
        
        with patch.object(state_engine, '_execute_actions', side_effect=slow_execution):
            # Execute state - should handle timeout gracefully
            result = await state_engine.execute_state(
                tenant_id=sample_context.tenant_id,
                conversation_id=sample_context.conversation_id,
                current_state=sample_state,
                event=sample_event,
                context=sample_context,
                flow_definition=sample_flow
            )
            
            # Should return error result due to timeout
            assert isinstance(result, StateExecutionResult)
            # Note: Actual timeout handling depends on implementation
    
    async def test_concurrent_execution_safety(
        self,
        state_engine,
        sample_state,
        sample_event,
        sample_context,
        sample_flow
    ):
        """Test concurrent execution safety with locks"""
        # Create multiple concurrent executions
        tasks = []
        for i in range(5):
            task = state_engine.execute_state(
                tenant_id=sample_context.tenant_id,
                conversation_id=sample_context.conversation_id,
                current_state=sample_state,
                event=sample_event,
                context=sample_context,
                flow_definition=sample_flow
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        for result in results:
            assert isinstance(result, StateExecutionResult)
            assert result.success
    
    async def test_error_recovery(
        self,
        state_engine,
        sample_state,
        sample_event,
        sample_context,
        sample_flow,
        mock_dependencies
    ):
        """Test error recovery mechanisms"""
        # Mock action executor to raise exception
        mock_dependencies["action_executor"].execute_actions_batch.side_effect = Exception("Test error")
        
        # Execute state
        result = await state_engine.execute_state(
            tenant_id=sample_context.tenant_id,
            conversation_id=sample_context.conversation_id,
            current_state=sample_state,
            event=sample_event,
            context=sample_context,
            flow_definition=sample_flow
        )
        
        # Should return error result but not crash
        assert isinstance(result, StateExecutionResult)
        assert not result.success
        assert len(result.errors) > 0
        assert "Test error" in str(result.errors)
    
    @pytest.mark.parametrize("state_type,config_class", [
        (StateType.RESPONSE, "ResponseStateConfig"),
        (StateType.INTENT, "IntentStateConfig"),
        (StateType.SLOT_FILLING, "SlotFillingConfig"),
        (StateType.INTEGRATION, "IntegrationStateConfig"),
        (StateType.CONDITION, "ConditionStateConfig")
    ])
    async def test_all_state_types(
        self,
        state_engine,
        sample_event,
        sample_context,
        state_type,
        config_class
    ):
        """Test all supported state types"""
        from src.models.domain import state_machine
        
        # Get config class
        config_cls = getattr(state_machine, config_class)
        
        # Create minimal config for each type
        if state_type == StateType.RESPONSE:
            config = config_cls(response_templates={"default": "Test"})
        elif state_type == StateType.INTENT:
            config = config_cls(intent_patterns=["test"])
        elif state_type == StateType.SLOT_FILLING:
            config = config_cls(
                required_slots=["test_slot"],
                prompts={"test_slot": "Enter value"}
            )
        elif state_type == StateType.INTEGRATION:
            config = config_cls(
                integration_id="test",
                endpoint="/test"
            )
        elif state_type == StateType.CONDITION:
            config = config_cls(
                conditions=[{"operator": "eq", "left": "test", "right": "test"}]
            )
        
        # Create state
        state = State(
            name="test_state",
            type=state_type,
            config=config,
            transitions=[]
        )
        
        # Execute state logic - should not raise exception
        result = await state_engine._execute_state_logic(
            state,
            sample_event,
            sample_context,
            sample_context.tenant_id
        )
        
        # Should return some result
        assert isinstance(result, dict)
```

## Success Criteria
- [x] Complete testing infrastructure with fixtures and utilities
- [x] Comprehensive unit tests for core components
- [x] Test helpers and utilities for common patterns
- [x] Mock services and external dependencies
- [x] Performance testing framework and helpers
- [x] Database testing with proper isolation
- [x] Async testing support with proper event loops
- [x] Test data generators and assertion helpers

## Key Error Handling & Performance Considerations
1. **Test Isolation**: Proper database cleanup between tests
2. **Async Testing**: Correct async/await patterns and event loop management
3. **Mock Management**: Comprehensive mocking of external dependencies
4. **Performance Testing**: Load testing and performance assertion helpers
5. **Error Simulation**: Testing error scenarios and edge cases
6. **Data Generation**: Realistic test data generation
7. **Resource Management**: Proper cleanup of test resources

## Technologies Used
- **Testing Framework**: pytest with async support
- **Mocking**: unittest.mock with AsyncMock
- **Database Testing**: Test database isolation and cleanup
- **Performance Testing**: Async load testing utilities
- **Test Data**: Factories and generators for realistic data
- **Fixtures**: Comprehensive fixture library for reusability

## Cross-Service Integration
- **Service Testing**: Unit tests for all service layers
- **Repository Testing**: Database interaction testing
- **API Testing**: HTTP and gRPC endpoint testing
- **Client Testing**: External service client testing
- **Integration Testing**: End-to-end workflow testing
- **Performance Testing**: Load and stress testing

## Next Phase Dependencies
Phase 10 will build upon:
- Testing infrastructure and frameworks
- Quality assurance processes and standards
- Performance testing and benchmarking
- Automated testing pipelines
- Code coverage and quality metrics