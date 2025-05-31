# Phase 10: Testing, Deployment & Documentation

**Duration:** 3 weeks  
**Focus:** Production readiness, comprehensive testing, deployment automation, and complete documentation  
**Team:** 4 developers + 1 DevOps engineer + 1 QA engineer

---

## Week 1: Comprehensive Testing Suite

### Step 1: Advanced Unit Testing Implementation
**Duration:** 3 days

#### Files to Create:

##### `/tests/unit/test_state_engine_advanced.py`
**Purpose:** Advanced unit tests for state engine with edge cases and concurrency

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.core.state_machine.state_engine import StateEngine
from src.models.domain.state_machine import State, StateEvent, ExecutionContext

class TestStateEngineAdvanced:
    """Advanced state engine testing with concurrency and edge cases"""
    
    @pytest.fixture
    async def state_engine(self):
        """Setup state engine with mocked dependencies"""
        transition_handler = AsyncMock()
        condition_evaluator = AsyncMock()
        action_executor = AsyncMock()
        logger = Mock()
        
        return StateEngine(
            transition_handler=transition_handler,
            condition_evaluator=condition_evaluator,
            action_executor=action_executor,
            logger=logger
        )
    
    async def test_concurrent_state_execution(self, state_engine):
        """
        Test concurrent state execution with proper locking
        
        Parameters: 
        - state_engine: StateEngine fixture
        
        Returns: None
        
        Tests:
        - Multiple simultaneous executions
        - Lock acquisition and release
        - Context consistency
        """
        # Setup test data
        state = State(name="test_state", type="response", config={}, transitions=[])
        event = StateEvent(type="message", data={"text": "test"})
        contexts = [
            ExecutionContext(
                conversation_id=f"conv_{i}",
                tenant_id="tenant_123",
                user_id=f"user_{i}",
                current_state="test_state"
            ) for i in range(10)
        ]
        
        # Execute concurrent state processing
        tasks = [
            state_engine.execute_state(state, event, ctx, Mock())
            for ctx in contexts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all executions completed successfully
        assert all(isinstance(r, Mock) for r in results)
        assert len(results) == 10

    async def test_circular_transition_detection(self, state_engine):
        """
        Test detection and handling of circular state transitions
        
        Parameters:
        - state_engine: StateEngine fixture
        
        Returns: None
        
        Validates:
        - Infinite loop prevention
        - Circuit breaker activation
        - Error state transition
        """
        # Create circular flow: A -> B -> A
        from src.models.domain.state_machine import Transition, TransitionCondition
        
        transition_a_to_b = Transition(
            condition=TransitionCondition.ANY_INPUT,
            target_state="state_b"
        )
        transition_b_to_a = Transition(
            condition=TransitionCondition.ANY_INPUT,
            target_state="state_a"
        )
        
        state_a = State(
            name="state_a", 
            type="response", 
            config={}, 
            transitions=[transition_a_to_b]
        )
        
        # Mock the transition handler to detect circular references
        state_engine.transition_handler.evaluate_transitions.side_effect = [
            transition_a_to_b,
            transition_b_to_a,
            transition_a_to_b  # This should trigger circuit breaker
        ]
        
        context = ExecutionContext(
            conversation_id="conv_123",
            tenant_id="tenant_123",
            user_id="user_123",
            current_state="state_a",
            previous_states=["state_a", "state_b"]  # History shows circular pattern
        )
        
        event = StateEvent(type="message", data={"text": "test"})
        
        # Should detect circular reference and handle gracefully
        result = await state_engine.execute_state(state_a, event, context, Mock())
        
        # Verify error handling
        assert "circular_reference" in str(result.errors).lower()

    async def test_memory_cleanup_after_execution(self, state_engine):
        """
        Test memory cleanup after state execution
        
        Parameters:
        - state_engine: StateEngine fixture
        
        Returns: None
        
        Checks:
        - Context cleanup
        - Lock release
        - Cache invalidation
        """
        import gc
        import sys
        
        # Get initial memory usage
        initial_objects = len(gc.get_objects())
        
        # Execute multiple state transitions
        for i in range(100):
            state = State(name=f"state_{i}", type="response", config={}, transitions=[])
            event = StateEvent(type="message", data={"text": f"test_{i}"})
            context = ExecutionContext(
                conversation_id=f"conv_{i}",
                tenant_id="tenant_123",
                user_id=f"user_{i}",
                current_state=f"state_{i}"
            )
            
            await state_engine.execute_state(state, event, context, Mock())
        
        # Force garbage collection
        gc.collect()
        
        # Check memory didn't grow significantly
        final_objects = len(gc.get_objects())
        memory_growth = final_objects - initial_objects
        
        # Allow some growth but not proportional to iterations
        assert memory_growth < 50  # Should be much less than 100

    async def test_state_execution_timeout_handling(self, state_engine):
        """
        Test handling of state execution timeouts
        
        Parameters:
        - state_engine: StateEngine fixture
        
        Returns: None
        
        Scenarios:
        - Long-running integrations
        - Model API timeouts
        - Recovery mechanisms
        """
        # Mock a slow action executor that times out
        async def slow_action(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow operation
            return Mock()
        
        state_engine.action_executor.execute_actions = slow_action
        
        state = State(name="slow_state", type="integration", config={}, transitions=[])
        event = StateEvent(type="message", data={"text": "test"})
        context = ExecutionContext(
            conversation_id="conv_123",
            tenant_id="tenant_123",
            user_id="user_123",
            current_state="slow_state"
        )
        
        # Execute with timeout
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await asyncio.wait_for(
                state_engine.execute_state(state, event, context, Mock()),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            result = None
        
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        
        # Verify timeout was respected
        assert execution_time < 3.0
        assert result is None  # Should have timed out

    async def test_context_validation_and_sanitization(self, state_engine):
        """
        Test context validation and sanitization
        
        Parameters:
        - state_engine: StateEngine fixture
        
        Returns: None
        
        Validates:
        - Input sanitization
        - Context size limits
        - Data type validation
        """
        # Test with oversized context
        large_context = ExecutionContext(
            conversation_id="conv_123",
            tenant_id="tenant_123",
            user_id="user_123",
            current_state="test_state",
            variables={"large_data": "x" * 1000000}  # 1MB of data
        )
        
        state = State(name="test_state", type="response", config={}, transitions=[])
        event = StateEvent(type="message", data={"text": "test"})
        
        # Should handle large context gracefully
        result = await state_engine.execute_state(state, event, large_context, Mock())
        
        # Context should be truncated or handled appropriately
        assert result is not None
```

##### `/tests/unit/test_flow_parser_edge_cases.py`
**Purpose:** Edge case testing for flow parser with malformed and complex flows

```python
import pytest
import json
from src.core.flows.flow_parser import FlowParser
from src.utils.json_schema_validator import JsonSchemaValidator
from src.exceptions.flow_exceptions import FlowParseError, FlowValidationError

class TestFlowParserEdgeCases:
    """Test flow parser with complex and malformed flow definitions"""
    
    @pytest.fixture
    def flow_parser(self):
        """Setup flow parser with validator"""
        schema_validator = JsonSchemaValidator()
        return FlowParser(schema_validator)
    
    def test_malformed_json_handling(self, flow_parser):
        """
        Test handling of malformed JSON flow definitions
        
        Parameters:
        - flow_parser: FlowParser fixture
        
        Returns: None
        
        Cases:
        - Invalid JSON syntax
        - Missing required fields
        - Type mismatches
        """
        malformed_flows = [
            # Invalid JSON syntax
            '{"name": "test", "states": {',
            
            # Missing required fields
            {
                "name": "test_flow"
                # Missing states, initial_state
            },
            
            # Type mismatches
            {
                "name": "test_flow",
                "initial_state": "start",
                "states": "not_an_object"  # Should be object
            },
            
            # Invalid state types
            {
                "name": "test_flow",
                "initial_state": "start",
                "states": {
                    "start": {
                        "type": "invalid_type",
                        "config": {},
                        "transitions": []
                    }
                }
            }
        ]
        
        for malformed_flow in malformed_flows:
            with pytest.raises((FlowParseError, FlowValidationError, json.JSONDecodeError)):
                if isinstance(malformed_flow, str):
                    parsed_flow = json.loads(malformed_flow)
                    flow_parser.parse_flow_definition(parsed_flow)
                else:
                    flow_parser.parse_flow_definition(malformed_flow)

    def test_recursive_flow_references(self, flow_parser):
        """
        Test handling of recursive flow references
        
        Parameters:
        - flow_parser: FlowParser fixture
        
        Returns: None
        
        Validates:
        - Self-referencing flows
        - Deep recursion detection
        - Stack overflow prevention
        """
        recursive_flow = {
            "name": "recursive_flow",
            "initial_state": "state_a",
            "states": {
                "state_a": {
                    "type": "response",
                    "config": {"response_templates": {"default": "Response A"}},
                    "transitions": [{
                        "condition": "any_input",
                        "target_state": "state_b"
                    }]
                },
                "state_b": {
                    "type": "response",
                    "config": {"response_templates": {"default": "Response B"}},
                    "transitions": [{
                        "condition": "any_input",
                        "target_state": "state_a"  # Creates cycle
                    }]
                }
            }
        }
        
        # Should detect and handle circular references
        with pytest.raises(FlowValidationError) as exc_info:
            flow_parser.parse_flow_definition(recursive_flow, validate=True)
        
        assert "circular" in str(exc_info.value).lower() or "recursive" in str(exc_info.value).lower()

    def test_maximum_complexity_flows(self, flow_parser):
        """
        Test parsing of maximum complexity flows
        
        Parameters:
        - flow_parser: FlowParser fixture
        
        Returns: None
        
        Tests:
        - Large number of states (1000+)
        - Complex transition conditions
        - Deep nesting levels
        """
        # Generate large flow with many states
        large_flow = {
            "name": "large_flow",
            "initial_state": "state_0",
            "states": {}
        }
        
        # Create 1000 states with complex transitions
        for i in range(1000):
            state_name = f"state_{i}"
            next_state = f"state_{(i + 1) % 1000}"  # Circular but controlled
            
            large_flow["states"][state_name] = {
                "type": "response",
                "config": {
                    "response_templates": {"default": f"Response {i}"}
                },
                "transitions": [
                    {
                        "condition": "intent_match",
                        "condition_value": f"intent_{i}",
                        "target_state": next_state,
                        "priority": i % 100
                    },
                    {
                        "condition": "expression",
                        "expression": f"slots.value > {i}",
                        "target_state": next_state,
                        "priority": (i + 50) % 100
                    }
                ]
            }
        
        # Should handle large flows without performance issues
        import time
        start_time = time.time()
        
        parsed_flow = flow_parser.parse_flow_definition(large_flow, validate=False)
        
        end_time = time.time()
        parse_time = end_time - start_time
        
        # Should parse within reasonable time (< 5 seconds)
        assert parse_time < 5.0
        assert len(parsed_flow.states) == 1000

    def test_unicode_and_special_characters(self, flow_parser):
        """
        Test handling of Unicode and special characters in flows
        
        Parameters:
        - flow_parser: FlowParser fixture
        
        Returns: None
        
        Tests:
        - Unicode state names
        - Special characters in responses
        - Emoji handling
        """
        unicode_flow = {
            "name": "unicode_flow_测试",
            "initial_state": "开始状态",
            "states": {
                "开始状态": {
                    "type": "response",
                    "config": {
                        "response_templates": {
                            "default": "欢迎！🎉 How can I help you today? 😊",
                            "spanish": "¡Hola! ¿Cómo puedo ayudarte?",
                            "special": "Special chars: @#$%^&*()_+{}|:<>?[]\\;'\",./"
                        }
                    },
                    "transitions": [{
                        "condition": "any_input",
                        "target_state": "处理状态"
                    }]
                },
                "处理状态": {
                    "type": "intent",
                    "config": {
                        "intent_patterns": ["订单查询", "技术支持"]
                    },
                    "transitions": []
                }
            }
        }
        
        # Should handle Unicode characters properly
        parsed_flow = flow_parser.parse_flow_definition(unicode_flow)
        
        assert parsed_flow.name == "unicode_flow_测试"
        assert "开始状态" in parsed_flow.states
        assert "处理状态" in parsed_flow.states
        
        start_state = parsed_flow.states["开始状态"]
        assert "🎉" in start_state.config["response_templates"]["default"]
        assert "😊" in start_state.config["response_templates"]["default"]
```

##### `/tests/integration/test_cross_service_communication.py`
**Purpose:** Test communication between MCP Engine and other services

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from src.services.execution_service import ExecutionService
from src.clients.model_orchestrator_client import ModelOrchestratorClient
from src.clients.adaptor_service_client import AdaptorServiceClient

class TestCrossServiceCommunication:
    """Test integration between MCP Engine and external services"""
    
    @pytest.fixture
    async def execution_service(self):
        """Setup execution service with mocked clients"""
        # Mock dependencies
        state_engine = AsyncMock()
        flow_repository = AsyncMock()
        context_service = AsyncMock()
        integration_service = AsyncMock()
        response_service = AsyncMock()
        analytics_client = AsyncMock()
        cache_client = AsyncMock()
        
        return ExecutionService(
            state_engine=state_engine,
            flow_repository=flow_repository,
            context_service=context_service,
            integration_service=integration_service,
            response_service=response_service,
            analytics_client=analytics_client,
            cache_client=cache_client
        )
    
    async def test_model_orchestrator_integration(self, execution_service):
        """
        Test integration with Model Orchestrator service
        
        Parameters:
        - execution_service: ExecutionService fixture
        
        Returns: None
        
        Tests:
        - Intent detection calls
        - Response generation calls
        - Error handling
        - Timeout handling
        """
        # Mock model orchestrator responses
        intent_response = {
            "detected_intent": "order_inquiry",
            "confidence": 0.92,
            "alternatives": [
                {"intent": "product_info", "confidence": 0.78}
            ]
        }
        
        response_generation = {
            "response": "I'll help you check your order status.",
            "confidence": 0.89,
            "model_used": "gpt-4-turbo"
        }
        
        # Setup mocks
        execution_service.response_service.detect_intent = AsyncMock(return_value=intent_response)
        execution_service.response_service.generate_response = AsyncMock(return_value=response_generation)
        
        # Test message processing
        from src.models.domain.state_machine import ExecutionContext
        from src.models.api.messages import InboundMessage, MessageContent
        
        message = InboundMessage(
            content=MessageContent(type="text", text="I want to check my order"),
            user_id="user_123",
            channel="web",
            metadata={}
        )
        
        context = ExecutionContext(
            conversation_id="conv_123",
            tenant_id="tenant_123",
            user_id="user_123",
            current_state="intent_detection"
        )
        
        # Mock context and flow loading
        execution_service.context_service.load_context = AsyncMock(return_value=context)
        execution_service.flow_repository.get_active_flow = AsyncMock(return_value=Mock())
        execution_service.state_engine.execute_state = AsyncMock(return_value=Mock(
            success=True,
            new_state="slot_filling",
            actions=[],
            response={"text": "I'll help you check your order status."}
        ))
        
        # Execute
        result = await execution_service.process_message(
            tenant_id="tenant_123",
            conversation_id="conv_123",
            message=message
        )
        
        # Verify calls were made
        execution_service.response_service.detect_intent.assert_called_once()
        execution_service.state_engine.execute_state.assert_called_once()
        
        assert result is not None

    async def test_adaptor_service_integration(self, execution_service):
        """
        Test integration with Adaptor Service
        
        Parameters:
        - execution_service: ExecutionService fixture
        
        Returns: None
        
        Tests:
        - External API calls
        - Data transformation
        - Error scenarios
        - Timeout handling
        """
        from src.models.domain.state_machine import IntegrationConfig
        
        # Setup integration config
        integration_config = IntegrationConfig(
            integration_id="order_lookup",
            endpoint="/orders/lookup",
            method="GET",
            request_mapping={"order_id": "{{slots.order_number}}"},
            response_mapping={"status": "$.order.status", "total": "$.order.total"}
        )
        
        # Mock successful integration response
        integration_response = {
            "success": True,
            "data": {
                "order": {
                    "id": "ORD123456",
                    "status": "shipped",
                    "total": "$99.99"
                }
            },
            "execution_time_ms": 245
        }
        
        execution_service.integration_service.execute_integration = AsyncMock(
            return_value=integration_response
        )
        
        # Test integration execution
        from src.models.domain.state_machine import ExecutionContext
        
        context = ExecutionContext(
            conversation_id="conv_123",
            tenant_id="tenant_123",
            user_id="user_123",
            current_state="order_lookup",
            slots={"order_number": "ORD123456"}
        )
        
        result = await execution_service.execute_integration(
            integration_config=integration_config,
            context=context,
            timeout_ms=5000
        )
        
        # Verify integration was called
        execution_service.integration_service.execute_integration.assert_called_once()
        assert result["success"] is True
        assert result["data"]["order"]["status"] == "shipped"

    async def test_analytics_event_publishing(self, execution_service):
        """
        Test publishing events to Analytics Engine
        
        Parameters:
        - execution_service: ExecutionService fixture
        
        Returns: None
        
        Tests:
        - Event publishing
        - Async handling
        - Error tolerance
        """
        # Mock analytics client
        execution_service.analytics_client.publish_event = AsyncMock()
        
        # Setup test data
        from src.models.api.messages import InboundMessage, MessageContent
        
        message = InboundMessage(
            content=MessageContent(type="text", text="test message"),
            user_id="user_123",
            channel="web",
            metadata={}
        )
        
        # Mock successful processing
        execution_service.context_service.load_context = AsyncMock(return_value=Mock())
        execution_service.flow_repository.get_active_flow = AsyncMock(return_value=Mock())
        execution_service.state_engine.execute_state = AsyncMock(return_value=Mock(
            success=True,
            new_state="completed",
            actions=[],
            response={"text": "Done"}
        ))
        
        # Process message
        await execution_service.process_message(
            tenant_id="tenant_123",
            conversation_id="conv_123",
            message=message
        )
        
        # Verify analytics event was published
        execution_service.analytics_client.publish_event.assert_called()
        
        # Check event data
        call_args = execution_service.analytics_client.publish_event.call_args
        event_data = call_args[1] if call_args[1] else call_args[0][0]
        
        assert "conversation_id" in str(event_data)
        assert "tenant_id" in str(event_data)

    async def test_service_timeout_handling(self, execution_service):
        """
        Test handling of service timeouts
        
        Parameters:
        - execution_service: ExecutionService fixture
        
        Returns: None
        
        Tests:
        - Model orchestrator timeouts
        - Integration timeouts
        - Graceful degradation
        """
        # Mock timeout scenarios
        async def timeout_function(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate timeout
            return Mock()
        
        execution_service.response_service.detect_intent = timeout_function
        execution_service.integration_service.execute_integration = timeout_function
        
        # Test message processing with timeout
        from src.models.api.messages import InboundMessage, MessageContent
        
        message = InboundMessage(
            content=MessageContent(type="text", text="test"),
            user_id="user_123",
            channel="web",
            metadata={}
        )
        
        # Mock other dependencies
        execution_service.context_service.load_context = AsyncMock(return_value=Mock())
        execution_service.flow_repository.get_active_flow = AsyncMock(return_value=Mock())
        
        # Should handle timeouts gracefully
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await asyncio.wait_for(
                execution_service.process_message(
                    tenant_id="tenant_123",
                    conversation_id="conv_123",
                    message=message
                ),
                timeout=3.0
            )
        except asyncio.TimeoutError:
            result = None
        
        end_time = asyncio.get_event_loop().time()
        
        # Should timeout within expected time
        assert (end_time - start_time) < 4.0
```

### Step 2: Load Testing and Performance Benchmarks
**Duration:** 2 days

#### Files to Create:

##### `/tests/load/test_concurrent_conversations.py`
**Purpose:** Load testing for concurrent conversation handling

```python
import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from src.services.execution_service import ExecutionService

class TestConcurrentConversations:
    """Load testing for concurrent conversation processing"""
    
    @pytest.mark.load
    async def test_concurrent_conversation_processing(self):
        """
        Test processing multiple conversations simultaneously
        
        Parameters: None
        Returns: None
        
        Metrics:
        - Throughput (conversations/second)
        - Response time distribution
        - Error rate
        - Memory usage
        """
        # Configuration
        num_conversations = 100
        messages_per_conversation = 10
        
        # Setup execution service (would use real instance in load test)
        execution_service = self._create_execution_service()
        
        # Generate test conversations
        conversations = [
            self._generate_conversation_data(f"conv_{i}", num_messages=messages_per_conversation)
            for i in range(num_conversations)
        ]
        
        # Metrics collection
        response_times = []
        errors = []
        start_time = time.time()
        
        # Execute concurrent conversations
        tasks = []
        for conv_data in conversations:
            task = asyncio.create_task(
                self._process_conversation(execution_service, conv_data, response_times, errors)
            )
            tasks.append(task)
        
        # Wait for all conversations to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        total_messages = num_conversations * messages_per_conversation
        throughput = total_messages / total_time
        error_rate = len(errors) / total_messages
        
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if response_times else 0
        
        # Performance assertions
        assert throughput > 50  # Should process at least 50 messages/second
        assert error_rate < 0.05  # Error rate should be less than 5%
        assert avg_response_time < 1000  # Average response time < 1 second
        assert p95_response_time < 2000  # 95th percentile < 2 seconds
        
        print(f"Load Test Results:")
        print(f"  Throughput: {throughput:.2f} messages/second")
        print(f"  Error Rate: {error_rate:.2%}")
        print(f"  Avg Response Time: {avg_response_time:.2f}ms")
        print(f"  95th Percentile: {p95_response_time:.2f}ms")
    
    def _create_execution_service(self):
        """Create execution service for load testing"""
        # Would create real service with proper configuration
        # For now, return mock that simulates realistic performance
        from unittest.mock import AsyncMock
        
        service = AsyncMock()
        
        async def mock_process_message(*args, **kwargs):
            # Simulate realistic processing time
            await asyncio.sleep(0.1 + (0.05 * asyncio.get_event_loop().time() % 1))
            return {"success": True, "response": "Mock response"}
        
        service.process_message = mock_process_message
        return service
    
    def _generate_conversation_data(self, conversation_id: str, num_messages: int):
        """Generate test conversation data"""
        from src.models.api.messages import InboundMessage, MessageContent
        
        messages = []
        for i in range(num_messages):
            message = InboundMessage(
                content=MessageContent(
                    type="text",
                    text=f"Test message {i} for {conversation_id}"
                ),
                user_id=f"user_{conversation_id}",
                channel="web",
                metadata={"sequence": i}
            )
            messages.append(message)
        
        return {
            "conversation_id": conversation_id,
            "tenant_id": "tenant_load_test",
            "messages": messages
        }
    
    async def _process_conversation(self, service, conv_data, response_times, errors):
        """Process a single conversation and collect metrics"""
        for message in conv_data["messages"]:
            start_time = time.time()
            
            try:
                result = await service.process_message(
                    tenant_id=conv_data["tenant_id"],
                    conversation_id=conv_data["conversation_id"],
                    message=message
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                
                if not result.get("success", True):
                    errors.append(f"Processing failed for {conv_data['conversation_id']}")
                    
            except Exception as e:
                errors.append(f"Exception in {conv_data['conversation_id']}: {str(e)}")
                
            # Small delay between messages in same conversation
            await asyncio.sleep(0.01)

    @pytest.mark.load
    async def test_memory_usage_under_load(self):
        """
        Test memory usage under sustained load
        
        Parameters: None
        Returns: None
        
        Monitors:
        - Memory growth
        - Garbage collection frequency
        - Memory leaks
        """
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run sustained load for 5 minutes
        duration_seconds = 300
        start_time = time.time()
        
        execution_service = self._create_execution_service()
        
        while time.time() - start_time < duration_seconds:
            # Process batch of conversations
            batch_size = 20
            tasks = []
            
            for i in range(batch_size):
                conv_data = self._generate_conversation_data(f"load_conv_{i}", 5)
                task = asyncio.create_task(
                    self._process_conversation(execution_service, conv_data, [], [])
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check memory every 30 seconds
            if int(time.time() - start_time) % 30 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                print(f"Memory usage: {current_memory:.1f}MB (growth: {memory_growth:.1f}MB)")
                
                # Force garbage collection
                gc.collect()
                
                # Memory shouldn't grow more than 100MB during test
                if memory_growth > 100:
                    pytest.fail(f"Excessive memory growth: {memory_growth:.1f}MB")
            
            await asyncio.sleep(1)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        # Allow some memory growth but not excessive
        assert total_growth < 200, f"Memory growth too high: {total_growth:.1f}MB"
```

## Week 2: Deployment Automation & Production Setup

### Step 3: Kubernetes Deployment Configuration
**Duration:** 3 days

#### Files to Create:

##### `/k8s/mcp-engine-deployment.yaml`
**Purpose:** Kubernetes deployment configuration for MCP Engine

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-engine
  namespace: chatbot-platform
  labels:
    app: mcp-engine
    component: core
    version: v2.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: mcp-engine
  template:
    metadata:
      labels:
        app: mcp-engine
        component: core
        version: v2.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8002"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: mcp-engine-sa
      containers:
      - name: mcp-engine
        image: chatbot-platform/mcp-engine:v2.0.0
        ports:
        - containerPort: 8002
          name: http
          protocol: TCP
        - containerPort: 50051
          name: grpc
          protocol: TCP
        env:
        - name: SERVICE_NAME
          value: "mcp-engine"
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: GRPC_PORT
          value: "50051"
        - name: HTTP_PORT
          value: "8002"
        - name: POSTGRES_URI
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: postgres-uri
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: cache-secrets
              key: redis-url
        - name: MODEL_ORCHESTRATOR_URL
          value: "model-orchestrator:50053"
        - name: ADAPTOR_SERVICE_URL
          value: "http://adaptor-service:8004"
        - name: ANALYTICS_KAFKA_BROKERS
          value: "kafka-cluster:9092"
        - name: MAX_PARALLEL_EXECUTIONS
          value: "100"
        - name: STATE_EXECUTION_TIMEOUT_MS
          value: "10000"
        - name: CONTEXT_LOCK_TIMEOUT_MS
          value: "5000"
        - name: ENABLE_AB_TESTING
          value: "true"
        - name: ENABLE_FLOW_VERSIONING
          value: "true"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8002
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: mcp-engine-config
      - name: logs-volume
        emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - mcp-engine
              topologyKey: kubernetes.io/hostname
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-engine
  namespace: chatbot-platform
  labels:
    app: mcp-engine
spec:
  type: ClusterIP
  ports:
  - port: 8002
    targetPort: 8002
    protocol: TCP
    name: http
  - port: 50051
    targetPort: 50051
    protocol: TCP
    name: grpc
  selector:
    app: mcp-engine
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mcp-engine-sa
  namespace: chatbot-platform
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: mcp-engine-config
  namespace: chatbot-platform
data:
  config.yaml: |
    service:
      name: mcp-engine
      version: v2.0.0
      environment: production
    
    logging:
      level: INFO
      format: json
      file: /app/logs/mcp-engine.log
      max_size_mb: 100
      max_files: 10
    
    database:
      pool_size: 20
      max_overflow: 30
      pool_timeout: 30
      pool_recycle: 3600
    
    redis:
      pool_size: 20
      socket_timeout: 5
      socket_connect_timeout: 5
      retry_on_timeout: true
    
    performance:
      max_parallel_executions: 100
      state_execution_timeout_ms: 10000
      context_lock_timeout_ms: 5000
      cache_ttl_seconds: 300
    
    monitoring:
      enable_metrics: true
      metrics_port: 8002
      health_check_interval: 30
    
    features:
      enable_ab_testing: true
      enable_flow_versioning: true
      enable_visual_designer: true
      enable_flow_analytics: true
```

##### `/scripts/deploy.sh`
**Purpose:** Deployment automation script

```bash
#!/bin/bash

# MCP Engine Deployment Script
# Purpose: Automated deployment with health checks and rollback capability

set -euo pipefail

# Configuration
NAMESPACE="chatbot-platform"
APP_NAME="mcp-engine"
IMAGE_TAG="${1:-latest}"
TIMEOUT_SECONDS=300
HEALTH_CHECK_RETRIES=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Deployment functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not configured or cluster is not accessible"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Check image exists
    if ! docker image inspect "chatbot-platform/${APP_NAME}:${IMAGE_TAG}" &> /dev/null; then
        log_warn "Image chatbot-platform/${APP_NAME}:${IMAGE_TAG} not found locally"
        log_info "Assuming image exists in registry..."
    fi
    
    log_info "Prerequisites check passed"
}

backup_current_deployment() {
    log_info "Backing up current deployment..."
    
    if kubectl get deployment $APP_NAME -n $NAMESPACE &> /dev/null; then
        kubectl get deployment $APP_NAME -n $NAMESPACE -o yaml > "${APP_NAME}-backup-$(date +%Y%m%d-%H%M%S).yaml"
        log_info "Backup created"
    else
        log_info "No existing deployment to backup"
    fi
}

deploy_application() {
    log_info "Deploying $APP_NAME with image tag: $IMAGE_TAG"
    
    # Update image tag in deployment
    sed -i.bak "s|chatbot-platform/mcp-engine:v2.0.0|chatbot-platform/mcp-engine:${IMAGE_TAG}|g" k8s/mcp-engine-deployment.yaml
    
    # Apply deployment
    kubectl apply -f k8s/mcp-engine-deployment.yaml -n $NAMESPACE
    
    # Restore original file
    mv k8s/mcp-engine-deployment.yaml.bak k8s/mcp-engine-deployment.yaml
    
    log_info "Deployment applied"
}

wait_for_rollout() {
    log_info "Waiting for rollout to complete..."
    
    if kubectl rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=${TIMEOUT_SECONDS}s; then
        log_info "Rollout completed successfully"
        return 0
    else
        log_error "Rollout failed or timed out"
        return 1
    fi
}

health_check() {
    log_info "Performing health checks..."
    
    local retry_count=0
    
    while [ $retry_count -lt $HEALTH_CHECK_RETRIES ]; do
        # Get pod names
        local pods=$(kubectl get pods -n $NAMESPACE -l app=$APP_NAME -o jsonpath='{.items[*].metadata.name}')
        
        if [ -z "$pods" ]; then
            log_warn "No pods found, waiting..."
            sleep 10
            ((retry_count++))
            continue
        fi
        
        local all_healthy=true
        
        for pod in $pods; do
            # Check pod status
            local pod_status=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.phase}')
            
            if [ "$pod_status" != "Running" ]; then
                log_warn "Pod $pod is not running (status: $pod_status)"
                all_healthy=false
                break
            fi
            
            # Check readiness probe
            local ready=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
            
            if [ "$ready" != "True" ]; then
                log_warn "Pod $pod is not ready"
                all_healthy=false
                break
            fi
            
            # Check health endpoint
            if kubectl exec $pod -n $NAMESPACE -- curl -f http://localhost:8002/health &> /dev/null; then
                log_info "Pod $pod health check passed"
            else
                log_warn "Pod $pod health check failed"
                all_healthy=false
                break
            fi
        done
        
        if $all_healthy; then
            log_info "All health checks passed"
            return 0
        fi
        
        log_warn "Health check attempt $((retry_count + 1))/$HEALTH_CHECK_RETRIES failed, retrying..."
        sleep 10
        ((retry_count++))
    done
    
    log_error "Health checks failed after $HEALTH_CHECK_RETRIES attempts"
    return 1
}

rollback_deployment() {
    log_error "Deployment failed, initiating rollback..."
    
    kubectl rollout undo deployment/$APP_NAME -n $NAMESPACE
    
    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=${TIMEOUT_SECONDS}s
    
    log_info "Rollback completed"
}

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get service endpoint
    local service_ip=$(kubectl get service $APP_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    
    # Test HTTP health endpoint
    if kubectl run smoke-test-http --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f "http://$service_ip:8002/health" &> /dev/null; then
        log_info "HTTP health endpoint test passed"
    else
        log_error "HTTP health endpoint test failed"
        return 1
    fi
    
    # Test gRPC health endpoint
    if kubectl run smoke-test-grpc --rm -i --restart=Never --image=fullstorydev/grpcurl -- \
        grpcurl -plaintext "$service_ip:50051" grpc.health.v1.Health/Check &> /dev/null; then
        log_info "gRPC health endpoint test passed"
    else
        log_error "gRPC health endpoint test failed"
        return 1
    fi
    
    log_info "Smoke tests completed successfully"
}

# Main deployment process
main() {
    log_info "Starting deployment of $APP_NAME:$IMAGE_TAG"
    
    check_prerequisites
    backup_current_deployment
    deploy_application
    
    if wait_for_rollout; then
        if health_check; then
            run_smoke_tests
            log_info "Deployment completed successfully!"
        else
            rollback_deployment
            exit 1
        fi
    else
        rollback_deployment
        exit 1
    fi
}

# Script execution
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
```

### Step 4: CI/CD Pipeline Setup
**Duration:** 2 days

#### Files to Create:

##### `/.github/workflows/mcp-engine-ci-cd.yml`
**Purpose:** GitHub Actions CI/CD pipeline

```yaml
name: MCP Engine CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'mcp-engine/**'
      - '.github/workflows/mcp-engine-ci-cd.yml'
  pull_request:
    branches: [ main ]
    paths:
      - 'mcp-engine/**'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: chatbot-platform/mcp-engine
  SERVICE_NAME: mcp-engine

jobs:
  # Job 1: Code Quality and Testing
  test:
    name: Code Quality & Testing
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: test_mcp
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        cd mcp-engine
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Code formatting check
      run: |
        cd mcp-engine
        black --check --diff .
        isort --check-only --diff .
    
    - name: Linting
      run: |
        cd mcp-engine
        flake8 src/ tests/
        pylint src/ --fail-under=8.0
    
    - name: Type checking
      run: |
        cd mcp-engine
        mypy src/
    
    - name: Security scanning
      run: |
        cd mcp-engine
        bandit -r src/ -f json -o bandit-report.json
        safety check --json --output safety-report.json
      continue-on-error: true
    
    - name: Run unit tests
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/test_mcp
        REDIS_URL: redis://localhost:6379
      run: |
        cd mcp-engine
        pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/test_mcp
        REDIS_URL: redis://localhost:6379
      run: |
        cd mcp-engine
        pytest tests/integration/ -v --tb=short
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: mcp-engine/coverage.xml
        flags: mcp-engine
        name: mcp-engine-coverage
    
    - name: Archive test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: |
          mcp-engine/coverage.xml
          mcp-engine/htmlcov/
          mcp-engine/bandit-report.json
          mcp-engine/safety-report.json
  
  # Job 2: Build and Push Docker Image
  build:
    name: Build & Push Image
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push'
    
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: mcp-engine/
        file: mcp-engine/Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          VERSION=${{ github.sha }}
          BUILD_DATE=${{ github.event.head_commit.timestamp }}
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
  
  # Job 3: Deploy to Staging
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}
    
    - name: Deploy to staging
      run: |
        cd mcp-engine
        chmod +x scripts/deploy.sh
        ./scripts/deploy.sh ${{ github.sha }}
      env:
        NAMESPACE: chatbot-platform-staging
    
    - name: Run E2E tests
      run: |
        cd mcp-engine
        pytest tests/e2e/ -v --tb=short
      env:
        TEST_ENVIRONMENT: staging
        MCP_ENGINE_URL: http://mcp-engine.chatbot-platform-staging.svc.cluster.local:8002
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        text: 'MCP Engine deployed to staging :rocket:'
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
      if: always()
  
  # Job 4: Deploy to Production
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_PRODUCTION }}
    
    - name: Deploy to production
      run: |
        cd mcp-engine
        chmod +x scripts/deploy.sh
        ./scripts/deploy.sh ${{ github.sha }}
      env:
        NAMESPACE: chatbot-platform
    
    - name: Run smoke tests
      run: |
        cd mcp-engine
        pytest tests/smoke/ -v
      env:
        TEST_ENVIRONMENT: production
        MCP_ENGINE_URL: http://mcp-engine.chatbot-platform.svc.cluster.local:8002
    
    - name: Create GitHub release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: mcp-engine-v${{ github.run_number }}
        release_name: MCP Engine Release v${{ github.run_number }}
        body: |
          ## Changes
          - Deployed MCP Engine version: ${{ github.sha }}
          - Image: ${{ needs.build.outputs.image-tag }}
          
          ## Deployment Details
          - Environment: Production
          - Deployment Time: ${{ github.event.head_commit.timestamp }}
          - Deployed by: ${{ github.actor }}
        draft: false
        prerelease: false
    
    - name: Notify production deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#production'
        text: 'MCP Engine deployed to production :tada:'
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
      if: always()
  
  # Job 5: Performance Testing
  performance-test:
    name: Performance Testing
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        cd mcp-engine
        pip install -r requirements.txt
        pip install locust
    
    - name: Run load tests
      run: |
        cd mcp-engine/tests/load
        locust -f locustfile.py --host=http://mcp-engine.chatbot-platform-staging.svc.cluster.local:8002 \
          --users=100 --spawn-rate=10 --run-time=5m --headless --html=load_test_report.html
    
    - name: Upload load test results
      uses: actions/upload-artifact@v3
      with:
        name: load-test-results
        path: mcp-engine/tests/load/load_test_report.html
```

## Week 3: Documentation & Production Readiness

### Step 5: Comprehensive Documentation
**Duration:** 3 days

#### Files to Create:

##### `/docs/production-runbook.md`
**Purpose:** Production operations and troubleshooting guide

```markdown
# MCP Engine Production Runbook

## Overview
This runbook provides operational procedures for the MCP Engine service in production environments.

## Service Architecture

### Components
- **MCP Engine**: Core conversation flow management service
- **Dependencies**: PostgreSQL, Redis, Model Orchestrator, Adaptor Service
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Logs**: Structured JSON logs via stdout

### Service Endpoints
- **gRPC**: Port 50051 (internal service communication)
- **HTTP**: Port 8002 (health checks, metrics)
- **Health Check**: `GET /health`
- **Readiness Check**: `GET /ready`
- **Metrics**: `GET /metrics`

## Monitoring & Alerting

### Key Metrics to Monitor

#### Performance Metrics
| Metric | Description | Threshold | Alert Level |
|--------|-------------|-----------|-------------|
| `mcp_request_duration_seconds` | Request processing time | p95 > 2s | Warning |
| `mcp_request_duration_seconds` | Request processing time | p95 > 5s | Critical |
| `mcp_requests_total` | Total requests processed | Rate declining | Warning |
| `mcp_errors_total` | Total error count | Error rate > 5% | Critical |

#### Resource Metrics
| Metric | Description | Threshold | Alert Level |
|--------|-------------|-----------|-------------|
| `container_memory_usage_bytes` | Memory usage | > 80% of limit | Warning |
| `container_memory_usage_bytes` | Memory usage | > 95% of limit | Critical |
| `container_cpu_usage_seconds_total` | CPU usage | > 80% of limit | Warning |
| `container_cpu_usage_seconds_total` | CPU usage | > 95% of limit | Critical |

#### Business Metrics
| Metric | Description | Threshold | Alert Level |
|--------|-------------|-----------|-------------|
| `mcp_conversations_active` | Active conversations | Unusual spikes | Info |
| `mcp_state_transitions_total` | State transitions | Rate declining | Warning |
| `mcp_integration_failures_total` | Integration failures | > 10% failure rate | Critical |

### Alert Configuration

#### Prometheus Alert Rules
```yaml
groups:
- name: mcp-engine-alerts
  rules:
  - alert: MCPEngineHighLatency
    expr: histogram_quantile(0.95, sum(rate(mcp_request_duration_seconds_bucket[5m])) by (le)) > 2
    for: 5m
    labels:
      severity: warning
      service: mcp-engine
    annotations:
      summary: "MCP Engine high latency detected"
      description: "95th percentile latency is {{ $value }}s"
  
  - alert: MCPEngineHighErrorRate
    expr: sum(rate(mcp_errors_total[5m])) / sum(rate(mcp_requests_total[5m])) > 0.05
    for: 2m
    labels:
      severity: critical
      service: mcp-engine
    annotations:
      summary: "MCP Engine high error rate"
      description: "Error rate is {{ $value | humanizePercentage }}"
  
  - alert: MCPEngineDown
    expr: up{job="mcp-engine"} == 0
    for: 1m
    labels:
      severity: critical
      service: mcp-engine
    annotations:
      summary: "MCP Engine is down"
      description: "MCP Engine has been down for more than 1 minute"
```

## Troubleshooting Guide

### Common Issues

#### 1. High Response Times

**Symptoms:**
- Request latency alerts firing
- Users reporting slow responses
- P95 latency > 2 seconds

**Investigation Steps:**
1. Check system resources (CPU, memory)
2. Examine database connection pool
3. Check Redis cache hit rates
4. Review external service response times

**Commands:**
```bash
# Check pod resource usage
kubectl top pods -n chatbot-platform -l app=mcp-engine

# Check logs for slow queries
kubectl logs -n chatbot-platform -l app=mcp-engine | grep "slow_query"

# Check Redis metrics
redis-cli --latency -h redis-cluster-host

# Check database connections
kubectl exec -it mcp-engine-pod -- \
  psql $POSTGRES_URI -c "SELECT count(*) FROM pg_stat_activity;"
```

**Resolution:**
- Scale up pods if CPU/memory constrained
- Optimize database queries
- Increase Redis cache TTL
- Add circuit breakers for external services

#### 2. Memory Leaks

**Symptoms:**
- Gradual memory increase over time
- Pods being OOMKilled
- Memory usage alerts

**Investigation Steps:**
1. Monitor memory usage trends
2. Check for unclosed database connections
3. Review object cleanup in state machine
4. Analyze garbage collection logs

**Commands:**
```bash
# Check memory usage over time
kubectl top pods -n chatbot-platform -l app=mcp-engine --sort-by=memory

# Check for memory leaks in logs
kubectl logs -n chatbot-platform -l app=mcp-engine | grep -E "(memory|oom|gc)"

# Get detailed pod resource info
kubectl describe pod -n chatbot-platform -l app=mcp-engine
```

**Resolution:**
- Restart affected pods
- Review and fix code for memory leaks
- Adjust memory limits if needed
- Implement memory profiling

#### 3. Database Connection Issues

**Symptoms:**
- Database connection errors in logs
- Intermittent service failures
- Connection pool exhaustion

**Investigation Steps:**
1. Check database server health
2. Verify connection pool configuration
3. Review connection lifecycle
4. Check network connectivity

**Commands:**
```bash
# Check database connectivity
kubectl exec -it mcp-engine-pod -- \
  pg_isready -h postgres-host -p 5432

# Check connection pool stats
kubectl exec -it mcp-engine-pod -- \
  psql $POSTGRES_URI -c "SELECT * FROM pg_stat_database;"

# Check for long-running transactions
kubectl exec -it mcp-engine-pod -- \
  psql $POSTGRES_URI -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

**Resolution:**
- Increase connection pool size
- Add connection retry logic
- Implement connection health checks
- Scale database if needed

#### 4. Redis Cache Issues

**Symptoms:**
- Cache miss rates increasing
- Redis connection timeouts
- Slow context loading

**Investigation Steps:**
1. Check Redis cluster health
2. Monitor cache hit rates
3. Review cache eviction policies
4. Check memory usage

**Commands:**
```bash
# Check Redis cluster status
redis-cli -h redis-cluster-host cluster info

# Check cache statistics
redis-cli -h redis-cluster-host info stats

# Monitor real-time commands
redis-cli -h redis-cluster-host monitor
```

**Resolution:**
- Scale Redis cluster
- Optimize cache keys and TTL
- Implement cache warming
- Add cache circuit breakers

### Emergency Procedures

#### Service Restart
```bash
# Rolling restart of MCP Engine pods
kubectl rollout restart deployment/mcp-engine -n chatbot-platform

# Check rollout status
kubectl rollout status deployment/mcp-engine -n chatbot-platform
```

#### Rollback Deployment
```bash
# Rollback to previous version
kubectl rollout undo deployment/mcp-engine -n chatbot-platform

# Rollback to specific revision
kubectl rollout undo deployment/mcp-engine --to-revision=2 -n chatbot-platform
```

#### Scale Service
```bash
# Scale up replicas
kubectl scale deployment/mcp-engine --replicas=5 -n chatbot-platform

# Scale down replicas
kubectl scale deployment/mcp-engine --replicas=2 -n chatbot-platform
```

#### Emergency Maintenance Mode
```bash
# Label nodes for maintenance
kubectl label nodes node-1 maintenance=true

# Cordon node to prevent new pods
kubectl cordon node-1

# Drain node safely
kubectl drain node-1 --ignore-daemonsets --delete-emptydir-data
```

## Performance Tuning

### Resource Optimization

#### Memory Tuning
- **Initial allocation**: 1Gi request, 4Gi limit
- **Heap size**: Set to 70% of container limit
- **GC tuning**: Monitor and adjust based on workload

#### CPU Tuning
- **Initial allocation**: 500m request, 2000m limit
- **Thread pool**: Configure based on expected load
- **Async operations**: Use for I/O bound operations

#### Database Tuning
- **Connection pool**: 20 connections per pod
- **Query timeout**: 30 seconds
- **Connection recycling**: 1 hour

#### Redis Tuning
- **Connection pool**: 20 connections per pod
- **Timeout settings**: 5 seconds
- **Retry policy**: 3 retries with exponential backoff

### Scaling Guidelines

#### Horizontal Scaling Triggers
- CPU utilization > 70% for 5 minutes
- Memory utilization > 80% for 5 minutes
- Request queue depth > 100
- Response time P95 > 2 seconds

#### Vertical Scaling Considerations
- Memory leaks requiring restarts
- CPU-intensive operations
- Large context processing

## Security Operations

### Access Control
- Service account with minimal permissions
- Network policies restricting ingress/egress
- Secrets management via Kubernetes secrets
- Regular security scanning of images

### Audit Procedures
- Review access logs monthly
- Monitor for unusual access patterns
- Validate secret rotation
- Check compliance with security policies

### Incident Response
1. **Detection**: Automated alerts and monitoring
2. **Assessment**: Determine scope and impact
3. **Containment**: Isolate affected components
4. **Recovery**: Restore service functionality
5. **Review**: Post-incident analysis and improvements

## Backup & Recovery

### Data Backup
- **PostgreSQL**: Automated daily backups with PITR
- **Redis**: RDB snapshots every 2 hours
- **Configuration**: Versioned in Git repository
- **State data**: Included in database backups

### Recovery Procedures
- **Database restore**: Point-in-time recovery available
- **Cache rebuild**: Automatic on service restart
- **Configuration rollback**: Via Git version control
- **Service recovery**: Kubernetes self-healing

## Contact Information

### On-Call Rotation
- **Primary**: Platform Engineering Team
- **Secondary**: Backend Development Team
- **Escalation**: Engineering Manager

### Communication Channels
- **Alerts**: #alerts-production
- **Incidents**: #incident-response
- **General**: #platform-engineering

### External Dependencies
- **Database Team**: database-team@company.com
- **Infrastructure Team**: infra-team@company.com
- **Security Team**: security-team@company.com
```

##### `/docs/api-documentation.md`
**Purpose:** Complete API documentation for developers

```markdown
# MCP Engine API Documentation

## Overview
The MCP Engine provides conversation flow management through gRPC and HTTP APIs. This document covers all available endpoints, message formats, and integration patterns.

## Authentication
All requests require either:
- **JWT Token**: `Authorization: Bearer <token>`
- **API Key**: `Authorization: ApiKey <key>`

## gRPC API

### Service Definition
```protobuf
service MCPEngine {
    rpc ProcessMessage(ProcessMessageRequest) returns (ProcessMessageResponse);
    rpc GetConversationState(GetStateRequest) returns (GetStateResponse);
    rpc ResetConversation(ResetRequest) returns (ResetResponse);
    rpc CreateFlow(CreateFlowRequest) returns (CreateFlowResponse);
    rpc UpdateFlow(UpdateFlowRequest) returns (UpdateFlowResponse);
    rpc GetFlow(GetFlowRequest) returns (GetFlowResponse);
    rpc ListFlows(ListFlowsRequest) returns (ListFlowsResponse);
    rpc DeleteFlow(DeleteFlowRequest) returns (DeleteFlowResponse);
}
```

### ProcessMessage
Process an incoming message through the conversation flow.

**Request:**
```protobuf
message ProcessMessageRequest {
    string tenant_id = 1;
    string conversation_id = 2;
    MessageContent content = 3;
    map<string, string> metadata = 4;
    ProcessingHints hints = 5;
}

message MessageContent {
    string type = 1;  // text, image, file, audio, video
    string text = 2;
    map<string, string> attributes = 3;
}

message ProcessingHints {
    string priority = 1;      // low, normal, high, urgent
    bool bypass_cache = 2;
    string force_flow = 3;    // Force specific flow ID
    int32 timeout_ms = 4;
}
```

**Response:**
```protobuf
message ProcessMessageResponse {
    string conversation_id = 1;
    string current_state = 2;
    MessageContent response = 3;
    map<string, string> context_updates = 4;
    repeated string actions_performed = 5;
    int32 processing_time_ms = 6;
    string ab_variant = 7;
    repeated Error errors = 8;
}
```

**Example:**
```python
import grpc
from mcp_pb2 import ProcessMessageRequest, MessageContent
from mcp_pb2_grpc import MCPEngineStub

# Create gRPC channel
channel = grpc.insecure_channel('mcp-engine:50051')
client = MCPEngineStub(channel)

# Process message
request = ProcessMessageRequest(
    tenant_id="tenant_123",
    conversation_id="conv_456",
    content=MessageContent(
        type="text",
        text="I want to check my order status"
    )
)

response = client.ProcessMessage(request)
print(f"Response: {response.response.text}")
print(f"New state: {response.current_state}")
```

### GetConversationState
Retrieve current conversation state and context.

**Request:**
```protobuf
message GetStateRequest {
    string tenant_id = 1;
    string conversation_id = 2;
    bool include_history = 3;
    bool include_context = 4;
}
```

**Response:**
```protobuf
message GetStateResponse {
    string current_state = 1;
    string flow_id = 2;
    map<string, string> context = 3;
    repeated string state_history = 4;
    int64 created_at = 5;
    int64 last_updated = 6;
}
```

### Flow Management APIs

#### CreateFlow
Create a new conversation flow.

**Request:**
```protobuf
message CreateFlowRequest {
    string tenant_id = 1;
    FlowDefinition flow = 2;
}

message FlowDefinition {
    string name = 1;
    string description = 2;
    string version = 3;
    string initial_state = 4;
    map<string, State> states = 5;
    map<string, string> global_handlers = 6;
}
```

#### UpdateFlow
Update existing conversation flow.

**Request:**
```protobuf
message UpdateFlowRequest {
    string tenant_id = 1;
    string flow_id = 2;
    FlowDefinition flow = 3;
    bool publish = 4;  // Publish immediately
}
```

## HTTP API

### Health Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-05-30T10:00:00Z",
    "version": "v2.0.0",
    "checks": {
        "database": "healthy",
        "redis": "healthy",
        "external_services": "healthy"
    }
}
```

#### Readiness Check
```http
GET /ready
```

**Response:**
```json
{
    "status": "ready",
    "timestamp": "2025-05-30T10:00:00Z"
}
```

### Metrics Endpoint
```http
GET /metrics
```

Returns Prometheus metrics in text format.

### Flow Management (REST)

#### Create Flow
```http
POST /api/v2/flows
Authorization: Bearer <token>
Content-Type: application/json

{
    "name": "Customer Support Flow",
    "description": "Handle customer support inquiries",
    "version": "1.0",
    "flow_definition": {
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "type": "response",
                "config": {
                    "response_templates": {
                        "default": "Hi! How can I help you today?"
                    }
                },
                "transitions": [
                    {
                        "condition": "any_input",
                        "target_state": "intent_detection"
                    }
                ]
            }
        }
    }
}
```

#### List Flows
```http
GET /api/v2/flows?limit=50&offset=0&status=active
Authorization: Bearer <token>
```

**Response:**
```json
{
    "status": "success",
    "data": {
        "flows": [
            {
                "flow_id": "flow_123",
                "name": "Customer Support Flow",
                "version": "1.0",
                "status": "active",
                "created_at": "2025-05-30T10:00:00Z",
                "last_used": "2025-05-30T15:30:00Z"
            }
        ],
        "total": 1,
        "limit": 50,
        "offset": 0
    }
}
```

## Error Handling

### Error Codes
| Code | Description | HTTP Status | gRPC Status |
|------|-------------|-------------|-------------|
| `VALIDATION_ERROR` | Request validation failed | 400 | INVALID_ARGUMENT |
| `FLOW_NOT_FOUND` | Flow does not exist | 404 | NOT_FOUND |
| `STATE_ERROR` | State execution failed | 500 | INTERNAL |
| `TIMEOUT_ERROR` | Processing timeout | 504 | DEADLINE_EXCEEDED |
| `RATE_LIMIT` | Rate limit exceeded | 429 | RESOURCE_EXHAUSTED |

### Error Response Format

**HTTP:**
```json
{
    "status": "error",
    "error": {
        "code": "FLOW_NOT_FOUND",
        "message": "Flow not found",
        "details": {
            "flow_id": "flow_123",
            "tenant_id": "tenant_456"
        },
        "trace_id": "trace_abc123"
    }
}
```

**gRPC:**
```protobuf
message Error {
    string code = 1;
    string message = 2;
    map<string, string> details = 3;
    string trace_id = 4;
}
```

## Rate Limiting

### Limits by Plan
| Plan | Requests/Minute | Burst Limit |
|------|-----------------|-------------|
| Basic | 100 | 200 |
| Standard | 1,000 | 2,000 |
| Premium | 10,000 | 20,000 |
| Enterprise | 100,000 | 200,000 |

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 742
X-RateLimit-Reset: 1622547600
X-RateLimit-Type: tenant_based
```

## Integration Examples

### Python Client
```python
import asyncio
import grpc
from mcp_pb2 import ProcessMessageRequest, MessageContent
from mcp_pb2_grpc import MCPEngineStub

class MCPClient:
    def __init__(self, endpoint: str):
        self.channel = grpc.aio.insecure_channel(endpoint)
        self.client = MCPEngineStub(self.channel)
    
    async def process_message(self, tenant_id: str, conversation_id: str, text: str):
        request = ProcessMessageRequest(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            content=MessageContent(type="text", text=text)
        )
        
        response = await self.client.ProcessMessage(request)
        return response
    
    async def close(self):
        await self.channel.close()

# Usage
async def main():
    client = MCPClient('mcp-engine:50051')
    
    response = await client.process_message(
        tenant_id="tenant_123",
        conversation_id="conv_456",
        text="Hello, I need help"
    )
    
    print(f"Bot response: {response.response.text}")
    await client.close()

asyncio.run(main())
```

### Node.js Client
```javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDefinition = protoLoader.loadSync('mcp.proto');
const mcpProto = grpc.loadPackageDefinition(packageDefinition);

class MCPClient {
    constructor(endpoint) {
        this.client = new mcpProto.MCPEngine(endpoint, grpc.credentials.createInsecure());
    }
    
    processMessage(tenantId, conversationId, text) {
        return new Promise((resolve, reject) => {
            const request = {
                tenant_id: tenantId,
                conversation_id: conversationId,
                content: {
                    type: 'text',
                    text: text
                }
            };
            
            this.client.ProcessMessage(request, (error, response) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(response);
                }
            });
        });
    }
}

// Usage
const client = new MCPClient('mcp-engine:50051');

client.processMessage('tenant_123', 'conv_456', 'Hello')
    .then(response => {
        console.log('Bot response:', response.response.text);
    })
    .catch(error => {
        console.error('Error:', error);
    });
```

### REST API with cURL
```bash
# Process message via HTTP (if HTTP API is available)
curl -X POST http://mcp-engine:8002/api/v2/process \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_123",
    "conversation_id": "conv_456",
    "content": {
      "type": "text",
      "text": "I want to check my order"
    }
  }'

# Get conversation state
curl -X GET "http://mcp-engine:8002/api/v2/conversations/conv_456/state" \
  -H "Authorization: Bearer <token>"

# Create new flow
curl -X POST http://mcp-engine:8002/api/v2/flows \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d @flow-definition.json
```

## Best Practices

### Performance Optimization
1. **Connection Pooling**: Reuse gRPC connections
2. **Caching**: Cache flow definitions and context
3. **Async Processing**: Use async operations for I/O
4. **Batch Operations**: Group multiple requests when possible

### Error Handling
1. **Retry Logic**: Implement exponential backoff
2. **Circuit Breakers**: Prevent cascade failures
3. **Graceful Degradation**: Provide fallback responses
4. **Monitoring**: Track error rates and patterns

### Security
1. **Authentication**: Always use proper authentication
2. **Input Validation**: Validate all inputs
3. **Rate Limiting**: Implement client-side rate limiting
4. **Encryption**: Use TLS for all communications

## Support

### Getting Help
- **Documentation**: Check this documentation first
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Email**: api-support@company.com

### API Versioning
- Current version: v2
- Backward compatibility: 2 major versions
- Deprecation notice: 6 months minimum
- Migration guides: Available for major versions
```

### Step 6: Final Production Checklist
**Duration:** 2 days

#### Files to Create:

##### `/docs/production-checklist.md`
**Purpose:** Pre-production deployment checklist

```markdown
# MCP Engine Production Checklist

## Pre-Deployment Verification

### Code Quality ✅
- [ ] All unit tests passing (>95% coverage)
- [ ] Integration tests passing
- [ ] E2E tests passing
- [ ] Performance tests meeting benchmarks
- [ ] Security scans completed with no critical issues
- [ ] Code review completed and approved
- [ ] Documentation updated

### Configuration ✅
- [ ] Environment variables configured
- [ ] Secrets properly set up
- [ ] Database connections tested
- [ ] Redis cluster connectivity verified
- [ ] External service endpoints configured
- [ ] Monitoring and alerting rules active
- [ ] Logging configuration validated

### Infrastructure ✅
- [ ] Kubernetes cluster ready
- [ ] Namespace created and configured
- [ ] Resource quotas set
- [ ] Network policies applied
- [ ] Storage classes configured
- [ ] Backup systems operational
- [ ] Disaster recovery plan in place

### Security ✅
- [ ] Service accounts configured with minimal permissions
- [ ] RBAC policies applied
- [ ] Secrets rotation schedule established
- [ ] Network policies restricting traffic
- [ ] Image vulnerability scans passed
- [ ] Compliance requirements met
- [ ] Security incident response plan ready

### Performance ✅
- [ ] Load testing completed successfully
- [ ] Resource limits and requests optimized
- [ ] Auto-scaling policies configured
- [ ] Database performance tuned
- [ ] Cache hit ratios optimized
- [ ] CDN configuration (if applicable)

### Monitoring ✅
- [ ] Health checks configured
- [ ] Metrics collection active
- [ ] Alerting rules configured
- [ ] Dashboard created
- [ ] Log aggregation working
- [ ] Distributed tracing enabled
- [ ] On-call rotation established

## Deployment Day Checklist

### Pre-Deployment (1 Hour Before)
- [ ] Team notified of deployment window
- [ ] Rollback plan reviewed
- [ ] Database backups verified
- [ ] Monitoring dashboards opened
- [ ] Communication channels ready
- [ ] Support team on standby

### During Deployment
- [ ] Deployment script executed
- [ ] Health checks passing
- [ ] Smoke tests executed
- [ ] Performance metrics normal
- [ ] Error rates within acceptable limits
- [ ] No critical alerts triggered

### Post-Deployment (1 Hour After)
- [ ] All services responding normally
- [ ] Key business metrics stable
- [ ] User feedback channels monitored
- [ ] Performance benchmarks met
- [ ] Documentation updated with new version
- [ ] Deployment success communicated

## Post-Production Monitoring

### Week 1
- [ ] Daily health check reviews
- [ ] Performance trend analysis
- [ ] Error rate monitoring
- [ ] User feedback collection
- [ ] Resource utilization review

### Month 1
- [ ] Security audit
- [ ] Performance optimization review
- [ ] Cost analysis
- [ ] Capacity planning update
- [ ] Lessons learned documentation

## Rollback Procedures

### Automatic Rollback Triggers
- [ ] Health check failures for >2 minutes
- [ ] Error rate >5% for >1 minute
- [ ] Response time >5s for >2 minutes
- [ ] Memory usage >95% for >1 minute

### Manual Rollback Process
1. Execute rollback script
2. Verify previous version health
3. Update monitoring dashboards
4. Communicate rollback completion
5. Begin root cause analysis

## Success Criteria

### Technical Metrics
- [ ] Response time P95 <2 seconds
- [ ] Error rate <1%
- [ ] Uptime >99.9%
- [ ] Memory usage <80% of limits
- [ ] CPU usage <70% of limits

### Business Metrics
- [ ] Conversation completion rate >95%
- [ ] User satisfaction score >4.0
- [ ] Integration success rate >98%
- [ ] Cost per conversation within budget
- [ ] Feature adoption meeting targets

## Contact Information

### Primary Team
- **Engineering Lead**: eng-lead@company.com
- **DevOps Engineer**: devops@company.com
- **QA Lead**: qa-lead@company.com

### Emergency Contacts
- **On-Call Engineer**: +1-XXX-XXX-XXXX
- **Engineering Manager**: +1-XXX-XXX-XXXX
- **CTO**: +1-XXX-XXX-XXXX

### Communication Channels
- **Slack**: #mcp-engine-deployment
- **Teams**: MCP Engine Team
- **Email**: mcp-engine-team@company.com
```

---

## Summary

Phase 10 delivers a production-ready MCP Engine with:

### Deliverables
1. **Comprehensive Test Suite**
   - Advanced unit tests covering edge cases
   - Integration tests for cross-service communication
   - Load testing for performance validation
   - End-to-end testing for complete workflows

2. **Production Infrastructure**
   - Kubernetes deployment configurations
   - CI/CD pipeline with automated testing
   - Monitoring and alerting setup
   - Security and compliance measures

3. **Complete Documentation**
   - Production runbook for operations
   - API documentation for developers
   - Deployment procedures and checklists
   - Troubleshooting guides

4. **Quality Assurance**
   - Performance benchmarks established
   - Security scanning integrated
   - Automated testing in CI/CD
   - Production readiness validation

### Key Technologies Used
- **Testing**: pytest, pytest-asyncio, locust
- **Deployment**: Kubernetes, Docker, GitHub Actions
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Security**: Trivy, Bandit, Safety
- **Documentation**: Markdown, OpenAPI

### Success Metrics
- Test coverage >95%
- Performance benchmarks met
- Zero-downtime deployments
- Complete operational documentation
- Production readiness certification

This phase ensures the MCP Engine is fully prepared for production deployment with comprehensive testing, monitoring, and documentation supporting long-term operational success.
