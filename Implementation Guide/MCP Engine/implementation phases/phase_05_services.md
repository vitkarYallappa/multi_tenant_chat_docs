# Phase 05: Business Logic Services Layer
**Duration**: Week 9-10 (Days 41-50)  
**Team Size**: 4-5 developers  
**Complexity**: Very High  

## Overview
Implement the comprehensive business logic services layer that orchestrates conversation processing, flow execution, context management, and integration handling. This layer serves as the main business logic coordinator between the API layer and the underlying repositories and core components.

## Step 13: Core Service Infrastructure (Days 41-43)

### Files to Create
```
src/
├── services/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── base_service.py
│   │   └── service_registry.py
│   ├── execution_service.py
│   ├── flow_service.py
│   ├── context_service.py
│   ├── integration_service.py
│   ├── intent_service.py
│   ├── slot_filling_service.py
│   ├── response_generation_service.py
│   ├── analytics_service.py
│   └── notification_service.py
```

### `/src/services/base/base_service.py`
**Purpose**: Base service class with common functionality and dependency injection
```python
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

from src.utils.logger import get_logger
from src.utils.metrics import MetricsCollector
from src.exceptions.base import MCPBaseException, ValidationError
from src.config.settings import settings

logger = get_logger(__name__)

class BaseService(ABC):
    """Base service class with common functionality"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = get_logger(f"service.{service_name}")
        self._initialized = False
        self._dependencies: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize service - to be overridden by subclasses"""
        if self._initialized:
            return
        
        await self._setup_dependencies()
        await self._validate_configuration()
        self._initialized = True
        
        self.logger.info(f"{self.service_name} service initialized")
    
    async def shutdown(self):
        """Shutdown service - to be overridden by subclasses"""
        await self._cleanup_resources()
        self._initialized = False
        
        self.logger.info(f"{self.service_name} service shutdown")
    
    @abstractmethod
    async def _setup_dependencies(self):
        """Setup service dependencies - must be implemented by subclasses"""
        pass
    
    async def _validate_configuration(self):
        """Validate service configuration"""
        # Base validation - can be overridden
        pass
    
    async def _cleanup_resources(self):
        """Cleanup service resources"""
        # Base cleanup - can be overridden
        pass
    
    def add_dependency(self, name: str, dependency: Any):
        """Add a dependency to the service"""
        self._dependencies[name] = dependency
        self.logger.debug(f"Dependency '{name}' added to {self.service_name}")
    
    def get_dependency(self, name: str) -> Any:
        """Get a dependency by name"""
        if name not in self._dependencies:
            raise ValueError(f"Dependency '{name}' not found in {self.service_name}")
        return self._dependencies[name]
    
    def has_dependency(self, name: str) -> bool:
        """Check if dependency exists"""
        return name in self._dependencies
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check - to be overridden by subclasses"""
        return {
            "service": self.service_name,
            "status": "healthy" if self._initialized else "unhealthy",
            "initialized": self._initialized,
            "dependencies": list(self._dependencies.keys())
        }
    
    def _validate_tenant_access(self, tenant_id: str, operation: str):
        """Validate tenant access for operation"""
        if not tenant_id:
            raise ValidationError("Tenant ID is required")
        
        # Add tenant-specific validation logic here
        self.logger.debug(
            "Tenant access validated",
            tenant_id=tenant_id,
            operation=operation
        )
    
    def _record_operation_metrics(
        self,
        operation: str,
        tenant_id: str,
        duration_ms: float,
        status: str = "success"
    ):
        """Record operation metrics"""
        if settings.features.enable_metrics:
            # Record service-specific metrics
            MetricsCollector.record_service_operation(
                service_name=self.service_name,
                operation=operation,
                tenant_id=tenant_id,
                duration_seconds=duration_ms / 1000,
                status=status
            )

class TransactionalService(BaseService):
    """Base service with transaction support"""
    
    def __init__(self, service_name: str):
        super().__init__(service_name)
        self._active_transactions: Dict[str, Any] = {}
    
    async def start_transaction(self, transaction_id: Optional[str] = None) -> str:
        """Start a new transaction"""
        if not transaction_id:
            transaction_id = str(uuid.uuid4())
        
        # Initialize transaction context
        transaction_context = {
            'id': transaction_id,
            'started_at': datetime.utcnow(),
            'operations': [],
            'status': 'active'
        }
        
        self._active_transactions[transaction_id] = transaction_context
        
        self.logger.debug("Transaction started", transaction_id=transaction_id)
        return transaction_id
    
    async def commit_transaction(self, transaction_id: str):
        """Commit transaction"""
        if transaction_id not in self._active_transactions:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        transaction = self._active_transactions[transaction_id]
        transaction['status'] = 'committed'
        transaction['committed_at'] = datetime.utcnow()
        
        # Cleanup transaction
        del self._active_transactions[transaction_id]
        
        self.logger.debug("Transaction committed", transaction_id=transaction_id)
    
    async def rollback_transaction(self, transaction_id: str):
        """Rollback transaction"""
        if transaction_id not in self._active_transactions:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        transaction = self._active_transactions[transaction_id]
        transaction['status'] = 'rolled_back'
        transaction['rolled_back_at'] = datetime.utcnow()
        
        # Perform rollback operations
        await self._execute_rollback_operations(transaction)
        
        # Cleanup transaction
        del self._active_transactions[transaction_id]
        
        self.logger.debug("Transaction rolled back", transaction_id=transaction_id)
    
    async def _execute_rollback_operations(self, transaction: Dict[str, Any]):
        """Execute rollback operations for transaction"""
        # To be implemented by subclasses
        pass
    
    def _add_transaction_operation(
        self,
        transaction_id: str,
        operation: str,
        data: Dict[str, Any]
    ):
        """Add operation to transaction"""
        if transaction_id in self._active_transactions:
            self._active_transactions[transaction_id]['operations'].append({
                'operation': operation,
                'data': data,
                'timestamp': datetime.utcnow()
            })

class CacheableService(BaseService):
    """Base service with caching support"""
    
    def __init__(self, service_name: str):
        super().__init__(service_name)
        self._cache_ttl = 300  # 5 minutes default
        self._cache_prefix = f"service:{service_name}"
    
    async def _setup_dependencies(self):
        """Setup cache dependencies"""
        from src.config.database import get_redis
        self._cache = await get_redis()
    
    async def _get_cached_value(
        self,
        cache_key: str,
        default: Any = None
    ) -> Any:
        """Get value from cache"""
        try:
            full_key = f"{self._cache_prefix}:{cache_key}"
            cached_value = await self._cache.get(full_key)
            
            if cached_value:
                import json
                return json.loads(cached_value)
            
            return default
            
        except Exception as e:
            self.logger.warning("Cache get failed", key=cache_key, error=e)
            return default
    
    async def _set_cached_value(
        self,
        cache_key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        try:
            full_key = f"{self._cache_prefix}:{cache_key}"
            ttl = ttl or self._cache_ttl
            
            import json
            serialized_value = json.dumps(value, default=str)
            
            await self._cache.setex(full_key, ttl, serialized_value)
            return True
            
        except Exception as e:
            self.logger.warning("Cache set failed", key=cache_key, error=e)
            return False
    
    async def _invalidate_cache(self, cache_key: str) -> bool:
        """Invalidate cache key"""
        try:
            full_key = f"{self._cache_prefix}:{cache_key}"
            await self._cache.delete(full_key)
            return True
            
        except Exception as e:
            self.logger.warning("Cache invalidation failed", key=cache_key, error=e)
            return False
    
    async def _invalidate_cache_pattern(self, pattern: str) -> int:
        """Invalidate all cache keys matching pattern"""
        try:
            full_pattern = f"{self._cache_prefix}:{pattern}"
            keys = await self._cache.keys(full_pattern)
            
            if keys:
                await self._cache.delete(*keys)
                return len(keys)
            
            return 0
            
        except Exception as e:
            self.logger.warning("Cache pattern invalidation failed", pattern=pattern, error=e)
            return 0
```

### `/src/services/execution_service.py`
**Purpose**: Main service for orchestrating conversation flow execution
```python
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
import asyncio

from src.services.base.base_service import TransactionalService, CacheableService
from src.core.state_machine.state_engine import StateEngine
from src.models.domain.execution_context import ExecutionContext
from src.models.domain.events import StateEvent, ProcessingResult
from src.models.domain.flow_definition import FlowDefinition
from src.models.domain.enums import ConversationStatus, MessageDirection
from src.exceptions.state_exceptions import StateExecutionError, FlowNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ExecutionService(TransactionalService, CacheableService):
    """Service for orchestrating conversation flow execution"""
    
    def __init__(self):
        super().__init__("execution")
        self._execution_locks: Dict[str, asyncio.Lock] = {}
        self._cache_ttl = 600  # 10 minutes for execution cache
    
    async def _setup_dependencies(self):
        """Setup service dependencies"""
        await super()._setup_dependencies()
        
        # Import services to avoid circular imports
        from src.services.flow_service import FlowService
        from src.services.context_service import ContextService
        from src.services.integration_service import IntegrationService
        from src.services.response_generation_service import ResponseGenerationService
        from src.services.analytics_service import AnalyticsService
        
        # Setup core components
        from src.core.state_machine.state_engine import StateEngine
        from src.core.state_machine.transition_handler import TransitionHandler
        from src.core.state_machine.condition_evaluator import ConditionEvaluator
        from src.core.state_machine.action_executor import ActionExecutor
        from src.core.state_machine.execution_context_manager import ExecutionContextManager
        
        # Initialize core components
        transition_handler = TransitionHandler()
        condition_evaluator = ConditionEvaluator()
        action_executor = ActionExecutor()
        context_manager = ExecutionContextManager()
        
        self.state_engine = StateEngine(
            transition_handler=transition_handler,
            condition_evaluator=condition_evaluator,
            action_executor=action_executor,
            context_manager=context_manager
        )
        
        # Initialize services
        self.flow_service = FlowService()
        self.context_service = ContextService()
        self.integration_service = IntegrationService()
        self.response_service = ResponseGenerationService()
        self.analytics_service = AnalyticsService()
        
        # Initialize all dependencies
        await self.state_engine.initialize()
        await self.flow_service.initialize()
        await self.context_service.initialize()
        await self.integration_service.initialize()
        await self.response_service.initialize()
        await self.analytics_service.initialize()
    
    async def process_message(
        self,
        tenant_id: str,
        conversation_id: str,
        message_content: Dict[str, Any],
        user_id: str,
        channel: str,
        session_id: Optional[str] = None,
        processing_hints: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Main entry point for message processing
        
        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            message_content: Message content and metadata
            user_id: User identifier
            channel: Communication channel
            session_id: Optional session identifier
            processing_hints: Optional processing hints
            
        Returns:
            Processing result with response and state updates
        """
        start_time = datetime.utcnow()
        execution_id = str(uuid.uuid4())
        
        processing_logger = self.logger.bind_context(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            user_id=user_id,
            execution_id=execution_id
        )
        
        try:
            # Validate inputs
            self._validate_tenant_access(tenant_id, "process_message")
            self._validate_message_input(message_content, user_id, channel)
            
            processing_logger.info("Starting message processing", channel=channel)
            
            # Acquire execution lock
            async with self._get_execution_lock(conversation_id):
                
                # Load or create execution context
                context = await self._load_or_create_context(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    channel=channel,
                    session_id=session_id
                )
                
                # Get active flow
                flow_definition = await self._get_active_flow(
                    tenant_id=tenant_id,
                    context=context,
                    processing_hints=processing_hints
                )
                
                # Create state event from message
                event = self._create_state_event(message_content, user_id, channel)
                
                # Get current state
                current_state = await self._get_current_state(context, flow_definition)
                
                # Execute state machine
                execution_result = await self.state_engine.execute_state(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    current_state=current_state,
                    event=event,
                    context=context,
                    flow_definition=flow_definition
                )
                
                # Process execution result
                processing_result = await self._process_execution_result(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    execution_result=execution_result,
                    context=context,
                    flow_definition=flow_definition,
                    original_message=message_content
                )
                
                # Update context with results
                await self._update_execution_context(
                    context=context,
                    execution_result=execution_result,
                    processing_result=processing_result
                )
                
                # Send analytics events
                await self._send_analytics_events(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    execution_result=execution_result,
                    processing_result=processing_result,
                    execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                )
                
                # Record metrics
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._record_operation_metrics(
                    operation="process_message",
                    tenant_id=tenant_id,
                    duration_ms=execution_time,
                    status="success"
                )
                
                processing_logger.info(
                    "Message processing completed",
                    execution_time_ms=int(execution_time),
                    new_state=execution_result.new_state,
                    actions_executed=len(execution_result.actions)
                )
                
                return processing_result
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self._record_operation_metrics(
                operation="process_message",
                tenant_id=tenant_id,
                duration_ms=execution_time,
                status="error"
            )
            
            processing_logger.error(
                "Message processing failed",
                error=e,
                execution_time_ms=int(execution_time)
            )
            
            # Return error response
            return ProcessingResult(
                conversation_id=conversation_id,
                current_state="error",
                response={
                    "text": "I'm sorry, I encountered an error processing your message. Please try again.",
                    "type": "text"
                },
                context_updates={},
                actions_performed=[],
                processing_time_ms=int(execution_time),
                success=False,
                error=str(e)
            )
    
    async def _load_or_create_context(
        self,
        tenant_id: str,
        conversation_id: str,
        user_id: str,
        channel: str,
        session_id: Optional[str] = None
    ) -> ExecutionContext:
        """Load existing context or create new one"""
        
        # Try to load existing context
        context = await self.context_service.get_conversation_context(
            tenant_id=tenant_id,
            conversation_id=conversation_id
        )
        
        if context:
            # Update activity timestamp
            context.last_activity = datetime.utcnow()
            return context
        
        # Create new context
        return await self.context_service.create_conversation_context(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            user_id=user_id,
            channel=channel,
            session_id=session_id or conversation_id
        )
    
    async def _get_active_flow(
        self,
        tenant_id: str,
        context: ExecutionContext,
        processing_hints: Optional[Dict[str, Any]] = None
    ) -> FlowDefinition:
        """Get active flow for conversation"""
        
        # Check for forced flow in processing hints
        if processing_hints and processing_hints.get("force_flow"):
            flow_id = processing_hints["force_flow"]
            flow = await self.flow_service.get_flow_by_id(tenant_id, flow_id)
            if flow:
                return flow
        
        # Check if context has active flow
        if context.flow_id:
            # Try to get cached flow first
            cache_key = f"flow:{tenant_id}:{context.flow_id}"
            cached_flow = await self._get_cached_value(cache_key)
            
            if cached_flow:
                return FlowDefinition.parse_obj(cached_flow)
            
            # Load flow from service
            flow = await self.flow_service.get_flow_by_id(tenant_id, context.flow_id)
            if flow:
                # Cache the flow
                await self._set_cached_value(cache_key, flow.dict())
                return flow
        
        # Get default flow for tenant
        default_flow = await self.flow_service.get_default_flow(tenant_id)
        if not default_flow:
            raise FlowNotFoundError(f"No default flow found for tenant {tenant_id}")
        
        # Cache the flow
        cache_key = f"flow:{tenant_id}:{default_flow.flow_id}"
        await self._set_cached_value(cache_key, default_flow.dict())
        
        return default_flow
    
    async def _get_current_state(
        self,
        context: ExecutionContext,
        flow_definition: FlowDefinition
    ) -> 'State':
        """Get current state from context and flow definition"""
        
        current_state_name = context.current_state or flow_definition.initial_state
        
        if current_state_name not in flow_definition.states:
            self.logger.warning(
                "Current state not found in flow, using initial state",
                current_state=current_state_name,
                flow_id=flow_definition.flow_id
            )
            current_state_name = flow_definition.initial_state
        
        return flow_definition.states[current_state_name]
    
    def _create_state_event(
        self,
        message_content: Dict[str, Any],
        user_id: str,
        channel: str
    ) -> StateEvent:
        """Create state event from message content"""
        
        return StateEvent(
            type="message",
            data={
                "text": message_content.get("text", ""),
                "message_type": message_content.get("type", "text"),
                "payload": message_content.get("payload"),
                "user_id": user_id,
                "channel": channel,
                "metadata": message_content.get("metadata", {})
            },
            source="user_input"
        )
    
    async def _process_execution_result(
        self,
        tenant_id: str,
        conversation_id: str,
        execution_result: 'StateExecutionResult',
        context: ExecutionContext,
        flow_definition: FlowDefinition,
        original_message: Dict[str, Any]
    ) -> ProcessingResult:
        """Process state execution result and generate final response"""
        
        # Extract response from actions
        response_content = None
        actions_performed = []
        
        for action in execution_result.actions:
            actions_performed.append(action.get("type", "unknown"))
            
            if action.get("type") == "send_message":
                message_content = action.get("message_content", {})
                if message_content:
                    response_content = message_content
        
        # If no response from actions, check execution result
        if not response_content and execution_result.response:
            response_content = execution_result.response
        
        # Generate default response if none provided
        if not response_content:
            response_content = await self._generate_default_response(
                context=context,
                state_name=execution_result.new_state or context.current_state,
                flow_definition=flow_definition
            )
        
        # Build processing result
        return ProcessingResult(
            conversation_id=conversation_id,
            current_state=execution_result.new_state or context.current_state,
            response=response_content,
            context_updates=execution_result.context_updates,
            actions_performed=actions_performed,
            processing_time_ms=execution_result.execution_time_ms,
            success=execution_result.success,
            confidence_scores={},  # TODO: Add confidence scores from ML models
            ab_variant=None  # TODO: Add A/B testing support
        )
    
    async def _update_execution_context(
        self,
        context: ExecutionContext,
        execution_result: 'StateExecutionResult',
        processing_result: ProcessingResult
    ):
        """Update execution context with processing results"""
        
        # Update state
        if execution_result.new_state and execution_result.new_state != context.current_state:
            context.previous_states.append(context.current_state)
            context.current_state = execution_result.new_state
        
        # Update context variables
        context_updates = execution_result.context_updates
        if "variables" in context_updates:
            context.variables.update(context_updates["variables"])
        
        if "slots" in context_updates:
            context.slots.update(context_updates["slots"])
        
        # Update timestamps
        context.last_activity = datetime.utcnow()
        context.updated_at = datetime.utcnow()
        
        # Save updated context
        await self.context_service.update_conversation_context(context)
    
    async def _send_analytics_events(
        self,
        tenant_id: str,
        conversation_id: str,
        execution_result: 'StateExecutionResult',
        processing_result: ProcessingResult,
        execution_time_ms: int
    ):
        """Send analytics events"""
        
        try:
            # Message processed event
            await self.analytics_service.track_event(
                tenant_id=tenant_id,
                event_name="message_processed",
                properties={
                    "conversation_id": conversation_id,
                    "current_state": processing_result.current_state,
                    "execution_time_ms": execution_time_ms,
                    "actions_performed": processing_result.actions_performed,
                    "success": processing_result.success
                }
            )
            
            # State transition event
            if execution_result.new_state:
                await self.analytics_service.track_event(
                    tenant_id=tenant_id,
                    event_name="state_transition",
                    properties={
                        "conversation_id": conversation_id,
                        "from_state": execution_result.metadata.get("previous_state"),
                        "to_state": execution_result.new_state,
                        "transition_condition": execution_result.metadata.get("transition_taken")
                    }
                )
                
        except Exception as e:
            self.logger.warning("Failed to send analytics events", error=e)
    
    async def _generate_default_response(
        self,
        context: ExecutionContext,
        state_name: str,
        flow_definition: FlowDefinition
    ) -> Dict[str, Any]:
        """Generate default response when none is provided"""
        
        # This is a fallback response
        return {
            "text": "I understand. How can I help you further?",
            "type": "text"
        }
    
    def _validate_message_input(
        self,
        message_content: Dict[str, Any],
        user_id: str,
        channel: str
    ):
        """Validate message input parameters"""
        
        if not message_content:
            raise ValidationError("Message content is required")
        
        if not user_id:
            raise ValidationError("User ID is required")
        
        if not channel:
            raise ValidationError("Channel is required")
        
        # Validate message content structure
        if "text" not in message_content and "payload" not in message_content:
            raise ValidationError("Message must contain either text or payload")
    
    def _get_execution_lock(self, conversation_id: str) -> asyncio.Lock:
        """Get execution lock for conversation"""
        if conversation_id not in self._execution_locks:
            self._execution_locks[conversation_id] = asyncio.Lock()
        return self._execution_locks[conversation_id]
    
    async def reset_conversation(
        self,
        tenant_id: str,
        conversation_id: str,
        reason: str = "manual_reset"
    ) -> bool:
        """
        Reset conversation to initial state
        
        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            reason: Reason for reset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._validate_tenant_access(tenant_id, "reset_conversation")
            
            # Reset context
            context_reset = await self.context_service.reset_conversation_context(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                reason=reason
            )
            
            if context_reset:
                # Send analytics event
                await self.analytics_service.track_event(
                    tenant_id=tenant_id,
                    event_name="conversation_reset",
                    properties={
                        "conversation_id": conversation_id,
                        "reason": reason
                    }
                )
                
                self.logger.info(
                    "Conversation reset",
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    reason=reason
                )
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(
                "Failed to reset conversation",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                error=e
            )
            return False
    
    async def get_conversation_state(
        self,
        tenant_id: str,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get current conversation state
        
        Args:
            tenant_id: Tenant identifier
            conversation_id: Conversation identifier
            
        Returns:
            Conversation state information
        """
        try:
            self._validate_tenant_access(tenant_id, "get_conversation_state")
            
            context = await self.context_service.get_conversation_context(
                tenant_id=tenant_id,
                conversation_id=conversation_id
            )
            
            if not context:
                return None
            
            return {
                "conversation_id": conversation_id,
                "current_state": context.current_state,
                "flow_id": context.flow_id,
                "user_id": context.user_id,
                "slots": context.slots,
                "variables": context.variables,
                "created_at": context.created_at.isoformat(),
                "last_activity": context.last_activity.isoformat(),
                "message_count": len(context.previous_states) + 1
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to get conversation state",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                error=e
            )
            return None
```

## Step 14: Flow Management Service (Days 44-45)

### `/src/services/flow_service.py`
**Purpose**: Service for managing conversation flows and their lifecycle
```python
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from src.services.base.base_service import CacheableService
from src.repositories.flow_repository import FlowRepository
from src.models.domain.flow_definition import FlowDefinition
from src.models.domain.enums import FlowStatus
from src.exceptions.flow_exceptions import FlowNotFoundError, FlowValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FlowService(CacheableService):
    """Service for managing conversation flows"""
    
    def __init__(self):
        super().__init__("flow")
        self._cache_ttl = 300  # 5 minutes for flow cache
    
    async def _setup_dependencies(self):
        """Setup service dependencies"""
        await super()._setup_dependencies()
        self.flow_repository = FlowRepository()
    
    async def create_flow(
        self,
        tenant_id: str,
        name: str,
        flow_definition: Dict[str, Any],
        description: Optional[str] = None,
        version: str = "1.0",
        created_by: Optional[str] = None
    ) -> FlowDefinition:
        """
        Create a new conversation flow
        
        Args:
            tenant_id: Tenant identifier
            name: Flow name
            flow_definition: Complete flow definition
            description: Optional description
            version: Flow version
            created_by: User creating the flow
            
        Returns:
            Created flow definition
        """
        try:
            self._validate_tenant_access(tenant_id, "create_flow")
            
            # Validate flow definition
            validated_flow = await self._validate_flow_definition(flow_definition)
            
            # Check for name conflicts
            existing_flow = await self.flow_repository.get_flow_by_name_and_version(
                tenant_id=tenant_id,
                name=name,
                version=version
            )
            
            if existing_flow:
                raise FlowValidationError(f"Flow with name '{name}' and version '{version}' already exists")
            
            # Create flow record
            created_by_uuid = uuid.UUID(created_by) if created_by else None
            
            flow_record = await self.flow_repository.create_flow(
                tenant_id=tenant_id,
                name=name,
                flow_definition=validated_flow,
                description=description,
                version=version,
                created_by=created_by_uuid
            )
            
            # Convert to domain model
            domain_flow = flow_record.to_domain_model()
            
            # Cache the flow
            cache_key = f"{tenant_id}:{domain_flow.flow_id}"
            await self._set_cached_value(cache_key, domain_flow.dict())
            
            self.logger.info(
                "Flow created",
                tenant_id=tenant_id,
                flow_id=domain_flow.flow_id,
                name=name,
                version=version
            )
            
            return domain_flow
            
        except Exception as e:
            self.logger.error(
                "Failed to create flow",
                tenant_id=tenant_id,
                name=name,
                error=e
            )
            raise
    
    async def get_flow_by_id(
        self,
        tenant_id: str,
        flow_id: str
    ) -> Optional[FlowDefinition]:
        """
        Get flow by ID
        
        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier
            
        Returns:
            Flow definition if found, None otherwise
        """
        try:
            self._validate_tenant_access(tenant_id, "get_flow")
            
            # Try cache first
            cache_key = f"{tenant_id}:{flow_id}"
            cached_flow = await self._get_cached_value(cache_key)
            
            if cached_flow:
                return FlowDefinition.parse_obj(cached_flow)
            
            # Load from repository
            flow_record = await self.flow_repository.get_by_tenant_and_id(
                tenant_id=tenant_id,
                record_id=flow_id
            )
            
            if not flow_record:
                return None
            
            # Convert to domain model
            domain_flow = flow_record.to_domain_model()
            
            # Cache the flow
            await self._set_cached_value(cache_key, domain_flow.dict())
            
            return domain_flow
            
        except Exception as e:
            self.logger.error(
                "Failed to get flow by ID",
                tenant_id=tenant_id,
                flow_id=flow_id,
                error=e
            )
            raise
    
    async def get_default_flow(
        self,
        tenant_id: str
    ) -> Optional[FlowDefinition]:
        """
        Get default flow for tenant
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Default flow definition if found, None otherwise
        """
        try:
            self._validate_tenant_access(tenant_id, "get_default_flow")
            
            # Try cache first
            cache_key = f"{tenant_id}:default"
            cached_flow = await self._get_cached_value(cache_key)
            
            if cached_flow:
                return FlowDefinition.parse_obj(cached_flow)
            
            # Load from repository
            flow_record = await self.flow_repository.get_default_flow(tenant_id=tenant_id)
            
            if not flow_record:
                return None
            
            # Convert to domain model
            domain_flow = flow_record.to_domain_model()
            
            # Cache the flow
            await self._set_cached_value(cache_key, domain_flow.dict())
            
            return domain_flow
            
        except Exception as e:
            self.logger.error(
                "Failed to get default flow",
                tenant_id=tenant_id,
                error=e
            )
            raise
    
    async def update_flow(
        self,
        tenant_id: str,
        flow_id: str,
        updates: Dict[str, Any],
        updated_by: Optional[str] = None
    ) -> Optional[FlowDefinition]:
        """
        Update flow
        
        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier
            updates: Update data
            updated_by: User updating the flow
            
        Returns:
            Updated flow definition if successful, None otherwise
        """
        try:
            self._validate_tenant_access(tenant_id, "update_flow")
            
            # Validate flow definition if provided
            if "flow_definition" in updates:
                updates["flow_definition"] = await self._validate_flow_definition(
                    updates["flow_definition"]
                )
            
            # Add audit information
            if updated_by:
                updates["updated_by"] = uuid.UUID(updated_by)
            
            # Update in repository
            flow_record = await self.flow_repository.update(
                record_id=flow_id,
                data=updates
            )
            
            if not flow_record:
                return None
            
            # Convert to domain model
            domain_flow = flow_record.to_domain_model()
            
            # Invalidate cache
            cache_key = f"{tenant_id}:{flow_id}"
            await self._invalidate_cache(cache_key)
            
            # If this was the default flow, invalidate default cache too
            if flow_record.is_default:
                await self._invalidate_cache(f"{tenant_id}:default")
            
            self.logger.info(
                "Flow updated",
                tenant_id=tenant_id,
                flow_id=flow_id,
                updates=list(updates.keys())
            )
            
            return domain_flow
            
        except Exception as e:
            self.logger.error(
                "Failed to update flow",
                tenant_id=tenant_id,
                flow_id=flow_id,
                error=e
            )
            raise
    
    async def publish_flow(
        self,
        tenant_id: str,
        flow_id: str,
        published_by: Optional[str] = None
    ) -> bool:
        """
        Publish a flow (change status to active)
        
        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier
            published_by: User publishing the flow
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._validate_tenant_access(tenant_id, "publish_flow")
            
            published_by_uuid = uuid.UUID(published_by) if published_by else None
            
            success = await self.flow_repository.publish_flow(
                tenant_id=tenant_id,
                flow_id=flow_id,
                published_by=published_by_uuid
            )
            
            if success:
                # Invalidate caches
                cache_key = f"{tenant_id}:{flow_id}"
                await self._invalidate_cache(cache_key)
                
                self.logger.info(
                    "Flow published",
                    tenant_id=tenant_id,
                    flow_id=flow_id
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Failed to publish flow",
                tenant_id=tenant_id,
                flow_id=flow_id,
                error=e
            )
            return False
    
    async def set_default_flow(
        self,
        tenant_id: str,
        flow_id: str
    ) -> bool:
        """
        Set flow as default for tenant
        
        Args:
            tenant_id: Tenant identifier
            flow_id: Flow identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._validate_tenant_access(tenant_id, "set_default_flow")
            
            success = await self.flow_repository.set_default_flow(
                tenant_id=tenant_id,
                flow_id=flow_id
            )
            
            if success:
                # Invalidate default flow cache
                await self._invalidate_cache(f"{tenant_id}:default")
                
                self.logger.info(
                    "Default flow set",
                    tenant_id=tenant_id,
                    flow_id=flow_id
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Failed to set default flow",
                tenant_id=tenant_id,
                flow_id=flow_id,
                error=e
            )
            return False
    
    async def list_flows(
        self,
        tenant_id: str,
        status_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[FlowDefinition]:
        """
        List flows for tenant
        
        Args:
            tenant_id: Tenant identifier
            status_filter: Optional status filter
            limit: Maximum number of flows
            offset: Offset for pagination
            
        Returns:
            List of flow definitions
        """
        try:
            self._validate_tenant_access(tenant_id, "list_flows")
            
            criteria = {}
            if status_filter:
                criteria["status"] = {"operator": "in", "value": status_filter}
            
            flow_records = await self.flow_repository.get_by_tenant(
                tenant_id=tenant_id,
                criteria=criteria,
                limit=limit,
                offset=offset,
                order_by="-updated_at"
            )
            
            # Convert to domain models
            flows = [record.to_domain_model() for record in flow_records]
            
            self.logger.debug(
                "Flows listed",
                tenant_id=tenant_id,
                count=len(flows),
                status_filter=status_filter
            )
            
            return flows
            
        except Exception as e:
            self.logger.error(
                "Failed to list flows",
                tenant_id=tenant_id,
                error=e
            )
            raise
    
    async def search_flows(
        self,
        tenant_id: str,
        search_term: str,
        status_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[FlowDefinition]:
        """
        Search flows by name, description, or tags
        
        Args:
            tenant_id: Tenant identifier
            search_term: Search term
            status_filter: Optional status filter
            limit: Maximum results
            offset: Results offset
            
        Returns:
            List of matching flow definitions
        """
        try:
            self._validate_tenant_access(tenant_id, "search_flows")
            
            flow_records = await self.flow_repository.search_flows(
                tenant_id=tenant_id,
                search_term=search_term,
                status_filter=status_filter,
                limit=limit,
                offset=offset
            )
            
            # Convert to domain models
            flows = [record.to_domain_model() for record in flow_records]
            
            self.logger.debug(
                "Flows searched",
                tenant_id=tenant_id,
                search_term=search_term,
                results_count=len(flows)
            )
            
            return flows
            
        except Exception as e:
            self.logger.error(
                "Failed to search flows",
                tenant_id=tenant_id,
                search_term=search_term,
                error=e
            )
            raise
    
    async def increment_flow_usage(
        self,
        flow_id: str
    ) -> bool:
        """
        Increment flow usage counter
        
        Args:
            flow_id: Flow identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.flow_repository.increment_flow_usage(flow_id)
        except Exception as e:
            self.logger.error(
                "Failed to increment flow usage",
                flow_id=flow_id,
                error=e
            )
            return False
    
    async def _validate_flow_definition(
        self,
        flow_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate flow definition structure and content
        
        Args:
            flow_definition: Flow definition to validate
            
        Returns:
            Validated flow definition
        """
        try:
            # Parse as domain model to validate structure
            flow_model = FlowDefinition.parse_obj({
                'flow_id': 'temp_id',
                'tenant_id': 'temp_tenant',
                'name': 'temp_name',
                'version': '1.0',
                **flow_definition
            })
            
            # Validate states exist
            if not flow_model.states:
                raise FlowValidationError("Flow must contain at least one state")
            
            # Validate initial state exists
            if flow_model.initial_state not in flow_model.states:
                raise FlowValidationError(f"Initial state '{flow_model.initial_state}' not found in states")
            
            # Validate state transitions reference valid states
            for state_name, state in flow_model.states.items():
                for transition in state.transitions:
                    if transition.target_state not in flow_model.states:
                        raise FlowValidationError(
                            f"Transition in state '{state_name}' references unknown state '{transition.target_state}'"
                        )
            
            # Return the original definition (without temp fields)
            return flow_definition
            
        except Exception as e:
            if isinstance(e, FlowValidationError):
                raise
            
            self.logger.error("Flow definition validation failed", error=e)
            raise FlowValidationError(f"Invalid flow definition: {str(e)}")
```

## Success Criteria
- [x] Complete service layer infrastructure with base classes
- [x] Execution service for orchestrating conversation processing
- [x] Flow service for managing conversation flows
- [x] Caching strategies for performance optimization
- [x] Transaction support for data consistency
- [x] Comprehensive error handling and validation
- [x] Service dependency injection and lifecycle management
- [x] Analytics integration for tracking and monitoring

## Key Error Handling & Performance Considerations
1. **Service Lifecycle**: Proper initialization and shutdown handling
2. **Dependency Injection**: Flexible service dependency management
3. **Caching Strategy**: Multi-layer caching with TTL and invalidation
4. **Transaction Support**: ACID transactions with rollback capabilities
5. **Error Recovery**: Comprehensive error handling with fallback responses
6. **Performance Metrics**: Detailed performance tracking and monitoring
7. **Resource Management**: Proper resource cleanup and connection pooling

## Technologies Used
- **Service Architecture**: Dependency injection with base service classes
- **Caching**: Redis with TTL and pattern-based invalidation
- **Transaction Management**: Database transactions with proper rollback
- **Async Processing**: asyncio for concurrent operations
- **Validation**: Pydantic models for data validation
- **Monitoring**: Comprehensive logging and metrics collection

## Cross-Service Integration
- **State Machine**: Core execution engine integration
- **Repositories**: Data access layer integration
- **External Services**: Model Orchestrator and Adaptor Service (placeholders)
- **Analytics**: Event tracking and metrics collection
- **Cache**: Distributed caching with Redis
- **Database**: Multi-database transaction support

## Next Phase Dependencies
Phase 6 will build upon:
- Service layer infrastructure and patterns
- Execution service for conversation processing
- Flow management capabilities
- Caching and performance optimizations
- Error handling and validation frameworks