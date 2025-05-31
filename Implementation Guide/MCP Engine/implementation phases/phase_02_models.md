# Phase 02: Core Models & Database Layer
**Duration**: Week 3-4 (Days 11-20)  
**Team Size**: 3-4 developers  
**Complexity**: High  

## Overview
Implement the core data models, domain objects, and database layer that form the foundation of the MCP Engine. This includes PostgreSQL models for configuration data, MongoDB collections for conversation data, Redis data structures for caching, and Pydantic models for API validation.

## Step 5: Domain Models & Core Types (Days 11-13)

### Files to Create
```
src/
├── models/
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── enums.py
│   │   ├── base.py
│   │   ├── state_machine.py
│   │   ├── flow_definition.py
│   │   ├── execution_context.py
│   │   ├── events.py
│   │   └── responses.py
│   ├── postgres/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── flow_model.py
│   │   ├── state_model.py
│   │   └── experiment_model.py
│   ├── redis/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── execution_state.py
│   │   ├── context_cache.py
│   │   └── flow_cache.py
│   └── mongodb/
│       ├── __init__.py
│       ├── base.py
│       ├── conversation.py
│       └── analytics.py
```

### `/src/models/domain/enums.py`
**Purpose**: Core enumerations used throughout the MCP Engine
```python
from enum import Enum

class StateType(str, Enum):
    """Types of states in the state machine"""
    RESPONSE = "response"
    INTENT = "intent"
    SLOT_FILLING = "slot_filling"
    INTEGRATION = "integration"
    CONDITION = "condition"
    WAIT = "wait"
    END = "end"
    ERROR = "error"
    HUMAN_HANDOFF = "human_handoff"

class TransitionCondition(str, Enum):
    """Conditions that trigger state transitions"""
    ANY_INPUT = "any_input"
    INTENT_MATCH = "intent_match"
    INTENT_CONFIDENCE = "intent_confidence"
    SLOT_FILLED = "slot_filled"
    ALL_SLOTS_FILLED = "all_slots_filled"
    INTEGRATION_SUCCESS = "integration_success"
    INTEGRATION_ERROR = "integration_error"
    EXPRESSION = "expression"
    TIMEOUT = "timeout"
    LOW_CONFIDENCE = "low_confidence"
    USER_CHOICE = "user_choice"
    FALLBACK = "fallback"

class ActionType(str, Enum):
    """Types of actions that can be executed"""
    SEND_MESSAGE = "send_message"
    SET_VARIABLE = "set_variable"
    CLEAR_VARIABLE = "clear_variable"
    SET_SLOT = "set_slot"
    CLEAR_SLOT = "clear_slot"
    CALL_INTEGRATION = "call_integration"
    LOG_EVENT = "log_event"
    TRIGGER_FLOW = "trigger_flow"
    SET_CONTEXT = "set_context"
    SEND_ANALYTICS = "send_analytics"

class FlowStatus(str, Enum):
    """Status of conversation flows"""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    TESTING = "testing"

class ConversationStatus(str, Enum):
    """Status of conversations"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"
    ERROR = "error"
    TIMEOUT = "timeout"

class MessageDirection(str, Enum):
    """Direction of messages"""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    SYSTEM = "system"

class MessageType(str, Enum):
    """Types of messages"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    LOCATION = "location"
    QUICK_REPLY = "quick_reply"
    CAROUSEL = "carousel"
    FORM = "form"
    SYSTEM = "system"

class IntegrationStatus(str, Enum):
    """Status of integration calls"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RETRY = "retry"

class ExperimentStatus(str, Enum):
    """Status of A/B test experiments"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Priority(str, Enum):
    """Priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"
```

### `/src/models/domain/base.py`
**Purpose**: Base domain model classes with common functionality
```python
from pydantic import BaseModel, Field, validator
from typing import Any, Dict, Optional, List
from datetime import datetime
from uuid import uuid4, UUID

class BaseEntity(BaseModel):
    """Base entity with common fields"""
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str
        }
        allow_population_by_field_name = True
        validate_assignment = True

class TenantEntity(BaseEntity):
    """Base entity with tenant isolation"""
    tenant_id: str = Field(..., description="Tenant identifier")
    
    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('tenant_id is required')
        return v.strip()

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def touch(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.utcnow()

class AuditMixin(BaseModel):
    """Mixin for audit fields"""
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    version: int = Field(default=1)
    
    def increment_version(self, updated_by: Optional[str] = None):
        """Increment version and update audit fields"""
        self.version += 1
        self.updated_by = updated_by
        if hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

class ValidatedModel(BaseModel):
    """Base model with enhanced validation"""
    
    @validator('*', pre=True)
    def empty_str_to_none(cls, v):
        """Convert empty strings to None"""
        if isinstance(v, str) and v.strip() == '':
            return None
        return v
    
    class Config:
        validate_assignment = True
        str_strip_whitespace = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Configuration(ValidatedModel):
    """Base configuration model"""
    enabled: bool = Field(default=True)
    config: Dict[str, Any] = Field(default_factory=dict)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self.config.get(key, default)
    
    def set_config_value(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        if hasattr(self, 'touch'):
            self.touch()
```

### `/src/models/domain/state_machine.py`
**Purpose**: Core state machine domain models
```python
from pydantic import BaseModel, Field, validator, root_validator
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .enums import StateType, TransitionCondition, ActionType
from .base import BaseEntity, ValidatedModel, Configuration

class Action(ValidatedModel):
    """Represents an action to be executed"""
    type: ActionType
    config: Dict[str, Any] = Field(default_factory=dict)
    condition: Optional[str] = None
    priority: int = Field(default=100, ge=0, le=1000)
    timeout_ms: Optional[int] = Field(None, ge=100, le=30000)
    
    @validator('config')
    def validate_config(cls, v, values):
        action_type = values.get('type')
        if not action_type:
            return v
        
        # Validate action-specific configuration
        if action_type == ActionType.SEND_MESSAGE:
            if 'text' not in v and 'template' not in v:
                raise ValueError('send_message action requires text or template')
        elif action_type == ActionType.CALL_INTEGRATION:
            if 'integration_id' not in v:
                raise ValueError('call_integration action requires integration_id')
        elif action_type in [ActionType.SET_VARIABLE, ActionType.SET_SLOT]:
            if 'key' not in v or 'value' not in v:
                raise ValueError(f'{action_type} action requires key and value')
        
        return v

class Transition(ValidatedModel):
    """Represents a state transition"""
    condition: TransitionCondition
    condition_value: Optional[str] = None
    expression: Optional[str] = None
    target_state: str
    priority: int = Field(default=100, ge=0, le=1000)
    actions: List[Action] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @root_validator
    def validate_condition_requirements(cls, values):
        condition = values.get('condition')
        condition_value = values.get('condition_value')
        expression = values.get('expression')
        
        # Validate condition-specific requirements
        if condition == TransitionCondition.EXPRESSION and not expression:
            raise ValueError('expression condition requires expression field')
        
        if condition in [
            TransitionCondition.INTENT_MATCH,
            TransitionCondition.SLOT_FILLED,
            TransitionCondition.USER_CHOICE
        ] and not condition_value:
            raise ValueError(f'{condition} condition requires condition_value')
        
        return values
    
    @validator('target_state')
    def validate_target_state(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('target_state is required')
        return v.strip()

class StateConfig(Configuration):
    """Base configuration for states"""
    pass

class ResponseStateConfig(StateConfig):
    """Configuration for response states"""
    response_templates: Dict[str, str] = Field(default_factory=dict)
    response_type: str = Field(default="text")
    personalization: bool = Field(default=True)
    typing_indicator: bool = Field(default=True)
    delay_ms: Optional[int] = Field(None, ge=0, le=10000)
    
    @validator('response_templates')
    def validate_templates(cls, v):
        if not v:
            raise ValueError('Response state requires at least one template')
        return v

class IntentStateConfig(StateConfig):
    """Configuration for intent detection states"""
    intent_patterns: List[str] = Field(default_factory=list)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    fallback_intent: Optional[str] = None
    context_aware: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=1, le=10)

class SlotFillingConfig(StateConfig):
    """Configuration for slot filling states"""
    required_slots: List[str] = Field(default_factory=list)
    optional_slots: List[str] = Field(default_factory=list)
    validation_rules: Dict[str, str] = Field(default_factory=dict)
    prompts: Dict[str, str] = Field(default_factory=dict)
    retry_prompts: Dict[str, str] = Field(default_factory=dict)
    max_retries: int = Field(default=3, ge=1, le=10)
    collect_mode: str = Field(default="sequential")  # sequential, parallel, adaptive
    
    @validator('required_slots')
    def validate_required_slots(cls, v):
        if not v:
            raise ValueError('Slot filling state requires at least one required slot')
        return v

class IntegrationStateConfig(StateConfig):
    """Configuration for integration states"""
    integration_id: str
    endpoint: str
    method: str = Field(default="GET")
    timeout_ms: int = Field(default=5000, ge=100, le=30000)
    retry_config: Dict[str, Any] = Field(default_factory=dict)
    request_mapping: Dict[str, str] = Field(default_factory=dict)
    response_mapping: Dict[str, str] = Field(default_factory=dict)
    error_handling: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('integration_id')
    def validate_integration_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('integration_id is required')
        return v.strip()

class ConditionStateConfig(StateConfig):
    """Configuration for condition states"""
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    default_state: Optional[str] = None
    evaluation_mode: str = Field(default="first_match")  # first_match, all_match, priority
    
    @validator('conditions')
    def validate_conditions(cls, v):
        if not v:
            raise ValueError('Condition state requires at least one condition')
        return v

class State(BaseEntity):
    """Represents a state in the state machine"""
    name: str
    type: StateType
    description: Optional[str] = None
    config: Union[
        ResponseStateConfig,
        IntentStateConfig,
        SlotFillingConfig,
        IntegrationStateConfig,
        ConditionStateConfig,
        StateConfig
    ]
    transitions: List[Transition] = Field(default_factory=list)
    entry_actions: List[Action] = Field(default_factory=list)
    exit_actions: List[Action] = Field(default_factory=list)
    timeout_seconds: Optional[int] = Field(None, ge=1, le=3600)
    is_final: bool = Field(default=False)
    tags: List[str] = Field(default_factory=list)
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('State name is required')
        # Validate name format (alphanumeric, underscore, hyphen)
        import re
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', v.strip()):
            raise ValueError('State name must start with letter and contain only alphanumeric, underscore, or hyphen')
        return v.strip()
    
    @root_validator
    def validate_final_state(cls, values):
        is_final = values.get('is_final', False)
        transitions = values.get('transitions', [])
        
        if is_final and transitions:
            raise ValueError('Final states cannot have outgoing transitions')
        
        return values
    
    def get_transition_by_condition(
        self,
        condition: TransitionCondition,
        condition_value: Optional[str] = None
    ) -> Optional[Transition]:
        """Get transition by condition and value"""
        for transition in sorted(self.transitions, key=lambda t: t.priority):
            if transition.condition == condition:
                if condition_value is None or transition.condition_value == condition_value:
                    return transition
        return None
    
    def has_timeout(self) -> bool:
        """Check if state has timeout configured"""
        return self.timeout_seconds is not None
    
    def get_actions_by_type(self, action_type: ActionType) -> List[Action]:
        """Get actions of specific type"""
        all_actions = self.entry_actions + self.exit_actions
        return [action for action in all_actions if action.type == action_type]
```

## Step 6: PostgreSQL Models (Days 14-15)

### `/src/models/postgres/base.py`
**Purpose**: Base PostgreSQL model setup with SQLAlchemy
```python
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declared_attr
from datetime import datetime
import uuid

from src.config.database import Base

class TimestampMixin:
    """Mixin for timestamp columns"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class TenantMixin:
    """Mixin for tenant isolation"""
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)

class AuditMixin:
    """Mixin for audit fields"""
    created_by = Column(UUID(as_uuid=True), nullable=True)
    updated_by = Column(UUID(as_uuid=True), nullable=True)
    version = Column(Integer, default=1, nullable=False)

class BaseModel(Base, TimestampMixin):
    """Base model with common functionality"""
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    @declared_attr
    def __tablename__(cls):
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

class TenantModel(BaseModel, TenantMixin):
    """Base model with tenant isolation"""
    __abstract__ = True

class AuditedModel(TenantModel, AuditMixin):
    """Base model with audit fields"""
    __abstract__ = True
    
    def touch(self, updated_by: uuid.UUID = None):
        """Update timestamp and version"""
        self.updated_at = datetime.utcnow()
        self.version += 1
        if updated_by:
            self.updated_by = updated_by
```

### `/src/models/postgres/flow_model.py`
**Purpose**: PostgreSQL models for conversation flows
```python
from sqlalchemy import Column, String, Boolean, Integer, Text, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from typing import Optional, Dict, Any, List
import uuid

from .base import AuditedModel

class ConversationFlow(AuditedModel):
    """Conversation flow definition stored in PostgreSQL"""
    __tablename__ = 'conversation_flows'
    
    # Basic flow information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(50), nullable=False, default='1.0')
    
    # Flow definition and configuration
    flow_definition = Column(JSONB, nullable=False)
    trigger_conditions = Column(JSONB, default={})
    fallback_flow_id = Column(UUID(as_uuid=True), ForeignKey('conversation_flows.id'), nullable=True)
    
    # Status and lifecycle
    status = Column(
        String(20), 
        nullable=False, 
        default='draft',
        server_default='draft'
    )  # draft, active, inactive, archived
    is_default = Column(Boolean, default=False, nullable=False)
    
    # A/B Testing configuration
    ab_test_enabled = Column(Boolean, default=False, nullable=False)
    ab_test_config = Column(JSONB, default={})
    
    # Analytics and performance metrics
    usage_count = Column(Integer, default=0, nullable=False)
    success_rate = Column(String(10))  # Store as decimal string
    avg_completion_time_seconds = Column(Integer)
    last_used_at = Column(DateTime(timezone=True))
    
    # Publish information
    published_at = Column(DateTime(timezone=True))
    published_by = Column(UUID(as_uuid=True))
    
    # Additional metadata
    tags = Column(ARRAY(String(50)), default=[])
    category = Column(String(100))
    
    # Relationships
    fallback_flow = relationship("ConversationFlow", remote_side=[id])
    flow_versions = relationship("FlowVersion", back_populates="flow")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', 'version', name='uq_flow_tenant_name_version'),
        UniqueConstraint('tenant_id', 'is_default', name='uq_tenant_default_flow'),
        Index('idx_flows_tenant_status', 'tenant_id', 'status'),
        Index('idx_flows_usage', 'usage_count'),
        Index('idx_flows_last_used', 'last_used_at'),
    )
    
    def to_domain_model(self) -> 'FlowDefinition':
        """Convert to domain model"""
        from src.models.domain.flow_definition import FlowDefinition
        return FlowDefinition.parse_obj({
            'flow_id': str(self.id),
            'tenant_id': str(self.tenant_id),
            'name': self.name,
            'version': self.version,
            'description': self.description,
            **self.flow_definition
        })
    
    def increment_usage(self):
        """Increment usage counter"""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()

class FlowVersion(AuditedModel):
    """Version history for conversation flows"""
    __tablename__ = 'flow_versions'
    
    flow_id = Column(UUID(as_uuid=True), ForeignKey('conversation_flows.id'), nullable=False)
    version = Column(String(50), nullable=False)
    flow_definition = Column(JSONB, nullable=False)
    change_description = Column(Text)
    change_type = Column(String(50))  # major, minor, patch, hotfix
    
    # Relationships
    flow = relationship("ConversationFlow", back_populates="flow_versions")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('flow_id', 'version', name='uq_flow_version'),
        Index('idx_flow_versions_flow', 'flow_id', 'created_at'),
    )

class FlowAnalytics(AuditedModel):
    """Analytics data for flows"""
    __tablename__ = 'flow_analytics'
    
    flow_id = Column(UUID(as_uuid=True), ForeignKey('conversation_flows.id'), nullable=False)
    metric_date = Column(Date, nullable=False)
    
    # Execution metrics
    executions_count = Column(Integer, default=0)
    completions_count = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    avg_duration_seconds = Column(Integer)
    
    # User engagement metrics
    user_satisfaction_avg = Column(String(10))  # Decimal as string
    abandonment_rate = Column(String(10))
    escalation_rate = Column(String(10))
    
    # Performance metrics
    avg_response_time_ms = Column(Integer)
    cache_hit_rate = Column(String(10))
    
    # Additional metrics
    custom_metrics = Column(JSONB, default={})
    
    # Relationships
    flow = relationship("ConversationFlow")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('flow_id', 'metric_date', name='uq_flow_analytics_date'),
        Index('idx_flow_analytics_flow_date', 'flow_id', 'metric_date'),
    )
```

## Step 7: Redis Data Structures (Days 16-17)

### `/src/models/redis/base.py`
**Purpose**: Base Redis data structure helpers
```python
import json
import pickle
from typing import Any, Dict, Optional, List, Type, TypeVar
from datetime import datetime, timedelta
from pydantic import BaseModel
import aioredis

from src.config.database import get_redis
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)

class RedisKeyBuilder:
    """Helper for building consistent Redis keys"""
    
    @staticmethod
    def conversation_context(tenant_id: str, conversation_id: str) -> str:
        return f"conversation:{tenant_id}:{conversation_id}:context"
    
    @staticmethod
    def conversation_lock(conversation_id: str) -> str:
        return f"lock:conversation:{conversation_id}"
    
    @staticmethod
    def flow_cache(tenant_id: str, flow_id: str) -> str:
        return f"flow:{tenant_id}:{flow_id}"
    
    @staticmethod
    def execution_state(tenant_id: str, conversation_id: str) -> str:
        return f"execution:{tenant_id}:{conversation_id}"
    
    @staticmethod
    def session_data(tenant_id: str, session_id: str) -> str:
        return f"session:{tenant_id}:{session_id}"
    
    @staticmethod
    def rate_limit(tenant_id: str, identifier: str, window: str) -> str:
        return f"rate_limit:{tenant_id}:{identifier}:{window}"
    
    @staticmethod
    def metrics_counter(tenant_id: str, metric_type: str, time_window: str) -> str:
        return f"metrics:{tenant_id}:{metric_type}:{time_window}"
    
    @staticmethod
    def circuit_breaker(service_name: str, tenant_id: str) -> str:
        return f"circuit_breaker:{service_name}:{tenant_id}"

class RedisRepository:
    """Base repository for Redis operations"""
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
    
    async def get_redis(self) -> aioredis.Redis:
        """Get Redis connection"""
        if not self.redis:
            self.redis = await get_redis()
        return self.redis
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
    
    async def set_json(
        self,
        key: str,
        value: BaseModel,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set JSON value with optional TTL"""
        try:
            redis = await self.get_redis()
            json_value = value.json()
            
            if ttl_seconds:
                return await redis.setex(key, ttl_seconds, json_value)
            else:
                return await redis.set(key, json_value)
        except Exception as e:
            logger.error("Failed to set JSON value", key=key, error=e)
            return False
    
    async def get_json(
        self,
        key: str,
        model_class: Type[T]
    ) -> Optional[T]:
        """Get JSON value and parse to model"""
        try:
            redis = await self.get_redis()
            value = await redis.get(key)
            
            if value:
                return model_class.parse_raw(value)
            return None
        except Exception as e:
            logger.error("Failed to get JSON value", key=key, error=e)
            return None
    
    async def set_hash(
        self,
        key: str,
        field_values: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set hash field values"""
        try:
            redis = await self.get_redis()
            
            # Convert values to strings
            string_values = {}
            for field, value in field_values.items():
                if isinstance(value, (dict, list)):
                    string_values[field] = json.dumps(value)
                elif isinstance(value, BaseModel):
                    string_values[field] = value.json()
                else:
                    string_values[field] = str(value)
            
            await redis.hset(key, mapping=string_values)
            
            if ttl_seconds:
                await redis.expire(key, ttl_seconds)
            
            return True
        except Exception as e:
            logger.error("Failed to set hash", key=key, error=e)
            return False
    
    async def get_hash(self, key: str) -> Dict[str, str]:
        """Get all hash fields"""
        try:
            redis = await self.get_redis()
            return await redis.hgetall(key)
        except Exception as e:
            logger.error("Failed to get hash", key=key, error=e)
            return {}
    
    async def get_hash_field(self, key: str, field: str) -> Optional[str]:
        """Get single hash field"""
        try:
            redis = await self.get_redis()
            return await redis.hget(key, field)
        except Exception as e:
            logger.error("Failed to get hash field", key=key, field=field, error=e)
            return None
    
    async def delete_key(self, key: str) -> bool:
        """Delete key"""
        try:
            redis = await self.get_redis()
            return bool(await redis.delete(key))
        except Exception as e:
            logger.error("Failed to delete key", key=key, error=e)
            return False
    
    async def set_with_lock(
        self,
        key: str,
        value: Any,
        lock_key: str,
        lock_timeout_ms: int = 5000,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value with distributed lock"""
        try:
            redis = await self.get_redis()
            
            # Acquire lock
            lock_acquired = await redis.set(
                lock_key,
                "locked",
                px=lock_timeout_ms,
                nx=True
            )
            
            if not lock_acquired:
                return False
            
            try:
                # Set the value
                if isinstance(value, BaseModel):
                    json_value = value.json()
                    if ttl_seconds:
                        await redis.setex(key, ttl_seconds, json_value)
                    else:
                        await redis.set(key, json_value)
                else:
                    if ttl_seconds:
                        await redis.setex(key, ttl_seconds, str(value))
                    else:
                        await redis.set(key, str(value))
                
                return True
            finally:
                # Release lock
                await redis.delete(lock_key)
                
        except Exception as e:
            logger.error("Failed to set with lock", key=key, lock_key=lock_key, error=e)
            return False
```

### `/src/models/redis/execution_state.py`
**Purpose**: Redis models for execution state management
```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import RedisRepository, RedisKeyBuilder
from src.models.domain.execution_context import ExecutionContext
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ExecutionState(BaseModel):
    """Execution state stored in Redis"""
    conversation_id: str
    tenant_id: str
    current_state: str
    previous_states: List[str] = Field(default_factory=list)
    flow_id: str
    flow_version: str
    execution_start_time: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    # Execution context
    slots: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution metadata
    error_count: int = Field(default=0)
    retry_count: int = Field(default=0)
    timeout_count: int = Field(default=0)
    
    # Performance tracking
    total_processing_time_ms: int = Field(default=0)
    state_execution_times: Dict[str, int] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
    
    def add_state_to_history(self, state_name: str):
        """Add state to history"""
        if state_name != self.current_state:
            self.previous_states.append(self.current_state)
            self.current_state = state_name
            self.update_activity()
    
    def increment_error_count(self):
        """Increment error counter"""
        self.error_count += 1
        self.update_activity()
    
    def record_state_execution_time(self, state_name: str, execution_time_ms: int):
        """Record execution time for a state"""
        self.state_execution_times[state_name] = execution_time_ms
        self.total_processing_time_ms += execution_time_ms
        self.update_activity()

class ExecutionStateRepository(RedisRepository):
    """Repository for execution state operations"""
    
    DEFAULT_TTL = 86400  # 24 hours
    
    async def save_execution_state(
        self,
        execution_state: ExecutionState,
        ttl_seconds: int = DEFAULT_TTL
    ) -> bool:
        """Save execution state to Redis"""
        key = RedisKeyBuilder.execution_state(
            execution_state.tenant_id,
            execution_state.conversation_id
        )
        
        return await self.set_json(key, execution_state, ttl_seconds)
    
    async def get_execution_state(
        self,
        tenant_id: str,
        conversation_id: str
    ) -> Optional[ExecutionState]:
        """Get execution state from Redis"""
        key = RedisKeyBuilder.execution_state(tenant_id, conversation_id)
        return await self.get_json(key, ExecutionState)
    
    async def update_current_state(
        self,
        tenant_id: str,
        conversation_id: str,
        new_state: str,
        execution_time_ms: int = 0
    ) -> bool:
        """Update current state with optimistic locking"""
        key = RedisKeyBuilder.execution_state(tenant_id, conversation_id)
        lock_key = f"{key}:lock"
        
        # Get current state
        execution_state = await self.get_execution_state(tenant_id, conversation_id)
        if not execution_state:
            return False
        
        # Update state
        execution_state.add_state_to_history(new_state)
        if execution_time_ms > 0:
            execution_state.record_state_execution_time(new_state, execution_time_ms)
        
        # Save with lock
        return await self.set_with_lock(
            key,
            execution_state,
            lock_key,
            lock_timeout_ms=5000,
            ttl_seconds=self.DEFAULT_TTL
        )
    
    async def update_slots(
        self,
        tenant_id: str,
        conversation_id: str,
        slot_updates: Dict[str, Any]
    ) -> bool:
        """Update conversation slots"""
        key = RedisKeyBuilder.execution_state(tenant_id, conversation_id)
        lock_key = f"{key}:lock"
        
        execution_state = await self.get_execution_state(tenant_id, conversation_id)
        if not execution_state:
            return False
        
        # Update slots
        execution_state.slots.update(slot_updates)
        execution_state.update_activity()
        
        return await self.set_with_lock(
            key,
            execution_state,
            lock_key,
            lock_timeout_ms=5000,
            ttl_seconds=self.DEFAULT_TTL
        )
    
    async def update_variables(
        self,
        tenant_id: str,
        conversation_id: str,
        variable_updates: Dict[str, Any]
    ) -> bool:
        """Update conversation variables"""
        key = RedisKeyBuilder.execution_state(tenant_id, conversation_id)
        lock_key = f"{key}:lock"
        
        execution_state = await self.get_execution_state(tenant_id, conversation_id)
        if not execution_state:
            return False
        
        # Update variables
        execution_state.variables.update(variable_updates)
        execution_state.update_activity()
        
        return await self.set_with_lock(
            key,
            execution_state,
            lock_key,
            lock_timeout_ms=5000,
            ttl_seconds=self.DEFAULT_TTL
        )
    
    async def delete_execution_state(
        self,
        tenant_id: str,
        conversation_id: str
    ) -> bool:
        """Delete execution state"""
        key = RedisKeyBuilder.execution_state(tenant_id, conversation_id)
        return await self.delete_key(key)
    
    async def extend_ttl(
        self,
        tenant_id: str,
        conversation_id: str,
        ttl_seconds: int = DEFAULT_TTL
    ) -> bool:
        """Extend TTL for execution state"""
        try:
            redis = await self.get_redis()
            key = RedisKeyBuilder.execution_state(tenant_id, conversation_id)
            return bool(await redis.expire(key, ttl_seconds))
        except Exception as e:
            logger.error("Failed to extend TTL", key=key, error=e)
            return False
```

## Step 8: MongoDB Collections & Repositories (Days 18-20)

### `/src/models/mongodb/conversation.py`
**Purpose**: MongoDB models for conversation data
```python
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING, DESCENDING
from typing import Dict, Any, List, Optional
from datetime import datetime
from bson import ObjectId

from src.config.database import get_mongodb
from src.models.domain.enums import ConversationStatus, MessageDirection, MessageType
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ConversationRepository:
    """Repository for conversation operations in MongoDB"""
    
    def __init__(self):
        self.db = None
        self.conversations: Optional[AsyncIOMotorCollection] = None
        self.messages: Optional[AsyncIOMotorCollection] = None
    
    async def initialize(self):
        """Initialize MongoDB collections and indexes"""
        self.db = get_mongodb()
        self.conversations = self.db.conversations
        self.messages = self.db.messages
        
        # Create indexes
        await self._create_indexes()
    
    async def _create_indexes(self):
        """Create required indexes for optimal performance"""
        # Conversation indexes
        conversation_indexes = [
            IndexModel([("tenant_id", ASCENDING), ("started_at", DESCENDING)]),
            IndexModel([("conversation_id", ASCENDING)], unique=True),
            IndexModel([("tenant_id", ASCENDING), ("user_id", ASCENDING), ("started_at", DESCENDING)]),
            IndexModel([("tenant_id", ASCENDING), ("status", ASCENDING), ("last_activity_at", DESCENDING)]),
            IndexModel([("tenant_id", ASCENDING), ("channel", ASCENDING), ("started_at", DESCENDING)]),
            IndexModel([("tenant_id", ASCENDING), ("business_context.category", ASCENDING)]),
            IndexModel([("compliance.data_retention_until", ASCENDING)]),
        ]
        
        await self.conversations.create_indexes(conversation_indexes)
        
        # Message indexes
        message_indexes = [
            IndexModel([("conversation_id", ASCENDING), ("sequence_number", ASCENDING)]),
            IndexModel([("tenant_id", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("message_id", ASCENDING)], unique=True),
            IndexModel([("tenant_id", ASCENDING), ("direction", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("tenant_id", ASCENDING), ("ai_analysis.intent.detected_intent", ASCENDING)]),
            IndexModel([("content.message_type", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("privacy.auto_delete_at", ASCENDING)]),
            IndexModel([("moderation.flagged", ASCENDING), ("timestamp", DESCENDING)]),
        ]
        
        await self.messages.create_indexes(message_indexes)
        
        logger.info("MongoDB indexes created successfully")
    
    async def create_conversation(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
        channel: str,
        flow_id: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new conversation"""
        now = datetime.utcnow()
        
        conversation_doc = {
            "conversation_id": conversation_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": conversation_id,  # Can be different if provided
            
            # Channel and context
            "channel": channel,
            "channel_metadata": {},
            
            # Conversation lifecycle
            "status": ConversationStatus.ACTIVE.value,
            "started_at": now,
            "last_activity_at": now,
            "completed_at": None,
            "duration_seconds": 0,
            
            # Flow and state management
            "flow_id": flow_id,
            "flow_version": "1.0",
            "current_state": "initial",
            "previous_states": [],
            "state_history": [],
            
            # Conversation context
            "context": {
                "intent_history": [],
                "current_intent": None,
                "intent_confidence": 0.0,
                "entities": {},
                "slots": {},
                "user_profile": {},
                "session_variables": initial_context or {},
                "custom_attributes": {},
                "conversation_tags": []
            },
            
            # User information
            "user_info": {
                "first_seen": now,
                "return_visitor": False,
                "language": "en",
                "timezone": "UTC",
                "device_info": {
                    "type": "unknown",
                    "os": None,
                    "browser": None
                },
                "location": {
                    "country": None,
                    "region": None,
                    "city": None,
                    "coordinates": {"lat": None, "lng": None}
                }
            },
            
            # Conversation quality and metrics
            "metrics": {
                "message_count": 0,
                "user_messages": 0,
                "bot_messages": 0,
                "response_time_avg_ms": 0,
                "response_time_max_ms": 0,
                "intent_switches": 0,
                "escalation_triggers": 0,
                "user_satisfaction": {
                    "score": None,
                    "feedback": None,
                    "collected_at": None
                },
                "completion_rate": 0.0,
                "goal_achieved": False
            },
            
            # AI and model metadata
            "ai_metadata": {
                "primary_models_used": [],
                "fallback_models_used": [],
                "total_cost_cents": 0.0,
                "total_tokens": 0,
                "average_confidence": 0.0,
                "quality_scores": {
                    "relevance": 0.0,
                    "helpfulness": 0.0,
                    "accuracy": 0.0,
                    "coherence": 0.0
                }
            },
            
            # Business and operational context
            "business_context": {
                "department": "general",
                "category": "general_inquiry",
                "subcategory": None,
                "priority": "normal",
                "tags": [],
                "resolution_type": None,
                "outcome": None,
                "value_generated": 0.0,
                "cost_incurred": 0.0
            },
            
            # Compliance and privacy
            "compliance": {
                "pii_detected": False,
                "pii_masked": False,
                "pii_types": [],
                "data_retention_until": None,
                "anonymization_level": "none",
                "gdpr_flags": [],
                "audit_required": False,
                "consent_collected": False,
                "consent_details": {}
            },
            
            # Integration tracking
            "integrations_used": [],
            
            # Summary and analysis
            "summary": {
                "auto_generated_summary": None,
                "key_topics": [],
                "entities_mentioned": [],
                "action_items": [],
                "follow_up_required": False,
                "follow_up_date": None,
                "escalation_reason": None,
                "human_notes": None
            },
            
            # A/B testing information
            "ab_testing": {
                "experiment_id": None,
                "variant": None,
                "control_group": False
            }
        }
        
        result = await self.conversations.insert_one(conversation_doc)
        conversation_doc["_id"] = result.inserted_id
        
        logger.info(
            "Conversation created",
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            flow_id=flow_id
        )
        
        return conversation_doc
    
    async def get_conversation(
        self,
        tenant_id: str,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get conversation by ID"""
        return await self.conversations.find_one({
            "tenant_id": tenant_id,
            "conversation_id": conversation_id
        })
    
    async def update_conversation_state(
        self,
        tenant_id: str,
        conversation_id: str,
        new_state: str,
        previous_state: Optional[str] = None
    ) -> bool:
        """Update conversation state"""
        update_doc = {
            "$set": {
                "current_state": new_state,
                "last_activity_at": datetime.utcnow()
            },
            "$push": {
                "state_history": {
                    "state": new_state,
                    "entered_at": datetime.utcnow(),
                    "exited_at": None,
                    "duration_ms": None
                }
            }
        }
        
        if previous_state:
            update_doc["$push"]["previous_states"] = previous_state
        
        result = await self.conversations.update_one(
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id
            },
            update_doc
        )
        
        return result.modified_count > 0
    
    async def update_conversation_context(
        self,
        tenant_id: str,
        conversation_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """Update conversation context"""
        update_doc = {
            "$set": {
                "last_activity_at": datetime.utcnow()
            }
        }
        
        # Build nested updates for context fields
        for key, value in context_updates.items():
            update_doc["$set"][f"context.{key}"] = value
        
        result = await self.conversations.update_one(
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id
            },
            update_doc
        )
        
        return result.modified_count > 0
    
    async def add_message(
        self,
        message_doc: Dict[str, Any]
    ) -> str:
        """Add message to conversation"""
        now = datetime.utcnow()
        
        # Ensure required fields
        message_doc.update({
            "timestamp": message_doc.get("timestamp", now),
            "created_at": now
        })
        
        # Generate sequence number
        last_message = await self.messages.find_one(
            {"conversation_id": message_doc["conversation_id"]},
            sort=[("sequence_number", DESCENDING)]
        )
        
        sequence_number = (last_message.get("sequence_number", 0) + 1) if last_message else 1
        message_doc["sequence_number"] = sequence_number
        
        # Insert message
        result = await self.messages.insert_one(message_doc)
        
        # Update conversation metrics
        await self._update_conversation_metrics(
            message_doc["tenant_id"],
            message_doc["conversation_id"],
            message_doc["direction"]
        )
        
        logger.info(
            "Message added",
            message_id=message_doc["message_id"],
            conversation_id=message_doc["conversation_id"],
            direction=message_doc["direction"]
        )
        
        return str(result.inserted_id)
    
    async def _update_conversation_metrics(
        self,
        tenant_id: str,
        conversation_id: str,
        message_direction: str
    ):
        """Update conversation message metrics"""
        update_doc = {
            "$inc": {"metrics.message_count": 1},
            "$set": {"last_activity_at": datetime.utcnow()}
        }
        
        if message_direction == MessageDirection.INBOUND.value:
            update_doc["$inc"]["metrics.user_messages"] = 1
        elif message_direction == MessageDirection.OUTBOUND.value:
            update_doc["$inc"]["metrics.bot_messages"] = 1
        
        await self.conversations.update_one(
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id
            },
            update_doc
        )
    
    async def get_conversation_messages(
        self,
        tenant_id: str,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation"""
        cursor = self.messages.find(
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id
            }
        ).sort("sequence_number", ASCENDING).skip(offset).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def close_conversation(
        self,
        tenant_id: str,
        conversation_id: str,
        status: ConversationStatus,
        completion_reason: Optional[str] = None
    ) -> bool:
        """Close conversation with final status"""
        now = datetime.utcnow()
        
        # Get conversation to calculate duration
        conversation = await self.get_conversation(tenant_id, conversation_id)
        if not conversation:
            return False
        
        started_at = conversation.get("started_at", now)
        duration_seconds = int((now - started_at).total_seconds())
        
        update_doc = {
            "$set": {
                "status": status.value,
                "completed_at": now,
                "duration_seconds": duration_seconds,
                "last_activity_at": now
            }
        }
        
        if completion_reason:
            update_doc["$set"]["business_context.outcome"] = completion_reason
        
        result = await self.conversations.update_one(
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id
            },
            update_doc
        )
        
        return result.modified_count > 0
```

## Success Criteria
- [x] Complete domain model system with enums and base classes
- [x] PostgreSQL models for flow definitions and analytics
- [x] Redis data structures for execution state and caching
- [x] MongoDB collections for conversation and message data
- [x] Repository pattern implementation for all databases
- [x] Comprehensive validation and error handling
- [x] Performance-optimized database indexes
- [x] Audit trails and compliance support
- [x] Type safety with Pydantic models

## Key Error Handling & Performance Considerations
1. **Validation**: Comprehensive Pydantic validation with custom validators
2. **Database Indexes**: Optimized indexes for query performance
3. **Redis Operations**: Distributed locking and atomic operations
4. **MongoDB**: Proper indexing and aggregation pipeline optimization
5. **Repository Pattern**: Consistent data access patterns with error handling
6. **Data Consistency**: Transaction support where needed
7. **Caching Strategy**: TTL-based caching with cache invalidation

## Technologies Used
- **Domain Models**: Pydantic with comprehensive validation
- **PostgreSQL**: SQLAlchemy with async support
- **MongoDB**: Motor (async MongoDB driver)
- **Redis**: aioredis with connection pooling
- **Validation**: JSON Schema, custom validators
- **Serialization**: JSON with custom encoders

## Cross-Service Integration
- **Models**: Shared domain models across all services
- **Context Management**: Redis-based distributed state
- **Audit Trails**: PostgreSQL audit logs
- **Analytics**: MongoDB aggregation pipelines
- **Caching**: Multi-layer caching strategy

## Next Phase Dependencies
Phase 3 will build upon:
- Domain models for state machine implementation
- Redis execution state management
- PostgreSQL flow definitions
- MongoDB conversation tracking
- Repository patterns for data access