# MCP Engine Architecture Diagrams

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "External Channels"
        WEB[Web Chat]
        WA[WhatsApp]
        MSG[Messenger]
        SLACK[Slack]
        VOICE[Voice]
    end
    
    subgraph "API Gateway Layer"
        LB[Load Balancer]
        API[API Gateway]
    end
    
    subgraph "MCP Engine Core"
        CHAT[Chat Service<br/>Message Ingestion]
        MCP[MCP Engine<br/>State Machine]
        MODEL[Model Orchestrator<br/>AI Processing]
        ADAPT[Adaptor Service<br/>Integrations]
        SEC[Security Hub<br/>Auth & RBAC]
        ANALYTICS[Analytics Engine<br/>Metrics & BI]
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL<br/>Config & Metadata)]
        MONGO[(MongoDB<br/>Conversations)]
        REDIS[(Redis Cluster<br/>Cache & Sessions)]
        TSDB[(TimescaleDB<br/>Analytics)]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI GPT]
        ANTHROPIC[Anthropic Claude]
        CRM[CRM Systems]
        ERP[ERP Systems]
    end
    
    WEB --> LB
    WA --> LB
    MSG --> LB
    SLACK --> LB
    VOICE --> LB
    
    LB --> API
    API --> CHAT
    
    CHAT <--> MCP
    MCP <--> MODEL
    MCP <--> ADAPT
    MCP <--> SEC
    CHAT --> ANALYTICS
    MCP --> ANALYTICS
    
    MODEL --> OPENAI
    MODEL --> ANTHROPIC
    ADAPT --> CRM
    ADAPT --> ERP
    
    CHAT --> MONGO
    CHAT --> REDIS
    MCP --> PG
    MCP --> REDIS
    SEC --> PG
    ANALYTICS --> TSDB
    
    classDef service fill:#e1f5fe
    classDef database fill:#f3e5f5
    classDef external fill:#fff3e0
    
    class CHAT,MCP,MODEL,ADAPT,SEC,ANALYTICS service
    class PG,MONGO,REDIS,TSDB database
    class WEB,WA,MSG,SLACK,VOICE,OPENAI,ANTHROPIC,CRM,ERP external
```

## 2. Message Processing Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Chat as Chat Service
    participant MCP as MCP Engine
    participant Model as Model Orchestrator
    participant Adapt as Adaptor Service
    participant Redis
    participant Mongo as MongoDB
    
    User->>Chat: Send Message
    Chat->>Redis: Get/Create Session
    Chat->>Mongo: Store Message
    Chat->>MCP: Process Message (gRPC)
    
    MCP->>Redis: Acquire Context Lock
    MCP->>Redis: Load Conversation Context
    
    alt State: Intent Detection
        MCP->>Model: Detect Intent & Extract Entities
        Model-->>MCP: Intent + Entities + Confidence
    else State: Integration Call
        MCP->>Adapt: Execute Integration
        Adapt-->>MCP: Integration Result
    else State: Response Generation
        MCP->>Model: Generate Response
        Model-->>MCP: Generated Text + Metadata
    end
    
    MCP->>MCP: Execute State Machine
    MCP->>MCP: Evaluate Transitions
    MCP->>Redis: Update Context
    MCP->>Redis: Release Context Lock
    
    MCP-->>Chat: Processing Result + Next State
    Chat->>Mongo: Store Response
    Chat->>Redis: Update Session
    Chat-->>User: Send Response
    
    Note over Chat,Mongo: Background: Analytics Event
    Chat->>+Analytics: Track Message Processed
    Analytics->>TimescaleDB: Store Metrics
```

## 3. State Machine Execution Flow

```mermaid
sequenceDiagram
    participant Engine as State Engine
    participant Validator as State Validator
    participant Condition as Condition Evaluator
    participant Action as Action Executor
    participant Context as Context Manager
    
    Engine->>Engine: Acquire Execution Lock
    Engine->>Validator: Validate Current State
    Validator-->>Engine: State Valid
    
    Engine->>Engine: Execute Entry Actions
    Engine->>Action: Execute Actions List
    Action-->>Engine: Actions Results
    
    Engine->>Engine: Execute State Logic
    
    alt Response State
        Engine->>Engine: Select Response Template
        Engine->>Engine: Apply Personalization
    else Intent State
        Engine->>Model: Call Intent Detection
    else Slot Filling State
        Engine->>Engine: Extract & Validate Slots
    else Integration State
        Engine->>Adaptor: Call Integration
    end
    
    Engine->>Condition: Evaluate All Transitions
    loop For Each Transition
        Condition->>Condition: Check Condition
        Condition->>Context: Access Context Data
    end
    Condition-->>Engine: Matching Transition
    
    alt Transition Found
        Engine->>Action: Execute Transition Actions
        Engine->>Engine: Execute Exit Actions
        Engine->>Context: Update Context
        Engine->>Engine: Move to New State
    else No Transition
        Engine->>Engine: Stay in Current State
    end
    
    Engine->>Engine: Release Execution Lock
    Engine-->>Caller: State Execution Result
```

## 4. Service Layer Class Diagram

```mermaid
classDiagram
    class BaseService {
        <<abstract>>
        +service_name: str
        +logger: Logger
        +dependencies: Dict
        +initialize()* async
        +shutdown()* async
        +health_check()* async
        +add_dependency(name, dependency)
        +get_dependency(name)
    }
    
    class ExecutionService {
        +state_engine: StateEngine
        +flow_service: FlowService
        +context_service: ContextService
        +process_message(tenant_id, conversation_id, message) async
        +reset_conversation(tenant_id, conversation_id) async
        +get_conversation_state(tenant_id, conversation_id) async
    }
    
    class FlowService {
        +flow_repository: FlowRepository
        +create_flow(tenant_id, name, definition) async
        +get_flow_by_id(tenant_id, flow_id) async
        +get_default_flow(tenant_id) async
        +publish_flow(tenant_id, flow_id) async
        +set_default_flow(tenant_id, flow_id) async
    }
    
    class ContextService {
        +context_repository: ContextRepository
        +get_conversation_context(tenant_id, conversation_id) async
        +create_conversation_context(tenant_id, conversation_id) async
        +update_conversation_context(context) async
        +reset_conversation_context(tenant_id, conversation_id) async
    }
    
    class StateEngine {
        +transition_handler: TransitionHandler
        +condition_evaluator: ConditionEvaluator
        +action_executor: ActionExecutor
        +execute_state(current_state, event, context, flow) async
        +evaluate_transitions(state, event, context) async
    }
    
    class ConditionEvaluator {
        +operators: Dict
        +evaluate_transition_condition(transition, event, context) async
        +evaluate_condition(condition_def, context, event) async
        +evaluate_expression(expression, context) async
    }
    
    class ActionExecutor {
        +action_handlers: Dict
        +execute_action(action, context, event) async
        +execute_actions_batch(actions, context, event) async
    }
    
    class ModelOrchestratorClient {
        +detect_intent(text, context, config) async
        +extract_entities(text, entity_types, context) async
        +generate_response(intent, entities, context) async
        +analyze_sentiment(text, config) async
    }
    
    class AdaptorServiceClient {
        +execute_integration(integration_id, endpoint, data) async
        +test_integration(integration_id) async
        +get_integration_status(integration_id) async
    }
    
    BaseService <|-- ExecutionService
    BaseService <|-- FlowService
    BaseService <|-- ContextService
    
    ExecutionService --> StateEngine
    ExecutionService --> FlowService
    ExecutionService --> ContextService
    ExecutionService --> ModelOrchestratorClient
    ExecutionService --> AdaptorServiceClient
    
    StateEngine --> ConditionEvaluator
    StateEngine --> ActionExecutor
    
    FlowService --> FlowRepository
    ContextService --> ContextRepository
```

## 5. Database Architecture & Data Flow

```mermaid
graph TB
    subgraph "Application Layer"
        API[API Layer]
        SERVICES[Service Layer]
        REPOS[Repository Layer]
    end
    
    subgraph "PostgreSQL - Configuration & Metadata"
        PG_TENANTS[tenants]
        PG_USERS[tenant_users]
        PG_FLOWS[conversation_flows]
        PG_INTEGRATIONS[integrations]
        PG_API_KEYS[api_keys]
    end
    
    subgraph "MongoDB - Conversations & Messages"
        MONGO_CONV[conversations]
        MONGO_MSG[messages]
        MONGO_KB[knowledge_base]
    end
    
    subgraph "Redis Cluster - Cache & Sessions"
        REDIS_SESSION[session:{tenant}:{session_id}]
        REDIS_CONTEXT[conversation:{tenant}:{conv_id}]
        REDIS_RATE[rate_limit:{identifier}]
        REDIS_CACHE[cache:{service}:{key}]
    end
    
    subgraph "TimescaleDB - Analytics"
        TS_SYSTEM[system_metrics]
        TS_CONV[conversation_analytics]
        TS_MODEL[model_usage_analytics]
        TS_CUSTOM[custom_metrics]
    end
    
    API --> SERVICES
    SERVICES --> REPOS
    
    REPOS --> PG_TENANTS
    REPOS --> PG_USERS
    REPOS --> PG_FLOWS
    REPOS --> PG_INTEGRATIONS
    REPOS --> PG_API_KEYS
    
    REPOS --> MONGO_CONV
    REPOS --> MONGO_MSG
    REPOS --> MONGO_KB
    
    SERVICES --> REDIS_SESSION
    SERVICES --> REDIS_CONTEXT
    API --> REDIS_RATE
    SERVICES --> REDIS_CACHE
    
    SERVICES --> TS_SYSTEM
    SERVICES --> TS_CONV
    SERVICES --> TS_MODEL
    SERVICES --> TS_CUSTOM
    
    classDef postgres fill:#336791,color:#fff
    classDef mongo fill:#4db33d,color:#fff
    classDef redis fill:#d82c20,color:#fff
    classDef timescale fill:#ffa500,color:#fff
    classDef app fill:#2196f3,color:#fff
    
    class PG_TENANTS,PG_USERS,PG_FLOWS,PG_INTEGRATIONS,PG_API_KEYS postgres
    class MONGO_CONV,MONGO_MSG,MONGO_KB mongo
    class REDIS_SESSION,REDIS_CONTEXT,REDIS_RATE,REDIS_CACHE redis
    class TS_SYSTEM,TS_CONV,TS_MODEL,TS_CUSTOM timescale
    class API,SERVICES,REPOS app
```

## Data Flow Patterns

### 1. **Configuration Data Flow**
- **PostgreSQL** → Services → Redis Cache → Application Logic
- Tenant configs, flow definitions, integration settings

### 2. **Conversation Data Flow**
- **User Input** → Chat Service → MongoDB (persist) → MCP Engine
- **Context Updates** → Redis (real-time) → MongoDB (persistence)

### 3. **Analytics Data Flow**
- **Real-time Events** → Kafka → Analytics Engine → TimescaleDB
- **Metrics Collection** → Redis (counters) → TimescaleDB (aggregation)

### 4. **Cache Strategy**
- **L1**: Redis (hot data, sessions, rate limits)
- **L2**: Application memory (parsed configs, templates)
- **Write-through**: Critical data written to persistent storage
- **Write-behind**: Analytics data batched for performance

