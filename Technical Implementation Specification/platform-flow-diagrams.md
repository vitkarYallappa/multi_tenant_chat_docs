# Multi-Tenant AI Chatbot Platform - Key Flow Diagrams

## 1. End-to-End Message Processing Flow

This diagram shows the complete journey of a user message through all services, from initial receipt to final response delivery.

```mermaid
sequenceDiagram
    participant User
    participant ChatService as Chat Service
    participant SecurityHub as Security Hub
    participant MCPEngine as MCP Engine
    participant ModelOrch as Model Orchestrator
    participant AdaptorSvc as Adaptor Service
    participant Analytics as Analytics Engine
    participant Redis
    participant MongoDB
    participant Kafka

    %% Initial Message Receipt
    User->>ChatService: Send Message
    ChatService->>SecurityHub: Validate Token
    SecurityHub->>Redis: Check Session
    SecurityHub-->>ChatService: Auth Context
    
    %% Message Processing
    ChatService->>Redis: Get/Create Conversation
    ChatService->>MongoDB: Store Inbound Message
    ChatService->>Kafka: Publish Message Event
    ChatService->>MCPEngine: Process Message (gRPC)
    
    %% State Machine Execution
    MCPEngine->>Redis: Load Conversation State
    MCPEngine->>MCPEngine: Execute Current State
    
    alt Intent Detection State
        MCPEngine->>ModelOrch: Detect Intent
        ModelOrch->>ModelOrch: Route to Best Model
        ModelOrch-->>MCPEngine: Intent + Confidence
    else Integration State
        MCPEngine->>AdaptorSvc: Execute Integration
        AdaptorSvc->>AdaptorSvc: Transform Data
        AdaptorSvc-->>MCPEngine: Integration Result
    else Response Generation State
        MCPEngine->>ModelOrch: Generate Response
        ModelOrch->>ModelOrch: Select Provider & Model
        ModelOrch-->>MCPEngine: Generated Response
    end
    
    %% State Transition
    MCPEngine->>MCPEngine: Evaluate Transitions
    MCPEngine->>Redis: Update Conversation State
    MCPEngine->>Kafka: Publish State Event
    MCPEngine-->>ChatService: Response + New State
    
    %% Response Delivery
    ChatService->>ChatService: Format for Channel
    ChatService->>MongoDB: Store Outbound Message
    ChatService->>Kafka: Publish Response Event
    ChatService-->>User: Deliver Response
    
    %% Async Analytics
    Kafka-->>Analytics: Process Events
    Analytics->>Analytics: Extract Metrics
    Analytics->>Analytics: Update Dashboards
```

## 2. Authentication and Authorization Flow

This diagram illustrates the complete security flow from login through API request authorization.

```mermaid
flowchart TB
    subgraph User Authentication
        A[User Login Request] --> B{MFA Required?}
        B -->|Yes| C[Send MFA Challenge]
        B -->|No| D[Validate Credentials]
        C --> E[Verify MFA Code]
        E --> F{Valid?}
        F -->|No| C
        F -->|Yes| G[Generate Tokens]
        D --> G
    end
    
    subgraph Token Management
        G --> H[Create JWT Tokens]
        H --> I[Store Session in Redis]
        H --> J[Set Refresh Token]
        H --> K[Return Token Pair]
    end
    
    subgraph API Authorization
        L[API Request] --> M[Extract Token]
        M --> N{Token in Blacklist?}
        N -->|Yes| O[Reject Request]
        N -->|No| P[Verify Signature]
        P --> Q{Valid?}
        Q -->|No| O
        Q -->|Yes| R[Check Expiration]
        R --> S{Expired?}
        S -->|Yes| T[Try Refresh]
        S -->|No| U[Load Permissions]
    end
    
    subgraph Permission Check
        U --> V[Get User Context]
        V --> W[Check RBAC Rules]
        W --> X[Evaluate Policies]
        X --> Y{Authorized?}
        Y -->|No| Z[Return 403]
        Y -->|Yes| AA[Add Context to Request]
        AA --> AB[Process Request]
    end
    
    subgraph Audit Trail
        AB --> AC[Log Access Event]
        O --> AC
        Z --> AC
        AC --> AD[Send to Analytics]
        AC --> AE[Store Audit Log]
    end
```

## 3. Model Orchestration and Fallback Flow

This diagram shows how the Model Orchestrator intelligently routes requests and handles failures.

```mermaid
flowchart LR
    subgraph Request Processing
        A[Model Request] --> B[Load Tenant Config]
        B --> C{Check Cache}
        C -->|Hit| D[Return Cached]
        C -->|Miss| E[Route Request]
    end
    
    subgraph Routing Decision
        E --> F[Calculate Costs]
        F --> G[Check Performance]
        G --> H[Apply Strategy]
        H --> I{Strategy Type}
        
        I -->|Cost| J[Sort by Cost]
        I -->|Performance| K[Sort by Quality]
        I -->|Balanced| L[Combined Score]
        
        J --> M[Select Provider]
        K --> M
        L --> M
    end
    
    subgraph Provider Execution
        M --> N[Check Health]
        N --> O{Healthy?}
        O -->|No| P[Next Provider]
        O -->|Yes| Q[Check Rate Limit]
        Q --> R{Within Limit?}
        R -->|No| P
        R -->|Yes| S[Execute Request]
        
        S --> T{Success?}
        T -->|No| U[Log Error]
        U --> P
        T -->|Yes| V[Process Response]
    end
    
    subgraph Fallback Chain
        P --> W{More Providers?}
        W -->|Yes| N
        W -->|No| X[All Failed]
        X --> Y[Return Error]
    end
    
    subgraph Response Processing
        V --> Z[Calculate Usage]
        Z --> AA[Update Metrics]
        AA --> AB[Cache Response]
        AB --> AC[Return Result]
    end
    
    subgraph Cost Tracking
        Z --> AD[Track Costs]
        AD --> AE[Check Limits]
        AE --> AF{Over Limit?}
        AF -->|Yes| AG[Send Alert]
        AF -->|No| AH[Update Dashboard]
    end
```

## 4. Integration Execution Flow

This diagram details how external integrations are executed with the Adaptor Service.

```mermaid
stateDiagram-v2
    [*] --> ValidateRequest
    
    ValidateRequest --> LoadIntegration: Valid
    ValidateRequest --> ReturnError: Invalid
    
    LoadIntegration --> CheckCache: Config Loaded
    CheckCache --> ReturnCached: Cache Hit
    CheckCache --> TransformRequest: Cache Miss
    
    TransformRequest --> ApplyMapping: Input Transformed
    ApplyMapping --> AddAuthentication: Mappings Applied
    
    AddAuthentication --> CheckCircuitBreaker: Auth Added
    CheckCircuitBreaker --> ExecuteRequest: Closed
    CheckCircuitBreaker --> ReturnFallback: Open
    
    ExecuteRequest --> Success: 200 OK
    ExecuteRequest --> RateLimit: 429
    ExecuteRequest --> ServerError: 5xx
    ExecuteRequest --> ClientError: 4xx
    ExecuteRequest --> Timeout: Timeout
    
    RateLimit --> WaitAndRetry: Retry After
    ServerError --> RetryWithBackoff: Retryable
    Timeout --> RetryWithBackoff: Retryable
    
    WaitAndRetry --> ExecuteRequest: After Wait
    RetryWithBackoff --> ExecuteRequest: Retry < Max
    RetryWithBackoff --> MarkUnhealthy: Max Retries
    
    Success --> TransformResponse: Parse Response
    TransformResponse --> ApplyResponseMapping: Transformed
    ApplyResponseMapping --> UpdateCache: Mapped
    UpdateCache --> ReturnSuccess: Cached
    
    ClientError --> LogError: Not Retryable
    MarkUnhealthy --> OpenCircuitBreaker: Failed
    OpenCircuitBreaker --> ReturnError: Circuit Open
    
    ReturnSuccess --> UpdateMetrics: Success
    ReturnError --> UpdateMetrics: Failure
    ReturnFallback --> UpdateMetrics: Fallback
    ReturnCached --> UpdateMetrics: Cached
    
    UpdateMetrics --> [*]
```

## 5. Real-time Analytics Pipeline Flow

This diagram shows how events flow through the analytics system for real-time processing and insights.

```mermaid
flowchart TB
    subgraph Event Sources
        A1[Chat Service] --> K1[conversation.events]
        A2[MCP Engine] --> K1
        A3[Model Orchestrator] --> K2[model.usage]
        A4[Adaptor Service] --> K3[integration.events]
        A5[Security Hub] --> K4[security.events]
    end
    
    subgraph Kafka Topics
        K1 --> EP[Event Processor]
        K2 --> EP
        K3 --> EP
        K4 --> EP
    end
    
    subgraph Stream Processing
        EP --> V1{Validate Schema}
        V1 -->|Invalid| DLQ[Dead Letter Queue]
        V1 -->|Valid| E1[Enrich Event]
        
        E1 --> E2[Add Context]
        E2 --> E3[Add Metadata]
        E3 --> E4[Calculate Derived Fields]
        
        E4 --> SP[Stream Processor]
    end
    
    subgraph Real-time Analysis
        SP --> W1[Window Aggregation]
        SP --> AD[Anomaly Detection]
        SP --> PM[Pattern Matching]
        
        W1 --> AGG[Aggregations]
        AGG --> M1[1-min Window]
        AGG --> M2[5-min Window]
        AGG --> M3[1-hour Window]
        
        AD --> AL{Anomaly Found?}
        AL -->|Yes| ALERT[Generate Alert]
        AL -->|No| CONT[Continue]
        
        PM --> PAT[Pattern Store]
        PAT --> INS[Generate Insights]
    end
    
    subgraph Storage Layer
        M1 --> R[Redis Cache]
        M2 --> R
        M3 --> TS[TimescaleDB]
        
        CONT --> TS
        INS --> TS
        
        TS --> MV[Materialized Views]
        MV --> DASH[Dashboards]
    end
    
    subgraph ML Pipeline
        TS --> FE[Feature Engineering]
        FE --> ML1[Satisfaction Model]
        FE --> ML2[Churn Model]
        FE --> ML3[Intent Predictor]
        
        ML1 --> PRED[Predictions]
        ML2 --> PRED
        ML3 --> PRED
        
        PRED --> API[Analytics API]
    end
    
    subgraph Outputs
        DASH --> U1[Real-time Dashboard]
        ALERT --> U2[Alert Manager]
        API --> U3[Business Users]
        TS --> U4[Reports]
    end
```

---

## Bonus: Complete Request Lifecycle Flow

This comprehensive diagram shows how all services work together for a complete conversation lifecycle.

```mermaid
journey
    title User Conversation Journey
    
    section Authentication
      User Opens Chat: 5: User
      Request Auth Token: 3: Chat Service, Security Hub
      Validate Session: 4: Security Hub
      Create Session: 4: Security Hub, Redis
    
    section Message Processing
      Send Message: 5: User
      Validate & Normalize: 4: Chat Service
      Store Message: 3: Chat Service, MongoDB
      Route to MCP: 4: Chat Service, MCP Engine
      Load Flow State: 4: MCP Engine, Redis
    
    section Intelligence Layer
      Detect Intent: 4: MCP Engine, Model Orchestrator
      Route to Best Model: 5: Model Orchestrator
      Process with AI: 4: Model Orchestrator
      Check Integration Need: 3: MCP Engine
      Execute Integration: 3: MCP Engine, Adaptor Service
      Transform Data: 4: Adaptor Service
    
    section Response Generation
      Generate Response: 4: MCP Engine, Model Orchestrator
      Update State: 4: MCP Engine, Redis
      Format Response: 4: Chat Service
      Deliver to User: 5: Chat Service, User
    
    section Analytics
      Publish Events: 3: All Services, Kafka
      Process Streams: 4: Analytics Engine
      Update Metrics: 4: Analytics Engine
      Generate Insights: 5: Analytics Engine
      Update Dashboards: 5: Analytics Engine
```

---

## Key Integration Points

### 1. **Synchronous Communications (gRPC/REST)**
- Chat Service → MCP Engine (Message Processing)
- MCP Engine → Model Orchestrator (AI Operations)
- MCP Engine → Adaptor Service (Integrations)
- All Services → Security Hub (Auth/AuthZ)

### 2. **Asynchronous Communications (Kafka)**
- All Services → Analytics Engine (Events)
- MCP Engine → Chat Service (State Updates)
- Security Hub → Analytics Engine (Audit Logs)

### 3. **Shared Storage (Redis)**
- Session Management
- Conversation State
- Rate Limiting
- Response Caching

### 4. **Persistent Storage**
- MongoDB: Conversations, Messages
- PostgreSQL: Configuration, Users
- TimescaleDB: Analytics, Metrics

These flow diagrams illustrate the complex interactions between services while maintaining clean separation of concerns and enabling scalability at each layer.
