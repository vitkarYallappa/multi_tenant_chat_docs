# Multi-Tenant AI Chatbot Platform - Architecture Diagrams

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph "External Clients"
        WEB[Web Chat Widget]
        WHATSAPP[WhatsApp Business]
        SLACK[Slack]
        TEAMS[Microsoft Teams]
        MESSENGER[Facebook Messenger]
    end

    subgraph "Load Balancer & Gateway"
        LB[Load Balancer]
        GATEWAY[API Gateway]
    end

    subgraph "Chat Service (Core)"
        subgraph "API Layer"
            ROUTES[FastAPI Routes]
            MIDDLEWARE[Middleware Stack]
            VALIDATORS[Request Validators]
        end
        
        subgraph "Service Layer"
            MSG_SVC[Message Service]
            CONV_SVC[Conversation Service]
            SESS_SVC[Session Service]
            CHAN_SVC[Channel Service]
            DELIVERY_SVC[Delivery Service]
            AUDIT_SVC[Audit Service]
        end
        
        subgraph "Core Logic"
            PROCESSORS[Message Processors]
            CHANNELS[Channel Implementations]
            NORMALIZERS[Content Normalizers]
            PIPELINE[Processing Pipeline]
        end
        
        subgraph "Data Layer"
            REPOS[Repositories]
            MODELS[Data Models]
        end
    end

    subgraph "External Services"
        MCP[MCP Engine<br/>Response Generation]
        SECURITY[Security Hub<br/>Authentication]
        ANALYTICS[Analytics Engine<br/>Metrics & Events]
    end

    subgraph "Infrastructure"
        subgraph "Databases"
            MONGO[(MongoDB<br/>Conversations & Messages)]
            REDIS[(Redis<br/>Sessions & Cache)]
        end
        
        subgraph "Message Queue"
            KAFKA[Kafka<br/>Event Streaming]
        end
        
        subgraph "Monitoring"
            METRICS[Metrics Collection]
            LOGS[Centralized Logging]
            HEALTH[Health Checks]
        end
    end

    subgraph "Webhook Endpoints"
        WH_ROUTES[Webhook Routes]
        WH_PROCESSORS[Webhook Processors]
        WH_SECURITY[Signature Validation]
    end

    %% External connections
    WEB --> LB
    WHATSAPP --> WH_ROUTES
    SLACK --> WH_ROUTES
    TEAMS --> WH_ROUTES
    MESSENGER --> WH_ROUTES

    %% Internal flow
    LB --> GATEWAY
    GATEWAY --> ROUTES
    ROUTES --> MIDDLEWARE
    MIDDLEWARE --> VALIDATORS
    VALIDATORS --> MSG_SVC

    %% Service dependencies
    MSG_SVC --> CONV_SVC
    MSG_SVC --> SESS_SVC
    MSG_SVC --> CHAN_SVC
    MSG_SVC --> PROCESSORS
    MSG_SVC --> CHANNELS
    
    %% Data layer
    MSG_SVC --> REPOS
    REPOS --> MONGO
    REPOS --> REDIS
    
    %% External services
    MSG_SVC -.->|gRPC| MCP
    MIDDLEWARE -.->|gRPC| SECURITY
    AUDIT_SVC -.->|Events| ANALYTICS
    
    %% Event streaming
    MSG_SVC --> KAFKA
    KAFKA --> ANALYTICS
    
    %% Webhooks
    WH_ROUTES --> WH_SECURITY
    WH_SECURITY --> WH_PROCESSORS
    WH_PROCESSORS --> MSG_SVC

    %% Monitoring
    MSG_SVC --> METRICS
    MSG_SVC --> LOGS
    ROUTES --> HEALTH

    style ROUTES fill:#e1f5fe
    style MSG_SVC fill:#f3e5f5
    style MCP fill:#fff3e0
    style MONGO fill:#e8f5e8
    style KAFKA fill:#fff8e1
```

## 2. Message Processing Flow

```mermaid
sequenceDiagram
    participant Client as External Client
    participant API as API Gateway
    participant Auth as Auth Middleware
    participant Routes as Chat Routes
    participant MsgSvc as Message Service
    participant ConvSvc as Conversation Service
    participant Processor as Text Processor
    participant Channel as Channel Handler
    participant MCP as MCP Engine
    participant DB as MongoDB
    participant Cache as Redis
    participant Events as Kafka Events

    Client->>API: Send Message Request
    API->>Auth: Validate JWT Token
    Auth->>Auth: Extract User Context
    Auth->>Routes: Authenticated Request
    
    Routes->>Routes: Validate Request Schema
    Routes->>MsgSvc: Process Message
    
    Note over MsgSvc: Message Processing Pipeline
    MsgSvc->>ConvSvc: Get/Create Conversation
    ConvSvc->>DB: Query/Create Conversation
    DB-->>ConvSvc: Conversation Data
    ConvSvc-->>MsgSvc: Conversation Context
    
    MsgSvc->>Cache: Get Session Data
    Cache-->>MsgSvc: Session Context
    
    MsgSvc->>Processor: Process Message Content
    Processor->>Processor: Normalize & Extract Entities
    Processor->>Processor: Language Detection
    Processor->>Processor: Content Analysis
    Processor-->>MsgSvc: Processing Result
    
    MsgSvc->>DB: Store Incoming Message
    
    MsgSvc->>MCP: Generate Response
    Note over MCP: AI Processing
    MCP-->>MsgSvc: Bot Response
    
    MsgSvc->>DB: Store Outgoing Message
    
    MsgSvc->>Channel: Deliver Response
    Channel->>Channel: Format for Channel
    Channel->>Client: Send Response
    Channel-->>MsgSvc: Delivery Status
    
    MsgSvc->>Cache: Update Session
    MsgSvc->>Events: Publish Events
    
    Events->>Events: Message Received Event
    Events->>Events: Message Sent Event
    Events->>Events: Conversation Updated Event
    
    MsgSvc-->>Routes: Processing Result
    Routes-->>API: Response
    API-->>Client: Final Response

    Note over Client,Events: Async Event Processing
    Events-->>Analytics: Analytics Events
```

## 3. Service Layer Architecture

```mermaid
graph TB

    subgraph API_Layer
        CHAT_API[Chat Routes]
        CONV_API[Conversation Routes]
        WEBHOOK_API[Webhook Routes]
        HEALTH_API[Health Routes]
    end

    subgraph Middleware_Stack
        AUTH_MW[Auth Middleware]
        RATE_MW[Rate Limit Middleware]
        TENANT_MW[Tenant Middleware]
        ERROR_MW[Error Handler]
        LOG_MW[Logging Middleware]
    end

    subgraph Service_Layer
        direction TB

        subgraph Core_Services
            MSG_SERVICE[Message Service\n- process_message\n- handle_webhook\n- generate_response]
            CONV_SERVICE[Conversation Service\n- get_conversation_history\n- list_conversations\n- close_conversation]
            SESS_SERVICE[Session Service\n- create_session\n- update_context\n- cleanup_expired]
            CHAN_SERVICE[Channel Service\n- send_message\n- validate_recipient\n- process_webhook]
        end

        subgraph Supporting_Services
            DELIVERY_SERVICE[Delivery Service\n- track_delivery\n- retry_failed\n- update_status]
            AUDIT_SERVICE[Audit Service\n- log_activity\n- compliance_check\n- generate_reports]
        end
    end

    subgraph Repository_Layer
        CONV_REPO[Conversation Repository\n- create\n- get_by_id\n- update]
        MSG_REPO[Message Repository\n- create\n- get_by_conversation\n- get_last_message]
        SESS_REPO[Session Repository\n- create_session\n- get_session\n- extend_session]
        RATE_REPO[Rate Limit Repository\n- check_rate_limit\n- increment_counter\n- reset_limits]
    end

    subgraph Data_Sources
        MONGODB[(MongoDB\n- Conversations\n- Messages\n- Users)]
        REDIS[(Redis\n- Sessions\n- Rate Limits\n- Cache)]
    end

    subgraph External_Dependencies
        MCP_CLIENT[MCP Engine Client\n- process_message\n- update_context]
        SECURITY_CLIENT[Security Hub Client\n- validate_token\n- check_permissions]
        EVENT_PUBLISHER[Event Publisher\n- publish_event\n- publish_batch]
    end

    %% API to Middleware
    CHAT_API --> AUTH_MW
    CONV_API --> AUTH_MW
    WEBHOOK_API --> RATE_MW

    %% Middleware Chain
    AUTH_MW --> RATE_MW
    RATE_MW --> TENANT_MW
    TENANT_MW --> ERROR_MW
    ERROR_MW --> LOG_MW

    %% Middleware to Services
    LOG_MW --> MSG_SERVICE
    LOG_MW --> CONV_SERVICE
    LOG_MW --> SESS_SERVICE

    %% Service Dependencies
    MSG_SERVICE --> CONV_SERVICE
    MSG_SERVICE --> SESS_SERVICE
    MSG_SERVICE --> CHAN_SERVICE
    MSG_SERVICE --> DELIVERY_SERVICE
    MSG_SERVICE --> AUDIT_SERVICE

    %% Services to Repositories
    MSG_SERVICE --> CONV_REPO
    MSG_SERVICE --> MSG_REPO
    CONV_SERVICE --> CONV_REPO
    CONV_SERVICE --> MSG_REPO
    SESS_SERVICE --> SESS_REPO
    RATE_MW --> RATE_REPO

    %% Repositories to Data
    CONV_REPO --> MONGODB
    MSG_REPO --> MONGODB
    SESS_REPO --> REDIS
    RATE_REPO --> REDIS

    %% External Dependencies
    MSG_SERVICE -.->|gRPC| MCP_CLIENT
    AUTH_MW -.->|gRPC| SECURITY_CLIENT
    MSG_SERVICE --> EVENT_PUBLISHER
    AUDIT_SERVICE --> EVENT_PUBLISHER

    %% Styles
    style MSG_SERVICE fill:#e3f2fd
    style CONV_SERVICE fill:#f3e5f5
    style SESS_SERVICE fill:#e8f5e8
    style MONGODB fill:#ffebee
    style REDIS fill:#fff3e0

```

## 4. Channel & Processor Architecture

```mermaid
graph TB
    subgraph "Channel Factory"
        CHAN_FACTORY["Channel Factory<br>– get_channel [type]<br>– register_channel<br>– validate_config"]
    end

    subgraph "Channel Implementations"
        BASE_CHANNEL["Base Channel<br><i>Abstract Interface</i>"]

        WEB_CHANNEL["Web Channel<br>– HTTP / WebSocket<br>– Real-time messaging<br>– File uploads"]

        WHATSAPP_CHANNEL["WhatsApp Channel<br>– Business API<br>– Media support<br>– Template messages"]

        SLACK_CHANNEL["Slack Channel<br>– Bot API<br>– Interactive buttons<br>– Threaded responses"]

        TEAMS_CHANNEL["Teams Channel<br>– Bot Framework<br>– Cards & actions<br>– Meeting integration"]

        MESSENGER_CHANNEL["Messenger Channel<br>– Graph API<br>– Persistent menu<br>– Quick replies"]
    end

    subgraph "Processor Factory"
        PROC_FACTORY["Processor Factory<br>– get_processor [type]<br>– register_processor<br>– validate_input"]
    end

    subgraph "Message Processors"
        BASE_PROCESSOR["Base Processor<br><i>Abstract Interface</i>"]

        TEXT_PROCESSOR["Text Processor<br>– Language detection<br>– Entity extraction<br>– Sentiment analysis<br>– Content safety"]

        MEDIA_PROCESSOR["Media Processor<br>– File validation<br>– Metadata extraction<br>– Virus scanning<br>– Format conversion"]

        LOCATION_PROCESSOR["Location Processor<br>– Coordinate validation<br>– Geocoding<br>– Privacy checks"]
    end

    subgraph "Normalizers"
        MSG_NORMALIZER["Message Normalizer<br>– Standardize format<br>– Remove noise<br>– Apply rules"]

        CONTENT_NORMALIZER["Content Normalizer<br>– Text cleaning<br>– Encoding fixes<br>– Character limits"]

        METADATA_NORMALIZER["Metadata Normalizer<br>– Extract headers<br>– User agent parsing<br>– IP validation"]
    end

    subgraph "Processing Pipeline"
        PIPELINE["Processing Pipeline<br>– Sequential stages<br>– Error handling<br>– Performance tracking"]

        subgraph "Pipeline Stages"
            VALIDATION["Input Validation"]
            NORMALIZATION["Content Normalization"]
            PROCESSING["Content Processing"]
            ENRICHMENT["Data Enrichment"]
            ANALYSIS["AI Analysis"]
        end
    end

    subgraph "Configuration"
        CHAN_CONFIG["Channel Config<br>– API credentials<br>– Rate limits<br>– Feature flags"]

        PROC_CONFIG["Processor Config<br>– Language models<br>– Safety thresholds<br>– Processing rules"]
    end

    %% Factory relationships
    CHAN_FACTORY --> BASE_CHANNEL
    PROC_FACTORY --> BASE_PROCESSOR

    %% Channel inheritance
    BASE_CHANNEL --> WEB_CHANNEL
    BASE_CHANNEL --> WHATSAPP_CHANNEL
    BASE_CHANNEL --> SLACK_CHANNEL
    BASE_CHANNEL --> TEAMS_CHANNEL
    BASE_CHANNEL --> MESSENGER_CHANNEL

    %% Processor inheritance
    BASE_PROCESSOR --> TEXT_PROCESSOR
    BASE_PROCESSOR --> MEDIA_PROCESSOR
    BASE_PROCESSOR --> LOCATION_PROCESSOR

    %% Pipeline flow
    PIPELINE --> VALIDATION
    VALIDATION --> NORMALIZATION
    NORMALIZATION --> PROCESSING
    PROCESSING --> ENRICHMENT
    ENRICHMENT --> ANALYSIS

    %% Normalizer usage
    NORMALIZATION --> MSG_NORMALIZER
    NORMALIZATION --> CONTENT_NORMALIZER
    NORMALIZATION --> METADATA_NORMALIZER

    %% Processor usage in pipeline
    PROCESSING --> TEXT_PROCESSOR
    PROCESSING --> MEDIA_PROCESSOR
    PROCESSING --> LOCATION_PROCESSOR

    %% Configuration
    CHAN_FACTORY --> CHAN_CONFIG
    PROC_FACTORY --> PROC_CONFIG

    %% Channel to processor flow
    WEB_CHANNEL -.-> PIPELINE
    WHATSAPP_CHANNEL -.-> PIPELINE
    SLACK_CHANNEL -.-> PIPELINE
    TEAMS_CHANNEL -.-> PIPELINE
    MESSENGER_CHANNEL -.-> PIPELINE

    %% Styles
    style BASE_CHANNEL fill:#e1f5fe
    style BASE_PROCESSOR fill:#f3e5f5
    style PIPELINE fill:#e8f5e8
    style CHAN_FACTORY fill:#fff3e0
    style PROC_FACTORY fill:#fff3e0

```

## 5. Event & Integration Architecture

```mermaid
graph TB
    %% === Event Sources ===
    subgraph "Event Sources"
        MSG_SERVICE["Message Service"]
        CONV_SERVICE["Conversation Service"]
        SESS_SERVICE["Session Service"]
        WEBHOOK_PROC["Webhook Processors"]
        HEALTH_CHECK["Health Checks"]
    end

    %% === Event Management ===
    subgraph "Event Management"
        EVENT_MANAGER["Event Manager<br>• register_handler()<br>• publish()<br>• process_event()"]
        EVENT_ROUTER["Event Router<br>• Route by type<br>• Load balancing<br>• Dead letter queue"]
    end

    %% === Event Types ===
    subgraph "Event Types"
        subgraph "Domain Events"
            MSG_RECEIVED["Message Received Event"]
            MSG_SENT["Message Sent Event"]
            MSG_DELIVERED["Message Delivered Event"]
            CONV_STARTED["Conversation Started Event"]
            CONV_ENDED["Conversation Ended Event"]
            CONV_UPDATED["Conversation Updated Event"]
        end

        subgraph "System Events"
            HEALTH_EVENT["Service Health Event"]
            ERROR_EVENT["Error Occurred Event"]
            PERFORMANCE_EVENT["Performance Event"]
        end

        subgraph "Integration Events"
            WEBHOOK_EVENT["Webhook Received Event"]
            API_CALL_EVENT["External API Call Event"]
            AUTH_EVENT["Authentication Event"]
        end
    end

    %% === Event Infrastructure ===
    subgraph "Event Infrastructure"
        KAFKA_PRODUCER["Kafka Producer<br>• Reliable publishing<br>• Batching<br>• Partitioning"]
        KAFKA_CONSUMER["Kafka Consumer<br>• Group processing<br>• Offset management<br>• Error handling"]
        KAFKA_TOPICS["Kafka Topics<br>• chat.message.*<br>• chat.conversation.*<br>• system.*<br>• integration.*"]
    end

    %% === Event Handlers ===
    subgraph "Event Handlers"
        ANALYTICS_HANDLER["Analytics Handler<br>• Metrics aggregation<br>• KPI calculation<br>• Real-time dashboards"]
        NOTIFICATION_HANDLER["Notification Handler<br>• Email alerts<br>• SMS notifications<br>• Slack alerts"]
        AUDIT_HANDLER["Audit Handler<br>• Compliance logging<br>• Security monitoring<br>• Data retention"]
        WEBHOOK_HANDLER["Webhook Handler<br>• External notifications<br>• Third-party integration<br>• Custom triggers"]
    end

    %% === External Integrations ===
    subgraph "External Integrations"
        direction TB

        subgraph "gRPC Clients"
            MCP_CLIENT["MCP Engine Client<br>• Message processing<br>• Context updates<br>• Health checks"]
            SECURITY_CLIENT["Security Hub Client<br>• Token validation<br>• Permission checks<br>• User management"]
        end

        subgraph "Webhook Systems"
            WEBHOOK_SENDER["Webhook Sender<br>• Delivery queue<br>• Retry logic<br>• Signature generation"]
            WEBHOOK_RECEIVER["Webhook Receiver<br>• Signature validation<br>• Rate limiting<br>• Processing queue"]
        end

        subgraph "External Services"
            ANALYTICS_ENGINE["Analytics Engine"]
            NOTIFICATION_SERVICE["Notification Service"]
            COMPLIANCE_SERVICE["Compliance Service"]
            THIRD_PARTY_APIS["Third-party APIs"]
        end
    end

    %% === Dead Letter Queue ===
    subgraph "Dead Letter Queue"
        DLQ["Dead Letter Queue<br>• Failed events<br>• Retry mechanism<br>• Manual intervention"]
    end

    %% === Event Flow ===
    MSG_SERVICE --> EVENT_MANAGER
    CONV_SERVICE --> EVENT_MANAGER
    SESS_SERVICE --> EVENT_MANAGER
    WEBHOOK_PROC --> EVENT_MANAGER

    EVENT_MANAGER --> KAFKA_PRODUCER
    KAFKA_PRODUCER --> KAFKA_TOPICS
    KAFKA_TOPICS --> KAFKA_CONSUMER
    KAFKA_CONSUMER --> EVENT_ROUTER

    EVENT_ROUTER --> ANALYTICS_HANDLER
    EVENT_ROUTER --> NOTIFICATION_HANDLER
    EVENT_ROUTER --> AUDIT_HANDLER
    EVENT_ROUTER --> WEBHOOK_HANDLER
    EVENT_ROUTER --> DLQ
    DLQ -.-> EVENT_ROUTER

    ANALYTICS_HANDLER --> ANALYTICS_ENGINE
    NOTIFICATION_HANDLER --> NOTIFICATION_SERVICE
    AUDIT_HANDLER --> COMPLIANCE_SERVICE
    WEBHOOK_HANDLER --> WEBHOOK_SENDER

    WEBHOOK_SENDER --> THIRD_PARTY_APIS
    THIRD_PARTY_APIS --> WEBHOOK_RECEIVER

    MSG_SERVICE -.->|gRPC| MCP_CLIENT
    EVENT_MANAGER -.->|gRPC| SECURITY_CLIENT
    MCP_CLIENT -.-> MSG_SERVICE
    SECURITY_CLIENT -.-> EVENT_MANAGER

    %% === Styles ===
    style EVENT_MANAGER fill:#e3f2fd
    style KAFKA_TOPICS fill:#fff8e1
    style ANALYTICS_HANDLER fill:#e8f5e8
    style MCP_CLIENT fill:#ffebee
    style DLQ fill:#fce4ec

```

## Architecture Summary

### Key Components Overview

1. **API Layer**: FastAPI-based REST endpoints with comprehensive middleware stack
2. **Service Layer**: Business logic orchestration with dependency injection
3. **Core Logic**: Channel abstractions and message processors with factory patterns
4. **Data Layer**: MongoDB for persistence, Redis for caching and sessions
5. **Event System**: Kafka-based event streaming for real-time analytics
6. **External Integration**: gRPC clients and webhook handling for external services

### Design Principles

- **Multi-tenancy**: Tenant isolation at all levels
- **Scalability**: Horizontal scaling with stateless services
- **Reliability**: Circuit breakers, retries, and dead letter queues
- **Security**: JWT authentication, rate limiting, content validation
- **Observability**: Comprehensive logging, metrics, and health checks
- **Extensibility**: Factory patterns and plugin architecture

### Technology Stack

- **Framework**: FastAPI with async/await
- **Databases**: MongoDB (primary), Redis (cache/sessions)
- **Messaging**: Apache Kafka for event streaming
- **Integration**: gRPC for service communication
- **Monitoring**: Structured logging with metrics collection
- **Deployment**: Kubernetes-ready with containerization