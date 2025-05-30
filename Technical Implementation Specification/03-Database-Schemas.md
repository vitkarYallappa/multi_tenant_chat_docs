# Database Schemas
## Multi-Tenant AI Chatbot Platform

**Document:** 03-Database-Schemas.md  
**Version:** 2.0  
**Last Updated:** May 30, 2025

---

## Table of Contents

1. [Database Strategy](#database-strategy)
2. [PostgreSQL Schemas](#postgresql-schemas)
3. [MongoDB Collections](#mongodb-collections)
4. [Redis Data Structures](#redis-data-structures)
5. [TimescaleDB Tables](#timescaledb-tables)
6. [Data Migration Strategy](#data-migration-strategy)
7. [Backup and Recovery](#backup-and-recovery)

---

## Database Strategy

### Database Selection Rationale

| Database | Purpose | Rationale |
|----------|---------|-----------|
| **PostgreSQL 15+** | Configuration & Transactional Data | ACID compliance, mature ecosystem, JSON support |
| **MongoDB 7+** | Conversation & Message Storage | Flexible schema, horizontal scaling, document queries |
| **Redis 7 Cluster** | Caching & Session Management | High performance, rich data structures, pub/sub |
| **TimescaleDB 2.12+** | Time-Series Analytics | Time-series optimization, PostgreSQL compatibility |

### Data Distribution Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA DISTRIBUTION                         │
└─────────────────────────────────────────────────────────────────┘

PostgreSQL (OLTP):
├── Tenant configurations and metadata
├── User profiles and authentication
├── Integration configurations  
├── System settings and admin data
└── Audit logs and compliance data

MongoDB (Document Store):
├── Conversation threads and metadata
├── Message content and attachments
├── Knowledge base documents
├── Search indexes and analytics
└── File metadata and references

Redis (Cache/Session):
├── Active conversation state
├── User session data
├── Rate limiting counters
├── Configuration cache
└── Real-time analytics

TimescaleDB (Time-Series):
├── Performance metrics over time
├── Business analytics data
├── Usage tracking and billing
├── System monitoring data
└── Custom tenant metrics
```

---

## PostgreSQL Schemas

### Core Tenant Management

#### Tenants Table
```sql
CREATE TABLE tenants (
    tenant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    subdomain VARCHAR(100) UNIQUE,
    status VARCHAR(20) DEFAULT 'active' 
        CHECK (status IN ('active', 'suspended', 'deleted', 'trial')),
    plan_type VARCHAR(50) DEFAULT 'starter' 
        CHECK (plan_type IN ('starter', 'professional', 'enterprise', 'custom')),
    
    -- Subscription and billing
    stripe_customer_id VARCHAR(100),
    subscription_id VARCHAR(100),
    billing_email VARCHAR(320),
    billing_cycle VARCHAR(20) DEFAULT 'monthly',
    trial_ends_at TIMESTAMP WITH TIME ZONE,
    
    -- Feature configuration
    features JSONB DEFAULT '{}',
    quotas JSONB DEFAULT '{
        "conversations_per_month": 10000,
        "api_calls_per_minute": 100,
        "storage_gb": 10,
        "integrations": 5,
        "team_members": 5,
        "channels": ["web", "whatsapp"]
    }',
    
    -- Compliance and security
    compliance_level VARCHAR(50) DEFAULT 'standard',
    data_residency VARCHAR(10) DEFAULT 'us',
    encryption_key_id VARCHAR(100),
    
    -- Customization
    branding JSONB DEFAULT '{}',
    custom_domain VARCHAR(255),
    white_label_enabled BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP WITH TIME ZONE,
    
    -- Organization info
    contact_info JSONB DEFAULT '{}',
    organization_size VARCHAR(20),
    industry VARCHAR(100),
    
    CONSTRAINT valid_billing_email 
        CHECK (billing_email ~* '^[A-Za-z0-9._%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Indexes for performance
CREATE INDEX idx_tenants_status ON tenants(status) WHERE status = 'active';
CREATE INDEX idx_tenants_plan ON tenants(plan_type);
CREATE INDEX idx_tenants_subdomain ON tenants(subdomain) WHERE subdomain IS NOT NULL;
CREATE INDEX idx_tenants_activity ON tenants(last_activity_at DESC);
```

#### Users Table
```sql
CREATE TABLE tenant_users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    
    -- Basic authentication
    email VARCHAR(320) NOT NULL,
    username VARCHAR(100),
    password_hash VARCHAR(255),
    salt VARCHAR(255),
    
    -- Multi-factor authentication
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(100),
    mfa_backup_codes TEXT[],
    
    -- Profile information
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    avatar_url VARCHAR(500),
    phone VARCHAR(20),
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(5) DEFAULT 'en',
    
    -- Role and permissions
    role VARCHAR(50) DEFAULT 'member' 
        CHECK (role IN ('owner', 'admin', 'developer', 'manager', 'member', 'viewer')),
    permissions JSONB DEFAULT '[]',
    custom_permissions JSONB DEFAULT '{}',
    
    -- Account status
    status VARCHAR(20) DEFAULT 'active' 
        CHECK (status IN ('active', 'suspended', 'deleted', 'pending')),
    email_verified BOOLEAN DEFAULT FALSE,
    email_verification_token VARCHAR(255),
    password_reset_token VARCHAR(255),
    password_reset_expires TIMESTAMP WITH TIME ZONE,
    
    -- Security tracking
    last_login_at TIMESTAMP WITH TIME ZONE,
    last_password_change TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional data
    profile JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    
    UNIQUE(tenant_id, email),
    UNIQUE(tenant_id, username) WHERE username IS NOT NULL
);

-- Indexes
CREATE INDEX idx_tenant_users_tenant ON tenant_users(tenant_id);
CREATE INDEX idx_tenant_users_email ON tenant_users(email);
CREATE INDEX idx_tenant_users_status ON tenant_users(status) WHERE status = 'active';
CREATE INDEX idx_tenant_users_role ON tenant_users(tenant_id, role);
```

#### API Keys Table
```sql
CREATE TABLE api_keys (
    key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    created_by UUID REFERENCES tenant_users(user_id),
    
    -- Key information
    name VARCHAR(255) NOT NULL,
    description TEXT,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_prefix VARCHAR(20) NOT NULL, -- First 16 chars for identification
    
    -- Permissions and limits
    permissions JSONB DEFAULT '[]',
    scopes JSONB DEFAULT '[]',
    rate_limit_per_minute INTEGER DEFAULT 1000,
    daily_quota INTEGER,
    monthly_quota INTEGER,
    
    -- Security settings
    allowed_ips INET[],
    allowed_origins TEXT[],
    require_https BOOLEAN DEFAULT TRUE,
    
    -- Status and expiration
    status VARCHAR(20) DEFAULT 'active' 
        CHECK (status IN ('active', 'revoked', 'expired')),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Usage tracking
    last_used_at TIMESTAMP WITH TIME ZONE,
    total_requests BIGINT DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_status ON api_keys(status) WHERE status = 'active';
```

### Conversation Flow Management

#### Conversation Flows Table
```sql
CREATE TABLE conversation_flows (
    flow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    
    -- Flow metadata
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(50) DEFAULT '1.0',
    
    -- Flow definition and configuration
    flow_definition JSONB NOT NULL,
    trigger_conditions JSONB DEFAULT '{}',
    fallback_flow_id UUID REFERENCES conversation_flows(flow_id),
    
    -- Status and lifecycle
    status VARCHAR(20) DEFAULT 'draft' 
        CHECK (status IN ('draft', 'active', 'inactive', 'archived')),
    is_default BOOLEAN DEFAULT FALSE,
    
    -- A/B Testing
    ab_test_enabled BOOLEAN DEFAULT FALSE,
    ab_test_config JSONB DEFAULT '{}',
    
    -- Analytics and performance
    usage_count BIGINT DEFAULT 0,
    success_rate DECIMAL(5,4),
    avg_completion_time INTERVAL,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit trail
    created_by UUID REFERENCES tenant_users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES tenant_users(user_id),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(tenant_id, name, version),
    UNIQUE(tenant_id, is_default) WHERE is_default = TRUE
);

-- Indexes
CREATE INDEX idx_flows_tenant ON conversation_flows(tenant_id);
CREATE INDEX idx_flows_status ON conversation_flows(status);
CREATE INDEX idx_flows_default ON conversation_flows(tenant_id, is_default) WHERE is_default = TRUE;
CREATE INDEX idx_flows_usage ON conversation_flows(usage_count DESC);
```

### Integration Management

#### Integrations Table
```sql
CREATE TABLE integrations (
    integration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    
    -- Integration metadata
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(100) NOT NULL, -- rest_api, graphql, webhook, database, etc.
    category VARCHAR(100), -- crm, ecommerce, support, analytics, etc.
    version VARCHAR(50) DEFAULT '1.0',
    
    -- Configuration
    configuration JSONB NOT NULL,
    credentials JSONB, -- Encrypted sensitive data
    endpoint_url VARCHAR(1000),
    
    -- Security and connection settings
    authentication_type VARCHAR(50), -- oauth2, api_key, basic_auth, jwt
    ssl_verification BOOLEAN DEFAULT TRUE,
    timeout_seconds INTEGER DEFAULT 30,
    retry_config JSONB DEFAULT '{
        "max_retries": 3,
        "backoff_strategy": "exponential",
        "initial_delay_ms": 1000
    }',
    
    -- Status and health monitoring
    status VARCHAR(20) DEFAULT 'active' 
        CHECK (status IN ('active', 'inactive', 'error', 'testing')),
    health_status VARCHAR(20) DEFAULT 'unknown',
    last_health_check TIMESTAMP WITH TIME ZONE,
    consecutive_failures INTEGER DEFAULT 0,
    last_error TEXT,
    last_success_at TIMESTAMP WITH TIME ZONE,
    
    -- Performance metrics
    total_calls BIGINT DEFAULT 0,
    successful_calls BIGINT DEFAULT 0,
    failed_calls BIGINT DEFAULT 0,
    avg_response_time_ms INTEGER,
    
    -- Testing and validation
    test_cases JSONB DEFAULT '[]',
    last_tested_at TIMESTAMP WITH TIME ZONE,
    test_results JSONB,
    
    -- Audit information
    created_by UUID REFERENCES tenant_users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES tenant_users(user_id),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(tenant_id, name)
);

-- Indexes
CREATE INDEX idx_integrations_tenant ON integrations(tenant_id);
CREATE INDEX idx_integrations_status ON integrations(status);
CREATE INDEX idx_integrations_type ON integrations(type);
CREATE INDEX idx_integrations_health ON integrations(health_status);
CREATE INDEX idx_integrations_category ON integrations(category);
```

### Model Configurations

#### Model Configurations Table
```sql
CREATE TABLE model_configurations (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    
    -- Configuration metadata
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_default BOOLEAN DEFAULT FALSE,
    
    -- Model preferences and routing
    model_preferences JSONB NOT NULL DEFAULT '{
        "intent_detection": {
            "primary": {"provider": "openai", "model": "gpt-4-turbo"},
            "fallback": {"provider": "anthropic", "model": "claude-3-haiku"}
        },
        "response_generation": {
            "primary": {"provider": "anthropic", "model": "claude-3-sonnet"},
            "fallback": {"provider": "openai", "model": "gpt-3.5-turbo"}
        },
        "entity_extraction": {
            "primary": {"provider": "openai", "model": "gpt-4-turbo"}
        },
        "sentiment_analysis": {
            "primary": {"provider": "huggingface", "model": "sentiment-roberta"}
        }
    }',
    
    -- Cost and performance controls
    cost_limits JSONB DEFAULT '{
        "daily_limit_cents": 1000,
        "monthly_limit_cents": 10000,
        "per_request_limit_cents": 10,
        "cost_optimization_enabled": true
    }',
    
    performance_targets JSONB DEFAULT '{
        "max_latency_ms": 3000,
        "quality_threshold": 0.8,
        "fallback_threshold": 0.6
    }',
    
    -- Advanced settings
    advanced_settings JSONB DEFAULT '{
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }',
    
    -- Usage tracking
    usage_stats JSONB DEFAULT '{}',
    total_requests BIGINT DEFAULT 0,
    total_cost_cents BIGINT DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(tenant_id, name),
    UNIQUE(tenant_id, is_default) WHERE is_default = TRUE
);

-- Indexes
CREATE INDEX idx_model_configs_tenant ON model_configurations(tenant_id);
CREATE INDEX idx_model_configs_default ON model_configurations(tenant_id, is_default) WHERE is_default = TRUE;
CREATE INDEX idx_model_configs_usage ON model_configurations(last_used_at DESC);
```

### Analytics and Monitoring

#### Usage Metrics Table
```sql
CREATE TABLE usage_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    
    -- Time dimensions
    metric_date DATE NOT NULL,
    metric_hour INTEGER CHECK (metric_hour >= 0 AND metric_hour <= 23),
    
    -- Metric classification
    metric_type VARCHAR(100) NOT NULL,
    metric_category VARCHAR(50), -- api, model, storage, integration, conversation
    metric_subcategory VARCHAR(50),
    
    -- Metric values and aggregations
    metric_value BIGINT DEFAULT 0,
    metric_sum BIGINT DEFAULT 0,
    metric_count INTEGER DEFAULT 0,
    metric_avg DECIMAL(10,4),
    metric_max BIGINT,
    metric_min BIGINT,
    
    -- Additional dimensions for analysis
    channel VARCHAR(50),
    model_provider VARCHAR(50),
    integration_name VARCHAR(255),
    user_segment VARCHAR(50),
    
    -- Cost and resource tracking
    cost_cents DECIMAL(10,4) DEFAULT 0,
    tokens_used BIGINT DEFAULT 0,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint to prevent duplicates
    UNIQUE(tenant_id, metric_date, metric_hour, metric_type, metric_category, 
           channel, model_provider, integration_name, user_segment)
);

-- Indexes for performance
CREATE INDEX idx_usage_metrics_tenant_date ON usage_metrics(tenant_id, metric_date DESC);
CREATE INDEX idx_usage_metrics_type ON usage_metrics(metric_type, metric_date DESC);
CREATE INDEX idx_usage_metrics_category ON usage_metrics(metric_category, metric_date DESC);
CREATE INDEX idx_usage_metrics_cost ON usage_metrics(tenant_id, cost_cents DESC);
```

#### Audit Logs Table
```sql
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    user_id UUID REFERENCES tenant_users(user_id) ON DELETE SET NULL,
    
    -- Event classification
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(50), -- auth, config, data, admin, api
    event_subcategory VARCHAR(50),
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    
    -- Event details
    action VARCHAR(100) NOT NULL,
    description TEXT,
    result VARCHAR(20) DEFAULT 'success', -- success, failure, error
    
    -- Request context
    ip_address INET,
    user_agent TEXT,
    request_id UUID,
    session_id UUID,
    
    -- Change tracking (for update events)
    old_values JSONB,
    new_values JSONB,
    
    -- Error information
    error_code VARCHAR(50),
    error_message TEXT,
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for audit queries
CREATE INDEX idx_audit_tenant_time ON audit_logs(tenant_id, created_at DESC);
CREATE INDEX idx_audit_user_time ON audit_logs(user_id, created_at DESC);
CREATE INDEX idx_audit_event_type ON audit_logs(event_type, created_at DESC);
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_ip ON audit_logs(ip_address, created_at DESC);
```

### Database Functions and Triggers

#### Updated Timestamp Trigger
```sql
-- Function to update updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to relevant tables
CREATE TRIGGER update_tenants_updated_at 
    BEFORE UPDATE ON tenants 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tenant_users_updated_at 
    BEFORE UPDATE ON tenant_users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_keys_updated_at 
    BEFORE UPDATE ON api_keys 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversation_flows_updated_at 
    BEFORE UPDATE ON conversation_flows 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_integrations_updated_at 
    BEFORE UPDATE ON integrations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_configurations_updated_at 
    BEFORE UPDATE ON model_configurations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

#### Tenant Activity Tracking
```sql
-- Function to update tenant last_activity_at
CREATE OR REPLACE FUNCTION update_tenant_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE tenants 
    SET last_activity_at = CURRENT_TIMESTAMP 
    WHERE tenant_id = NEW.tenant_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to relevant activity tables
CREATE TRIGGER update_tenant_activity_on_user_login 
    AFTER UPDATE OF last_login_at ON tenant_users 
    FOR EACH ROW EXECUTE FUNCTION update_tenant_activity();
```

---

## MongoDB Collections

### Conversations Collection

```javascript
// Conversations Collection Schema
{
  "_id": ObjectId,
  "conversation_id": "uuid_v4",
  "tenant_id": "uuid_v4",
  "user_id": "string",
  "session_id": "uuid_v4",
  
  // Channel and context
  "channel": "web|whatsapp|messenger|slack|teams|voice|sms",
  "channel_metadata": {
    "platform_user_id": "string",
    "platform_channel_id": "string",
    "thread_id": "string",
    "bot_id": "string",
    "workspace_id": "string"
  },
  
  // Conversation lifecycle
  "status": "active|completed|abandoned|escalated|error",
  "started_at": ISODate,
  "last_activity_at": ISODate,
  "completed_at": ISODate,
  "duration_seconds": NumberLong,
  
  // Flow and state management
  "flow_id": "uuid_v4",
  "flow_version": "string",
  "current_state": "string",
  "previous_states": ["array_of_state_names"],
  "state_history": [
    {
      "state": "string",
      "entered_at": ISODate,
      "exited_at": ISODate,
      "duration_ms": NumberLong
    }
  ],
  
  // Conversation context
  "context": {
    "intent_history": ["array_of_intents"],
    "current_intent": "string",
    "intent_confidence": NumberDecimal,
    "entities": {},
    "slots": {},
    "user_profile": {},
    "session_variables": {},
    "custom_attributes": {},
    "conversation_tags": ["array_of_tags"]
  },
  
  // User information (privacy compliant)
  "user_info": {
    "first_seen": ISODate,
    "return_visitor": Boolean,
    "language": "ISO_639-1",
    "timezone": "string",
    "device_info": {
      "type": "mobile|desktop|tablet|voice",
      "os": "string",
      "browser": "string"
    },
    "location": {
      "country": "string",
      "region": "string",
      "city": "string", // Anonymized if needed
      "coordinates": {
        "lat": NumberDecimal,
        "lng": NumberDecimal
      }
    }
  },
  
  // Conversation quality and metrics
  "metrics": {
    "message_count": NumberInt,
    "user_messages": NumberInt,
    "bot_messages": NumberInt,
    "response_time_avg_ms": NumberInt,
    "response_time_max_ms": NumberInt,
    "intent_switches": NumberInt,
    "escalation_triggers": NumberInt,
    "user_satisfaction": {
      "score": NumberDecimal, // 1-5 rating
      "feedback": "string",
      "collected_at": ISODate
    },
    "completion_rate": NumberDecimal,
    "goal_achieved": Boolean
  },
  
  // AI and model metadata
  "ai_metadata": {
    "primary_models_used": ["string"],
    "fallback_models_used": ["string"],
    "total_cost_cents": NumberDecimal,
    "total_tokens": NumberLong,
    "average_confidence": NumberDecimal,
    "quality_scores": {
      "relevance": NumberDecimal,
      "helpfulness": NumberDecimal,
      "accuracy": NumberDecimal,
      "coherence": NumberDecimal
    }
  },
  
  // Business and operational context
  "business_context": {
    "department": "sales|support|marketing|general",
    "category": "string",
    "subcategory": "string",
    "priority": "low|normal|high|urgent|critical",
    "tags": ["array_of_business_tags"],
    "resolution_type": "automated|escalated|abandoned|timeout",
    "outcome": "resolved|unresolved|pending|escalated",
    "value_generated": NumberDecimal,
    "cost_incurred": NumberDecimal
  },
  
  // Compliance and privacy
  "compliance": {
    "pii_detected": Boolean,
    "pii_masked": Boolean,
    "pii_types": ["email", "phone", "ssn", "credit_card"],
    "data_retention_until": ISODate,
    "anonymization_level": "none|partial|full",
    "gdpr_flags": ["array_of_flags"],
    "audit_required": Boolean,
    "consent_collected": Boolean,
    "consent_details": {}
  },
  
  // Integration and external system data
  "integrations_used": [
    {
      "integration_id": "uuid",
      "integration_name": "string",
      "calls_made": NumberInt,
      "success_rate": NumberDecimal,
      "total_cost_cents": NumberDecimal
    }
  ],
  
  // Summary and analysis
  "summary": {
    "auto_generated_summary": "string",
    "key_topics": ["array_of_topics"],
    "entities_mentioned": ["array_of_entities"],
    "action_items": ["array_of_actions"],
    "follow_up_required": Boolean,
    "follow_up_date": ISODate,
    "escalation_reason": "string",
    "human_notes": "string"
  },
  
  // A/B testing information
  "ab_testing": {
    "experiment_id": "string",
    "variant": "string",
    "control_group": Boolean
  }
}

// Indexes for Conversations Collection
db.conversations.createIndex({"tenant_id": 1, "started_at": -1});
db.conversations.createIndex({"conversation_id": 1});
db.conversations.createIndex({"tenant_id": 1, "user_id": 1, "started_at": -1});
db.conversations.createIndex({"tenant_id": 1, "status": 1, "last_activity_at": -1});
db.conversations.createIndex({"tenant_id": 1, "channel": 1, "started_at": -1});
db.conversations.createIndex({"tenant_id": 1, "business_context.category": 1});
db.conversations.createIndex({"tenant_id": 1, "business_context.outcome": 1});
db.conversations.createIndex({"compliance.data_retention_until": 1}); // For cleanup
```

### Messages Collection

```javascript
// Messages Collection Schema
{
  "_id": ObjectId,
  "message_id": "uuid_v4",
  "conversation_id": "uuid_v4",
  "tenant_id": "uuid_v4",
  "user_id": "string",
  
  // Message metadata
  "sequence_number": NumberInt,
  "direction": "inbound|outbound",
  "timestamp": ISODate,
  "channel": "string",
  "message_type": "text|image|file|audio|video|location|quick_reply|carousel|form|system",
  
  // Content with comprehensive structure
  "content": {
    // Text content
    "text": "string",
    "original_text": "string", // Before any processing/translation
    "translated_text": "string", // If translation applied
    "language": "ISO_639-1",
    "language_confidence": NumberDecimal,
    
    // Rich media content
    "media": {
      "url": "string",
      "secure_url": "string",
      "type": "string", // MIME type
      "size_bytes": NumberLong,
      "duration_ms": NumberLong, // For audio/video
      "dimensions": {
        "width": NumberInt,
        "height": NumberInt
      },
      "thumbnail_url": "string",
      "alt_text": "string",
      "caption": "string",
      // Audio/Video specific
      "transcript": "string",
      "transcript_confidence": NumberDecimal,
      // Image specific
      "ocr_text": "string",
      "ocr_confidence": NumberDecimal,
      "detected_objects": ["array_of_objects"],
      // File specific
      "filename": "string",
      "file_extension": "string"
    },
    
    // Location data
    "location": {
      "latitude": NumberDecimal,
      "longitude": NumberDecimal,
      "accuracy_meters": NumberInt,
      "address": "string",
      "place_name": "string",
      "place_id": "string"
    },
    
    // Interactive elements
    "quick_replies": [
      {
        "title": "string",
        "payload": "string",
        "content_type": "text|location|phone|email",
        "clicked": Boolean,
        "click_timestamp": ISODate
      }
    ],
    
    "buttons": [
      {
        "type": "postback|url|phone|share|login",
        "title": "string",
        "payload": "string",
        "url": "string",
        "clicked": Boolean,
        "click_timestamp": ISODate
      }
    ],
    
    "carousel": [
      {
        "title": "string",
        "subtitle": "string",
        "image_url": "string",
        "buttons": ["array_of_buttons"]
      }
    ],
    
    // Form data
    "form": {
      "form_id": "string",
      "form_data": {},
      "validation_status": "valid|invalid|pending",
      "submitted_at": ISODate
    }
  },
  
  // AI processing results
  "ai_analysis": {
    // Intent detection
    "intent": {
      "detected_intent": "string",
      "confidence": NumberDecimal,
      "alternatives": [
        {
          "intent": "string",
          "confidence": NumberDecimal
        }
      ]
    },
    
    // Entity extraction
    "entities": [
      {
        "entity": "string",
        "value": "string",
        "start": NumberInt,
        "end": NumberInt,
        "confidence": NumberDecimal,
        "resolution": {},
        "source": "user_input|context|integration"
      }
    ],
    
    // Sentiment analysis
    "sentiment": {
      "label": "positive|negative|neutral",
      "score": NumberDecimal, // -1 to 1
      "confidence": NumberDecimal,
      "emotions": {
        "joy": NumberDecimal,
        "anger": NumberDecimal,
        "fear": NumberDecimal,
        "sadness": NumberDecimal,
        "surprise": NumberDecimal,
        "disgust": NumberDecimal
      }
    },
    
    // Topic and keyword extraction
    "topics": ["array_of_topics"],
    "keywords": ["array_of_keywords"],
    "categories": ["array_of_categories"],
    
    // Content analysis
    "toxicity": {
      "is_toxic": Boolean,
      "toxicity_score": NumberDecimal,
      "categories": ["harassment", "hate_speech", "spam"]
    },
    
    "quality": {
      "grammar_score": NumberDecimal,
      "readability_score": NumberDecimal,
      "completeness_score": NumberDecimal
    }
  },
  
  // Generation metadata (for outbound messages)
  "generation_metadata": {
    "model_provider": "string",
    "model_name": "string",
    "model_version": "string",
    "generation_config": {
      "temperature": NumberDecimal,
      "max_tokens": NumberInt,
      "top_p": NumberDecimal,
      "frequency_penalty": NumberDecimal,
      "presence_penalty": NumberDecimal
    },
    "tokens_used": {
      "input": NumberInt,
      "output": NumberInt,
      "total": NumberInt
    },
    "cost_cents": NumberDecimal,
    "generation_time_ms": NumberInt,
    "fallback_used": Boolean,
    "fallback_reason": "string",
    "template_used": "string",
    "personalization_applied": Boolean,
    "a_b_variant": "string"
  },
  
  // Channel-specific metadata
  "channel_metadata": {
    "platform_message_id": "string",
    "platform_timestamp": ISODate,
    "thread_id": "string",
    "parent_message_id": "string",
    "reply_to_message_id": "string",
    "forwarded": Boolean,
    "forwarded_from": "string",
    "edited": Boolean,
    "edited_at": ISODate,
    "delivery_status": "sent|delivered|read|failed",
    "delivery_timestamp": ISODate,
    "read_timestamp": ISODate,
    "delivery_attempts": NumberInt,
    "failure_reason": "string"
  },
  
  // Processing pipeline information
  "processing": {
    "pipeline_version": "string",
    "processing_stages": [
      {
        "stage": "string",
        "status": "success|error|skipped",
        "duration_ms": NumberInt,
        "error_details": "string",
        "started_at": ISODate,
        "completed_at": ISODate
      }
    ],
    "total_processing_time_ms": NumberInt,
    "queue_time_ms": NumberInt,
    "priority": "low|normal|high|urgent|critical",
    "retry_count": NumberInt,
    "last_retry_at": ISODate
  },
  
  // Quality assurance and feedback
  "quality_assurance": {
    "automated_quality_score": NumberDecimal,
    "human_quality_rating": NumberInt, // 1-5
    "quality_feedback": "string",
    "reported_issues": ["accuracy", "relevance", "tone", "grammar"],
    "improvement_suggestions": "string",
    "reviewed_by": "string",
    "reviewed_at": ISODate,
    "approved": Boolean
  },
  
  // Moderation and compliance
  "moderation": {
    "flagged": Boolean,
    "flags": ["spam", "inappropriate", "pii", "toxic", "off_topic"],
    "auto_moderated": Boolean,
    "human_reviewed": Boolean,
    "approved": Boolean,
    "moderator_id": "string",
    "moderator_notes": "string",
    "moderated_at": ISODate,
    "escalated": Boolean,
    "escalation_reason": "string"
  },
  
  // PII and privacy
  "privacy": {
    "contains_pii": Boolean,
    "pii_types": ["email", "phone", "ssn", "credit_card", "address"],
    "masked_content": "string", // PII-masked version
    "anonymization_level": "none|partial|full",
    "retention_category": "standard|extended|permanent",
    "auto_delete_at": ISODate
  }
}

// Indexes for Messages Collection
db.messages.createIndex({"conversation_id": 1, "sequence_number": 1});
db.messages.createIndex({"tenant_id": 1, "timestamp": -1});
db.messages.createIndex({"message_id": 1});
db.messages.createIndex({"tenant_id": 1, "direction": 1, "timestamp": -1});
db.messages.createIndex({"tenant_id": 1, "ai_analysis.intent.detected_intent": 1});
db.messages.createIndex({"content.message_type": 1, "timestamp": -1});
db.messages.createIndex({"privacy.auto_delete_at": 1}); // For cleanup
db.messages.createIndex({"moderation.flagged": 1, "timestamp": -1});
```

---

## Redis Data Structures

### Session Management

```redis
# Session Data Pattern
Key: session:{tenant_id}:{session_id}
Type: Hash
TTL: 3600 seconds (1 hour)
Fields:
  conversation_id: "uuid"
  user_id: "string"
  channel: "web|whatsapp|messenger"
  created_at: "timestamp"
  last_activity: "timestamp"
  context: "json_string"
  preferences: "json_string"

# Example Commands
HSET session:tenant123:sess456 conversation_id conv789
HSET session:tenant123:sess456 user_id user123
HSET session:tenant123:sess456 channel web
EXPIRE session:tenant123:sess456 3600
```

### Conversation State Management

```redis
# Conversation Context Pattern
Key: conversation:{tenant_id}:{conversation_id}:context
Type: Hash
TTL: 86400 seconds (24 hours)
Fields:
  current_state: "string"
  previous_state: "string"
  intent_stack: "json_array"
  slots: "json_object"
  variables: "json_object"
  flow_id: "uuid"
  last_updated: "timestamp"

# State Machine Execution Lock
Key: lock:conversation:{conversation_id}
Type: String
TTL: 30 seconds
Value: "processing_node_id"

# Example Commands
HSET conversation:tenant123:conv789:context current_state "waiting_for_order_number"
HSET conversation:tenant123:conv789:context slots '{"intent":"order_inquiry"}'
EXPIRE conversation:tenant123:conv789:context 86400
```

### Rate Limiting

```redis
# API Rate Limiting Pattern
Key: rate_limit:{tenant_id}:{api_key}:{window}
Type: Sorted Set
TTL: window_duration
Score: timestamp
Member: request_id

# Sliding Window Rate Limiting
Key: rate_limit:tenant123:key456:minute:202405301430
ZADD rate_limit:tenant123:key456:minute:202405301430 1717076400123 "req_uuid1"
ZADD rate_limit:tenant123:key456:minute:202405301430 1717076401456 "req_uuid2"
EXPIRE rate_limit:tenant123:key456:minute:202405301430 60

# Check rate limit
ZCOUNT rate_limit:tenant123:key456:minute:202405301430 1717076340000 1717076400000

# Token Bucket Rate Limiting
Key: token_bucket:{tenant_id}:{api_key}
Type: Hash
Fields:
  tokens: "current_token_count"
  last_refill: "timestamp"
  capacity: "bucket_capacity"
  refill_rate: "tokens_per_second"
```

### Caching Strategies

```redis
# Response Caching Pattern
Key: cache:response:{intent}:{context_hash}
Type: String (JSON)
TTL: 1800 seconds (30 minutes)
Value: "json_response_object"

# Configuration Caching
Key: cache:config:{tenant_id}:{config_type}
Type: Hash
TTL: 300 seconds (5 minutes)
Fields:
  data: "json_config_data"
  version: "config_version"
  last_updated: "timestamp"

# Model Response Caching
Key: cache:model:{model_provider}:{input_hash}
Type: Hash
TTL: 3600 seconds (1 hour)
Fields:
  response: "model_response"
  confidence: "confidence_score"
  cost_cents: "api_cost"
  created_at: "timestamp"
  hit_count: "usage_counter"
```

### Real-time Analytics

```redis
# Real-time Counters
Key: metrics:{tenant_id}:{metric_type}:{time_window}
Type: Hash
TTL: window_duration * 2
Fields:
  count: "counter_value"
  sum: "sum_value"
  min: "minimum_value"
  max: "maximum_value"
  avg: "average_value"

# Live Dashboard Metrics
Key: dashboard:{tenant_id}:realtime
Type: Hash
TTL: 300 seconds (5 minutes)
Fields:
  active_conversations: "count"
  messages_per_minute: "rate"
  average_response_time: "milliseconds"
  error_rate: "percentage"
  top_intents: "json_array"
  last_updated: "timestamp"
```

### Circuit Breaker State

```redis
# Circuit Breaker Pattern
Key: circuit_breaker:{service_name}:{tenant_id}
Type: Hash
TTL: 3600 seconds (1 hour)
Fields:
  state: "closed|open|half_open"
  failure_count: "number"
  success_count: "number"
  last_failure_time: "timestamp"
  last_success_time: "timestamp"
  next_attempt_time: "timestamp"
  
# Integration Health Monitoring
Key: integration_health:{integration_id}
Type: Hash
TTL: 600 seconds (10 minutes)
Fields:
  status: "healthy|degraded|unhealthy"
  last_check: "timestamp"
  response_time_ms: "number"
  error_rate: "percentage"
  consecutive_failures: "count"
```

---

## TimescaleDB Tables

### Performance Metrics

```sql
-- System Performance Metrics
CREATE TABLE system_metrics (
    time TIMESTAMPTZ NOT NULL,
    tenant_id UUID,
    service_name VARCHAR(100),
    metric_name VARCHAR(100),
    metric_value DOUBLE PRECISION,
    tags JSONB,
    
    PRIMARY KEY (time, tenant_id, service_name, metric_name)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('system_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Add indexes for common queries
CREATE INDEX idx_system_metrics_tenant_time ON system_metrics (tenant_id, time DESC);
CREATE INDEX idx_system_metrics_service ON system_metrics (service_name, time DESC);
CREATE INDEX idx_system_metrics_name ON system_metrics (metric_name, time DESC);
```

### Business Analytics

```sql
-- Conversation Analytics
CREATE TABLE conversation_analytics (
    time TIMESTAMPTZ NOT NULL,
    tenant_id UUID NOT NULL,
    channel VARCHAR(50),
    conversations_started INTEGER DEFAULT 0,
    conversations_completed INTEGER DEFAULT 0,
    messages_sent INTEGER DEFAULT 0,
    messages_received INTEGER DEFAULT 0,
    avg_response_time_ms DOUBLE PRECISION,
    completion_rate DOUBLE PRECISION,
    satisfaction_score DOUBLE PRECISION,
    escalation_rate DOUBLE PRECISION,
    cost_total_cents DOUBLE PRECISION,
    
    PRIMARY KEY (time, tenant_id, channel)
);

SELECT create_hypertable('conversation_analytics', 'time', chunk_time_interval => INTERVAL '1 day');

-- Model Usage Analytics
CREATE TABLE model_usage_analytics (
    time TIMESTAMPTZ NOT NULL,
    tenant_id UUID NOT NULL,
    model_provider VARCHAR(50),
    model_name VARCHAR(100),
    operation_type VARCHAR(50),
    request_count INTEGER DEFAULT 0,
    token_count BIGINT DEFAULT 0,
    cost_cents DOUBLE PRECISION DEFAULT 0,
    avg_latency_ms DOUBLE PRECISION,
    error_count INTEGER DEFAULT 0,
    
    PRIMARY KEY (time, tenant_id, model_provider, model_name, operation_type)
);

SELECT create_hypertable('model_usage_analytics', 'time', chunk_time_interval => INTERVAL '1 day');
```

### Custom Tenant Metrics

```sql
-- Custom Metrics Table
CREATE TABLE custom_metrics (
    time TIMESTAMPTZ NOT NULL,
    tenant_id UUID NOT NULL,
    metric_name VARCHAR(200) NOT NULL,
    metric_value DOUBLE PRECISION,
    dimensions JSONB,
    
    PRIMARY KEY (time, tenant_id, metric_name)
);

SELECT create_hypertable('custom_metrics', 'time', chunk_time_interval => INTERVAL '1 day');

-- Retention policies for different metric types
SELECT add_retention_policy('system_metrics', INTERVAL '90 days');
SELECT add_retention_policy('conversation_analytics', INTERVAL '2 years');
SELECT add_retention_policy('model_usage_analytics', INTERVAL '1 year');
SELECT add_retention_policy('custom_metrics', INTERVAL '6 months');
```

---

## Data Migration Strategy

### Version Control for Schemas

```sql
-- Migration tracking table
CREATE TABLE schema_migrations (
    migration_id VARCHAR(255) PRIMARY KEY,
    description TEXT,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    rollback_sql TEXT
);

-- Example migration structure
-- migrations/001_initial_schema.sql
-- migrations/002_add_mfa_support.sql
-- migrations/003_add_integration_marketplace.sql
```

### Data Migration Scripts

```python
# Example migration script
class Migration_002_Add_MFA_Support:
    def up(self):
        """Apply migration"""
        return """
        ALTER TABLE tenant_users 
        ADD COLUMN mfa_enabled BOOLEAN DEFAULT FALSE,
        ADD COLUMN mfa_secret VARCHAR(100),
        ADD COLUMN mfa_backup_codes TEXT[];
        
        CREATE INDEX idx_tenant_users_mfa ON tenant_users(mfa_enabled) 
        WHERE mfa_enabled = TRUE;
        """
    
    def down(self):
        """Rollback migration"""
        return """
        ALTER TABLE tenant_users 
        DROP COLUMN mfa_enabled,
        DROP COLUMN mfa_secret,
        DROP COLUMN mfa_backup_codes;
        """
```

---

## Backup and Recovery

### Backup Strategy

```yaml
PostgreSQL Backup:
  Type: Continuous WAL archiving + periodic full backups
  Frequency: 
    - Full backup: Daily at 2 AM UTC
    - WAL archiving: Continuous
  Retention: 30 days full + 90 days compressed
  Storage: AWS S3 with cross-region replication
  
MongoDB Backup:
  Type: Replica set with oplog + periodic snapshots
  Frequency:
    - Snapshot: Every 6 hours
    - Oplog backup: Continuous
  Retention: 30 days
  Storage: AWS S3 with versioning
  
Redis Backup:
  Type: RDB snapshots + AOF
  Frequency: Every 2 hours
  Retention: 7 days
  Storage: AWS S3
  
TimescaleDB Backup:
  Type: Continuous WAL + compressed chunks
  Frequency: 
    - Recent data: Continuous
    - Historical data: Compressed daily
  Retention: 90 days recent + 2 years compressed
```

### Recovery Procedures

```bash
# PostgreSQL Point-in-Time Recovery
pg_basebackup -h source_host -D /backup/base -U replication -v -P
# Restore to specific point in time
postgresql.conf: restore_command = 'aws s3 cp s3://backups/wal/%f %p'
recovery.conf: recovery_target_time = '2025-05-30 10:00:00'

# MongoDB Replica Set Recovery
mongorestore --host replica_set/host1:27017,host2:27017,host3:27017 \
  --archive=backup.archive --gzip

# Redis Recovery
redis-server --dbfilename dump.rdb --dir /backup/redis/
```



**Document Maintainer:** Database Architecture Team  
**Review Schedule:** Monthly during development, quarterly in production  
**Related Documents:** System Architecture, API Specifications, Security Implementation