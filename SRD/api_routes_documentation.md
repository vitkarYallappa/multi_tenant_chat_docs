# API Routes Documentation
## Multi-Tenant AI Chatbot Platform - Complete Route Specifications

**Version:** 2.0  
**Last Updated:** June 1, 2025  
**Document Type:** Technical Implementation Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Chat Routes (`chat_routes.py`)](#chat-routes)
3. [Conversation Routes (`conversation_routes.py`)](#conversation-routes)
4. [Session Routes (`session_routes.py`)](#session-routes)
5. [Health Routes (`health_routes.py`)](#health-routes)
6. [Webhook Routes (`webhook_routes.py`)](#webhook-routes)
7. [Tenant Routes (`tenant_routes.py`)](#tenant-routes)
8. [Common Patterns & Error Handling](#common-patterns--error-handling)

---

## Overview

The Chat Service API is organized into specialized route modules, each handling specific aspects of the chatbot platform. All routes follow RESTful conventions and include comprehensive error handling, authentication, and tenant isolation.

### Base URL Structure
```
https://api.chatbot-platform.com/api/v2/{route_group}/
```

### Common Headers
```http
Content-Type: application/json
Authorization: Bearer {jwt_token}
X-Tenant-ID: {tenant_uuid}
X-Request-ID: {optional_trace_id}
```

---

## Chat Routes (`chat_routes.py`)

**Purpose:** Core message processing and real-time chat functionality

### Route Overview

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/chat/message` | POST | Send message and get bot response | Required |
| `/chat/message/{message_id}` | GET | Retrieve specific message details | Required |
| `/chat/message/{message_id}/retry` | POST | Retry failed message processing | Required |
| `/chat/typing` | POST | Send typing indicators | Required |
| `/chat/feedback` | POST | Submit user feedback for response | Required |

### Detailed Examples

#### 1. Send Message - Basic Text Query

**Scenario:** Customer asks about store hours on website chat widget

```http
POST /api/v2/chat/message
Content-Type: application/json
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
X-Tenant-ID: 12345678-1234-1234-1234-123456789012

{
  "message_id": "msg_98765432-1234-1234-1234-123456789012",
  "conversation_id": "conv_11111111-2222-3333-4444-555555555555",
  "user_id": "user_customer123",
  "channel": "web",
  "content": {
    "type": "text",
    "text": "What are your store hours?",
    "language": "en"
  },
  "channel_metadata": {
    "platform_user_id": "web_visitor_456",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "referrer": "https://mystore.com/products"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "message_id": "msg_98765432-1234-1234-1234-123456789012",
    "conversation_id": "conv_11111111-2222-3333-4444-555555555555",
    "response": {
      "type": "text",
      "text": "Our store hours are Monday-Friday 9 AM to 8 PM, Saturday 10 AM to 6 PM, and Sunday 12 PM to 5 PM. Is there anything specific you'd like to know?",
      "quick_replies": [
        {
          "title": "Store Location",
          "payload": "get_store_location"
        },
        {
          "title": "Holiday Hours",
          "payload": "get_holiday_hours"
        }
      ]
    },
    "conversation_state": {
      "current_intent": "store_hours_inquiry",
      "next_expected_input": "follow_up_question",
      "conversation_stage": "information_provided"
    },
    "processing_metadata": {
      "model_used": "gpt-4-turbo",
      "processing_time_ms": 287,
      "confidence_score": 0.94
    }
  }
}
```

#### 2. Send Message with Media - WhatsApp Image

**Scenario:** Customer sends a photo of a damaged product via WhatsApp

```http
POST /api/v2/chat/message

{
  "message_id": "msg_img_123456789",
  "user_id": "whatsapp_+1234567890",
  "channel": "whatsapp",
  "content": {
    "type": "image",
    "text": "My order arrived damaged, please help!",
    "media": {
      "url": "https://media.whatsapp.com/v1/media/abc123def456",
      "type": "image/jpeg",
      "size_bytes": 245760,
      "alt_text": "Damaged product package"
    }
  },
  "channel_metadata": {
    "platform_message_id": "wamid.abc123def456",
    "platform_user_id": "+1234567890"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "response": {
      "type": "text",
      "text": "I'm sorry to see your order arrived damaged! I can see the issue in your photo. Let me help you with a replacement or refund. Could you please provide your order number?",
      "quick_replies": [
        {
          "title": "I have my order number",
          "payload": "provide_order_number"
        },
        {
          "title": "Help me find it",
          "payload": "help_find_order"
        }
      ]
    },
    "conversation_state": {
      "current_intent": "damaged_product_complaint",
      "entities": {
        "issue_type": "damaged_product",
        "evidence_provided": true
      }
    }
  }
}
```

#### 3. Typing Indicator

**Scenario:** Show user that bot is processing a complex request

```http
POST /api/v2/chat/typing

{
  "conversation_id": "conv_11111111-2222-3333-4444-555555555555",
  "user_id": "user_customer123",
  "channel": "web",
  "typing_duration_ms": 3000
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "typing_sent": true,
    "duration_ms": 3000
  }
}
```

---

## Conversation Routes (`conversation_routes.py`)

**Purpose:** Manage conversation lifecycle, history, and analytics

### Route Overview

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/conversations` | GET | List conversations with filters | Required |
| `/conversations/{id}` | GET | Get conversation details | Required |
| `/conversations/{id}/export` | POST | Export conversation data | Required |
| `/conversations/{id}/transfer` | POST | Transfer to human agent | Required |
| `/conversations/{id}/close` | POST | Close conversation | Required |
| `/conversations/{id}/reopen` | POST | Reopen closed conversation | Required |
| `/conversations/{id}/summary` | GET | Get conversation summary | Required |
| `/conversations/{id}/analytics` | GET | Get conversation analytics | Required |

### Detailed Examples

#### 1. List Conversations with Filters

**Scenario:** Support manager wants to see all active WhatsApp conversations from today

```http
GET /api/v2/conversations?status=active&channel=whatsapp&date_from=2025-06-01&limit=50&sort=last_activity_desc
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "conversations": [
      {
        "conversation_id": "conv_active_001",
        "user_id": "whatsapp_+1234567890",
        "channel": "whatsapp",
        "status": "active",
        "started_at": "2025-06-01T10:30:00Z",
        "last_activity_at": "2025-06-01T14:22:00Z",
        "message_count": 8,
        "current_intent": "order_status_inquiry",
        "summary": {
          "primary_issue": "Order tracking",
          "resolution_status": "in_progress",
          "customer_satisfaction": null
        }
      },
      {
        "conversation_id": "conv_active_002",
        "user_id": "whatsapp_+9876543210",
        "channel": "whatsapp",
        "status": "active",
        "started_at": "2025-06-01T12:15:00Z",
        "last_activity_at": "2025-06-01T14:20:00Z",
        "message_count": 12,
        "current_intent": "technical_support",
        "summary": {
          "primary_issue": "Login problems",
          "resolution_status": "escalated",
          "customer_satisfaction": null
        }
      }
    ],
    "pagination": {
      "total": 127,
      "page": 1,
      "page_size": 50,
      "has_next": true
    }
  }
}
```

#### 2. Get Conversation Details

**Scenario:** Customer service agent reviews full conversation history

```http
GET /api/v2/conversations/conv_11111111-2222-3333-4444-555555555555?include_messages=true&message_limit=100
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "conversation_id": "conv_11111111-2222-3333-4444-555555555555",
    "user_id": "user_customer123",
    "channel": "web",
    "status": "active",
    "started_at": "2025-06-01T10:00:00Z",
    "last_activity_at": "2025-06-01T10:15:00Z",
    "context": {
      "current_intent": "order_inquiry",
      "entities": {
        "order_number": "ORD123456",
        "customer_email": "john@example.com"
      }
    },
    "messages": [
      {
        "message_id": "msg_001",
        "direction": "inbound",
        "timestamp": "2025-06-01T10:00:00Z",
        "content": {
          "type": "text",
          "text": "Hello, I need help with my order"
        }
      },
      {
        "message_id": "msg_002",
        "direction": "outbound",
        "timestamp": "2025-06-01T10:00:01Z",
        "content": {
          "type": "text",
          "text": "Hi! I'd be happy to help you with your order. Could you please provide your order number?"
        }
      }
    ],
    "summary": {
      "total_messages": 8,
      "duration_minutes": 15,
      "resolution_status": "in_progress",
      "escalation_count": 0
    }
  }
}
```

#### 3. Export Conversation

**Scenario:** Compliance team needs conversation data for audit

```http
POST /api/v2/conversations/conv_11111111-2222-3333-4444-555555555555/export

{
  "format": "json",
  "include_metadata": true,
  "include_analytics": true,
  "date_range": {
    "start": "2025-06-01T00:00:00Z",
    "end": "2025-06-01T23:59:59Z"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "export_id": "export_789",
    "download_url": "https://exports.chatbot-platform.com/conversations/export_789.json",
    "expires_at": "2025-06-02T10:00:00Z",
    "file_size_bytes": 52428800
  }
}
```

#### 4. Transfer to Human Agent

**Scenario:** Bot cannot resolve customer issue and needs human intervention

```http
POST /api/v2/conversations/conv_11111111-2222-3333-4444-555555555555/transfer

{
  "transfer_reason": "complex_technical_issue",
  "transfer_note": "Customer has login issues that require account verification",
  "department": "technical_support",
  "priority": "high",
  "agent_id": "agent_john_doe"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "transfer_id": "transfer_456",
    "assigned_agent": {
      "agent_id": "agent_john_doe",
      "name": "John Doe",
      "department": "technical_support"
    },
    "estimated_wait_time_minutes": 5,
    "transfer_message": "I'm connecting you with our technical support specialist John who can help resolve your login issue. Estimated wait time: 5 minutes."
  }
}
```

---

## Session Routes (`session_routes.py`)

**Purpose:** Manage user sessions, authentication state, and session-related data

### Route Overview

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/sessions` | POST | Create new session | Optional |
| `/sessions/{id}` | GET | Get session details | Required |
| `/sessions/{id}` | PUT | Update session data | Required |
| `/sessions/{id}` | DELETE | End session | Required |
| `/sessions/{id}/extend` | POST | Extend session timeout | Required |
| `/sessions/{id}/context` | GET | Get session context | Required |
| `/sessions/{id}/context` | PUT | Update session context | Required |

### Detailed Examples

#### 1. Create New Session

**Scenario:** User starts chatting on website - anonymous session creation

```http
POST /api/v2/sessions

{
  "channel": "web",
  "user_id": "anonymous_visitor_123",
  "initial_context": {
    "page_url": "https://mystore.com/products/widget-pro",
    "referrer": "https://google.com/search?q=premium+widgets",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "device_type": "desktop",
    "timezone": "America/New_York"
  },
  "session_config": {
    "timeout_minutes": 30,
    "enable_context_persistence": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "session_id": "sess_abc123def456",
    "user_id": "anonymous_visitor_123",
    "channel": "web",
    "created_at": "2025-06-01T14:30:00Z",
    "expires_at": "2025-06-01T15:00:00Z",
    "session_token": "st_987654321fedcba",
    "context": {
      "page_url": "https://mystore.com/products/widget-pro",
      "product_interest": "widget-pro",
      "visitor_type": "new"
    }
  }
}
```

#### 2. Get Session Details

**Scenario:** Retrieve current session information for context

```http
GET /api/v2/sessions/sess_abc123def456
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "session_id": "sess_abc123def456",
    "user_id": "anonymous_visitor_123",
    "channel": "web",
    "status": "active",
    "created_at": "2025-06-01T14:30:00Z",
    "last_activity": "2025-06-01T14:45:00Z",
    "expires_at": "2025-06-01T15:15:00Z",
    "conversation_count": 1,
    "message_count": 6,
    "context": {
      "current_page": "https://mystore.com/products/widget-pro",
      "session_duration_minutes": 15,
      "pages_visited": [
        "https://mystore.com/",
        "https://mystore.com/products",
        "https://mystore.com/products/widget-pro"
      ],
      "interests": ["widget-pro", "premium-features"],
      "intent_history": ["product_inquiry", "pricing_question"]
    }
  }
}
```

#### 3. Update Session Context

**Scenario:** User navigates to different page, update session context

```http
PUT /api/v2/sessions/sess_abc123def456/context

{
  "context_updates": {
    "current_page": "https://mystore.com/cart",
    "cart_items": [
      {
        "product_id": "widget-pro",
        "quantity": 2,
        "price": 99.99
      }
    ],
    "checkout_stage": "cart_review"
  },
  "append_to_history": {
    "page_visits": "https://mystore.com/cart",
    "user_actions": "added_to_cart"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "session_id": "sess_abc123def456",
    "context_updated": true,
    "updated_fields": ["current_page", "cart_items", "checkout_stage"],
    "new_context": {
      "current_page": "https://mystore.com/cart",
      "cart_items": [
        {
          "product_id": "widget-pro",
          "quantity": 2,
          "price": 99.99
        }
      ],
      "checkout_stage": "cart_review",
      "pages_visited": [
        "https://mystore.com/",
        "https://mystore.com/products",
        "https://mystore.com/products/widget-pro",
        "https://mystore.com/cart"
      ]
    }
  }
}
```

#### 4. Extend Session

**Scenario:** User is actively chatting, extend session to prevent timeout

```http
POST /api/v2/sessions/sess_abc123def456/extend

{
  "extend_by_minutes": 30,
  "reason": "active_conversation"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "session_id": "sess_abc123def456",
    "previous_expiry": "2025-06-01T15:00:00Z",
    "new_expiry": "2025-06-01T15:30:00Z",
    "extended_by_minutes": 30
  }
}
```

---

## Health Routes (`health_routes.py`)

**Purpose:** System health monitoring, status checks, and diagnostics

### Route Overview

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/health` | GET | Basic health check | None |
| `/health/detailed` | GET | Detailed system status | Internal Only |
| `/health/dependencies` | GET | External dependency status | Internal Only |
| `/health/metrics` | GET | Performance metrics | Internal Only |
| `/health/readiness` | GET | Kubernetes readiness probe | None |
| `/health/liveness` | GET | Kubernetes liveness probe | None |

### Detailed Examples

#### 1. Basic Health Check

**Scenario:** Load balancer checks service availability

```http
GET /api/v2/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-01T14:30:00Z",
  "version": "2.1.0",
  "uptime_seconds": 86400,
  "environment": "production"
}
```

#### 2. Detailed Health Check

**Scenario:** Operations team monitors system components

```http
GET /api/v2/health/detailed
Authorization: Bearer {internal_monitoring_token}
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-01T14:30:00Z",
  "services": {
    "chat_service": {
      "status": "healthy",
      "response_time_ms": 45,
      "cpu_usage_percent": 23.5,
      "memory_usage_mb": 512,
      "active_connections": 1247
    },
    "mcp_engine": {
      "status": "healthy",
      "response_time_ms": 78,
      "active_state_machines": 892,
      "queue_size": 15
    },
    "model_orchestrator": {
      "status": "healthy",
      "response_time_ms": 234,
      "active_requests": 45,
      "models_available": 8
    }
  },
  "databases": {
    "mongodb": {
      "status": "healthy",
      "connection_pool_size": 45,
      "response_time_ms": 12,
      "disk_usage_percent": 34.2
    },
    "redis": {
      "status": "healthy",
      "memory_usage_mb": 2048,
      "connected_clients": 156,
      "keyspace_hits_ratio": 0.97
    },
    "postgresql": {
      "status": "healthy",
      "active_connections": 23,
      "response_time_ms": 8,
      "disk_usage_percent": 45.7
    }
  }
}
```

#### 3. Dependency Status Check

**Scenario:** Monitor external service dependencies

```http
GET /api/v2/health/dependencies
```

**Response:**
```json
{
  "status": "partial_degradation",
  "timestamp": "2025-06-01T14:30:00Z",
  "dependencies": {
    "openai_api": {
      "status": "healthy",
      "last_check": "2025-06-01T14:29:45Z",
      "response_time_ms": 287,
      "rate_limit_remaining": 4500
    },
    "anthropic_api": {
      "status": "healthy",
      "last_check": "2025-06-01T14:29:50Z",
      "response_time_ms": 345,
      "rate_limit_remaining": 2300
    },
    "whatsapp_api": {
      "status": "degraded",
      "last_check": "2025-06-01T14:29:30Z",
      "response_time_ms": 2300,
      "error": "High latency detected",
      "rate_limit_remaining": 8900
    },
    "slack_api": {
      "status": "healthy",
      "last_check": "2025-06-01T14:29:55Z",
      "response_time_ms": 156
    }
  },
  "overall_health_score": 0.92
}
```

#### 4. Performance Metrics

**Scenario:** Get real-time performance data for monitoring

```http
GET /api/v2/health/metrics?timeframe=1h
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-06-01T14:30:00Z",
  "timeframe": "1h",
  "metrics": {
    "requests_per_second": 1247.5,
    "average_response_time_ms": 287,
    "p95_response_time_ms": 456,
    "p99_response_time_ms": 1200,
    "error_rate_percent": 0.12,
    "active_conversations": 12450,
    "messages_processed": 89650,
    "cpu_usage_average": 45.2,
    "memory_usage_average": 67.8,
    "disk_usage_percent": 34.5
  }
}
```

---

## Webhook Routes (`webhook_routes.py`)

**Purpose:** Handle incoming webhooks from external channels and services

### Route Overview

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/webhooks/whatsapp` | POST | WhatsApp message webhooks | Token |
| `/webhooks/messenger` | POST | Facebook Messenger webhooks | Token |
| `/webhooks/slack` | POST | Slack event webhooks | Token |
| `/webhooks/teams` | POST | Microsoft Teams webhooks | Token |
| `/webhooks/telegram` | POST | Telegram bot webhooks | Token |
| `/webhooks/generic/{integration_id}` | POST | Custom integration webhooks | Token |
| `/webhooks/verify/{channel}` | GET | Webhook verification | Token |

### Detailed Examples

#### 1. WhatsApp Webhook - Incoming Message

**Scenario:** Customer sends message via WhatsApp Business API

```http
POST /api/v2/webhooks/whatsapp
X-Hub-Signature-256: sha256=abc123def456...
Content-Type: application/json

{
  "object": "whatsapp_business_account",
  "entry": [
    {
      "id": "1234567890",
      "changes": [
        {
          "value": {
            "messaging_product": "whatsapp",
            "metadata": {
              "display_phone_number": "15551234567",
              "phone_number_id": "987654321"
            },
            "messages": [
              {
                "from": "1234567890",
                "id": "wamid.abc123def456",
                "timestamp": "1672531200",
                "text": {
                  "body": "Hi, I need help with my order ORD123456"
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
```

**Response:**
```json
{
  "status": "received",
  "message_id": "msg_whatsapp_processed_789",
  "conversation_id": "conv_whatsapp_user_1234567890",
  "processing_time_ms": 145
}
```

#### 2. Slack Webhook - Slash Command

**Scenario:** User uses /chatbot command in Slack workspace

```http
POST /api/v2/webhooks/slack
Content-Type: application/x-www-form-urlencoded

token=verification_token_here
team_id=T1234567
team_domain=mycompany
channel_id=C2147483705
channel_name=general
user_id=U2147483697
user_name=john.doe
command=/chatbot
text=help with expense reports
response_url=https://hooks.slack.com/commands/1234/5678
trigger_id=13345224609.738474920.8088930838d88f008e0
```

**Response:**
```json
{
  "response_type": "in_channel",
  "text": "I can help you with expense reports! Here are the available options:",
  "attachments": [
    {
      "text": "What would you like to do?",
      "fallback": "Expense report options",
      "callback_id": "expense_report_menu",
      "color": "good",
      "actions": [
        {
          "name": "submit_expense",
          "text": "Submit New Expense",
          "type": "button",
          "value": "submit_new"
        },
        {
          "name": "check_status",
          "text": "Check Status",
          "type": "button",
          "value": "check_status"
        }
      ]
    }
  ]
}
```

#### 3. Generic Webhook - CRM Integration

**Scenario:** Customer data updated in CRM, trigger chatbot context update

```http
POST /api/v2/webhooks/generic/crm_integration_456
Authorization: Bearer webhook_token_abc123
Content-Type: application/json

{
  "event_type": "customer_updated",
  "timestamp": "2025-06-01T14:30:00Z",
  "customer_id": "cust_789",
  "data": {
    "customer_email": "john@example.com",
    "customer_tier": "premium",
    "recent_orders": [
      {
        "order_id": "ORD123456",
        "status": "shipped",
        "tracking_number": "TRK789012"
      }
    ],
    "support_tickets": [
      {
        "ticket_id": "TKT456",
        "status": "open",
        "priority": "high",
        "subject": "Login issues"
      }
    ]
  }
}
```

**Response:**
```json
{
  "status": "processed",
  "event_id": "evt_webhook_123",
  "actions_taken": [
    "updated_customer_profile",
    "refreshed_conversation_context",
    "triggered_proactive_notification"
  ],
  "affected_conversations": [
    "conv_customer_789_active"
  ]
}
```

#### 4. Webhook Verification

**Scenario:** Channel platform verifies webhook endpoint during setup

```http
GET /api/v2/webhooks/verify/whatsapp?hub.mode=subscribe&hub.challenge=challenge_string_123&hub.verify_token=my_verify_token
```

**Response:**
```
challenge_string_123
```

---

## Tenant Routes (`tenant_routes.py`)

**Purpose:** Multi-tenant management, configuration, and administration

### Route Overview

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/tenants` | GET | List all tenants (admin) | Admin Only |
| `/tenants` | POST | Create new tenant | Admin Only |
| `/tenants/{id}` | GET | Get tenant details | Tenant Admin |
| `/tenants/{id}` | PUT | Update tenant configuration | Tenant Admin |
| `/tenants/{id}` | DELETE | Delete tenant | Admin Only |
| `/tenants/{id}/users` | GET | List tenant users | Tenant Admin |
| `/tenants/{id}/users` | POST | Add user to tenant | Tenant Admin |
| `/tenants/{id}/config` | GET | Get tenant configuration | Tenant Admin |
| `/tenants/{id}/config` | PUT | Update tenant configuration | Tenant Admin |
| `/tenants/{id}/billing` | GET | Get billing information | Tenant Admin |
| `/tenants/{id}/usage` | GET | Get usage statistics | Tenant Admin |

### Detailed Examples

#### 1. Create New Tenant

**Scenario:** Platform admin creates new tenant for customer

```http
POST /api/v2/tenants
Authorization: Bearer {admin_token}
Content-Type: application/json

{
  "name": "Acme Corporation",
  "domain": "acme.com",
  "plan": "enterprise",
  "admin_user": {
    "email": "admin@acme.com",
    "first_name": "John",
    "last_name": "Smith",
    "role": "owner"
  },
  "configuration": {
    "default_language": "en",
    "timezone": "America/New_York",
    "business_hours": {
      "monday": {"start": "09:00", "end": "17:00"},
      "tuesday": {"start": "09:00", "end": "17:00"},
      "wednesday": {"start": "09:00", "end": "17:00"},
      "thursday": {"start": "09:00", "end": "17:00"},
      "friday": {"start": "09:00", "end": "17:00"},
      "saturday": {"closed": true},
      "sunday": {"closed": true}
    },
    "branding": {
      "primary_color": "#1f2937",
      "secondary_color": "#3b82f6",
      "logo_url": "https://acme.com/logo.png",
      "company_name": "Acme Corporation"
    }
  },
  "features": {
    "channels": ["web", "whatsapp", "slack"],
    "integrations_limit": 50,
    "users_limit": 100,
    "conversations_per_month": 100000
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "tenant_id": "tenant_acme_123456",
    "name": "Acme Corporation",
    "domain": "acme.com",
    "plan": "enterprise",
    "status": "active",
    "created_at": "2025-06-01T14:30:00Z",
    "admin_user": {
      "user_id": "user_acme_admin_789",
      "email": "admin@acme.com",
      "role": "owner",
      "invitation_sent": true
    },
    "api_keys": {
      "primary": "cb_live_acme123456789012345678901234567890ab",
      "test": "cb_test_acme123456789012345678901234567890ab"
    },
    "webhook_endpoints": {
      "incoming": "https://api.chatbot-platform.com/api/v2/webhooks/tenant/tenant_acme_123456",
      "outgoing": "https://acme.com/webhooks/chatbot"
    }
  }
}
```

#### 2. Get Tenant Configuration

**Scenario:** Tenant admin retrieves current configuration

```http
GET /api/v2/tenants/tenant_acme_123456/config
Authorization: Bearer {tenant_admin_token}
X-Tenant-ID: tenant_acme_123456
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "tenant_id": "tenant_acme_123456",
    "configuration": {
      "general": {
        "default_language": "en",
        "timezone": "America/New_York",
        "currency": "USD",
        "date_format": "MM/DD/YYYY"
      },
      "chatbot": {
        "default_greeting": "Hello! How can I help you today?",
        "fallback_response": "I'm sorry, I didn't understand. Could you please rephrase?",
        "escalation_trigger": "human_agent",
        "confidence_threshold": 0.7
      },
      "channels": {
        "web": {
          "enabled": true,
          "widget_color": "#1f2937",
          "position": "bottom_right",
          "welcome_message": "Welcome to Acme! How can we help?"
        },
        "whatsapp": {
          "enabled": true,
          "business_account_id": "123456789",
          "phone_number": "+15551234567"
        },
        "slack": {
          "enabled": true,
          "workspace_id": "T1234567",
          "bot_token": "xoxb-encrypted-token"
        }
      },
      "integrations": {
        "crm": {
          "provider": "salesforce",
          "enabled": true,
          "sync_customer_data": true
        },
        "helpdesk": {
          "provider": "zendesk",
          "enabled": true,
          "auto_create_tickets": true
        }
      },
      "ai_models": {
        "primary_provider": "openai",
        "primary_model": "gpt-4-turbo",
        "fallback_chain": ["anthropic:claude-3-sonnet", "rule_based"],
        "custom_instructions": "You are a helpful customer service agent for Acme Corporation..."
      }
    }
  }
}
```

#### 3. Update Tenant Configuration

**Scenario:** Tenant admin updates chatbot settings

```http
PUT /api/v2/tenants/tenant_acme_123456/config
Authorization: Bearer {tenant_admin_token}
X-Tenant-ID: tenant_acme_123456

{
  "configuration": {
    "chatbot": {
      "default_greeting": "Hi there! Welcome to Acme Corporation. How can I assist you today?",
      "confidence_threshold": 0.8,
      "enable_sentiment_analysis": true
    },
    "channels": {
      "web": {
        "widget_color": "#2563eb",
        "show_agent_avatars": true,
        "enable_file_upload": true
      }
    },
    "ai_models": {
      "temperature": 0.7,
      "max_tokens": 500,
      "custom_instructions": "You are a friendly and professional customer service agent for Acme Corporation. Always be helpful and concise in your responses."
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "updated_at": "2025-06-01T14:35:00Z",
    "configuration_version": "v2.1",
    "changes_applied": [
      "chatbot.default_greeting",
      "chatbot.confidence_threshold", 
      "chatbot.enable_sentiment_analysis",
      "channels.web.widget_color",
      "channels.web.show_agent_avatars",
      "channels.web.enable_file_upload",
      "ai_models.temperature",
      "ai_models.max_tokens",
      "ai_models.custom_instructions"
    ],
    "deployment_status": "pending",
    "estimated_deployment_time": "2-3 minutes"
  }
}
```

#### 4. Get Tenant Usage Statistics

**Scenario:** Tenant admin checks monthly usage for billing

```http
GET /api/v2/tenants/tenant_acme_123456/usage?period=current_month&detailed=true
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "tenant_id": "tenant_acme_123456",
    "period": "2025-06",
    "usage": {
      "conversations": {
        "total": 8450,
        "completed": 7623,
        "abandoned": 827,
        "escalated": 156
      },
      "messages": {
        "total": 45230,
        "inbound": 22615,
        "outbound": 22615,
        "by_channel": {
          "web": 28450,
          "whatsapp": 12340,
          "slack": 4440
        }
      },
      "ai_costs": {
        "total_cents": 2847,
        "by_provider": {
          "openai": 2234,
          "anthropic": 456,
          "fallback": 157
        },
        "by_model": {
          "gpt-4-turbo": 2234,
          "claude-3-sonnet": 456,
          "rule_based": 157
        }
      },
      "integrations": {
        "api_calls": 1250,
        "webhook_calls": 890,
        "data_sync_operations": 456
      }
    },
    "limits": {
      "conversations_limit": 100000,
      "conversations_used_percent": 8.45,
      "api_calls_limit": 50000,
      "api_calls_used_percent": 2.5
    },
    "billing_period": {
      "start": "2025-06-01T00:00:00Z",
      "end": "2025-06-30T23:59:59Z",
      "days_remaining": 20
    }
  }
}
```

#### 5. Add User to Tenant

**Scenario:** Tenant admin invites new team member

```http
POST /api/v2/tenants/tenant_acme_123456/users
Authorization: Bearer {tenant_admin_token}
X-Tenant-ID: tenant_acme_123456

{
  "email": "sarah.johnson@acme.com",
  "first_name": "Sarah",
  "last_name": "Johnson",
  "role": "developer",
  "permissions": [
    "conversations:read",
    "integrations:write",
    "analytics:read"
  ],
  "send_invitation": true,
  "invitation_message": "Welcome to our chatbot platform team!"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "user_id": "user_sarah_acme_456",
    "email": "sarah.johnson@acme.com",
    "role": "developer",
    "status": "invitation_sent",
    "invitation_expires_at": "2025-06-08T14:30:00Z",
    "permissions": [
      "conversations:read",
      "integrations:write", 
      "analytics:read"
    ],
    "created_at": "2025-06-01T14:30:00Z"
  }
}
```

---

## Common Patterns & Error Handling

### Authentication Patterns

#### JWT Token Authentication
```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### API Key Authentication
```http
Authorization: ApiKey cb_live_a1b2c3d4e5f6789012345678901234567890abcd
```

#### Webhook Token Authentication
```http
X-Hub-Signature-256: sha256=abc123def456...
```

### Error Response Format

All endpoints return consistent error responses:

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field": "email",
      "reason": "Invalid email format",
      "provided_value": "invalid-email"
    },
    "trace_id": "trace_abc123",
    "timestamp": "2025-06-01T14:30:00Z"
  },
  "meta": {
    "request_id": "req_uuid",
    "api_version": "v2"
  }
}
```

### Common HTTP Status Codes

| Status Code | Meaning | When Used |
|-------------|---------|-----------|
| `200` | Success | Request processed successfully |
| `201` | Created | Resource created successfully |
| `400` | Bad Request | Invalid request format or parameters |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource does not exist |
| `429` | Rate Limited | Too many requests |
| `500` | Internal Error | Server-side error |
| `503` | Service Unavailable | Service temporarily down |

### Rate Limiting Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 742
X-RateLimit-Reset: 1622547600
X-RateLimit-Type: api_key
```

### Pagination Parameters

```http
GET /api/v2/conversations?page=1&limit=50&sort=created_at&order=desc
```

### Filtering and Search

```http
GET /api/v2/conversations?status=active&channel=whatsapp&user_id=user123&date_from=2025-06-01&search=order
```

---

## Implementation Notes

### Security Considerations

1. **Always validate tenant isolation** - Every request must include X-Tenant-ID
2. **Rate limiting per tenant** - Different limits based on plan tier
3. **Input validation** - Sanitize all user inputs before processing
4. **Audit logging** - Log all admin and configuration changes

### Performance Optimizations

1. **Response caching** - Cache frequently accessed data (sessions, configs)
2. **Database indexing** - Proper indexes on conversation_id, tenant_id, timestamps
3. **Async processing** - Heavy operations should be queued
4. **Connection pooling** - Reuse database connections efficiently

### Monitoring and Alerting

1. **Health check endpoints** - Regular monitoring of service health
2. **Performance metrics** - Track response times and error rates
3. **Business metrics** - Monitor conversation success rates
4. **Error tracking** - Comprehensive error logging and alerting

---

**Last Updated:** June 1, 2025  
**Document Maintainer:** API Development Team  
**Review Schedule:** Weekly during development, monthly in production