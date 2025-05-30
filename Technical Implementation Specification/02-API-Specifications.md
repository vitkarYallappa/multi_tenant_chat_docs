# API Specifications
## Multi-Tenant AI Chatbot Platform

**Document:** 02-API-Specifications.md  
**Version:** 2.0  
**Last Updated:** May 30, 2025

---

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Chat Service API](#chat-service-api)
4. [MCP Engine API](#mcp-engine-api)
5. [Model Orchestrator API](#model-orchestrator-api)
6. [Adaptor Service API](#adaptor-service-api)
7. [Security Hub API](#security-hub-api)
8. [Analytics Engine API](#analytics-engine-api)
9. [Error Handling](#error-handling)
10. [Rate Limiting](#rate-limiting)

---

## API Overview

### Design Principles

1. **RESTful Design:** Standard HTTP methods and status codes
2. **API Versioning:** Version in URL path (e.g., `/api/v2/`)
3. **JSON Format:** All requests and responses in JSON
4. **Tenant Isolation:** Tenant ID in headers and all operations
5. **Idempotent Operations:** Safe to retry requests
6. **Comprehensive Error Messages:** Detailed error information

### Common Headers

```http
Content-Type: application/json
Accept: application/json
X-Tenant-ID: {tenant_uuid}
Authorization: Bearer {jwt_token} | ApiKey {api_key}
X-Request-ID: {uuid} (optional, for tracing)
User-Agent: {client_info}
```

### Standard Response Format

```json
{
  "status": "success|error",
  "data": {},
  "meta": {
    "request_id": "uuid",
    "timestamp": "ISO_8601",
    "version": "v2",
    "processing_time_ms": 123
  },
  "error": {
    "code": "string",
    "message": "string",
    "details": {},
    "trace_id": "string"
  }
}
```

---

## Authentication

### JWT Token Structure

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "iss": "chatbot-platform",
    "sub": "user_id",
    "aud": "chatbot-api",
    "exp": 1234567890,
    "iat": 1234564290,
    "tenant_id": "uuid",
    "user_role": "admin|developer|member|viewer",
    "permissions": ["conversations:read", "config:write"],
    "scopes": ["api:read", "api:write"],
    "rate_limit_tier": "premium|standard|basic"
  }
}
```

### API Key Format

```
Format: cb_{environment}_{32_character_hex}
Examples:
- cb_live_a1b2c3d4e5f6789012345678901234567890abcd
- cb_test_fedcba0987654321abcdef1234567890abcdef12
```

---

## Chat Service API

### Message Processing

#### Send Message
```http
POST /api/v2/chat/message
```

**Request Body:**
```json
{
  "message_id": "uuid_v4",
  "conversation_id": "uuid_v4",
  "user_id": "string",
  "session_id": "uuid_v4",
  "channel": "web|whatsapp|messenger|slack|teams|voice|sms",
  "timestamp": "2025-05-30T10:00:00Z",
  
  "content": {
    "type": "text|image|file|audio|video|location|quick_reply",
    "text": "Hello, I need help with my order",
    "language": "en",
    
    "media": {
      "url": "https://example.com/media/image.jpg",
      "type": "image/jpeg",
      "size_bytes": 245760,
      "alt_text": "Screenshot of error message"
    },
    
    "location": {
      "latitude": 37.7749,
      "longitude": -122.4194,
      "accuracy_meters": 10,
      "address": "123 Main St, San Francisco, CA"
    },
    
    "context": {
      "user_agent": "Mozilla/5.0...",
      "device_type": "mobile",
      "referrer": "https://example.com/product/123"
    }
  },
  
  "channel_metadata": {
    "platform_message_id": "msg_abc123",
    "platform_user_id": "user_xyz789",
    "thread_id": "thread_456"
  },
  
  "processing_hints": {
    "priority": "normal",
    "expected_response_type": "text",
    "bypass_automation": false,
    "require_human_review": false
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "message_id": "uuid_v4",
    "conversation_id": "uuid_v4",
    "response": {
      "type": "text",
      "text": "I'd be happy to help you with your order. Could you please provide your order number?",
      "confidence_score": 0.92,
      
      "quick_replies": [
        {
          "title": "Check Order Status",
          "payload": "check_order_status"
        },
        {
          "title": "Cancel Order",
          "payload": "cancel_order"
        }
      ]
    },
    
    "conversation_state": {
      "current_intent": "order_inquiry",
      "entities": {
        "order_type": "product_order"
      },
      "next_expected_input": "order_number",
      "conversation_stage": "information_gathering"
    },
    
    "processing_metadata": {
      "model_used": "gpt-4-turbo",
      "model_provider": "openai",
      "processing_time_ms": 287,
      "cost_cents": 1.25,
      "fallback_applied": false
    }
  },
  "meta": {
    "request_id": "req_uuid",
    "timestamp": "2025-05-30T10:00:00.287Z",
    "processing_time_ms": 287
  }
}
```

#### Get Conversation
```http
GET /api/v2/chat/conversations/{conversation_id}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "conversation_id": "uuid_v4",
    "user_id": "string",
    "channel": "web",
    "status": "active",
    "started_at": "2025-05-30T09:45:00Z",
    "last_activity_at": "2025-05-30T10:00:00Z",
    
    "messages": [
      {
        "message_id": "uuid_1",
        "direction": "inbound",
        "timestamp": "2025-05-30T09:45:00Z",
        "content": {
          "type": "text",
          "text": "Hello, I need help"
        }
      },
      {
        "message_id": "uuid_2", 
        "direction": "outbound",
        "timestamp": "2025-05-30T09:45:01Z",
        "content": {
          "type": "text",
          "text": "Hi! I'm here to help. What can I assist you with today?"
        }
      }
    ],
    
    "summary": {
      "message_count": 2,
      "user_messages": 1,
      "bot_messages": 1,
      "primary_intent": "general_inquiry",
      "resolution_status": "in_progress"
    }
  }
}
```

#### List Conversations
```http
GET /api/v2/chat/conversations?limit=50&offset=0&status=active&channel=web
```

#### Export Conversation
```http
POST /api/v2/chat/conversations/{conversation_id}/export
```

---

## MCP Engine API

### State Machine Management

#### Create Flow
```http
POST /api/v2/mcp/flows
```

**Request Body:**
```json
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
            "default": "Hi! How can I help you today?",
            "returning_user": "Welcome back! How can I assist you?"
          }
        },
        "transitions": {
          "user_responds": {
            "condition": "any_input",
            "target_state": "intent_detection"
          }
        }
      },
      
      "intent_detection": {
        "type": "intent",
        "config": {
          "intent_patterns": [
            "order_inquiry",
            "technical_support", 
            "billing_question",
            "general_info"
          ],
          "confidence_threshold": 0.7
        },
        "transitions": {
          "order_inquiry": {
            "condition": "intent_match",
            "condition_value": "order_inquiry",
            "target_state": "collect_order_info"
          },
          "technical_support": {
            "condition": "intent_match",
            "condition_value": "technical_support",
            "target_state": "technical_triage"
          },
          "fallback": {
            "condition": "low_confidence",
            "target_state": "clarification"
          }
        }
      },
      
      "collect_order_info": {
        "type": "slot_filling",
        "config": {
          "required_slots": ["order_number"],
          "optional_slots": ["email", "phone"],
          "validation_rules": {
            "order_number": "^ORD[0-9]{6}$"
          }
        },
        "transitions": {
          "slots_complete": {
            "condition": "all_required_slots_filled",
            "target_state": "lookup_order"
          }
        }
      },
      
      "lookup_order": {
        "type": "integration",
        "config": {
          "integration_id": "order_management_system",
          "endpoint": "/orders/lookup",
          "method": "GET",
          "timeout_ms": 5000
        },
        "transitions": {
          "success": {
            "condition": "integration_success",
            "target_state": "display_order_info"
          },
          "error": {
            "condition": "integration_error",
            "target_state": "order_lookup_error"
          }
        }
      }
    },
    
    "global_handlers": {
      "timeout": {
        "timeout_seconds": 300,
        "target_state": "session_timeout",
        "response": "I haven't heard from you in a while. Is there anything else I can help you with?"
      },
      "error": {
        "target_state": "error_handler",
        "response": "I'm sorry, something went wrong. Let me connect you with a human agent."
      }
    }
  },
  
  "ab_test_config": {
    "enabled": true,
    "variants": [
      {
        "name": "variant_a",
        "percentage": 50,
        "modifications": {
          "greeting.config.response_templates.default": "Hello! What brings you here today?"
        }
      },
      {
        "name": "variant_b", 
        "percentage": 50,
        "modifications": {}
      }
    ]
  }
}
```

#### Execute State Machine
```http
POST /api/v2/mcp/execute
```

**Request Body:**
```json
{
  "conversation_id": "uuid_v4",
  "current_state": "intent_detection",
  "event": {
    "type": "user_message",
    "data": {
      "text": "I want to check my order status",
      "intent": "order_inquiry",
      "confidence": 0.92,
      "entities": {
        "intent": "order_inquiry"
      }
    }
  },
  "context": {
    "slots": {},
    "variables": {},
    "history": ["greeting", "intent_detection"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "new_state": "collect_order_info",
    "actions": [
      {
        "type": "send_message",
        "content": {
          "text": "I'll help you check your order status. Please provide your order number.",
          "quick_replies": [
            {
              "title": "I don't have my order number",
              "payload": "no_order_number"
            }
          ]
        }
      }
    ],
    "context_updates": {
      "slots": {
        "intent": "order_inquiry"
      },
      "variables": {
        "conversation_stage": "information_gathering"
      }
    },
    "next_expected_inputs": ["order_number"]
  }
}
```

---

## Model Orchestrator API

### Model Management

#### Process Request
```http
POST /api/v2/model/process
```

**Request Body:**
```json
{
  "request_id": "uuid_v4",
  "operation": "response_generation",
  
  "input": {
    "text": "I want to check my order status",
    "language": "en",
    "conversation_context": {
      "previous_messages": [
        {
          "role": "user",
          "content": "Hello",
          "timestamp": "2025-05-30T09:45:00Z"
        },
        {
          "role": "assistant", 
          "content": "Hi! How can I help you today?",
          "timestamp": "2025-05-30T09:45:01Z"
        }
      ],
      "intent_history": ["greeting"],
      "entities": {},
      "user_profile": {
        "preferred_language": "en",
        "customer_tier": "premium"
      }
    }
  },
  
  "model_preferences": {
    "primary_provider": "openai",
    "primary_model": "gpt-4-turbo",
    "fallback_chain": ["anthropic:claude-3-sonnet", "rule_based"],
    "max_cost_cents": 5,
    "max_latency_ms": 3000,
    "temperature": 0.7,
    "max_tokens": 500
  },
  
  "output_requirements": {
    "format": "structured",
    "include_reasoning": false,
    "include_alternatives": false
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "result": {
      "operation": "response_generation",
      "primary_output": "I'd be happy to help you check your order status. Could you please provide your order number?",
      "confidence_score": 0.92,
      
      "structured_output": {
        "intent": "order_status_inquiry",
        "next_action": "collect_order_number",
        "response_type": "information_request",
        "suggested_quick_replies": [
          "I have my order number",
          "I don't have my order number",
          "Check by email address"
        ]
      }
    },
    
    "model_info": {
      "provider": "openai",
      "model": "gpt-4-turbo",
      "version": "gpt-4-turbo-2024-04-09",
      "fallback_used": false,
      "cost_cents": 1.25,
      "tokens": {
        "input": 156,
        "output": 42,
        "total": 198
      }
    },
    
    "performance_metrics": {
      "queue_time_ms": 12,
      "processing_time_ms": 287,
      "model_latency_ms": 245,
      "cache_hit": false
    }
  }
}
```

#### Get Model Status
```http
GET /api/v2/model/status
```

#### Update Model Configuration  
```http
PUT /api/v2/model/config/{config_id}
```

---

## Adaptor Service API

### Integration Management

#### Create Integration
```http
POST /api/v2/integrations
```

**Request Body:**
```json
{
  "name": "Shopify Store Integration",
  "description": "Connect to Shopify for order management",
  "type": "rest_api",
  "category": "ecommerce",
  
  "configuration": {
    "base_url": "https://mystore.myshopify.com/admin/api/2023-10",
    "authentication": {
      "type": "api_key",
      "api_key_header": "X-Shopify-Access-Token",
      "api_key": "{{SHOPIFY_ACCESS_TOKEN}}"
    },
    
    "endpoints": {
      "get_order": {
        "path": "/orders/{order_id}.json",
        "method": "GET",
        "timeout_ms": 5000,
        "retry_config": {
          "max_retries": 3,
          "backoff_strategy": "exponential"
        }
      },
      
      "list_orders": {
        "path": "/orders.json",
        "method": "GET",
        "query_params": {
          "status": "any",
          "limit": 50
        }
      }
    },
    
    "data_mapping": {
      "order_lookup": {
        "request_mapping": {
          "order_id": "{{conversation.slots.order_number}}"
        },
        "response_mapping": {
          "order_number": "$.order.order_number",
          "status": "$.order.fulfillment_status",
          "total": "$.order.total_price",
          "created_at": "$.order.created_at",
          "items": "$.order.line_items[*].{name: title, quantity: quantity, price: price}"
        }
      }
    }
  },
  
  "test_cases": [
    {
      "name": "Valid Order Lookup",
      "input": {
        "order_id": "1234567890"
      },
      "expected_output": {
        "status_code": 200,
        "response_fields": ["order_number", "status", "total"]
      }
    }
  ]
}
```

#### Execute Integration
```http
POST /api/v2/integrations/{integration_id}/execute
```

**Request Body:**
```json
{
  "execution_id": "uuid_v4",
  "endpoint": "get_order",
  "input_data": {
    "order_id": "1234567890",
    "context": {
      "user_id": "user_123",
      "conversation_id": "conv_456"
    }
  },
  "timeout_override_ms": 10000
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "execution_id": "uuid_v4",
    "result": {
      "status_code": 200,
      "response_data": {
        "order_number": "ORD123456",
        "status": "fulfilled",
        "total": "$99.99",
        "created_at": "2025-05-29T14:30:00Z",
        "items": [
          {
            "name": "Premium Widget",
            "quantity": 2,
            "price": "$49.99"
          }
        ]
      },
      "formatted_response": "Your order ORD123456 for $99.99 has been fulfilled and should arrive soon!"
    },
    
    "execution_metadata": {
      "execution_time_ms": 423,
      "retry_count": 0,
      "cache_hit": false,
      "cost_cents": 0.05
    }
  }
}
```

#### Test Integration
```http
POST /api/v2/integrations/{integration_id}/test
```

#### List Available Integrations
```http
GET /api/v2/integrations/marketplace?category=ecommerce&limit=20
```

---

## Security Hub API

### Authentication

#### Login
```http
POST /api/v2/auth/login
```

**Request Body:**
```json
{
  "email": "user@company.com",
  "password": "secure_password",
  "tenant_id": "uuid_v4",
  "mfa_token": "123456",
  "remember_me": false
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "scope": "api:read api:write",
    
    "user": {
      "user_id": "uuid_v4",
      "email": "user@company.com",
      "role": "admin",
      "permissions": ["conversations:read", "config:write"],
      "tenant_id": "uuid_v4"
    }
  }
}
```

#### Refresh Token
```http
POST /api/v2/auth/refresh
```

#### Logout
```http
POST /api/v2/auth/logout
```

### API Key Management

#### Create API Key
```http
POST /api/v2/auth/api-keys
```

**Request Body:**
```json
{
  "name": "Production API Key",
  "description": "For production integrations",
  "permissions": ["api:read", "api:write"],
  "scopes": ["conversations:read", "integrations:execute"],
  "rate_limit_per_minute": 1000,
  "expires_at": "2026-05-30T00:00:00Z",
  "allowed_ips": ["192.168.1.0/24"],
  "allowed_origins": ["https://myapp.com"]
}
```

#### List API Keys
```http
GET /api/v2/auth/api-keys
```

#### Revoke API Key
```http
DELETE /api/v2/auth/api-keys/{key_id}
```

---

## Analytics Engine API

### Metrics and Reporting

#### Get Conversation Metrics
```http
GET /api/v2/analytics/conversations?start_date=2025-05-01&end_date=2025-05-30&granularity=daily
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "metrics": [
      {
        "date": "2025-05-30",
        "conversations_started": 1247,
        "conversations_completed": 1156,
        "completion_rate": 0.927,
        "avg_response_time_ms": 287,
        "user_satisfaction": 4.2,
        "channels": {
          "web": 623,
          "whatsapp": 345,
          "messenger": 279
        }
      }
    ],
    
    "summary": {
      "total_conversations": 35642,
      "avg_completion_rate": 0.923,
      "avg_response_time_ms": 295,
      "top_intents": [
        "order_inquiry",
        "technical_support", 
        "billing_question"
      ]
    }
  }
}
```

#### Get Performance Metrics
```http
GET /api/v2/analytics/performance?service=chat-service&timeframe=24h
```

#### Custom Report
```http
POST /api/v2/analytics/reports
```

---

## Error Handling

### Error Response Format

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    },
    "trace_id": "trace_abc123",
    "timestamp": "2025-05-30T10:00:00Z"
  },
  "meta": {
    "request_id": "req_uuid",
    "api_version": "v2"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `AUTHENTICATION_FAILED` | 401 | Authentication credentials invalid |
| `AUTHORIZATION_FAILED` | 403 | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `TIMEOUT_ERROR` | 504 | Request timeout |

---

## Rate Limiting

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 742
X-RateLimit-Reset: 1622547600
X-RateLimit-Type: api_key
```

### Rate Limit Tiers

| Tier | Requests/Minute | Burst Limit | Cost per Request |
|------|-----------------|-------------|------------------|
| **Basic** | 100 | 200 | Standard |
| **Standard** | 1,000 | 2,000 | Standard |
| **Premium** | 10,000 | 20,000 | Priority |
| **Enterprise** | 100,000 | 200,000 | Priority |

### Rate Limit Exceeded Response

```json
{
  "status": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "remaining": 0,
      "reset_at": "2025-05-30T10:05:00Z"
    }
  }
}
```


**Document Maintainer:** API Development Team  
**Review Schedule:** Weekly during development, monthly in production  
**Related Documents:** System Architecture, Security Implementation