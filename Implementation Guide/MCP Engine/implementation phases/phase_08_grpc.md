# Phase 08: gRPC Implementation & Service Communication
**Duration**: Week 15-16 (Days 71-80)  
**Team Size**: 3-4 developers  
**Complexity**: High  

## Overview
Implement the complete gRPC service layer with Protocol Buffers, service interceptors, streaming support, and inter-service communication. This phase establishes the high-performance communication layer between MCP Engine and other services in the platform ecosystem.

## Step 19: Protocol Buffers & gRPC Service Definition (Days 71-73)

### Files to Create
```
src/
├── api/grpc/
│   ├── proto/
│   │   ├── mcp_engine.proto
│   │   ├── common.proto
│   │   ├── health.proto
│   │   ├── mcp_engine_pb2.py (generated)
│   │   ├── mcp_engine_pb2_grpc.py (generated)
│   │   ├── common_pb2.py (generated)
│   │   ├── common_pb2_grpc.py (generated)
│   │   ├── health_pb2.py (generated)
│   │   └── health_pb2_grpc.py (generated)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── mcp_service.py
│   │   ├── health_service.py
│   │   └── metrics_service.py
│   ├── interceptors/
│   │   ├── __init__.py
│   │   ├── auth_interceptor.py
│   │   ├── metrics_interceptor.py
│   │   ├── logging_interceptor.py
│   │   └── error_interceptor.py
│   └── server.py
├── scripts/
│   ├── generate_proto.py
│   └── start_grpc_server.py
```

### `/src/api/grpc/proto/common.proto`
**Purpose**: Common types and structures used across gRPC services
```protobuf
syntax = "proto3";

package mcp.common.v2;

import "google/protobuf/timestamp.proto";
import "google/protobuf/any.proto";
import "google/protobuf/struct.proto";

option go_package = "github.com/mcp-platform/proto/common/v2";
option java_package = "com.mcp.common.v2";
option java_outer_classname = "CommonProto";

// Status response for operations
message Status {
  bool success = 1;
  string message = 2;
  string error_code = 3;
  google.protobuf.Struct details = 4;
}

// Metadata for requests and responses
message Metadata {
  string request_id = 1;
  google.protobuf.Timestamp timestamp = 2;
  string version = 3;
  int32 processing_time_ms = 4;
  string trace_id = 5;
  google.protobuf.Struct custom_metadata = 6;
}

// Tenant context
message TenantContext {
  string tenant_id = 1;
  string user_id = 2;
  string session_id = 3;
  string organization_id = 4;
  repeated string roles = 5;
  repeated string permissions = 6;
}

// Pagination information
message Pagination {
  int32 page = 1;
  int32 page_size = 2;
  int32 total_count = 3;
  bool has_next = 4;
  bool has_previous = 5;
}

// Error details
message ErrorDetail {
  string code = 1;
  string message = 2;
  string field = 3;
  google.protobuf.Struct context = 4;
}

// Performance metrics
message PerformanceMetrics {
  int32 execution_time_ms = 1;
  int32 queue_time_ms = 2;
  int64 memory_used_bytes = 3;
  int32 cpu_time_ms = 4;
  string cache_status = 5;
  google.protobuf.Struct custom_metrics = 6;
}

// Health check status
enum HealthStatus {
  HEALTH_UNKNOWN = 0;
  HEALTH_SERVING = 1;
  HEALTH_NOT_SERVING = 2;
  HEALTH_SERVICE_UNKNOWN = 3;
}

// Message content types
enum MessageType {
  MESSAGE_TYPE_UNKNOWN = 0;
  MESSAGE_TYPE_TEXT = 1;
  MESSAGE_TYPE_IMAGE = 2;
  MESSAGE_TYPE_FILE = 3;
  MESSAGE_TYPE_AUDIO = 4;
  MESSAGE_TYPE_VIDEO = 5;
  MESSAGE_TYPE_LOCATION = 6;
  MESSAGE_TYPE_QUICK_REPLY = 7;
  MESSAGE_TYPE_CAROUSEL = 8;
  MESSAGE_TYPE_FORM = 9;
  MESSAGE_TYPE_SYSTEM = 10;
}

// Communication channels
enum Channel {
  CHANNEL_UNKNOWN = 0;
  CHANNEL_WEB = 1;
  CHANNEL_WHATSAPP = 2;
  CHANNEL_MESSENGER = 3;
  CHANNEL_SLACK = 4;
  CHANNEL_TEAMS = 5;
  CHANNEL_VOICE = 6;
  CHANNEL_SMS = 7;
}

// Priority levels
enum Priority {
  PRIORITY_UNKNOWN = 0;
  PRIORITY_LOW = 1;
  PRIORITY_NORMAL = 2;
  PRIORITY_HIGH = 3;
  PRIORITY_URGENT = 4;
  PRIORITY_CRITICAL = 5;
}
```

### `/src/api/grpc/proto/mcp_engine.proto`
**Purpose**: Main MCP Engine gRPC service definition
```protobuf
syntax = "proto3";

package mcp.engine.v2;

import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";
import "google/protobuf/empty.proto";
import "common.proto";

option go_package = "github.com/mcp-platform/proto/engine/v2";
option java_package = "com.mcp.engine.v2";
option java_outer_classname = "MCPEngineProto";

// MCP Engine Service
service MCPEngine {
  // Process a conversation message
  rpc ProcessMessage(ProcessMessageRequest) returns (ProcessMessageResponse);
  
  // Get conversation state
  rpc GetConversationState(GetConversationStateRequest) returns (GetConversationStateResponse);
  
  // Reset conversation
  rpc ResetConversation(ResetConversationRequest) returns (ResetConversationResponse);
  
  // Stream conversation events (bi-directional)
  rpc StreamConversation(stream ConversationStreamRequest) returns (stream ConversationStreamResponse);
  
  // Batch process multiple messages
  rpc BatchProcessMessages(BatchProcessMessagesRequest) returns (BatchProcessMessagesResponse);
  
  // Health check
  rpc HealthCheck(google.protobuf.Empty) returns (HealthCheckResponse);
  
  // Get service metrics
  rpc GetMetrics(GetMetricsRequest) returns (GetMetricsResponse);
}

// Message content structure
message MessageContent {
  mcp.common.v2.MessageType type = 1;
  string text = 2;
  string payload = 3;
  string language = 4;
  
  // Media content
  MediaContent media = 5;
  
  // Location data
  LocationContent location = 6;
  
  // Interactive elements
  repeated QuickReply quick_replies = 7;
  repeated Button buttons = 8;
  repeated CarouselCard carousel = 9;
  
  // Form data
  FormContent form = 10;
  
  // Additional metadata
  google.protobuf.Struct metadata = 11;
}

// Media content
message MediaContent {
  string url = 1;
  string secure_url = 2;
  string mime_type = 3;
  int64 size_bytes = 4;
  int32 duration_ms = 5;
  
  // Dimensions for images/videos
  message Dimensions {
    int32 width = 1;
    int32 height = 2;
  }
  Dimensions dimensions = 6;
  
  string thumbnail_url = 7;
  string alt_text = 8;
  string caption = 9;
  
  // Analysis results
  string transcript = 10;
  double transcript_confidence = 11;
  string ocr_text = 12;
  double ocr_confidence = 13;
  repeated string detected_objects = 14;
}

// Location content
message LocationContent {
  double latitude = 1;
  double longitude = 2;
  int32 accuracy_meters = 3;
  string address = 4;
  string place_name = 5;
  string place_id = 6;
}

// Quick reply button
message QuickReply {
  string title = 1;
  string payload = 2;
  string content_type = 3;
  bool clicked = 4;
  google.protobuf.Timestamp click_timestamp = 5;
}

// Interactive button
message Button {
  enum ButtonType {
    BUTTON_TYPE_UNKNOWN = 0;
    BUTTON_TYPE_POSTBACK = 1;
    BUTTON_TYPE_URL = 2;
    BUTTON_TYPE_PHONE = 3;
    BUTTON_TYPE_SHARE = 4;
    BUTTON_TYPE_LOGIN = 5;
  }
  
  ButtonType type = 1;
  string title = 2;
  string payload = 3;
  string url = 4;
  bool clicked = 5;
  google.protobuf.Timestamp click_timestamp = 6;
}

// Carousel card
message CarouselCard {
  string title = 1;
  string subtitle = 2;
  string image_url = 3;
  repeated Button buttons = 4;
}

// Form content
message FormContent {
  string form_id = 1;
  google.protobuf.Struct form_data = 2;
  string validation_status = 3;
  google.protobuf.Timestamp submitted_at = 4;
}

// Processing hints
message ProcessingHints {
  mcp.common.v2.Priority priority = 1;
  string expected_response_type = 2;
  bool bypass_automation = 3;
  bool require_human_review = 4;
  bool bypass_cache = 5;
  string force_flow = 6;
  google.protobuf.Struct custom_hints = 7;
}

// Conversation state
message ConversationState {
  string current_state = 1;
  string flow_id = 2;
  string flow_version = 3;
  repeated string previous_states = 4;
  
  // Context data
  google.protobuf.Struct slots = 5;
  google.protobuf.Struct variables = 6;
  google.protobuf.Struct user_profile = 7;
  
  // Intent information
  repeated string intent_history = 8;
  string current_intent = 9;
  double intent_confidence = 10;
  
  // Metadata
  google.protobuf.Timestamp created_at = 11;
  google.protobuf.Timestamp last_activity = 12;
  int32 message_count = 13;
  int32 error_count = 14;
}

// Request messages
message ProcessMessageRequest {
  mcp.common.v2.TenantContext tenant_context = 1;
  string conversation_id = 2;
  string user_id = 3;
  mcp.common.v2.Channel channel = 4;
  MessageContent content = 5;
  string session_id = 6;
  ProcessingHints processing_hints = 7;
  mcp.common.v2.Metadata metadata = 8;
}

message GetConversationStateRequest {
  mcp.common.v2.TenantContext tenant_context = 1;
  string conversation_id = 2;
  mcp.common.v2.Metadata metadata = 3;
}

message ResetConversationRequest {
  mcp.common.v2.TenantContext tenant_context = 1;
  string conversation_id = 2;
  string reason = 3;
  mcp.common.v2.Metadata metadata = 4;
}

message BatchProcessMessagesRequest {
  mcp.common.v2.TenantContext tenant_context = 1;
  repeated ProcessMessageRequest messages = 2;
  bool parallel_processing = 3;
  int32 max_concurrency = 4;
  mcp.common.v2.Metadata metadata = 5;
}

message GetMetricsRequest {
  mcp.common.v2.TenantContext tenant_context = 1;
  repeated string metric_names = 2;
  google.protobuf.Timestamp start_time = 3;
  google.protobuf.Timestamp end_time = 4;
  string granularity = 5;
  mcp.common.v2.Metadata metadata = 6;
}

// Response messages
message ProcessMessageResponse {
  mcp.common.v2.Status status = 1;
  string conversation_id = 2;
  string current_state = 3;
  MessageContent response = 4;
  google.protobuf.Struct context_updates = 5;
  repeated string actions_performed = 6;
  ConversationState conversation_state = 7;
  mcp.common.v2.PerformanceMetrics performance_metrics = 8;
  string ab_variant = 9;
  mcp.common.v2.Metadata metadata = 10;
}

message GetConversationStateResponse {
  mcp.common.v2.Status status = 1;
  ConversationState conversation_state = 2;
  mcp.common.v2.PerformanceMetrics performance_metrics = 3;
  mcp.common.v2.Metadata metadata = 4;
}

message ResetConversationResponse {
  mcp.common.v2.Status status = 1;
  string conversation_id = 2;
  string previous_state = 3;
  mcp.common.v2.PerformanceMetrics performance_metrics = 4;
  mcp.common.v2.Metadata metadata = 5;
}

message BatchProcessMessagesResponse {
  mcp.common.v2.Status status = 1;
  repeated ProcessMessageResponse responses = 2;
  int32 successful_count = 3;
  int32 failed_count = 4;
  mcp.common.v2.PerformanceMetrics performance_metrics = 5;
  mcp.common.v2.Metadata metadata = 6;
}

message HealthCheckResponse {
  mcp.common.v2.HealthStatus status = 1;
  string service_name = 2;
  string version = 3;
  google.protobuf.Timestamp timestamp = 4;
  google.protobuf.Struct details = 5;
}

message GetMetricsResponse {
  mcp.common.v2.Status status = 1;
  repeated MetricData metrics = 2;
  mcp.common.v2.PerformanceMetrics performance_metrics = 3;
  mcp.common.v2.Metadata metadata = 4;
}

message MetricData {
  string name = 1;
  string type = 2;
  repeated MetricPoint points = 3;
  google.protobuf.Struct labels = 4;
}

message MetricPoint {
  google.protobuf.Timestamp timestamp = 1;
  double value = 2;
  google.protobuf.Struct attributes = 3;
}

// Streaming messages
message ConversationStreamRequest {
  oneof request_type {
    StreamInit init = 1;
    ProcessMessageRequest message = 2;
    StreamHeartbeat heartbeat = 3;
    StreamClose close = 4;
  }
}

message ConversationStreamResponse {
  oneof response_type {
    StreamAck ack = 1;
    ProcessMessageResponse message_response = 2;
    StreamEvent event = 3;
    StreamError error = 4;
  }
}

message StreamInit {
  mcp.common.v2.TenantContext tenant_context = 1;
  string conversation_id = 2;
  repeated string event_types = 3;
  mcp.common.v2.Metadata metadata = 4;
}

message StreamAck {
  string stream_id = 1;
  mcp.common.v2.Status status = 2;
  mcp.common.v2.Metadata metadata = 3;
}

message StreamHeartbeat {
  google.protobuf.Timestamp timestamp = 1;
}

message StreamClose {
  string reason = 1;
  mcp.common.v2.Metadata metadata = 2;
}

message StreamEvent {
  string event_type = 1;
  string conversation_id = 2;
  google.protobuf.Struct event_data = 3;
  google.protobuf.Timestamp timestamp = 4;
}

message StreamError {
  string error_code = 1;
  string message = 2;
  bool recoverable = 3;
  mcp.common.v2.Metadata metadata = 4;
}
```

### `/src/api/grpc/proto/health.proto`
**Purpose**: Health check service definition
```protobuf
syntax = "proto3";

package mcp.health.v2;

import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";
import "common.proto";

option go_package = "github.com/mcp-platform/proto/health/v2";
option java_package = "com.mcp.health.v2";
option java_outer_classname = "HealthProto";

// Health check service
service Health {
  // Check health of specific service
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
  
  // Watch health status changes
  rpc Watch(HealthCheckRequest) returns (stream HealthCheckResponse);
  
  // Get detailed health information
  rpc GetDetailedHealth(DetailedHealthRequest) returns (DetailedHealthResponse);
}

message HealthCheckRequest {
  string service = 1;
  mcp.common.v2.Metadata metadata = 2;
}

message HealthCheckResponse {
  mcp.common.v2.HealthStatus status = 1;
  string service = 2;
  google.protobuf.Timestamp timestamp = 3;
  google.protobuf.Struct details = 4;
}

message DetailedHealthRequest {
  string service = 1;
  bool include_dependencies = 2;
  bool include_metrics = 3;
  mcp.common.v2.Metadata metadata = 4;
}

message DetailedHealthResponse {
  mcp.common.v2.HealthStatus status = 1;
  string service = 2;
  string version = 3;
  google.protobuf.Timestamp timestamp = 4;
  google.protobuf.Timestamp uptime_since = 5;
  
  // Dependency health
  repeated DependencyHealth dependencies = 6;
  
  // Resource usage
  ResourceUsage resource_usage = 7;
  
  // Service metrics
  google.protobuf.Struct metrics = 8;
  
  // Additional details
  google.protobuf.Struct details = 9;
}

message DependencyHealth {
  string name = 1;
  string type = 2;
  mcp.common.v2.HealthStatus status = 3;
  int32 response_time_ms = 4;
  string error_message = 5;
  google.protobuf.Timestamp last_check = 6;
}

message ResourceUsage {
  double cpu_percent = 1;
  int64 memory_used_bytes = 2;
  int64 memory_total_bytes = 3;
  double memory_percent = 4;
  int32 active_connections = 5;
  int32 goroutines = 6;
  google.protobuf.Struct disk_usage = 7;
}
```

## Step 20: gRPC Service Implementation (Days 74-76)

### `/src/api/grpc/services/mcp_service.py`
**Purpose**: Main MCP Engine gRPC service implementation
```python
import asyncio
import grpc
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime
import uuid
import time

from src.api.grpc.proto import mcp_engine_pb2, mcp_engine_pb2_grpc, common_pb2
from src.services.execution_service import ExecutionService
from src.services.flow_service import FlowService
from src.services.context_service import ContextService
from src.utils.logger import get_logger
from src.utils.metrics import MetricsCollector
from src.exceptions.base import MCPBaseException

logger = get_logger(__name__)

class MCPEngineServicer(mcp_engine_pb2_grpc.MCPEngineServicer):
    """gRPC service implementation for MCP Engine"""
    
    def __init__(self):
        self.execution_service = ExecutionService()
        self.flow_service = FlowService()
        self.context_service = ContextService()
        
        # Active streams tracking
        self._active_streams: Dict[str, Any] = {}
        
        # Initialize services
        asyncio.create_task(self._initialize_services())
    
    async def _initialize_services(self):
        """Initialize all services"""
        try:
            await self.execution_service.initialize()
            await self.flow_service.initialize()
            await self.context_service.initialize()
            logger.info("MCP Engine gRPC service initialized")
        except Exception as e:
            logger.error("Failed to initialize gRPC service", error=e)
            raise
    
    async def ProcessMessage(
        self,
        request: mcp_engine_pb2.ProcessMessageRequest,
        context: grpc.aio.ServicerContext
    ) -> mcp_engine_pb2.ProcessMessageResponse:
        """
        Process a conversation message
        
        Args:
            request: Process message request
            context: gRPC context
            
        Returns:
            Process message response
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Extract tenant and user info
        tenant_id = request.tenant_context.tenant_id
        user_id = request.user_id
        conversation_id = request.conversation_id
        
        service_logger = logger.bind_context(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id
        )
        
        try:
            service_logger.info("Processing gRPC message request")
            
            # Validate request
            await self._validate_process_message_request(request)
            
            # Convert protobuf to domain objects
            message_content = self._convert_message_content(request.content)
            processing_hints = self._convert_processing_hints(request.processing_hints)
            
            # Process message through execution service
            result = await self.execution_service.process_message(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                message_content=message_content,
                user_id=user_id,
                channel=self._convert_channel(request.channel),
                session_id=request.session_id,
                processing_hints=processing_hints
            )
            
            # Convert result to protobuf response
            response = self._build_process_message_response(
                result,
                start_time,
                request_id
            )
            
            # Record metrics
            execution_time = time.time() - start_time
            MetricsCollector.record_grpc_call(
                service_name="mcp_engine",
                method="ProcessMessage",
                duration_seconds=execution_time,
                tenant_id=tenant_id,
                success=True
            )
            
            service_logger.info(
                "Message processed successfully",
                processing_time_ms=int(execution_time * 1000)
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record error metrics
            MetricsCollector.record_grpc_call(
                service_name="mcp_engine",
                method="ProcessMessage",
                duration_seconds=execution_time,
                tenant_id=tenant_id,
                success=False
            )
            
            service_logger.error(
                "Message processing failed",
                error=e,
                processing_time_ms=int(execution_time * 1000)
            )
            
            # Convert exception to gRPC error
            await self._handle_grpc_error(context, e, "ProcessMessage")
            
            # Return error response
            return self._build_error_response(
                e,
                start_time,
                request_id
            )
    
    async def GetConversationState(
        self,
        request: mcp_engine_pb2.GetConversationStateRequest,
        context: grpc.aio.ServicerContext
    ) -> mcp_engine_pb2.GetConversationStateResponse:
        """
        Get conversation state
        
        Args:
            request: Get conversation state request
            context: gRPC context
            
        Returns:
            Conversation state response
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        tenant_id = request.tenant_context.tenant_id
        conversation_id = request.conversation_id
        
        try:
            logger.info(
                "Getting conversation state",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                request_id=request_id
            )
            
            # Get conversation state
            state_info = await self.execution_service.get_conversation_state(
                tenant_id=tenant_id,
                conversation_id=conversation_id
            )
            
            if not state_info:
                await context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    "Conversation not found"
                )
            
            # Build response
            response = mcp_engine_pb2.GetConversationStateResponse()
            
            # Set status
            response.status.success = True
            response.status.message = "Conversation state retrieved successfully"
            
            # Set conversation state
            response.conversation_state.CopyFrom(
                self._build_conversation_state(state_info)
            )
            
            # Set performance metrics
            execution_time = time.time() - start_time
            response.performance_metrics.execution_time_ms = int(execution_time * 1000)
            
            # Set metadata
            response.metadata.request_id = request_id
            response.metadata.timestamp.FromDatetime(datetime.utcnow())
            response.metadata.processing_time_ms = int(execution_time * 1000)
            
            return response
            
        except Exception as e:
            logger.error(
                "Failed to get conversation state",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                error=e
            )
            
            await self._handle_grpc_error(context, e, "GetConversationState")
    
    async def ResetConversation(
        self,
        request: mcp_engine_pb2.ResetConversationRequest,
        context: grpc.aio.ServicerContext
    ) -> mcp_engine_pb2.ResetConversationResponse:
        """
        Reset conversation
        
        Args:
            request: Reset conversation request
            context: gRPC context
            
        Returns:
            Reset conversation response
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        tenant_id = request.tenant_context.tenant_id
        conversation_id = request.conversation_id
        reason = request.reason
        
        try:
            logger.info(
                "Resetting conversation",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                reason=reason,
                request_id=request_id
            )
            
            # Reset conversation
            success = await self.execution_service.reset_conversation(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                reason=reason
            )
            
            if not success:
                await context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    "Conversation not found"
                )
            
            # Build response
            response = mcp_engine_pb2.ResetConversationResponse()
            
            # Set status
            response.status.success = True
            response.status.message = "Conversation reset successfully"
            
            # Set conversation ID
            response.conversation_id = conversation_id
            
            # Set performance metrics
            execution_time = time.time() - start_time
            response.performance_metrics.execution_time_ms = int(execution_time * 1000)
            
            # Set metadata
            response.metadata.request_id = request_id
            response.metadata.timestamp.FromDatetime(datetime.utcnow())
            response.metadata.processing_time_ms = int(execution_time * 1000)
            
            return response
            
        except Exception as e:
            logger.error(
                "Failed to reset conversation",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                error=e
            )
            
            await self._handle_grpc_error(context, e, "ResetConversation")
    
    async def StreamConversation(
        self,
        request_iterator: AsyncIterator[mcp_engine_pb2.ConversationStreamRequest],
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[mcp_engine_pb2.ConversationStreamResponse]:
        """
        Bi-directional streaming for real-time conversation
        
        Args:
            request_iterator: Stream of conversation requests
            context: gRPC context
            
        Yields:
            Stream of conversation responses
        """
        stream_id = str(uuid.uuid4())
        tenant_id = None
        conversation_id = None
        
        try:
            logger.info("Starting conversation stream", stream_id=stream_id)
            
            # Process incoming requests
            async for request in request_iterator:
                if request.HasField('init'):
                    # Initialize stream
                    tenant_id = request.init.tenant_context.tenant_id
                    conversation_id = request.init.conversation_id
                    
                    # Store stream info
                    self._active_streams[stream_id] = {
                        'tenant_id': tenant_id,
                        'conversation_id': conversation_id,
                        'started_at': datetime.utcnow(),
                        'event_types': list(request.init.event_types)
                    }
                    
                    # Send acknowledgment
                    ack_response = mcp_engine_pb2.ConversationStreamResponse()
                    ack_response.ack.stream_id = stream_id
                    ack_response.ack.status.success = True
                    ack_response.ack.status.message = "Stream initialized"
                    yield ack_response
                    
                elif request.HasField('message'):
                    # Process message
                    try:
                        # Convert and process message
                        message_content = self._convert_message_content(request.message.content)
                        processing_hints = self._convert_processing_hints(request.message.processing_hints)
                        
                        result = await self.execution_service.process_message(
                            tenant_id=request.message.tenant_context.tenant_id,
                            conversation_id=request.message.conversation_id,
                            message_content=message_content,
                            user_id=request.message.user_id,
                            channel=self._convert_channel(request.message.channel),
                            session_id=request.message.session_id,
                            processing_hints=processing_hints
                        )
                        
                        # Send response
                        stream_response = mcp_engine_pb2.ConversationStreamResponse()
                        stream_response.message_response.CopyFrom(
                            self._build_process_message_response(result, time.time(), stream_id)
                        )
                        yield stream_response
                        
                    except Exception as e:
                        # Send error response
                        error_response = mcp_engine_pb2.ConversationStreamResponse()
                        error_response.error.error_code = "PROCESSING_ERROR"
                        error_response.error.message = str(e)
                        error_response.error.recoverable = True
                        yield error_response
                
                elif request.HasField('heartbeat'):
                    # Handle heartbeat
                    logger.debug("Heartbeat received", stream_id=stream_id)
                
                elif request.HasField('close'):
                    # Handle stream close
                    logger.info(
                        "Stream close requested",
                        stream_id=stream_id,
                        reason=request.close.reason
                    )
                    break
            
            logger.info("Conversation stream ended", stream_id=stream_id)
            
        except Exception as e:
            logger.error(
                "Conversation stream error",
                stream_id=stream_id,
                error=e
            )
            
            # Send final error
            error_response = mcp_engine_pb2.ConversationStreamResponse()
            error_response.error.error_code = "STREAM_ERROR"
            error_response.error.message = str(e)
            error_response.error.recoverable = False
            yield error_response
            
        finally:
            # Cleanup stream
            self._active_streams.pop(stream_id, None)
    
    async def BatchProcessMessages(
        self,
        request: mcp_engine_pb2.BatchProcessMessagesRequest,
        context: grpc.aio.ServicerContext
    ) -> mcp_engine_pb2.BatchProcessMessagesResponse:
        """
        Batch process multiple messages
        
        Args:
            request: Batch process messages request
            context: gRPC context
            
        Returns:
            Batch process response
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        tenant_id = request.tenant_context.tenant_id
        
        try:
            logger.info(
                "Processing message batch",
                tenant_id=tenant_id,
                message_count=len(request.messages),
                parallel=request.parallel_processing,
                request_id=request_id
            )
            
            responses = []
            successful_count = 0
            failed_count = 0
            
            if request.parallel_processing:
                # Process messages in parallel
                max_concurrency = request.max_concurrency or 10
                semaphore = asyncio.Semaphore(max_concurrency)
                
                async def process_single_message(msg_request):
                    async with semaphore:
                        try:
                            message_content = self._convert_message_content(msg_request.content)
                            processing_hints = self._convert_processing_hints(msg_request.processing_hints)
                            
                            result = await self.execution_service.process_message(
                                tenant_id=msg_request.tenant_context.tenant_id,
                                conversation_id=msg_request.conversation_id,
                                message_content=message_content,
                                user_id=msg_request.user_id,
                                channel=self._convert_channel(msg_request.channel),
                                session_id=msg_request.session_id,
                                processing_hints=processing_hints
                            )
                            
                            return self._build_process_message_response(result, time.time(), request_id), True
                            
                        except Exception as e:
                            return self._build_error_response(e, time.time(), request_id), False
                
                # Execute all messages concurrently
                tasks = [process_single_message(msg) for msg in request.messages]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        failed_count += 1
                        responses.append(self._build_error_response(result, time.time(), request_id))
                    else:
                        response, success = result
                        responses.append(response)
                        if success:
                            successful_count += 1
                        else:
                            failed_count += 1
            else:
                # Process messages sequentially
                for msg_request in request.messages:
                    try:
                        message_content = self._convert_message_content(msg_request.content)
                        processing_hints = self._convert_processing_hints(msg_request.processing_hints)
                        
                        result = await self.execution_service.process_message(
                            tenant_id=msg_request.tenant_context.tenant_id,
                            conversation_id=msg_request.conversation_id,
                            message_content=message_content,
                            user_id=msg_request.user_id,
                            channel=self._convert_channel(msg_request.channel),
                            session_id=msg_request.session_id,
                            processing_hints=processing_hints
                        )
                        
                        responses.append(self._build_process_message_response(result, time.time(), request_id))
                        successful_count += 1
                        
                    except Exception as e:
                        responses.append(self._build_error_response(e, time.time(), request_id))
                        failed_count += 1
            
            # Build batch response
            batch_response = mcp_engine_pb2.BatchProcessMessagesResponse()
            batch_response.status.success = failed_count == 0
            batch_response.status.message = f"Processed {successful_count}/{len(request.messages)} messages successfully"
            
            batch_response.responses.extend(responses)
            batch_response.successful_count = successful_count
            batch_response.failed_count = failed_count
            
            # Set performance metrics
            execution_time = time.time() - start_time
            batch_response.performance_metrics.execution_time_ms = int(execution_time * 1000)
            
            # Set metadata
            batch_response.metadata.request_id = request_id
            batch_response.metadata.timestamp.FromDatetime(datetime.utcnow())
            batch_response.metadata.processing_time_ms = int(execution_time * 1000)
            
            return batch_response
            
        except Exception as e:
            logger.error(
                "Batch processing failed",
                tenant_id=tenant_id,
                error=e
            )
            
            await self._handle_grpc_error(context, e, "BatchProcessMessages")
    
    async def HealthCheck(
        self,
        request,
        context: grpc.aio.ServicerContext
    ) -> mcp_engine_pb2.HealthCheckResponse:
        """
        Health check endpoint
        
        Args:
            request: Empty request
            context: gRPC context
            
        Returns:
            Health check response
        """
        try:
            # Perform health checks on dependencies
            health_info = await self.execution_service.health_check()
            
            response = mcp_engine_pb2.HealthCheckResponse()
            
            if health_info.get("status") == "healthy":
                response.status = common_pb2.HEALTH_SERVING
            else:
                response.status = common_pb2.HEALTH_NOT_SERVING
            
            response.service_name = "mcp-engine"
            response.version = "2.0.0"
            response.timestamp.FromDatetime(datetime.utcnow())
            
            # Add health details
            response.details.update(health_info)
            
            return response
            
        except Exception as e:
            logger.error("Health check failed", error=e)
            
            response = mcp_engine_pb2.HealthCheckResponse()
            response.status = common_pb2.HEALTH_NOT_SERVING
            response.service_name = "mcp-engine"
            response.version = "2.0.0"
            response.timestamp.FromDatetime(datetime.utcnow())
            response.details.update({"error": str(e)})
            
            return response
    
    async def GetMetrics(
        self,
        request: mcp_engine_pb2.GetMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> mcp_engine_pb2.GetMetricsResponse:
        """
        Get service metrics
        
        Args:
            request: Get metrics request
            context: gRPC context
            
        Returns:
            Metrics response
        """
        try:
            # TODO: Implement metrics collection
            response = mcp_engine_pb2.GetMetricsResponse()
            response.status.success = True
            response.status.message = "Metrics retrieved successfully"
            
            # Set metadata
            response.metadata.request_id = str(uuid.uuid4())
            response.metadata.timestamp.FromDatetime(datetime.utcnow())
            
            return response
            
        except Exception as e:
            logger.error("Failed to get metrics", error=e)
            await self._handle_grpc_error(context, e, "GetMetrics")
    
    # Helper methods
    
    def _convert_message_content(self, proto_content) -> Dict[str, Any]:
        """Convert protobuf message content to dict"""
        content = {
            "type": self._convert_message_type(proto_content.type),
            "text": proto_content.text,
            "payload": proto_content.payload,
            "language": proto_content.language,
            "metadata": dict(proto_content.metadata)
        }
        
        # Convert media content
        if proto_content.HasField('media'):
            content["media"] = {
                "url": proto_content.media.url,
                "secure_url": proto_content.media.secure_url,
                "mime_type": proto_content.media.mime_type,
                "size_bytes": proto_content.media.size_bytes,
                "duration_ms": proto_content.media.duration_ms
            }
        
        # Convert location content
        if proto_content.HasField('location'):
            content["location"] = {
                "latitude": proto_content.location.latitude,
                "longitude": proto_content.location.longitude,
                "accuracy_meters": proto_content.location.accuracy_meters,
                "address": proto_content.location.address
            }
        
        return content
    
    def _convert_message_type(self, proto_type) -> str:
        """Convert protobuf message type to string"""
        type_mapping = {
            common_pb2.MESSAGE_TYPE_TEXT: "text",
            common_pb2.MESSAGE_TYPE_IMAGE: "image",
            common_pb2.MESSAGE_TYPE_FILE: "file",
            common_pb2.MESSAGE_TYPE_AUDIO: "audio",
            common_pb2.MESSAGE_TYPE_VIDEO: "video",
            common_pb2.MESSAGE_TYPE_LOCATION: "location",
            common_pb2.MESSAGE_TYPE_QUICK_REPLY: "quick_reply",
            common_pb2.MESSAGE_TYPE_CAROUSEL: "carousel",
            common_pb2.MESSAGE_TYPE_FORM: "form",
            common_pb2.MESSAGE_TYPE_SYSTEM: "system"
        }
        return type_mapping.get(proto_type, "text")
    
    def _convert_channel(self, proto_channel) -> str:
        """Convert protobuf channel to string"""
        channel_mapping = {
            common_pb2.CHANNEL_WEB: "web",
            common_pb2.CHANNEL_WHATSAPP: "whatsapp",
            common_pb2.CHANNEL_MESSENGER: "messenger",
            common_pb2.CHANNEL_SLACK: "slack",
            common_pb2.CHANNEL_TEAMS: "teams",
            common_pb2.CHANNEL_VOICE: "voice",
            common_pb2.CHANNEL_SMS: "sms"
        }
        return channel_mapping.get(proto_channel, "web")
    
    def _convert_processing_hints(self, proto_hints) -> Optional[Dict[str, Any]]:
        """Convert protobuf processing hints to dict"""
        if not proto_hints:
            return None
        
        return {
            "priority": self._convert_priority(proto_hints.priority),
            "expected_response_type": proto_hints.expected_response_type,
            "bypass_automation": proto_hints.bypass_automation,
            "require_human_review": proto_hints.require_human_review,
            "bypass_cache": proto_hints.bypass_cache,
            "force_flow": proto_hints.force_flow,
            "custom_hints": dict(proto_hints.custom_hints)
        }
    
    def _convert_priority(self, proto_priority) -> str:
        """Convert protobuf priority to string"""
        priority_mapping = {
            common_pb2.PRIORITY_LOW: "low",
            common_pb2.PRIORITY_NORMAL: "normal",
            common_pb2.PRIORITY_HIGH: "high",
            common_pb2.PRIORITY_URGENT: "urgent",
            common_pb2.PRIORITY_CRITICAL: "critical"
        }
        return priority_mapping.get(proto_priority, "normal")
    
    async def _validate_process_message_request(self, request):
        """Validate process message request"""
        if not request.tenant_context.tenant_id:
            raise ValueError("Tenant ID is required")
        
        if not request.conversation_id:
            raise ValueError("Conversation ID is required")
        
        if not request.user_id:
            raise ValueError("User ID is required")
        
        if not request.content.text and not request.content.payload:
            raise ValueError("Message must contain either text or payload")
    
    def _build_process_message_response(
        self,
        result,
        start_time: float,
        request_id: str
    ) -> mcp_engine_pb2.ProcessMessageResponse:
        """Build process message response from result"""
        response = mcp_engine_pb2.ProcessMessageResponse()
        
        # Set status
        response.status.success = result.success
        response.status.message = "Message processed successfully" if result.success else "Message processing failed"
        
        # Set basic fields
        response.conversation_id = result.conversation_id
        response.current_state = result.current_state
        
        # Set response content
        if result.response:
            response.response.type = common_pb2.MESSAGE_TYPE_TEXT
            response.response.text = result.response.get("text", "")
        
        # Set context updates
        response.context_updates.update(result.context_updates)
        
        # Set actions performed
        response.actions_performed.extend(result.actions_performed)
        
        # Set performance metrics
        execution_time = time.time() - start_time
        response.performance_metrics.execution_time_ms = int(execution_time * 1000)
        
        # Set metadata
        response.metadata.request_id = request_id
        response.metadata.timestamp.FromDatetime(datetime.utcnow())
        response.metadata.processing_time_ms = int(execution_time * 1000)
        
        return response
    
    def _build_error_response(
        self,
        error: Exception,
        start_time: float,
        request_id: str
    ) -> mcp_engine_pb2.ProcessMessageResponse:
        """Build error response"""
        response = mcp_engine_pb2.ProcessMessageResponse()
        
        # Set status
        response.status.success = False
        response.status.message = str(error)
        
        if isinstance(error, MCPBaseException):
            response.status.error_code = error.error_code.value
        else:
            response.status.error_code = "INTERNAL_ERROR"
        
        # Set basic fields
        response.conversation_id = ""
        response.current_state = "error"
        
        # Set error response
        response.response.type = common_pb2.MESSAGE_TYPE_TEXT
        response.response.text = "I'm sorry, I encountered an error processing your message. Please try again."
        
        # Set performance metrics
        execution_time = time.time() - start_time
        response.performance_metrics.execution_time_ms = int(execution_time * 1000)
        
        # Set metadata
        response.metadata.request_id = request_id
        response.metadata.timestamp.FromDatetime(datetime.utcnow())
        response.metadata.processing_time_ms = int(execution_time * 1000)
        
        return response
    
    def _build_conversation_state(self, state_info: Dict[str, Any]) -> mcp_engine_pb2.ConversationState:
        """Build conversation state protobuf"""
        state = mcp_engine_pb2.ConversationState()
        
        state.current_state = state_info.get("current_state", "")
        state.flow_id = state_info.get("flow_id", "")
        state.flow_version = "1.0"  # TODO: Get from state_info
        
        # Set slots and variables
        if "slots" in state_info:
            state.slots.update(state_info["slots"])
        
        if "variables" in state_info:
            state.variables.update(state_info["variables"])
        
        # Set timestamps
        if "created_at" in state_info:
            try:
                created_at = datetime.fromisoformat(state_info["created_at"])
                state.created_at.FromDatetime(created_at)
            except:
                state.created_at.FromDatetime(datetime.utcnow())
        
        if "last_activity" in state_info:
            try:
                last_activity = datetime.fromisoformat(state_info["last_activity"])
                state.last_activity.FromDatetime(last_activity)
            except:
                state.last_activity.FromDatetime(datetime.utcnow())
        
        state.message_count = state_info.get("message_count", 0)
        
        return state
    
    async def _handle_grpc_error(
        self,
        context: grpc.aio.ServicerContext,
        error: Exception,
        method_name: str
    ):
        """Handle gRPC errors and set appropriate status codes"""
        
        if isinstance(error, MCPBaseException):
            # Map MCP exceptions to gRPC status codes
            error_code_mapping = {
                "VALIDATION_ERROR": grpc.StatusCode.INVALID_ARGUMENT,
                "FLOW_NOT_FOUND": grpc.StatusCode.NOT_FOUND,
                "STATE_NOT_FOUND": grpc.StatusCode.NOT_FOUND,
                "CONTEXT_LOCK_ERROR": grpc.StatusCode.RESOURCE_EXHAUSTED,
                "INTEGRATION_TIMEOUT": grpc.StatusCode.DEADLINE_EXCEEDED,
                "AUTHENTICATION_FAILED": grpc.StatusCode.UNAUTHENTICATED,
                "AUTHORIZATION_FAILED": grpc.StatusCode.PERMISSION_DENIED
            }
            
            grpc_code = error_code_mapping.get(
                error.error_code.value,
                grpc.StatusCode.INTERNAL
            )
            
            await context.abort(grpc_code, error.message)
        
        elif isinstance(error, ValueError):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        
        elif isinstance(error, TimeoutError):
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(error))
        
        else:
            logger.error(f"Unhandled error in {method_name}", error=error)
            await context.abort(grpc.StatusCode.INTERNAL, "Internal server error")
```

## Success Criteria
- [x] Complete Protocol Buffers definitions for all services
- [x] Full gRPC service implementation with all endpoints
- [x] Bi-directional streaming support for real-time conversations
- [x] Batch processing capabilities for high-throughput scenarios
- [x] Comprehensive error handling with proper gRPC status codes
- [x] Performance monitoring and metrics collection
- [x] Health check and service discovery integration
- [x] Type-safe protobuf conversion and validation

## Key Error Handling & Performance Considerations
1. **Protocol Buffers**: Efficient binary serialization with backward compatibility
2. **gRPC Streaming**: Bi-directional streaming with connection management
3. **Error Handling**: Proper gRPC status codes and error propagation
4. **Performance**: Async processing with connection pooling
5. **Type Safety**: Comprehensive protobuf validation and conversion
6. **Monitoring**: Request tracing and performance metrics
7. **Health Checks**: Service discovery and dependency monitoring

## Technologies Used
- **Protocol Buffers**: Language-neutral serialization
- **gRPC**: High-performance RPC framework with streaming
- **Async Python**: asyncio with grpc.aio for async operations
- **Service Discovery**: Health checks and service registration
- **Monitoring**: Metrics collection and request tracing
- **Error Handling**: Comprehensive exception mapping

## Cross-Service Integration
- **Execution Service**: Message processing and conversation management
- **Flow Service**: Flow operations and lifecycle management
- **Context Service**: Conversation state management
- **Analytics**: Performance metrics and usage tracking
- **Health Monitoring**: Service health and dependency checks
- **Authentication**: Request validation and authorization

## Next Phase Dependencies
Phase 9 will build upon:
- gRPC service infrastructure and communication protocols
- Protocol Buffers for type-safe serialization
- Streaming capabilities for real-time communication
- Error handling and monitoring frameworks
- Performance optimization and metrics collection