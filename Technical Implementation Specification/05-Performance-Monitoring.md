# Performance Monitoring
## Multi-Tenant AI Chatbot Platform

**Document:** 05-Performance-Monitoring.md  
**Version:** 2.0  
**Last Updated:** May 30, 2025

---

## Table of Contents

1. [Monitoring Strategy Overview](#monitoring-strategy-overview)
2. [Metrics Collection](#metrics-collection)
3. [Performance Optimization](#performance-optimization)
4. [Alerting and Notification](#alerting-and-notification)
5. [Observability Implementation](#observability-implementation)
6. [Dashboard Configuration](#dashboard-configuration)
7. [Capacity Planning](#capacity-planning)
8. [Troubleshooting Guides](#troubleshooting-guides)

---

## Monitoring Strategy Overview

### Monitoring Philosophy

1. **Four Golden Signals:** Latency, Traffic, Errors, Saturation
2. **USE Method:** Utilization, Saturation, Errors for resources
3. **RED Method:** Rate, Errors, Duration for services
4. **Business Metrics:** Customer-centric performance indicators
5. **Proactive Monitoring:** Predict issues before they impact users

### Monitoring Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────┘

Data Collection:
├── Application Metrics (Prometheus)
├── Infrastructure Metrics (Node Exporter)
├── Business Metrics (Custom Exporters)
├── Log Aggregation (ELK Stack)
├── Distributed Tracing (Jaeger)
└── Real-User Monitoring (RUM)

Data Storage:
├── Prometheus (Short-term metrics)
├── TimescaleDB (Long-term metrics)
├── Elasticsearch (Logs)
└── S3 (Long-term storage)

Visualization:
├── Grafana (Technical dashboards)
├── Kibana (Log analysis)
├── Custom Dashboards (Business metrics)
└── Mobile Apps (Critical alerts)

Alerting:
├── AlertManager (Prometheus alerts)
├── PagerDuty (Incident management)
├── Slack (Team notifications)
└── Email (Non-critical alerts)
```

---

## Metrics Collection

### Application Metrics

#### Service-Level Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, Summary
from functools import wraps
import time
from datetime import datetime

# Create custom registry for better organization
REGISTRY = CollectorRegistry()

# HTTP Request Metrics
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status', 'tenant_id', 'service'],
    registry=REGISTRY
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'tenant_id', 'service'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')],
    registry=REGISTRY
)

HTTP_REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint', 'service'],
    buckets=[100, 1000, 10000, 100000, 1000000, float('inf')],
    registry=REGISTRY
)

# Business Metrics
CONVERSATIONS_ACTIVE = Gauge(
    'conversations_active_total',
    'Number of active conversations',
    ['tenant_id', 'channel'],
    registry=REGISTRY
)

CONVERSATIONS_STARTED = Counter(
    'conversations_started_total',
    'Total conversations started',
    ['tenant_id', 'channel'],
    registry=REGISTRY
)

CONVERSATIONS_COMPLETED = Counter(
    'conversations_completed_total',
    'Total conversations completed',
    ['tenant_id', 'channel', 'resolution_status'],
    registry=REGISTRY
)

MESSAGES_PROCESSED = Counter(
    'messages_processed_total',
    'Total messages processed',
    ['tenant_id', 'channel', 'direction', 'message_type'],
    registry=REGISTRY
)

# AI Model Metrics
MODEL_API_CALLS = Counter(
    'model_api_calls_total',
    'Total model API calls',
    ['provider', 'model', 'tenant_id', 'operation_type', 'status'],
    registry=REGISTRY
)

MODEL_API_DURATION = Histogram(
    'model_api_duration_seconds',
    'Model API call duration in seconds',
    ['provider', 'model', 'tenant_id', 'operation_type'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')],
    registry=REGISTRY
)

MODEL_API_COST = Counter(
    'model_api_cost_cents_total',
    'Total model API cost in cents',
    ['provider', 'model', 'tenant_id'],
    registry=REGISTRY
)

MODEL_API_TOKENS = Counter(
    'model_api_tokens_total',
    'Total tokens used',
    ['provider', 'model', 'tenant_id', 'token_type'],
    registry=REGISTRY
)

# Integration Metrics
INTEGRATION_CALLS = Counter(
    'integration_calls_total',
    'Total integration calls',
    ['tenant_id', 'integration_id', 'integration_type', 'status'],
    registry=REGISTRY
)

INTEGRATION_DURATION = Histogram(
    'integration_duration_seconds',
    'Integration call duration in seconds',
    ['tenant_id', 'integration_id', 'integration_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float('inf')],
    registry=REGISTRY
)

# Database Metrics
DATABASE_CONNECTIONS_ACTIVE = Gauge(
    'database_connections_active',
    'Active database connections',
    ['database_type', 'database_name'],
    registry=REGISTRY
)

DATABASE_QUERY_DURATION = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['database_type', 'operation', 'table'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, float('inf')],
    registry=REGISTRY
)

# Queue Metrics
QUEUE_SIZE = Gauge(
    'queue_size',
    'Number of items in queue',
    ['queue_name', 'tenant_id'],
    registry=REGISTRY
)

QUEUE_PROCESSING_DURATION = Histogram(
    'queue_processing_duration_seconds',
    'Queue item processing duration',
    ['queue_name', 'message_type'],
    registry=REGISTRY
)

# Error Metrics
ERRORS_TOTAL = Counter(
    'errors_total',
    'Total errors by type',
    ['error_type', 'service', 'tenant_id'],
    registry=REGISTRY
)

# Custom Metrics for Business Logic
class BusinessMetricsCollector:
    """Collector for business-specific metrics"""
    
    def __init__(self, redis_client, database):
        self.redis = redis_client
        self.db = database
    
    def track_conversation_started(self, tenant_id: str, channel: str):
        """Track when a conversation starts"""
        CONVERSATIONS_STARTED.labels(
            tenant_id=tenant_id,
            channel=channel
        ).inc()
        
        CONVERSATIONS_ACTIVE.labels(
            tenant_id=tenant_id,
            channel=channel
        ).inc()
    
    def track_conversation_completed(self, tenant_id: str, channel: str, 
                                   resolution_status: str):
        """Track when a conversation completes"""
        CONVERSATIONS_COMPLETED.labels(
            tenant_id=tenant_id,
            channel=channel,
            resolution_status=resolution_status
        ).inc()
        
        CONVERSATIONS_ACTIVE.labels(
            tenant_id=tenant_id,
            channel=channel
        ).dec()
    
    def track_message_processed(self, tenant_id: str, channel: str, 
                              direction: str, message_type: str):
        """Track message processing"""
        MESSAGES_PROCESSED.labels(
            tenant_id=tenant_id,
            channel=channel,
            direction=direction,
            message_type=message_type
        ).inc()
    
    def track_model_api_call(self, provider: str, model: str, tenant_id: str,
                           operation_type: str, duration: float, 
                           tokens_used: int, cost_cents: float, status: str):
        """Track model API usage"""
        MODEL_API_CALLS.labels(
            provider=provider,
            model=model,
            tenant_id=tenant_id,
            operation_type=operation_type,
            status=status
        ).inc()
        
        MODEL_API_DURATION.labels(
            provider=provider,
            model=model,
            tenant_id=tenant_id,
            operation_type=operation_type
        ).observe(duration)
        
        MODEL_API_TOKENS.labels(
            provider=provider,
            model=model,
            tenant_id=tenant_id,
            token_type='total'
        ).inc(tokens_used)
        
        MODEL_API_COST.labels(
            provider=provider,
            model=model,
            tenant_id=tenant_id
        ).inc(cost_cents)

# Decorators for automatic instrumentation
def track_execution_time(metric_name: str = None, labels: dict = None):
    """Decorator to track function execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                
                # Use function name if metric name not provided
                name = metric_name or f"{func.__module__}_{func.__name__}_duration_seconds"
                
                # Create or get existing histogram
                if name not in globals():
                    globals()[name] = Histogram(
                        name,
                        f'Execution time for {func.__name__}',
                        list(labels.keys()) + ['status'] if labels else ['status'],
                        registry=REGISTRY
                    )
                
                metric_labels = (labels or {}).copy()
                metric_labels['status'] = status
                globals()[name].labels(**metric_labels).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                
                name = metric_name or f"{func.__module__}_{func.__name__}_duration_seconds"
                
                if name not in globals():
                    globals()[name] = Histogram(
                        name,
                        f'Execution time for {func.__name__}',
                        list(labels.keys()) + ['status'] if labels else ['status'],
                        registry=REGISTRY
                    )
                
                metric_labels = (labels or {}).copy()
                metric_labels['status'] = status
                globals()[name].labels(**metric_labels).observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def track_api_calls(service_name: str):
    """Decorator to track API calls automatically"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            start_time = time.time()
            
            # Extract tenant ID from request
            tenant_id = request.headers.get("X-Tenant-ID", "unknown")
            method = request.method
            endpoint = request.url.path
            
            try:
                response = await func(request, *args, **kwargs)
                status_code = getattr(response, 'status_code', 200)
                
                # Track metrics
                HTTP_REQUESTS_TOTAL.labels(
                    method=method,
                    endpoint=endpoint,
                    status=str(status_code),
                    tenant_id=tenant_id,
                    service=service_name
                ).inc()
                
                return response
            except Exception as e:
                HTTP_REQUESTS_TOTAL.labels(
                    method=method,
                    endpoint=endpoint,
                    status="500",
                    tenant_id=tenant_id,
                    service=service_name
                ).inc()
                
                ERRORS_TOTAL.labels(
                    error_type=type(e).__name__,
                    service=service_name,
                    tenant_id=tenant_id
                ).inc()
                
                raise
            finally:
                duration = time.time() - start_time
                HTTP_REQUEST_DURATION.labels(
                    method=method,
                    endpoint=endpoint,
                    tenant_id=tenant_id,
                    service=service_name
                ).observe(duration)
        
        return wrapper
    return decorator
```

### Infrastructure Metrics

#### System Resource Monitoring

```yaml
# Prometheus Configuration for Infrastructure Monitoring
prometheus.yml:
  global:
    scrape_interval: 15s
    evaluation_interval: 15s
    external_labels:
      cluster: 'chatbot-platform'
      environment: 'production'

  rule_files:
    - "alert_rules.yml"
    - "recording_rules.yml"

  alerting:
    alertmanagers:
      - static_configs:
          - targets:
            - alertmanager:9093

  scrape_configs:
    # Application services
    - job_name: 'chat-service'
      static_configs:
        - targets: ['chat-service:8000']
      metrics_path: /metrics
      scrape_interval: 10s
      scrape_timeout: 5s
      
    - job_name: 'mcp-service'
      static_configs:
        - targets: ['mcp-service:8000']
      metrics_path: /metrics
      scrape_interval: 10s
      
    - job_name: 'model-orchestrator'
      static_configs:
        - targets: ['model-orchestrator:8000']
      metrics_path: /metrics
      scrape_interval: 10s
      
    - job_name: 'adaptor-service'
      static_configs:
        - targets: ['adaptor-service:8000']
      metrics_path: /metrics
      scrape_interval: 10s

    # Infrastructure monitoring
    - job_name: 'node-exporter'
      static_configs:
        - targets: ['node-exporter:9100']
      scrape_interval: 15s
      
    - job_name: 'postgres-exporter'
      static_configs:
        - targets: ['postgres-exporter:9187']
      scrape_interval: 30s
      
    - job_name: 'mongodb-exporter'
      static_configs:
        - targets: ['mongodb-exporter:9216']
      scrape_interval: 30s
      
    - job_name: 'redis-exporter'
      static_configs:
        - targets: ['redis-exporter:9121']
      scrape_interval: 30s
      
    - job_name: 'kafka-exporter'
      static_configs:
        - targets: ['kafka-exporter:9308']
      scrape_interval: 30s

    # Kubernetes monitoring
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
        - role: pod
      relabel_configs:
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
          action: replace
          target_label: __metrics_path__
          regex: (.+)
        - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          regex: ([^:]+)(?::\d+)?;(\d+)
          replacement: $1:$2
          target_label: __address__
```

#### Custom Infrastructure Monitoring

```python
import psutil
import time
from threading import Thread
from prometheus_client import Gauge, Counter
import redis
import pymongo
import psycopg2

class InfrastructureMonitor:
    """Custom infrastructure monitoring for specific platform needs"""
    
    def __init__(self, redis_client, postgres_conn, mongo_client):
        self.redis = redis_client
        self.postgres = postgres_conn
        self.mongo = mongo_client
        
        # Define infrastructure metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage', ['mount_point'])
        self.network_io = Counter('system_network_io_bytes_total', 'Network I/O bytes', ['direction'])
        
        # Database-specific metrics
        self.db_connections = Gauge('database_connections_current', 'Current database connections', ['database'])
        self.db_query_rate = Gauge('database_queries_per_second', 'Database queries per second', ['database'])
        self.db_slow_queries = Counter('database_slow_queries_total', 'Number of slow queries', ['database'])
        
        # Application-specific metrics
        self.active_websockets = Gauge('websocket_connections_active', 'Active WebSocket connections')
        self.message_queue_depth = Gauge('message_queue_depth', 'Message queue depth', ['queue_name'])
        self.cache_hit_rate = Gauge('cache_hit_rate_percent', 'Cache hit rate percentage', ['cache_type'])
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._collect_database_metrics()
                self._collect_application_metrics()
                time.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.percent)
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                self.disk_usage.labels(mount_point=partition.mountpoint).set(usage.percent)
            except PermissionError:
                pass
        
        # Network I/O
        network = psutil.net_io_counters()
        self.network_io.labels(direction='sent').inc(network.bytes_sent)
        self.network_io.labels(direction='received').inc(network.bytes_recv)
    
    def _collect_database_metrics(self):
        """Collect database-specific metrics"""
        
        # PostgreSQL metrics
        try:
            with self.postgres.cursor() as cursor:
                # Connection count
                cursor.execute("SELECT count(*) FROM pg_stat_activity")
                conn_count = cursor.fetchone()[0]
                self.db_connections.labels(database='postgresql').set(conn_count)
                
                # Query rate (queries per second)
                cursor.execute("""
                    SELECT sum(xact_commit + xact_rollback) 
                    FROM pg_stat_database 
                    WHERE datname NOT IN ('template0', 'template1', 'postgres')
                """)
                current_queries = cursor.fetchone()[0] or 0
                
                # Calculate rate (would need previous value stored)
                # This is simplified - in practice, store previous values
                self.db_query_rate.labels(database='postgresql').set(current_queries)
                
                # Slow queries (queries > 1 second)
                cursor.execute("""
                    SELECT count(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND now() - query_start > interval '1 second'
                """)
                slow_queries = cursor.fetchone()[0]
                if slow_queries > 0:
                    self.db_slow_queries.labels(database='postgresql').inc(slow_queries)
        except Exception as e:
            print(f"Error collecting PostgreSQL metrics: {e}")
        
        # MongoDB metrics
        try:
            server_status = self.mongo.admin.command("serverStatus")
            
            # Connection count
            current_conn = server_status['connections']['current']
            self.db_connections.labels(database='mongodb').set(current_conn)
            
            # Operation rate
            opcounts = server_status['opcounters']
            total_ops = sum(opcounts.values())
            self.db_query_rate.labels(database='mongodb').set(total_ops)
            
        except Exception as e:
            print(f"Error collecting MongoDB metrics: {e}")
        
        # Redis metrics
        try:
            redis_info = self.redis.info()
            
            # Connection count
            connected_clients = redis_info['connected_clients']
            self.db_connections.labels(database='redis').set(connected_clients)
            
            # Calculate hit rate
            keyspace_hits = redis_info.get('keyspace_hits', 0)
            keyspace_misses = redis_info.get('keyspace_misses', 0)
            total_requests = keyspace_hits + keyspace_misses
            
            if total_requests > 0:
                hit_rate = (keyspace_hits / total_requests) * 100
                self.cache_hit_rate.labels(cache_type='redis').set(hit_rate)
            
        except Exception as e:
            print(f"Error collecting Redis metrics: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Message queue depths
            queue_names = ['message.inbound.v1', 'message.outbound.v1', 'analytics.events.v1']
            
            for queue_name in queue_names:
                # This would integrate with your message queue system
                # For Kafka, you'd check consumer lag
                # For Redis queues, you'd check list length
                queue_depth = self._get_queue_depth(queue_name)
                self.message_queue_depth.labels(queue_name=queue_name).set(queue_depth)
            
            # Active WebSocket connections (if applicable)
            # This would integrate with your WebSocket management system
            active_ws = self._get_active_websocket_count()
            self.active_websockets.set(active_ws)
            
        except Exception as e:
            print(f"Error collecting application metrics: {e}")
    
    def _get_queue_depth(self, queue_name: str) -> int:
        """Get message queue depth - implement based on your queue system"""
        # Example for Redis-based queues
        try:
            return self.redis.llen(f"queue:{queue_name}")
        except:
            return 0
    
    def _get_active_websocket_count(self) -> int:
        """Get active WebSocket connection count"""
        # This would depend on your WebSocket implementation
        try:
            return int(self.redis.get("websocket:active_connections") or 0)
        except:
            return 0
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
```

---

## Performance Optimization

### Caching Strategy Implementation

```python
import hashlib
import json
import gzip
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class CacheConfig:
    ttl_seconds: int
    max_size_mb: float
    compression_threshold_kb: float = 100
    enable_compression: bool = True
    enable_encryption: bool = False

class IntelligentCacheService:
    """Advanced caching service with performance optimization"""
    
    def __init__(self, redis_client, local_cache_size_mb: float = 100):
        self.redis = redis_client
        self.local_cache = {}
        self.local_cache_size_mb = local_cache_size_mb
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0,
            "evictions": 0,
            "compressions": 0
        }
        
        # Cache configurations for different data types
        self.cache_configs = {
            "conversation_context": CacheConfig(
                ttl_seconds=3600,
                max_size_mb=2.0,
                enable_compression=True
            ),
            "user_session": CacheConfig(
                ttl_seconds=1800,
                max_size_mb=0.5,
                enable_compression=False
            ),
            "model_response": CacheConfig(
                ttl_seconds=7200,
                max_size_mb=5.0,
                enable_compression=True,
                compression_threshold_kb=50
            ),
            "tenant_config": CacheConfig(
                ttl_seconds=300,
                max_size_mb=1.0,
                enable_compression=False
            ),
            "integration_config": CacheConfig(
                ttl_seconds=600,
                max_size_mb=1.0,
                enable_compression=False
            ),
            "analytics_data": CacheConfig(
                ttl_seconds=900,
                max_size_mb=10.0,
                enable_compression=True
            )
        }
    
    def get(self, key: str, cache_type: str = "default") -> Optional[Any]:
        """Get value from cache with intelligent multi-level lookup"""
        
        # Level 1: Check local cache first (fastest)
        local_result = self._get_from_local_cache(key)
        if local_result is not None:
            self.cache_stats["hits"] += 1
            self.cache_stats["local_hits"] += 1
            return local_result
        
        # Level 2: Check Redis cache
        redis_result = self._get_from_redis_cache(key, cache_type)
        if redis_result is not None:
            self.cache_stats["hits"] += 1
            self.cache_stats["redis_hits"] += 1
            
            # Store in local cache for faster future access
            self._store_in_local_cache(key, redis_result, cache_type)
            return redis_result
        
        # Cache miss
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, cache_type: str = "default",
            custom_ttl: Optional[int] = None) -> bool:
        """Set value in cache with intelligent storage optimization"""
        
        config = self.cache_configs.get(cache_type, CacheConfig(ttl_seconds=300, max_size_mb=1.0))
        ttl = custom_ttl or config.ttl_seconds
        
        # Serialize and check size
        serialized = json.dumps(value, default=str).encode('utf-8')
        size_kb = len(serialized) / 1024
        size_mb = size_kb / 1024
        
        # Check size limits
        if size_mb > config.max_size_mb:
            print(f"Cache entry too large: {size_mb:.2f}MB > {config.max_size_mb}MB")
            return False
        
        # Apply compression if configured and threshold met
        compressed = False
        if (config.enable_compression and 
            size_kb > config.compression_threshold_kb):
            try:
                serialized = gzip.compress(serialized)
                compressed = True
                self.cache_stats["compressions"] += 1
            except Exception as e:
                print(f"Compression failed: {e}")
        
        # Store in Redis
        cache_metadata = {
            "compressed": compressed,
            "original_size": len(serialized),
            "cached_at": datetime.utcnow().isoformat(),
            "cache_type": cache_type
        }
        
        try:
            # Store data and metadata separately for better performance
            pipe = self.redis.pipeline()
            pipe.setex(key, ttl, serialized)
            pipe.setex(f"{key}:meta", ttl, json.dumps(cache_metadata))
            pipe.execute()
            
            # Also store in local cache
            self._store_in_local_cache(key, value, cache_type)
            
            return True
        except Exception as e:
            print(f"Failed to store in cache: {e}")
            return False
    
    def invalidate(self, pattern: str = None, cache_type: str = None) -> int:
        """Intelligent cache invalidation"""
        deleted_count = 0
        
        if pattern:
            # Pattern-based invalidation
            keys = self.redis.keys(pattern)
            if keys:
                deleted_count = self.redis.delete(*keys)
                
                # Also remove from local cache
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    if key_str in self.local_cache:
                        del self.local_cache[key_str]
        
        elif cache_type:
            # Type-based invalidation
            cache_type_pattern = f"*:{cache_type}:*"
            keys = self.redis.keys(cache_type_pattern)
            if keys:
                deleted_count = self.redis.delete(*keys)
        
        return deleted_count
    
    def _get_from_local_cache(self, key: str) -> Optional[Any]:
        """Get value from local in-memory cache"""
        if key not in self.local_cache:
            return None
        
        entry = self.local_cache[key]
        
        # Check expiration
        if entry["expires_at"] < datetime.utcnow():
            del self.local_cache[key]
            return None
        
        return entry["value"]
    
    def _get_from_redis_cache(self, key: str, cache_type: str) -> Optional[Any]:
        """Get value from Redis cache with decompression"""
        try:
            # Get data and metadata
            pipe = self.redis.pipeline()
            pipe.get(key)
            pipe.get(f"{key}:meta")
            results = pipe.execute()
            
            cached_data = results[0]
            metadata_str = results[1]
            
            if not cached_data:
                return None
            
            # Parse metadata
            metadata = {}
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                except:
                    pass
            
            # Decompress if needed
            if metadata.get("compressed", False):
                try:
                    cached_data = gzip.decompress(cached_data)
                except Exception as e:
                    print(f"Decompression failed: {e}")
                    return None
            
            # Deserialize
            value = json.loads(cached_data.decode('utf-8'))
            return value
            
        except Exception as e:
            print(f"Error retrieving from Redis cache: {e}")
            return None
    
    def _store_in_local_cache(self, key: str, value: Any, cache_type: str):
        """Store value in local cache with size management"""
        config = self.cache_configs.get(cache_type, CacheConfig(ttl_seconds=300, max_size_mb=1.0))
        
        # Calculate value size
        serialized_size = len(json.dumps(value, default=str).encode('utf-8'))
        
        # Check if we need to evict entries
        current_size = sum(entry["size"] for entry in self.local_cache.values())
        total_size_mb = (current_size + serialized_size) / (1024 * 1024)
        
        if total_size_mb > self.local_cache_size_mb:
            self._evict_local_cache_entries()
        
        # Store new entry
        self.local_cache[key] = {
            "value": value,
            "expires_at": datetime.utcnow() + timedelta(seconds=config.ttl_seconds),
            "size": serialized_size,
            "cache_type": cache_type,
            "created_at": datetime.utcnow()
        }
    
    def _evict_local_cache_entries(self):
        """Evict entries from local cache using LRU strategy"""
        if not self.local_cache:
            return
        
        # Sort by creation time (oldest first for LRU)
        sorted_entries = sorted(
            self.local_cache.items(),
            key=lambda x: x[1]["created_at"]
        )
        
        # Remove oldest 25% of entries
        entries_to_remove = max(1, len(sorted_entries) // 4)
        
        for i in range(entries_to_remove):
            key, _ = sorted_entries[i]
            del self.local_cache[key]
            self.cache_stats["evictions"] += 1
    
    def get_cache_statistics(self) -> Dict:
        """Get comprehensive cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        local_cache_size = sum(entry["size"] for entry in self.local_cache.values())
        
        try:
            redis_info = self.redis.info("memory")
            redis_memory_mb = redis_info.get("used_memory", 0) / (1024 * 1024)
        except:
            redis_memory_mb = 0
        
        return {
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "local_hits": self.cache_stats["local_hits"],
            "redis_hits": self.cache_stats["redis_hits"],
            "evictions": self.cache_stats["evictions"],
            "compressions": self.cache_stats["compressions"],
            "local_cache": {
                "entry_count": len(self.local_cache),
                "size_mb": round(local_cache_size / (1024 * 1024), 2),
                "max_size_mb": self.local_cache_size_mb
            },
            "redis_cache": {
                "memory_mb": round(redis_memory_mb, 2)
            }
        }

# Usage in FastAPI middleware
from fastapi import Request, Response
import asyncio

class CacheMiddleware:
    """Middleware for automatic response caching"""
    
    def __init__(self, cache_service: IntelligentCacheService):
        self.cache = cache_service
        
        # Define cacheable endpoints and their configurations
        self.cacheable_endpoints = {
            "/api/v2/tenants/{tenant_id}/config": {
                "cache_type": "tenant_config",
                "ttl": 300,  # 5 minutes
                "vary_headers": ["X-Tenant-ID"]
            },
            "/api/v2/analytics/conversations": {
                "cache_type": "analytics_data",
                "ttl": 900,  # 15 minutes
                "vary_headers": ["X-Tenant-ID"],
                "vary_params": ["start_date", "end_date", "granularity"]
            }
        }
    
    async def __call__(self, request: Request, call_next):
        # Check if endpoint is cacheable
        cache_config = self._get_cache_config(request)
        if not cache_config:
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request, cache_config)
        
        # Try to get cached response
        cached_response = self.cache.get(cache_key, cache_config["cache_type"])
        if cached_response:
            return Response(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers=cached_response["headers"],
                media_type=cached_response["media_type"]
            )
        
        # Get fresh response
        response = await call_next(request)
        
        # Cache successful responses
        if 200 <= response.status_code < 300:
            # Read response content
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            cached_data = {
                "content": response_body.decode(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "media_type": response.media_type
            }
            
            self.cache.set(
                cache_key,
                cached_data,
                cache_config["cache_type"],
                cache_config.get("ttl")
            )
            
            # Create new response with cached content
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.media_type
            )
        
        return response
    
    def _get_cache_config(self, request: Request) -> Optional[Dict]:
        """Get cache configuration for request"""
        path = request.url.path
        
        for pattern, config in self.cacheable_endpoints.items():
            # Simple pattern matching - in production, use more sophisticated routing
            if self._matches_pattern(path, pattern):
                return config
        
        return None
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches cache pattern"""
        # Simple implementation - in production, use proper path matching
        import re
        # Convert {param} to regex groups
        regex_pattern = re.sub(r'\{[^}]+\}', r'[^/]+', pattern)
        return re.match(f"^{regex_pattern}$", path) is not None
    
    def _generate_cache_key(self, request: Request, config: Dict) -> str:
        """Generate unique cache key for request"""
        key_components = [
            request.method,
            request.url.path
        ]
        
        # Add vary headers
        for header in config.get("vary_headers", []):
            value = request.headers.get(header, "")
            key_components.append(f"{header}:{value}")
        
        # Add query parameters
        for param in config.get("vary_params", []):
            value = request.query_params.get(param, "")
            key_components.append(f"{param}:{value}")
        
        # Create hash of components
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
```

**Document Maintainer:** DevOps and SRE Team  
**Review Schedule:** Weekly during development, monthly in production  
**Related Documents:** System Architecture, Database Schemas, Security Implementation