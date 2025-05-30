# Configuration Management
## Multi-Tenant AI Chatbot Platform

**Document:** 09-Configuration-Management.md  
**Version:** 2.0  
**Last Updated:** May 30, 2025

---

## Table of Contents

1. [Configuration Architecture Overview](#configuration-architecture-overview)
2. [Environment-Specific Configuration](#environment-specific-configuration)
3. [Secret Management](#secret-management)
4. [Feature Flag System](#feature-flag-system)
5. [Configuration Validation](#configuration-validation)
6. [Dynamic Configuration Updates](#dynamic-configuration-updates)
7. [Configuration Versioning](#configuration-versioning)
8. [Infrastructure as Code](#infrastructure-as-code)
9. [Configuration Security](#configuration-security)
10. [Monitoring and Auditing](#monitoring-and-auditing)

---

## Configuration Architecture Overview

### Configuration Philosophy

1. **Environment Parity:** Consistent configuration structure across all environments
2. **Security First:** Sensitive data encrypted and access-controlled
3. **Version Controlled:** All configuration changes tracked and auditable
4. **Dynamic Updates:** Runtime configuration changes without service restarts
5. **Validation Driven:** Schema-based validation prevents configuration errors
6. **Least Privilege:** Granular access control for configuration management

### Configuration Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION HIERARCHY                      │
└─────────────────────────────────────────────────────────────────┘

Global Configuration:
├── Platform Settings (shared across all tenants)
├── Infrastructure Configuration
├── Security Policies
└── Compliance Settings

Environment Configuration:
├── Development Settings
├── Staging Settings
├── Production Settings
└── Disaster Recovery Settings

Service Configuration:
├── Chat Service Config
├── MCP Engine Config
├── Model Orchestrator Config
├── Adaptor Service Config
├── Security Hub Config
└── Analytics Engine Config

Tenant Configuration:
├── Tenant-Specific Settings
├── Feature Flags per Tenant
├── Integration Configurations
├── Conversation Flow Definitions
└── Custom Branding Settings

Runtime Configuration:
├── Feature Flags
├── Rate Limits
├── Circuit Breaker Settings
├── Cache Configuration
└── Monitoring Thresholds
```

### Configuration Storage Strategy

| Configuration Type | Storage | Update Frequency | Caching Strategy |
|-------------------|---------|------------------|------------------|
| **Infrastructure** | Kubernetes ConfigMaps/Secrets | Deployment-time | None |
| **Application** | Database + Redis Cache | Runtime | 5-minute TTL |
| **Feature Flags** | Database + Redis | Real-time | 30-second TTL |
| **Tenant Settings** | Database | Runtime | 10-minute TTL |
| **Secrets** | HashiCorp Vault | On-demand | None |

---

## Environment-Specific Configuration

### Configuration Management Service

```python
import os
import yaml
import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import hashlib
from datetime import datetime, timedelta

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DR = "disaster_recovery"

class ConfigurationScope(Enum):
    GLOBAL = "global"
    ENVIRONMENT = "environment"
    SERVICE = "service"
    TENANT = "tenant"

@dataclass
class ConfigurationMetadata:
    """Metadata for configuration entries"""
    version: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    environment: Environment
    scope: ConfigurationScope
    checksum: str
    description: Optional[str] = None
    tags: List[str] = None

@dataclass
class ConfigurationEntry:
    """Individual configuration entry"""
    key: str
    value: Any
    metadata: ConfigurationMetadata
    encrypted: bool = False
    sensitive: bool = False
    schema_version: str = "1.0"

class ConfigurationManager:
    """Centralized configuration management service"""
    
    def __init__(self, database, redis_client, vault_client, encryption_service):
        self.db = database
        self.redis = redis_client
        self.vault = vault_client
        self.encryption = encryption_service
        self.cache_prefix = "config:"
        self.notification_channels = []
        
        # Configuration schemas for validation
        self.schemas = {}
        self.load_schemas()
    
    async def get_configuration(self, key: str, environment: Environment, 
                              scope: ConfigurationScope, 
                              tenant_id: Optional[str] = None,
                              use_cache: bool = True) -> Optional[Any]:
        """Get configuration value with caching and fallback"""
        
        # Build cache key
        cache_key = self._build_cache_key(key, environment, scope, tenant_id)
        
        # Try cache first
        if use_cache:
            cached_value = await self._get_from_cache(cache_key)
            if cached_value is not None:
                return cached_value
        
        # Get from database
        config_entry = await self._get_from_database(key, environment, scope, tenant_id)
        
        if config_entry:
            # Decrypt if necessary
            value = await self._decrypt_if_needed(config_entry)
            
            # Cache the result
            if use_cache:
                await self._set_cache(cache_key, value, ttl=300)  # 5 minutes
            
            return value
        
        # Try fallback hierarchy
        return await self._get_with_fallback(key, environment, scope, tenant_id)
    
    async def set_configuration(self, key: str, value: Any, environment: Environment,
                              scope: ConfigurationScope, user_id: str,
                              tenant_id: Optional[str] = None,
                              description: Optional[str] = None,
                              sensitive: bool = False) -> bool:
        """Set configuration value with validation and versioning"""
        
        # Validate configuration
        validation_result = await self._validate_configuration(key, value, scope)
        if not validation_result.valid:
            raise ConfigurationValidationError(f"Validation failed: {validation_result.errors}")
        
        # Encrypt if sensitive
        final_value = value
        encrypted = False
        if sensitive:
            final_value = await self.encryption.encrypt_field(str(value), "configuration")
            encrypted = True
        
        # Create metadata
        metadata = ConfigurationMetadata(
            version=self._generate_version(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by=user_id,
            updated_by=user_id,
            environment=environment,
            scope=scope,
            checksum=self._calculate_checksum(value),
            description=description
        )
        
        # Create configuration entry
        config_entry = ConfigurationEntry(
            key=key,
            value=final_value,
            metadata=metadata,
            encrypted=encrypted,
            sensitive=sensitive
        )
        
        # Store in database with versioning
        success = await self._store_configuration(config_entry, tenant_id)
        
        if success:
            # Invalidate cache
            cache_key = self._build_cache_key(key, environment, scope, tenant_id)
            await self._invalidate_cache(cache_key)
            
            # Notify subscribers
            await self._notify_configuration_change(key, environment, scope, tenant_id)
            
            # Log audit event
            await self._log_configuration_change(config_entry, tenant_id, "SET")
        
        return success
    
    async def delete_configuration(self, key: str, environment: Environment,
                                 scope: ConfigurationScope, user_id: str,
                                 tenant_id: Optional[str] = None) -> bool:
        """Delete configuration entry"""
        
        # Get current value for audit
        current_entry = await self._get_from_database(key, environment, scope, tenant_id)
        
        # Delete from database
        success = await self._delete_from_database(key, environment, scope, tenant_id)
        
        if success:
            # Invalidate cache
            cache_key = self._build_cache_key(key, environment, scope, tenant_id)
            await self._invalidate_cache(cache_key)
            
            # Notify subscribers
            await self._notify_configuration_change(key, environment, scope, tenant_id)
            
            # Log audit event
            if current_entry:
                await self._log_configuration_change(current_entry, tenant_id, "DELETE")
        
        return success
    
    async def get_configuration_history(self, key: str, environment: Environment,
                                      scope: ConfigurationScope,
                                      tenant_id: Optional[str] = None,
                                      limit: int = 50) -> List[ConfigurationEntry]:
        """Get configuration change history"""
        
        query = """
        SELECT key, value, metadata, encrypted, sensitive, created_at
        FROM configuration_history 
        WHERE key = $1 AND environment = $2 AND scope = $3
        """
        params = [key, environment.value, scope.value]
        
        if tenant_id:
            query += " AND tenant_id = $4"
            params.append(tenant_id)
        
        query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            history = []
            for row in rows:
                metadata = ConfigurationMetadata(**json.loads(row['metadata']))
                entry = ConfigurationEntry(
                    key=row['key'],
                    value=row['value'],
                    metadata=metadata,
                    encrypted=row['encrypted'],
                    sensitive=row['sensitive']
                )
                history.append(entry)
            
            return history
    
    async def rollback_configuration(self, key: str, environment: Environment,
                                   scope: ConfigurationScope, target_version: str,
                                   user_id: str, tenant_id: Optional[str] = None) -> bool:
        """Rollback configuration to a specific version"""
        
        # Get target version
        history = await self.get_configuration_history(key, environment, scope, tenant_id)
        target_entry = None
        
        for entry in history:
            if entry.metadata.version == target_version:
                target_entry = entry
                break
        
        if not target_entry:
            raise ConfigurationError(f"Version {target_version} not found")
        
        # Decrypt value if needed
        value = await self._decrypt_if_needed(target_entry)
        
        # Set as current configuration
        return await self.set_configuration(
            key=key,
            value=value,
            environment=environment,
            scope=scope,
            user_id=user_id,
            tenant_id=tenant_id,
            description=f"Rollback to version {target_version}",
            sensitive=target_entry.sensitive
        )
    
    async def bulk_update_configuration(self, updates: List[Dict[str, Any]], 
                                      user_id: str) -> Dict[str, Any]:
        """Perform bulk configuration updates atomically"""
        
        results = {
            'total': len(updates),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        # Start transaction
        async with self.db.transaction():
            for update in updates:
                try:
                    await self.set_configuration(
                        key=update['key'],
                        value=update['value'],
                        environment=Environment(update['environment']),
                        scope=ConfigurationScope(update['scope']),
                        user_id=user_id,
                        tenant_id=update.get('tenant_id'),
                        description=update.get('description'),
                        sensitive=update.get('sensitive', False)
                    )
                    results['successful'] += 1
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append({
                        'key': update['key'],
                        'error': str(e)
                    })
        
        return results
    
    def _build_cache_key(self, key: str, environment: Environment, 
                        scope: ConfigurationScope, tenant_id: Optional[str]) -> str:
        """Build cache key for configuration"""
        parts = [self.cache_prefix, environment.value, scope.value, key]
        if tenant_id:
            parts.append(tenant_id)
        return ":".join(parts)
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get configuration from Redis cache"""
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception:
            pass
        return None
    
    async def _set_cache(self, cache_key: str, value: Any, ttl: int):
        """Set configuration in Redis cache"""
        try:
            await self.redis.setex(cache_key, ttl, json.dumps(value, default=str))
        except Exception:
            pass
    
    async def _invalidate_cache(self, cache_key: str):
        """Invalidate configuration cache"""
        try:
            await self.redis.delete(cache_key)
            # Also invalidate pattern-based cache entries
            pattern = cache_key.replace(self.cache_prefix, f"{self.cache_prefix}*")
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
        except Exception:
            pass
    
    async def _get_from_database(self, key: str, environment: Environment,
                               scope: ConfigurationScope, 
                               tenant_id: Optional[str]) -> Optional[ConfigurationEntry]:
        """Get configuration from database"""
        
        query = """
        SELECT key, value, metadata, encrypted, sensitive, schema_version
        FROM configurations 
        WHERE key = $1 AND environment = $2 AND scope = $3 AND active = true
        """
        params = [key, environment.value, scope.value]
        
        if tenant_id:
            query += " AND tenant_id = $4"
            params.append(tenant_id)
        else:
            query += " AND tenant_id IS NULL"
        
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            
            if row:
                metadata = ConfigurationMetadata(**json.loads(row['metadata']))
                return ConfigurationEntry(
                    key=row['key'],
                    value=row['value'],
                    metadata=metadata,
                    encrypted=row['encrypted'],
                    sensitive=row['sensitive'],
                    schema_version=row['schema_version']
                )
        
        return None
    
    async def _store_configuration(self, config_entry: ConfigurationEntry, 
                                 tenant_id: Optional[str]) -> bool:
        """Store configuration in database with versioning"""
        
        try:
            async with self.db.transaction():
                # Archive current version
                await self._archive_current_version(config_entry.key, 
                                                   config_entry.metadata.environment,
                                                   config_entry.metadata.scope, 
                                                   tenant_id)
                
                # Insert new version
                query = """
                INSERT INTO configurations 
                (key, value, metadata, encrypted, sensitive, schema_version, 
                 environment, scope, tenant_id, active, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, true, $10, $11)
                """
                
                await self.db.execute(
                    query,
                    config_entry.key,
                    config_entry.value,
                    json.dumps(asdict(config_entry.metadata), default=str),
                    config_entry.encrypted,
                    config_entry.sensitive,
                    config_entry.schema_version,
                    config_entry.metadata.environment.value,
                    config_entry.metadata.scope.value,
                    tenant_id,
                    config_entry.metadata.created_at,
                    config_entry.metadata.updated_at
                )
            
            return True
        except Exception as e:
            print(f"Error storing configuration: {e}")
            return False
    
    async def _archive_current_version(self, key: str, environment: Environment,
                                     scope: ConfigurationScope, 
                                     tenant_id: Optional[str]):
        """Archive current configuration version"""
        
        # Move current active version to history
        query = """
        INSERT INTO configuration_history 
        SELECT * FROM configurations 
        WHERE key = $1 AND environment = $2 AND scope = $3 AND active = true
        """
        params = [key, environment.value, scope.value]
        
        if tenant_id:
            query += " AND tenant_id = $4"
            params.append(tenant_id)
        else:
            query += " AND tenant_id IS NULL"
        
        await self.db.execute(query, *params)
        
        # Mark current version as inactive
        update_query = """
        UPDATE configurations 
        SET active = false 
        WHERE key = $1 AND environment = $2 AND scope = $3 AND active = true
        """
        update_params = [key, environment.value, scope.value]
        
        if tenant_id:
            update_query += " AND tenant_id = $4"
            update_params.append(tenant_id)
        else:
            update_query += " AND tenant_id IS NULL"
        
        await self.db.execute(update_query, *update_params)
    
    async def _get_with_fallback(self, key: str, environment: Environment,
                               scope: ConfigurationScope, 
                               tenant_id: Optional[str]) -> Optional[Any]:
        """Get configuration with fallback hierarchy"""
        
        # Fallback order: tenant -> service -> environment -> global
        fallback_chain = []
        
        if scope == ConfigurationScope.TENANT and tenant_id:
            fallback_chain.extend([
                (ConfigurationScope.SERVICE, None),
                (ConfigurationScope.ENVIRONMENT, None),
                (ConfigurationScope.GLOBAL, None)
            ])
        elif scope == ConfigurationScope.SERVICE:
            fallback_chain.extend([
                (ConfigurationScope.ENVIRONMENT, None),
                (ConfigurationScope.GLOBAL, None)
            ])
        elif scope == ConfigurationScope.ENVIRONMENT:
            fallback_chain.append((ConfigurationScope.GLOBAL, None))
        
        # Try fallback chain
        for fallback_scope, fallback_tenant in fallback_chain:
            config_entry = await self._get_from_database(key, environment, 
                                                        fallback_scope, fallback_tenant)
            if config_entry:
                value = await self._decrypt_if_needed(config_entry)
                return value
        
        return None
    
    async def _decrypt_if_needed(self, config_entry: ConfigurationEntry) -> Any:
        """Decrypt configuration value if encrypted"""
        if config_entry.encrypted:
            try:
                return await self.encryption.decrypt_field(config_entry.value, "configuration")
            except Exception:
                return None
        return config_entry.value
    
    def _calculate_checksum(self, value: Any) -> str:
        """Calculate checksum for configuration value"""
        value_str = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(value_str.encode()).hexdigest()
    
    def _generate_version(self) -> str:
        """Generate version string for configuration"""
        return f"v{int(datetime.utcnow().timestamp())}"
    
    async def _notify_configuration_change(self, key: str, environment: Environment,
                                         scope: ConfigurationScope, tenant_id: Optional[str]):
        """Notify subscribers of configuration changes"""
        
        change_event = {
            'key': key,
            'environment': environment.value,
            'scope': scope.value,
            'tenant_id': tenant_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Publish to Redis for real-time notifications
        channel = f"config_changes:{environment.value}"
        await self.redis.publish(channel, json.dumps(change_event))
        
        # Notify registered handlers
        for handler in self.notification_channels:
            try:
                await handler(change_event)
            except Exception as e:
                print(f"Error in notification handler: {e}")
    
    async def _log_configuration_change(self, config_entry: ConfigurationEntry,
                                      tenant_id: Optional[str], action: str):
        """Log configuration change for audit purposes"""
        
        audit_entry = {
            'action': action,
            'key': config_entry.key,
            'environment': config_entry.metadata.environment.value,
            'scope': config_entry.metadata.scope.value,
            'tenant_id': tenant_id,
            'version': config_entry.metadata.version,
            'user_id': config_entry.metadata.updated_by,
            'timestamp': datetime.utcnow().isoformat(),
            'checksum': config_entry.metadata.checksum,
            'sensitive': config_entry.sensitive
        }
        
        # Store in audit log
        query = """
        INSERT INTO configuration_audit_log 
        (action, key, environment, scope, tenant_id, version, user_id, 
         timestamp, checksum, sensitive, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        
        await self.db.execute(
            query,
            audit_entry['action'],
            audit_entry['key'],
            audit_entry['environment'],
            audit_entry['scope'],
            audit_entry['tenant_id'],
            audit_entry['version'],
            audit_entry['user_id'],
            audit_entry['timestamp'],
            audit_entry['checksum'],
            audit_entry['sensitive'],
            json.dumps(audit_entry)
        )

class ConfigurationValidationError(Exception):
    """Exception raised when configuration validation fails"""
    pass

class ConfigurationError(Exception):
    """General configuration error"""
    pass
```

### Environment Configuration Templates

```yaml
# config/environments/development.yaml
environment:
  name: development
  debug: true
  log_level: DEBUG
  
database:
  postgresql:
    host: localhost
    port: 5432
    database: chatbot_dev
    pool_size: 5
    max_overflow: 10
  
  mongodb:
    host: localhost
    port: 27017
    database: chatbot_dev
    replica_set: null
  
  redis:
    host: localhost
    port: 6379
    database: 0
    cluster_mode: false

services:
  chat_service:
    replicas: 1
    resources:
      requests:
        memory: "256Mi"
        cpu: "250m"
      limits:
        memory: "512Mi"
        cpu: "500m"
  
  mcp_engine:
    replicas: 1
    resources:
      requests:
        memory: "512Mi"
        cpu: "250m"
      limits:
        memory: "1Gi"
        cpu: "500m"

external_apis:
  openai:
    base_url: https://api.openai.com/v1
    timeout: 30
    max_retries: 3
  
  anthropic:
    base_url: https://api.anthropic.com/v1
    timeout: 30
    max_retries: 3

monitoring:
  metrics_enabled: true
  tracing_enabled: true
  log_level: DEBUG
  sample_rate: 1.0

security:
  cors_origins:
    - http://localhost:3000
    - http://localhost:8000
  csrf_protection: false
  rate_limiting:
    enabled: true
    requests_per_minute: 1000

---

# config/environments/production.yaml
environment:
  name: production
  debug: false
  log_level: INFO
  
database:
  postgresql:
    host: chatbot-prod-db.cluster-xyz.us-east-1.rds.amazonaws.com
    port: 5432
    database: chatbot_production
    pool_size: 20
    max_overflow: 30
    ssl_mode: require
  
  mongodb:
    host: chatbot-prod-mongo.cluster-xyz.us-east-1.docdb.amazonaws.com
    port: 27017
    database: chatbot_production
    replica_set: rs0
    ssl: true
  
  redis:
    cluster_endpoints:
      - chatbot-prod-redis-001.xyz.cache.amazonaws.com:6379
      - chatbot-prod-redis-002.xyz.cache.amazonaws.com:6379
      - chatbot-prod-redis-003.xyz.cache.amazonaws.com:6379
    cluster_mode: true
    ssl: true

services:
  chat_service:
    replicas: 10
    resources:
      requests:
        memory: "1Gi"
        cpu: "500m"
      limits:
        memory: "2Gi"
        cpu: "1000m"
    
    autoscaling:
      enabled: true
      min_replicas: 10
      max_replicas: 100
      target_cpu_utilization: 70
      target_memory_utilization: 80
  
  mcp_engine:
    replicas: 5
    resources:
      requests:
        memory: "2Gi"
        cpu: "500m"
      limits:
        memory: "4Gi"
        cpu: "1000m"
    
    autoscaling:
      enabled: true
      min_replicas: 5
      max_replicas: 50

external_apis:
  openai:
    base_url: https://api.openai.com/v1
    timeout: 30
    max_retries: 3
    rate_limit_per_minute: 10000
  
  anthropic:
    base_url: https://api.anthropic.com/v1
    timeout: 30
    max_retries: 3
    rate_limit_per_minute: 5000

monitoring:
  metrics_enabled: true
  tracing_enabled: true
  log_level: INFO
  sample_rate: 0.1
  retention_days: 30

security:
  cors_origins:
    - https://app.chatbot-platform.com
    - https://admin.chatbot-platform.com
  csrf_protection: true
  rate_limiting:
    enabled: true
    requests_per_minute: 10000
  
  compliance:
    gdpr_enabled: true
    hipaa_enabled: false
    soc2_enabled: true

backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention_days: 30
  destinations:
    - s3://chatbot-backups/production/
```

---

## Secret Management

### Vault Integration

```python
import hvac
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class SecretMetadata:
    """Metadata for secret entries"""
    created_at: datetime
    created_by: str
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    expires_at: Optional[datetime] = None
    rotation_required: bool = False
    tags: List[str] = None

class VaultSecretManager:
    """HashiCorp Vault integration for secret management"""
    
    def __init__(self, vault_url: str, vault_token: str, mount_point: str = "secret"):
        self.client = hvac.Client(url=vault_url, token=vault_token)
        self.mount_point = mount_point
        self.secret_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def store_secret(self, path: str, secret_data: Dict[str, Any],
                          metadata: Optional[SecretMetadata] = None) -> bool:
        """Store secret in Vault"""
        
        if not metadata:
            metadata = SecretMetadata(
                created_at=datetime.utcnow(),
                created_by="system"
            )
        
        # Prepare secret payload
        payload = {
            'data': secret_data,
            'metadata': {
                'created_at': metadata.created_at.isoformat(),
                'created_by': metadata.created_by,
                'access_count': metadata.access_count,
                'tags': metadata.tags or []
            }
        }
        
        if metadata.expires_at:
            payload['metadata']['expires_at'] = metadata.expires_at.isoformat()
        
        try:
            # Store in Vault (KV v2)
            response = self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=payload,
                mount_point=self.mount_point
            )
            
            # Log secret creation
            await self._log_secret_access(path, "CREATE", metadata.created_by)
            
            return response is not None
        except Exception as e:
            print(f"Error storing secret: {e}")
            return False
    
    async def get_secret(self, path: str, user_id: str = "system") -> Optional[Dict[str, Any]]:
        """Retrieve secret from Vault with caching"""
        
        # Check cache first
        cache_key = f"{path}:{user_id}"
        if cache_key in self.secret_cache:
            cache_entry = self.secret_cache[cache_key]
            if cache_entry['expires_at'] > datetime.utcnow():
                await self._log_secret_access(path, "ACCESS_CACHED", user_id)
                return cache_entry['data']
            else:
                del self.secret_cache[cache_key]
        
        try:
            # Retrieve from Vault
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.mount_point
            )
            
            if response and 'data' in response:
                secret_data = response['data']['data']
                
                # Update access tracking
                await self._update_access_metadata(path, user_id)
                
                # Cache the secret
                self.secret_cache[cache_key] = {
                    'data': secret_data,
                    'expires_at': datetime.utcnow() + timedelta(seconds=self.cache_ttl)
                }
                
                # Log access
                await self._log_secret_access(path, "ACCESS", user_id)
                
                return secret_data
        except Exception as e:
            print(f"Error retrieving secret: {e}")
        
        return None
    
    async def rotate_secret(self, path: str, new_secret_data: Dict[str, Any],
                          user_id: str) -> bool:
        """Rotate secret with new values"""
        
        # Get current secret for backup
        current_secret = await self.get_secret(path, user_id)
        
        if current_secret:
            # Backup current version
            backup_path = f"{path}/backup/{int(datetime.utcnow().timestamp())}"
            await self.store_secret(backup_path, current_secret)
        
        # Store new secret
        metadata = SecretMetadata(
            created_at=datetime.utcnow(),
            created_by=user_id,
            rotation_required=False
        )
        
        success = await self.store_secret(path, new_secret_data, metadata)
        
        if success:
            # Invalidate cache
            cache_keys_to_remove = [key for key in self.secret_cache.keys() if key.startswith(f"{path}:")]
            for key in cache_keys_to_remove:
                del self.secret_cache[key]
            
            # Log rotation
            await self._log_secret_access(path, "ROTATE", user_id)
        
        return success
    
    async def delete_secret(self, path: str, user_id: str) -> bool:
        """Delete secret from Vault"""
        
        try:
            # Soft delete (mark as deleted)
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path,
                mount_point=self.mount_point
            )
            
            # Remove from cache
            cache_keys_to_remove = [key for key in self.secret_cache.keys() if key.startswith(f"{path}:")]
            for key in cache_keys_to_remove:
                del self.secret_cache[key]
            
            # Log deletion
            await self._log_secret_access(path, "DELETE", user_id)
            
            return True
        except Exception as e:
            print(f"Error deleting secret: {e}")
            return False
    
    async def list_secrets(self, path_prefix: str = "") -> List[str]:
        """List secrets at path"""
        
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path=path_prefix,
                mount_point=self.mount_point
            )
            
            if response and 'data' in response and 'keys' in response['data']:
                return response['data']['keys']
        except Exception as e:
            print(f"Error listing secrets: {e}")
        
        return []
    
    async def get_secret_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """Get secret metadata without retrieving the secret value"""
        
        try:
            response = self.client.secrets.kv.v2.read_secret_metadata(
                path=path,
                mount_point=self.mount_point
            )
            
            if response and 'data' in response:
                return response['data']
        except Exception as e:
            print(f"Error retrieving secret metadata: {e}")
        
        return None
    
    async def _update_access_metadata(self, path: str, user_id: str):
        """Update secret access metadata"""
        
        # Get current metadata
        metadata = await self.get_secret_metadata(path)
        
        if metadata:
            # Update access count and last access time
            current_metadata = metadata.get('custom_metadata', {})
            access_count = int(current_metadata.get('access_count', 0)) + 1
            
            # Update metadata
            updated_metadata = {
                **current_metadata,
                'access_count': str(access_count),
                'last_accessed': datetime.utcnow().isoformat(),
                'last_accessed_by': user_id
            }
            
            try:
                self.client.secrets.kv.v2.update_metadata(
                    path=path,
                    custom_metadata=updated_metadata,
                    mount_point=self.mount_point
                )
            except Exception as e:
                print(f"Error updating secret metadata: {e}")
    
    async def _log_secret_access(self, path: str, action: str, user_id: str):
        """Log secret access for audit purposes"""
        
        log_entry = {
            'path': path,
            'action': action,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat(),
            'ip_address': None  # Would be filled by actual request context
        }
        
        # In production, this would write to a secure audit log
        print(f"Secret access log: {log_entry}")

class SecretRotationService:
    """Automated secret rotation service"""
    
    def __init__(self, vault_manager: VaultSecretManager, 
                 configuration_manager: ConfigurationManager):
        self.vault = vault_manager
        self.config = configuration_manager
        self.rotation_policies = {}
    
    def register_rotation_policy(self, secret_path: str, rotation_interval_days: int,
                                notification_days: int = 7):
        """Register automatic rotation policy for a secret"""
        
        self.rotation_policies[secret_path] = {
            'rotation_interval_days': rotation_interval_days,
            'notification_days': notification_days,
            'last_rotated': None,
            'next_rotation': None
        }
    
    async def check_rotation_requirements(self) -> List[Dict[str, Any]]:
        """Check which secrets need rotation"""
        
        rotation_needed = []
        current_time = datetime.utcnow()
        
        for secret_path, policy in self.rotation_policies.items():
            metadata = await self.vault.get_secret_metadata(secret_path)
            
            if metadata:
                created_time = datetime.fromisoformat(
                    metadata.get('created_time', current_time.isoformat())
                )
                
                days_since_creation = (current_time - created_time).days
                
                if days_since_creation >= policy['rotation_interval_days']:
                    rotation_needed.append({
                        'secret_path': secret_path,
                        'days_overdue': days_since_creation - policy['rotation_interval_days'],
                        'created_time': created_time,
                        'policy': policy
                    })
                elif days_since_creation >= (policy['rotation_interval_days'] - policy['notification_days']):
                    # Notify about upcoming rotation
                    rotation_needed.append({
                        'secret_path': secret_path,
                        'notification_only': True,
                        'days_until_rotation': policy['rotation_interval_days'] - days_since_creation,
                        'policy': policy
                    })
        
        return rotation_needed
    
    async def rotate_database_credentials(self, secret_path: str) -> bool:
        """Rotate database credentials automatically"""
        
        # Get current credentials
        current_creds = await self.vault.get_secret(secret_path)
        
        if not current_creds:
            return False
        
        # Generate new password
        new_password = self._generate_secure_password()
        
        # Update database with new credentials
        # This would depend on the specific database type
        success = await self._update_database_password(
            current_creds['username'],
            new_password,
            current_creds.get('host'),
            current_creds.get('database')
        )
        
        if success:
            # Store new credentials in Vault
            new_creds = {**current_creds, 'password': new_password}
            return await self.vault.rotate_secret(secret_path, new_creds, "rotation_service")
        
        return False
    
    async def rotate_api_keys(self, secret_path: str, provider: str) -> bool:
        """Rotate API keys for external services"""
        
        # This would depend on the specific provider's API
        # Example for a generic API key rotation
        
        current_keys = await self.vault.get_secret(secret_path)
        
        if not current_keys:
            return False
        
        # Generate new API key through provider's API
        new_api_key = await self._generate_new_api_key(provider, current_keys)
        
        if new_api_key:
            # Test new API key
            if await self._test_api_key(provider, new_api_key):
                # Store new key
                new_keys = {**current_keys, 'api_key': new_api_key}
                success = await self.vault.rotate_secret(secret_path, new_keys, "rotation_service")
                
                if success:
                    # Revoke old API key
                    await self._revoke_old_api_key(provider, current_keys['api_key'])
                
                return success
        
        return False
    
    def _generate_secure_password(self, length: int = 32) -> str:
        """Generate cryptographically secure password"""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    async def _update_database_password(self, username: str, new_password: str,
                                      host: str, database: str) -> bool:
        """Update database password"""
        # Implementation would depend on database type
        # This is a placeholder
        return True
    
    async def _generate_new_api_key(self, provider: str, current_keys: Dict[str, Any]) -> Optional[str]:
        """Generate new API key through provider"""
        # Implementation would depend on provider
        # This is a placeholder
        return None
    
    async def _test_api_key(self, provider: str, api_key: str) -> bool:
        """Test new API key"""
        # Implementation would test the API key
        # This is a placeholder
        return True
    
    async def _revoke_old_api_key(self, provider: str, old_api_key: str) -> bool:
        """Revoke old API key"""
        # Implementation would revoke the old key
        # This is a placeholder
        return True
```

---

## Feature Flag System

### Dynamic Feature Management

```python
import asyncio
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

class FeatureFlagType(Enum):
    BOOLEAN = "boolean"
    STRING = "string"
    NUMBER = "number"
    JSON = "json"

class TargetingRule(Enum):
    USER_ID = "user_id"
    TENANT_ID = "tenant_id"
    USER_ROLE = "user_role"
    TENANT_PLAN = "tenant_plan"
    PERCENTAGE = "percentage"
    CUSTOM = "custom"

@dataclass
class FeatureFlagVariant:
    """Feature flag variant configuration"""
    name: str
    value: Any
    weight: float = 0.0
    description: Optional[str] = None

@dataclass
class FeatureFlagRule:
    """Feature flag targeting rule"""
    rule_type: TargetingRule
    operator: str  # equals, not_equals, in, not_in, greater_than, less_than
    values: List[Any]
    variant: str

@dataclass
class FeatureFlag:
    """Feature flag definition"""
    key: str
    name: str
    description: str
    flag_type: FeatureFlagType
    enabled: bool
    default_variant: str
    variants: List[FeatureFlagVariant]
    targeting_rules: List[FeatureFlagRule]
    
    # Metadata
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str
    environment: str
    
    # Rollout configuration
    rollout_percentage: float = 0.0
    rollout_start_time: Optional[datetime] = None
    rollout_end_time: Optional[datetime] = None
    
    # Analytics
    impression_count: int = 0
    last_evaluated: Optional[datetime] = None

class FeatureFlagService:
    """Dynamic feature flag management service"""
    
    def __init__(self, database, redis_client):
        self.db = database
        self.redis = redis_client
        self.cache_prefix = "feature_flags:"
        self.cache_ttl = 30  # 30 seconds
        self.evaluation_callbacks = []
        self.flag_cache = {}
    
    async def create_feature_flag(self, flag: FeatureFlag) -> bool:
        """Create a new feature flag"""
        
        try:
            # Validate flag configuration
            validation_result = self._validate_feature_flag(flag)
            if not validation_result['valid']:
                raise ValueError(f"Invalid feature flag: {validation_result['errors']}")
            
            # Store in database
            query = """
            INSERT INTO feature_flags 
            (key, name, description, flag_type, enabled, default_variant, variants, 
             targeting_rules, created_at, created_by, updated_at, updated_by, 
             environment, rollout_percentage, rollout_start_time, rollout_end_time)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """
            
            await self.db.execute(
                query,
                flag.key,
                flag.name,
                flag.description,
                flag.flag_type.value,
                flag.enabled,
                flag.default_variant,
                json.dumps([asdict(v) for v in flag.variants]),
                json.dumps([asdict(r) for r in flag.targeting_rules]),
                flag.created_at,
                flag.created_by,
                flag.updated_at,
                flag.updated_by,
                flag.environment,
                flag.rollout_percentage,
                flag.rollout_start_time,
                flag.rollout_end_time
            )
            
            # Invalidate cache
            await self._invalidate_flag_cache(flag.key, flag.environment)
            
            # Log flag creation
            await self._log_flag_event(flag.key, "CREATE", flag.created_by, 
                                     {"flag_data": asdict(flag)})
            
            return True
        except Exception as e:
            print(f"Error creating feature flag: {e}")
            return False
    
    async def update_feature_flag(self, flag: FeatureFlag) -> bool:
        """Update existing feature flag"""
        
        try:
            # Validate flag configuration
            validation_result = self._validate_feature_flag(flag)
            if not validation_result['valid']:
                raise ValueError(f"Invalid feature flag: {validation_result['errors']}")
            
            # Update in database
            query = """
            UPDATE feature_flags 
            SET name = $2, description = $3, flag_type = $4, enabled = $5, 
                default_variant = $6, variants = $7, targeting_rules = $8, 
                updated_at = $9, updated_by = $10, rollout_percentage = $11,
                rollout_start_time = $12, rollout_end_time = $13
            WHERE key = $1 AND environment = $14
            """
            
            result = await self.db.execute(
                query,
                flag.key,
                flag.name,
                flag.description,
                flag.flag_type.value,
                flag.enabled,
                flag.default_variant,
                json.dumps([asdict(v) for v in flag.variants]),
                json.dumps([asdict(r) for r in flag.targeting_rules]),
                flag.updated_at,
                flag.updated_by,
                flag.rollout_percentage,
                flag.rollout_start_time,
                flag.rollout_end_time,
                flag.environment
            )
            
            if result:
                # Invalidate cache
                await self._invalidate_flag_cache(flag.key, flag.environment)
                
                # Log flag update
                await self._log_flag_event(flag.key, "UPDATE", flag.updated_by,
                                         {"flag_data": asdict(flag)})
                
                return True
        except Exception as e:
            print(f"Error updating feature flag: {e}")
        
        return False
    
    async def evaluate_feature_flag(self, flag_key: str, environment: str,
                                   context: Dict[str, Any]) -> Any:
        """Evaluate feature flag for given context"""
        
        # Get feature flag
        flag = await self._get_feature_flag(flag_key, environment)
        
        if not flag or not flag.enabled:
            return None
        
        # Check rollout timing
        if not self._is_rollout_active(flag):
            return None
        
        # Find matching variant
        variant_name = await self._evaluate_targeting_rules(flag, context)
        
        # Get variant value
        variant = next((v for v in flag.variants if v.name == variant_name), None)
        
        if variant:
            # Track impression
            await self._track_flag_impression(flag.key, environment, variant.name, context)
            
            # Execute callbacks
            for callback in self.evaluation_callbacks:
                try:
                    await callback(flag.key, variant.value, context)
                except Exception as e:
                    print(f"Error in evaluation callback: {e}")
            
            return variant.value
        
        return None
    
    async def get_all_flags_for_context(self, environment: str, 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Get all feature flags evaluated for a specific context"""
        
        # Get all flags for environment
        flags = await self._get_all_flags(environment)
        
        evaluated_flags = {}
        
        for flag in flags:
            if flag.enabled and self._is_rollout_active(flag):
                variant_name = await self._evaluate_targeting_rules(flag, context)
                variant = next((v for v in flag.variants if v.name == variant_name), None)
                
                if variant:
                    evaluated_flags[flag.key] = variant.value
                    # Track impression
                    await self._track_flag_impression(flag.key, environment, variant.name, context)
        
        return evaluated_flags
    
    async def _get_feature_flag(self, flag_key: str, environment: str) -> Optional[FeatureFlag]:
        """Get feature flag with caching"""
        
        cache_key = f"{self.cache_prefix}{environment}:{flag_key}"
        
        # Try cache first
        if cache_key in self.flag_cache:
            cache_entry = self.flag_cache[cache_key]
            if cache_entry['expires_at'] > datetime.utcnow():
                return cache_entry['flag']
            else:
                del self.flag_cache[cache_key]
        
        # Get from database
        query = """
        SELECT * FROM feature_flags 
        WHERE key = $1 AND environment = $2
        """
        
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query, flag_key, environment)
            
            if row:
                # Parse variants and rules
                variants = [
                    FeatureFlagVariant(**v) 
                    for v in json.loads(row['variants'])
                ]
                
                rules = [
                    FeatureFlagRule(
                        rule_type=TargetingRule(r['rule_type']),
                        operator=r['operator'],
                        values=r['values'],
                        variant=r['variant']
                    )
                    for r in json.loads(row['targeting_rules'])
                ]
                
                flag = FeatureFlag(
                    key=row['key'],
                    name=row['name'],
                    description=row['description'],
                    flag_type=FeatureFlagType(row['flag_type']),
                    enabled=row['enabled'],
                    default_variant=row['default_variant'],
                    variants=variants,
                    targeting_rules=rules,
                    created_at=row['created_at'],
                    created_by=row['created_by'],
                    updated_at=row['updated_at'],
                    updated_by=row['updated_by'],
                    environment=row['environment'],
                    rollout_percentage=row['rollout_percentage'],
                    rollout_start_time=row['rollout_start_time'],
                    rollout_end_time=row['rollout_end_time'],
                    impression_count=row['impression_count'],
                    last_evaluated=row['last_evaluated']
                )
                
                # Cache the flag
                self.flag_cache[cache_key] = {
                    'flag': flag,
                    'expires_at': datetime.utcnow() + timedelta(seconds=self.cache_ttl)
                }
                
                return flag
        
        return None
    
    async def _get_all_flags(self, environment: str) -> List[FeatureFlag]:
        """Get all feature flags for environment"""
        
        query = """
        SELECT * FROM feature_flags 
        WHERE environment = $1 AND enabled = true
        ORDER BY key
        """
        
        flags = []
        
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, environment)
            
            for row in rows:
                # Parse variants and rules
                variants = [
                    FeatureFlagVariant(**v) 
                    for v in json.loads(row['variants'])
                ]
                
                rules = [
                    FeatureFlagRule(
                        rule_type=TargetingRule(r['rule_type']),
                        operator=r['operator'],
                        values=r['values'],
                        variant=r['variant']
                    )
                    for r in json.loads(row['targeting_rules'])
                ]
                
                flag = FeatureFlag(
                    key=row['key'],
                    name=row['name'],
                    description=row['description'],
                    flag_type=FeatureFlagType(row['flag_type']),
                    enabled=row['enabled'],
                    default_variant=row['default_variant'],
                    variants=variants,
                    targeting_rules=rules,
                    created_at=row['created_at'],
                    created_by=row['created_by'],
                    updated_at=row['updated_at'],
                    updated_by=row['updated_by'],
                    environment=row['environment'],
                    rollout_percentage=row['rollout_percentage'],
                    rollout_start_time=row['rollout_start_time'],
                    rollout_end_time=row['rollout_end_time'],
                    impression_count=row['impression_count'],
                    last_evaluated=row['last_evaluated']
                )
                
                flags.append(flag)
        
        return flags
    
    async def _evaluate_targeting_rules(self, flag: FeatureFlag, 
                                       context: Dict[str, Any]) -> str:
        """Evaluate targeting rules to determine variant"""
        
        # Check targeting rules in order
        for rule in flag.targeting_rules:
            if self._evaluate_rule(rule, context):
                return rule.variant
        
        # Check percentage rollout
        if flag.rollout_percentage > 0:
            # Use consistent hashing for percentage rollout
            user_hash = self._calculate_user_hash(context, flag.key)
            if user_hash <= flag.rollout_percentage:
                # Find first non-default variant or return default
                non_default_variants = [v for v in flag.variants if v.name != flag.default_variant]
                if non_default_variants:
                    return non_default_variants[0].name
        
        # Return default variant
        return flag.default_variant
    
    def _evaluate_rule(self, rule: FeatureFlagRule, context: Dict[str, Any]) -> bool:
        """Evaluate a single targeting rule"""
        
        if rule.rule_type == TargetingRule.USER_ID:
            user_id = context.get('user_id')
            return self._evaluate_operator(rule.operator, user_id, rule.values)
        
        elif rule.rule_type == TargetingRule.TENANT_ID:
            tenant_id = context.get('tenant_id')
            return self._evaluate_operator(rule.operator, tenant_id, rule.values)
        
        elif rule.rule_type == TargetingRule.USER_ROLE:
            user_role = context.get('user_role')
            return self._evaluate_operator(rule.operator, user_role, rule.values)
        
        elif rule.rule_type == TargetingRule.TENANT_PLAN:
            tenant_plan = context.get('tenant_plan')
            return self._evaluate_operator(rule.operator, tenant_plan, rule.values)
        
        elif rule.rule_type == TargetingRule.PERCENTAGE:
            # Percentage-based targeting
            percentage = float(rule.values[0]) if rule.values else 0
            user_hash = self._calculate_user_hash(context, rule.variant)
            return user_hash <= percentage
        
        elif rule.rule_type == TargetingRule.CUSTOM:
            # Custom rule evaluation (would be extended based on needs)
            return self._evaluate_custom_rule(rule, context)
        
        return False
    
    def _evaluate_operator(self, operator: str, actual_value: Any, 
                          expected_values: List[Any]) -> bool:
        """Evaluate comparison operator"""
        
        if operator == "equals":
            return actual_value in expected_values
        elif operator == "not_equals":
            return actual_value not in expected_values
        elif operator == "in":
            return actual_value in expected_values
        elif operator == "not_in":
            return actual_value not in expected_values
        elif operator == "greater_than":
            return float(actual_value) > float(expected_values[0]) if expected_values else False
        elif operator == "less_than":
            return float(actual_value) < float(expected_values[0]) if expected_values else False
        elif operator == "contains":
            return any(str(expected) in str(actual_value) for expected in expected_values)
        elif operator == "starts_with":
            return any(str(actual_value).startswith(str(expected)) for expected in expected_values)
        elif operator == "ends_with":
            return any(str(actual_value).endswith(str(expected)) for expected in expected_values)
        
        return False
    
    def _calculate_user_hash(self, context: Dict[str, Any], salt: str) -> float:
        """Calculate consistent hash for user (0-100)"""
        import hashlib
        
        user_id = context.get('user_id', 'anonymous')
        hash_input = f"{user_id}:{salt}".encode()
        hash_value = hashlib.md5(hash_input).hexdigest()
        
        # Convert to percentage (0-100)
        return (int(hash_value[:8], 16) % 10000) / 100
    
    def _evaluate_custom_rule(self, rule: FeatureFlagRule, context: Dict[str, Any]) -> bool:
        """Evaluate custom targeting rule"""
        # This would be extended based on specific business needs
        return False
    
    def _is_rollout_active(self, flag: FeatureFlag) -> bool:
        """Check if flag rollout is currently active"""
        current_time = datetime.utcnow()
        
        if flag.rollout_start_time and current_time < flag.rollout_start_time:
            return False
        
        if flag.rollout_end_time and current_time > flag.rollout_end_time:
            return False
        
        return True
    
    def _validate_feature_flag(self, flag: FeatureFlag) -> Dict[str, Any]:
        """Validate feature flag configuration"""
        errors = []
        
        # Check required fields
        if not flag.key:
            errors.append("Flag key is required")
        
        if not flag.name:
            errors.append("Flag name is required")
        
        if not flag.variants:
            errors.append("At least one variant is required")
        
        # Check default variant exists
        variant_names = [v.name for v in flag.variants]
        if flag.default_variant not in variant_names:
            errors.append(f"Default variant '{flag.default_variant}' not found in variants")
        
        # Validate targeting rules
        for rule in flag.targeting_rules:
            if rule.variant not in variant_names:
                errors.append(f"Rule variant '{rule.variant}' not found in variants")
        
        # Validate rollout percentage
        if not (0 <= flag.rollout_percentage <= 100):
            errors.append("Rollout percentage must be between 0 and 100")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _track_flag_impression(self, flag_key: str, environment: str,
                                   variant_name: str, context: Dict[str, Any]):
        """Track feature flag impression for analytics"""
        
        impression_data = {
            'flag_key': flag_key,
            'environment': environment,
            'variant': variant_name,
            'user_id': context.get('user_id'),
            'tenant_id': context.get('tenant_id'),
            'timestamp': datetime.utcnow().isoformat(),
            'context': context
        }
        
        # Store impression in analytics system
        try:
            await self.redis.lpush(
                "feature_flag_impressions",
                json.dumps(impression_data)
            )
            
            # Update impression count
            await self.db.execute(
                "UPDATE feature_flags SET impression_count = impression_count + 1, "
                "last_evaluated = $1 WHERE key = $2 AND environment = $3",
                datetime.utcnow(), flag_key, environment
            )
        except Exception as e:
            print(f"Error tracking flag impression: {e}")
    
    async def _invalidate_flag_cache(self, flag_key: str, environment: str):
        """Invalidate feature flag cache"""
        cache_key = f"{self.cache_prefix}{environment}:{flag_key}"
        if cache_key in self.flag_cache:
            del self.flag_cache[cache_key]
        
        # Also publish cache invalidation to other instances
        await self.redis.publish(
            "flag_cache_invalidation",
            json.dumps({'flag_key': flag_key, 'environment': environment})
        )
    
    async def _log_flag_event(self, flag_key: str, action: str, user_id: str,
                            metadata: Dict[str, Any]):
        """Log feature flag events for audit"""
        
        log_entry = {
            'flag_key': flag_key,
            'action': action,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata
        }
        
        # Store in audit log
        await self.db.execute(
            "INSERT INTO feature_flag_audit_log "
            "(flag_key, action, user_id, timestamp, metadata) "
            "VALUES ($1, $2, $3, $4, $5)",
            flag_key, action, user_id, log_entry['timestamp'], json.dumps(metadata)
        )
    
    def register_evaluation_callback(self, callback: Callable):
        """Register callback for flag evaluations"""
        self.evaluation_callbacks.append(callback)

# Example usage in application code
class FeatureFlagClient:
    """Client for easy feature flag usage in application code"""
    
    def __init__(self, feature_flag_service: FeatureFlagService, environment: str):
        self.service = feature_flag_service
        self.environment = environment
    
    async def is_enabled(self, flag_key: str, context: Dict[str, Any]) -> bool:
        """Check if feature flag is enabled (boolean flags)"""
        value = await self.service.evaluate_feature_flag(flag_key, self.environment, context)
        return bool(value) if value is not None else False
    
    async def get_string(self, flag_key: str, context: Dict[str, Any], 
                        default: str = "") -> str:
        """Get string value from feature flag"""
        value = await self.service.evaluate_feature_flag(flag_key, self.environment, context)
        return str(value) if value is not None else default
    
    async def get_number(self, flag_key: str, context: Dict[str, Any], 
                        default: float = 0) -> float:
        """Get number value from feature flag"""
        value = await self.service.evaluate_feature_flag(flag_key, self.environment, context)
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    async def get_json(self, flag_key: str, context: Dict[str, Any], 
                      default: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get JSON object from feature flag"""
        value = await self.service.evaluate_feature_flag(flag_key, self.environment, context)
        if isinstance(value, dict):
            return value
        return default or {}

# Example feature flag definitions
async def create_example_feature_flags(service: FeatureFlagService):
    """Create example feature flags"""
    
    # Boolean feature flag
    new_ui_flag = FeatureFlag(
        key="new_ui_enabled",
        name="New UI Design",
        description="Enable new user interface design",
        flag_type=FeatureFlagType.BOOLEAN,
        enabled=True,
        default_variant="off",
        variants=[
            FeatureFlagVariant(name="off", value=False, description="Old UI"),
            FeatureFlagVariant(name="on", value=True, description="New UI")
        ],
        targeting_rules=[
            FeatureFlagRule(
                rule_type=TargetingRule.TENANT_PLAN,
                operator="in",
                values=["enterprise", "professional"],
                variant="on"
            )
        ],
        created_at=datetime.utcnow(),
        created_by="admin",
        updated_at=datetime.utcnow(),
        updated_by="admin",
        environment="production",
        rollout_percentage=25.0  # 25% rollout
    )
    
    # Configuration feature flag
    model_config_flag = FeatureFlag(
        key="model_configuration",
        name="AI Model Configuration",
        description="Dynamic AI model configuration",
        flag_type=FeatureFlagType.JSON,
        enabled=True,
        default_variant="standard",
        variants=[
            FeatureFlagVariant(
                name="standard",
                value={
                    "primary_model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                description="Standard model configuration"
            ),
            FeatureFlagVariant(
                name="premium",
                value={
                    "primary_model": "gpt-4-turbo",
                    "temperature": 0.8,
                    "max_tokens": 1000
                },
                description="Premium model configuration"
            )
        ],
        targeting_rules=[
            FeatureFlagRule(
                rule_type=TargetingRule.TENANT_PLAN,
                operator="equals",
                values=["enterprise"],
                variant="premium"
            )
        ],
        created_at=datetime.utcnow(),
        created_by="admin",
        updated_at=datetime.utcnow(),
        updated_by="admin",
        environment="production"
    )
    
    await service.create_feature_flag(new_ui_flag)
    await service.create_feature_flag(model_config_flag)
```

---

## Configuration Validation

### Schema-Based Validation

```python
import jsonschema
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of configuration validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class ConfigurationValidator:
    """Schema-based configuration validation"""
    
    def __init__(self):
        self.schemas = {}
        self.custom_validators = {}
        self.load_default_schemas()
    
    def load_default_schemas(self):
        """Load default configuration schemas"""
        
        # Database configuration schema
        self.schemas['database'] = {
            "type": "object",
            "required": ["postgresql", "mongodb", "redis"],
            "properties": {
                "postgresql": {
                    "type": "object",
                    "required": ["host", "port", "database"],
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "database": {"type": "string"},
                        "username": {"type": "string"},
                        "password": {"type": "string"},
                        "pool_size": {"type": "integer", "minimum": 1, "maximum": 100},
                        "ssl_mode": {"type": "string", "enum": ["disable", "require", "verify-ca", "verify-full"]}
                    }
                },
                "mongodb": {
                    "type": "object",
                    "required": ["host", "port", "database"],
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "database": {"type": "string"},
                        "replica_set": {"type": ["string", "null"]},
                        "ssl": {"type": "boolean"}
                    }
                },
                "redis": {
                    "type": "object",
                    "oneOf": [
                        {
                            "properties": {
                                "host": {"type": "string"},
                                "port": {"type": "integer"},
                                "database": {"type": "integer"},
                                "cluster_mode": {"const": False}
                            },
                            "required": ["host", "port"]
                        },
                        {
                            "properties": {
                                "cluster_endpoints": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "cluster_mode": {"const": True}
                            },
                            "required": ["cluster_endpoints"]
                        }
                    ]
                }
            }
        }
        
        # Service configuration schema
        self.schemas['services'] = {
            "type": "object",
            "patternProperties": {
                "^[a-z_]+$": {
                    "type": "object",
                    "properties": {
                        "replicas": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                        },
                        "resources": {
                            "type": "object",
                            "properties": {
                                "requests": {
                                    "type": "object",
                                    "properties": {
                                        "memory": {"type": "string", "pattern": "^[0-9]+(Mi|Gi)$"},
                                        "cpu": {"type": "string", "pattern": "^[0-9]+(m)?$"}
                                    }
                                },
                                "limits": {
                                    "type": "object",
                                    "properties": {
                                        "memory": {"type": "string", "pattern": "^[0-9]+(Mi|Gi)$"},
                                        "cpu": {"type": "string", "pattern": "^[0-9]+(m)?$"}
                                    }
                                }
                            }
                        },
                        "autoscaling": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "min_replicas": {"type": "integer", "minimum": 1},
                                "max_replicas": {"type": "integer", "minimum": 1},
                                "target_cpu_utilization": {"type": "integer", "minimum": 1, "maximum": 100},
                                "target_memory_utilization": {"type": "integer", "minimum": 1, "maximum": 100}
                            }
                        }
                    }
                }
            }
        }
        
        # External API configuration schema
        self.schemas['external_apis'] = {
            "type": "object",
            "patternProperties": {
                "^[a-z_]+$": {
                    "type": "object",
                    "required": ["base_url"],
                    "properties": {
                        "base_url": {
                            "type": "string",
                            "format": "uri"
                        },
                        "timeout": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 300
                        },
                        "max_retries": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10
                        },
                        "rate_limit_per_minute": {
                            "type": "integer",
                            "minimum": 1
                        }
                    }
                }
            }
        }
        
        # Security configuration schema
        self.schemas['security'] = {
            "type": "object",
            "properties": {
                "cors_origins": {
                    "type": "array",
                    "items": {"type": "string", "format": "uri"}
                },
                "csrf_protection": {"type": "boolean"},
                "rate_limiting": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "requests_per_minute": {"type": "integer", "minimum": 1}
                    }
                },
                "compliance": {
                    "type": "object",
                    "properties": {
                        "gdpr_enabled": {"type": "boolean"},
                        "hipaa_enabled": {"type": "boolean"},
                        "soc2_enabled": {"type": "boolean"}
                    }
                }
            }
        }
    
    def validate_configuration(self, config: Dict[str, Any], 
                             schema_name: str) -> ValidationResult:
        """Validate configuration against schema"""
        
        errors = []
        warnings = []
        suggestions = []
        
        if schema_name not in self.schemas:
            errors.append(f"Unknown schema: {schema_name}")
            return ValidationResult(False, errors, warnings, suggestions)
        
        schema = self.schemas[schema_name]
        
        try:
            # JSON Schema validation
            jsonschema.validate(config, schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")
        
        # Custom validations
        if schema_name in self.custom_validators:
            custom_result = self.custom_validators[schema_name](config)
            errors.extend(custom_result.get('errors', []))
            warnings.extend(custom_result.get('warnings', []))
            suggestions.extend(custom_result.get('suggestions', []))
        
        # Cross-field validations
        cross_field_result = self._validate_cross_fields(config, schema_name)
        errors.extend(cross_field_result.get('errors', []))
        warnings.extend(cross_field_result.get('warnings', []))
        suggestions.extend(cross_field_result.get('suggestions', []))
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def register_custom_validator(self, schema_name: str, validator_func):
        """Register custom validation function"""
        self.custom_validators[schema_name] = validator_func
    
    def _validate_cross_fields(self, config: Dict[str, Any], 
                              schema_name: str) -> Dict[str, List[str]]:
        """Validate cross-field dependencies"""
        
        errors = []
        warnings = []
        suggestions = []
        
        if schema_name == 'services':
            # Validate autoscaling configuration
            for service_name, service_config in config.items():
                autoscaling = service_config.get('autoscaling', {})
                if autoscaling.get('enabled'):
                    min_replicas = autoscaling.get('min_replicas', 1)
                    max_replicas = autoscaling.get('max_replicas', 1)
                    current_replicas = service_config.get('replicas', 1)
                    
                    if min_replicas > max_replicas:
                        errors.append(f"{service_name}: min_replicas cannot be greater than max_replicas")
                    
                    if current_replicas < min_replicas:
                        warnings.append(f"{service_name}: current replicas ({current_replicas}) is less than min_replicas ({min_replicas})")
                    
                    if current_replicas > max_replicas:
                        warnings.append(f"{service_name}: current replicas ({current_replicas}) is greater than max_replicas ({max_replicas})")
        
        elif schema_name == 'database':
            # Validate database configuration consistency
            postgresql = config.get('postgresql', {})
            if postgresql.get('ssl_mode') == 'require':
                suggestions.append("Consider using 'verify-full' SSL mode for better security")
            
            redis = config.get('redis', {})
            if redis.get('cluster_mode') and 'host' in redis:
                errors.append("Redis cluster mode cannot use single host configuration")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions
        }
    
    def validate_environment_consistency(self, dev_config: Dict[str, Any],
                                       staging_config: Dict[str, Any],
                                       prod_config: Dict[str, Any]) -> ValidationResult:
        """Validate consistency across environments"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Check structural consistency
        dev_keys = set(self._flatten_keys(dev_config))
        staging_keys = set(self._flatten_keys(staging_config))
        prod_keys = set(self._flatten_keys(prod_config))
        
        # Keys that exist in one environment but not others
        all_keys = dev_keys | staging_keys | prod_keys
        
        for key in all_keys:
            environments = []
            if key in dev_keys:
                environments.append('development')
            if key in staging_keys:
                environments.append('staging')
            if key in prod_keys:
                environments.append('production')
            
            if len(environments) < 3:
                missing = [env for env in ['development', 'staging', 'production'] if env not in environments]
                warnings.append(f"Configuration key '{key}' missing in environments: {', '.join(missing)}")
        
        # Check value type consistency
        common_keys = dev_keys & staging_keys & prod_keys
        for key in common_keys:
            dev_value = self._get_nested_value(dev_config, key)
            staging_value = self._get_nested_value(staging_config, key)
            prod_value = self._get_nested_value(prod_config, key)
            
            types = [type(dev_value), type(staging_value), type(prod_value)]
            if len(set(types)) > 1:
                warnings.append(f"Configuration key '{key}' has different types across environments")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _flatten_keys(self, config: Dict[str, Any], prefix: str = "") -> List[str]:
        """Flatten nested configuration keys"""
        keys = []
        
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            
            if isinstance(value, dict):
                keys.extend(self._flatten_keys(value, full_key))
        
        return keys
    
    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
        """Get value from nested configuration using dot notation"""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value

# Example custom validators
def validate_database_performance(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Custom validator for database performance settings"""
    
    warnings = []
    suggestions = []
    
    postgresql = config.get('postgresql', {})
    pool_size = postgresql.get('pool_size', 5)
    
    if pool_size < 10:
        suggestions.append("Consider increasing PostgreSQL pool_size for better performance in production")
    elif pool_size > 50:
        warnings.append("High PostgreSQL pool_size may cause resource issues")
    
    return {
        'errors': [],
        'warnings': warnings,
        'suggestions': suggestions
    }

def validate_security_settings(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Custom validator for security settings"""
    
    errors = []
    warnings = []
    suggestions = []
    
    cors_origins = config.get('cors_origins', [])
    
    # Check for wildcard CORS origins
    if '*' in cors_origins:
        errors.append("Wildcard CORS origins are not allowed in production")
    
    # Check for HTTP origins in production
    for origin in cors_origins:
        if origin.startswith('http://') and 'localhost' not in origin:
            warnings.append(f"HTTP origin detected: {origin}. Consider using HTTPS")
    
    # Check CSRF protection
    if not config.get('csrf_protection', False):
        warnings.append("CSRF protection is disabled")
    
    return {
        'errors': errors,
        'warnings': warnings,
        'suggestions': suggestions
    }
```

---

## Dynamic Configuration Updates

### Real-Time Configuration Changes

```python
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import websockets

class ConfigurationWatcher:
    """Watch for configuration changes and notify subscribers"""
    
    def __init__(self, redis_client, configuration_manager):
        self.redis = redis_client
        self.config_manager = configuration_manager
        self.subscribers = {}
        self.websocket_connections = set()
        self.watching = False
    
    async def start_watching(self):
        """Start watching for configuration changes"""
        self.watching = True
        
        # Subscribe to Redis pub/sub for config changes
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("config_changes:*")
        
        try:
            while self.watching:
                message = await pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    await self._handle_config_change(message)
        except Exception as e:
            print(f"Error in configuration watcher: {e}")
        finally:
            await pubsub.unsubscribe("config_changes:*")
    
    def stop_watching(self):
        """Stop watching for configuration changes"""
        self.watching = False
    
    async def _handle_config_change(self, message):
        """Handle configuration change notification"""
        try:
            change_data = json.loads(message['data'])
            
            # Notify registered subscribers
            await self._notify_subscribers(change_data)
            
            # Notify WebSocket connections
            await self._notify_websocket_clients(change_data)
            
        except Exception as e:
            print(f"Error handling config change: {e}")
    
    def subscribe_to_changes(self, pattern: str, callback: Callable):
        """Subscribe to configuration changes matching pattern"""
        if pattern not in self.subscribers:
            self.subscribers[pattern] = []
        self.subscribers[pattern].append(callback)
    
    def unsubscribe_from_changes(self, pattern: str, callback: Callable):
        """Unsubscribe from configuration changes"""
        if pattern in self.subscribers:
            try:
                self.subscribers[pattern].remove(callback)
            except ValueError:
                pass
    
    async def _notify_subscribers(self, change_data: Dict[str, Any]):
        """Notify registered subscribers of configuration changes"""
        
        for pattern, callbacks in self.subscribers.items():
            if self._matches_pattern(change_data['key'], pattern):
                for callback in callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(change_data)
                        else:
                            callback(change_data)
                    except Exception as e:
                        print(f"Error in subscriber callback: {e}")
    
    async def _notify_websocket_clients(self, change_data: Dict[str, Any]):
        """Notify WebSocket clients of configuration changes"""
        
        if self.websocket_connections:
            message = {
                'type': 'config_change',
                'data': change_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Send to all connected clients
            disconnected = set()
            for websocket in self.websocket_connections:
                try:
                    await websocket.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(websocket)
                except Exception as e:
                    print(f"Error sending to WebSocket client: {e}")
                    disconnected.add(websocket)
            
            # Remove disconnected clients
            self.websocket_connections -= disconnected
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches subscription pattern"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def add_websocket_connection(self, websocket):
        """Add WebSocket connection for real-time updates"""
        self.websocket_connections.add(websocket)
    
    async def remove_websocket_connection(self, websocket):
        """Remove WebSocket connection"""
        self.websocket_connections.discard(websocket)

class ConfigurationHotReload:
    """Hot reload configuration without service restart"""
    
    def __init__(self, services: Dict[str, Any]):
        self.services = services
        self.reload_handlers = {}
    
    def register_reload_handler(self, config_pattern: str, handler: Callable):
        """Register handler for configuration hot reload"""
        if config_pattern not in self.reload_handlers:
            self.reload_handlers[config_pattern] = []
        self.reload_handlers[config_pattern].append(handler)
    
    async def handle_configuration_change(self, change_data: Dict[str, Any]):
        """Handle configuration change and trigger hot reload"""
        
        config_key = change_data['key']
        
        # Find matching reload handlers
        for pattern, handlers in self.reload_handlers.items():
            if self._matches_pattern(config_key, pattern):
                for handler in handlers:
                    try:
                        await self._execute_reload_handler(handler, change_data)
                    except Exception as e:
                        print(f"Error in reload handler for {pattern}: {e}")
    
    async def _execute_reload_handler(self, handler: Callable, change_data: Dict[str, Any]):
        """Execute reload handler safely"""
        
        if asyncio.iscoroutinefunction(handler):
            await handler(change_data)
        else:
            # Run in thread pool for sync handlers
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, handler, change_data)
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches handler pattern"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)

# Example hot reload handlers
async def reload_database_pool(change_data: Dict[str, Any]):
    """Hot reload database connection pool"""
    
    if 'pool_size' in change_data['key']:
        # Get new pool size from configuration
        new_pool_size = change_data.get('new_value')
        
        if new_pool_size:
            # Update database pool (implementation specific)
            print(f"Updating database pool size to {new_pool_size}")
            # database_pool.resize(new_pool_size)

async def reload_rate_limits(change_data: Dict[str, Any]):
    """Hot reload rate limiting configuration"""
    
    if 'rate_limit' in change_data['key']:
        new_rate_limit = change_data.get('new_value')
        
        if new_rate_limit:
            # Update rate limiter
            print(f"Updating rate limit to {new_rate_limit}")
            # rate_limiter.update_limits(new_rate_limit)

async def reload_external_api_config(change_data: Dict[str, Any]):
    """Hot reload external API configuration"""
    
    if 'external_apis' in change_data['key']:
        api_name = change_data['key'].split('.')[-2]  # Extract API name
        
        # Reload API client configuration
        print(f"Reloading API configuration for {api_name}")
        # api_clients[api_name].reload_config()

class ConfigurationDeployment:
    """Manage configuration deployments across environments"""
    
    def __init__(self, configuration_manager: ConfigurationManager):
        self.config_manager = configuration_manager
        self.deployment_history = []
    
    async def deploy_configuration_batch(self, configurations: List[Dict[str, Any]],
                                       environment: str, user_id: str,
                                       deployment_strategy: str = "rolling") -> Dict[str, Any]:
        """Deploy batch of configurations with specified strategy"""
        
        deployment_id = f"deploy_{int(datetime.utcnow().timestamp())}"
        
        deployment_record = {
            'deployment_id': deployment_id,
            'environment': environment,
            'user_id': user_id,
            'strategy': deployment_strategy,
            'configurations': configurations,
            'started_at': datetime.utcnow(),
            'status': 'in_progress',
            'results': []
        }
        
        self.deployment_history.append(deployment_record)
        
        try:
            if deployment_strategy == "rolling":
                result = await self._rolling_deployment(configurations, environment, user_id)
            elif deployment_strategy == "blue_green":
                result = await self._blue_green_deployment(configurations, environment, user_id)
            elif deployment_strategy == "canary":
                result = await self._canary_deployment(configurations, environment, user_id)
            else:
                result = await self._immediate_deployment(configurations, environment, user_id)
            
            deployment_record['status'] = 'completed' if result['success'] else 'failed'
            deployment_record['completed_at'] = datetime.utcnow()
            deployment_record['results'] = result
            
            return result
            
        except Exception as e:
            deployment_record['status'] = 'error'
            deployment_record['error'] = str(e)
            deployment_record['completed_at'] = datetime.utcnow()
            
            return {
                'success': False,
                'error': str(e),
                'deployment_id': deployment_id
            }
    
    async def _rolling_deployment(self, configurations: List[Dict[str, Any]],
                                environment: str, user_id: str) -> Dict[str, Any]:
        """Deploy configurations with rolling strategy"""
        
        results = []
        failed_configs = []
        
        for config in configurations:
            try:
                success = await self.config_manager.set_configuration(
                    key=config['key'],
                    value=config['value'],
                    environment=Environment(environment),
                    scope=ConfigurationScope(config['scope']),
                    user_id=user_id,
                    tenant_id=config.get('tenant_id'),
                    description=config.get('description'),
                    sensitive=config.get('sensitive', False)
                )
                
                if success:
                    results.append({'key': config['key'], 'status': 'success'})
                    
                    # Wait for propagation
                    await asyncio.sleep(0.5)
                else:
                    failed_configs.append(config['key'])
                    results.append({'key': config['key'], 'status': 'failed'})
                    
            except Exception as e:
                failed_configs.append(config['key'])
                results.append({'key': config['key'], 'status': 'error', 'error': str(e)})
        
        return {
            'success': len(failed_configs) == 0,
            'total_configs': len(configurations),
            'successful_configs': len(configurations) - len(failed_configs),
            'failed_configs': failed_configs,
            'results': results
        }
    
    async def _blue_green_deployment(self, configurations: List[Dict[str, Any]],
                                   environment: str, user_id: str) -> Dict[str, Any]:
        """Deploy configurations with blue-green strategy"""
        
        # Create staging environment copy
        staging_env = f"{environment}_staging"
        
        # Deploy to staging first
        staging_result = await self._immediate_deployment(configurations, staging_env, user_id)
        
        if staging_result['success']:
            # Validate staging environment
            validation_result = await self._validate_environment_health(staging_env)
            
            if validation_result['healthy']:
                # Switch traffic to staging (green)
                await self._switch_environment(environment, staging_env)
                
                # Clean up old environment
                await self._cleanup_old_environment(f"{environment}_old")
                
                return staging_result
            else:
                # Rollback staging changes
                await self._cleanup_old_environment(staging_env)
                
                return {
                    'success': False,
                    'error': 'Environment validation failed',
                    'validation_result': validation_result
                }
        else:
            return staging_result
    
    async def _canary_deployment(self, configurations: List[Dict[str, Any]],
                                environment: str, user_id: str) -> Dict[str, Any]:
        """Deploy configurations with canary strategy"""
        
        # Deploy to canary subset (10% of services)
        canary_result = await self._deploy_to_canary(configurations, environment, user_id)
        
        if canary_result['success']:
            # Monitor canary for 5 minutes
            monitoring_result = await self._monitor_canary_deployment(environment, 300)
            
            if monitoring_result['healthy']:
                # Deploy to remaining services
                return await self._complete_canary_deployment(configurations, environment, user_id)
            else:
                # Rollback canary
                await self._rollback_canary_deployment(environment)
                
                return {
                    'success': False,
                    'error': 'Canary deployment failed health checks',
                    'monitoring_result': monitoring_result
                }
        else:
            return canary_result
    
    async def _immediate_deployment(self, configurations: List[Dict[str, Any]],
                                  environment: str, user_id: str) -> Dict[str, Any]:
        """Deploy configurations immediately"""
        
        return await self.config_manager.bulk_update_configuration(configurations, user_id)
    
    async def _validate_environment_health(self, environment: str) -> Dict[str, Any]:
        """Validate environment health after deployment"""
        
        # This would check service health, metrics, etc.
        # Placeholder implementation
        return {
            'healthy': True,
            'checks': [
                {'service': 'chat-service', 'status': 'healthy'},
                {'service': 'mcp-engine', 'status': 'healthy'}
            ]
        }
    
    async def _switch_environment(self, current_env: str, new_env: str):
        """Switch traffic from current to new environment"""
        # Placeholder for environment switching logic
        print(f"Switching from {current_env} to {new_env}")
    
    async def _cleanup_old_environment(self, environment: str):
        """Clean up old environment configuration"""
        # Placeholder for cleanup logic
        print(f"Cleaning up environment {environment}")
    
    async def _deploy_to_canary(self, configurations: List[Dict[str, Any]],
                              environment: str, user_id: str) -> Dict[str, Any]:
        """Deploy to canary subset"""
        # Placeholder for canary deployment
        return {'success': True, 'canary_deployed': True}
    
    async def _monitor_canary_deployment(self, environment: str, duration_seconds: int) -> Dict[str, Any]:
        """Monitor canary deployment health"""
        # Placeholder for canary monitoring
        await asyncio.sleep(1)  # Simulate monitoring
        return {'healthy': True, 'metrics': {}}
    
    async def _complete_canary_deployment(self, configurations: List[Dict[str, Any]],
                                        environment: str, user_id: str) -> Dict[str, Any]:
        """Complete canary deployment to all services"""
        return await self._immediate_deployment(configurations, environment, user_id)
    
    async def _rollback_canary_deployment(self, environment: str):
        """Rollback canary deployment"""
        print(f"Rolling back canary deployment in {environment}")
```

---

## Configuration Versioning

### Git-Based Configuration Management

```python
import git
import os
import yaml
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConfigurationCommit:
    """Configuration commit information"""
    commit_hash: str
    author: str
    message: str
    timestamp: datetime
    changed_files: List[str]
    environment: str

class GitConfigurationManager:
    """Git-based configuration version control"""
    
    def __init__(self, repo_path: str, remote_url: Optional[str] = None):
        self.repo_path = repo_path
        self.remote_url = remote_url
        self.repo = None
        self._init_repository()
    
    def _init_repository(self):
        """Initialize or open Git repository"""
        
        if os.path.exists(os.path.join(self.repo_path, '.git')):
            # Open existing repository
            self.repo = git.Repo(self.repo_path)
        else:
            # Initialize new repository
            os.makedirs(self.repo_path, exist_ok=True)
            self.repo = git.Repo.init(self.repo_path)
            
            # Add remote if specified
            if self.remote_url:
                self.repo.create_remote('origin', self.remote_url)
            
            # Create initial commit
            self._create_initial_structure()
    
    def _create_initial_structure(self):
        """Create initial configuration structure"""
        
        directories = [
            'environments/development',
            'environments/staging', 
            'environments/production',
            'services',
            'secrets',
            'schemas'
        ]
        
        for directory in directories:
            dir_path = os.path.join(self.repo_path, directory)
            os.makedirs(dir_path, exist_ok=True)
            
            # Create .gitkeep file
            gitkeep_path = os.path.join(dir_path, '.gitkeep')
            with open(gitkeep_path, 'w') as f:
                f.write('')
        
        # Create .gitignore
        gitignore_content = """
# Sensitive files
secrets/*.yml
secrets/*.yaml
secrets/*.json
*.key
*.pem

# Environment-specific overrides
local.yml
local.yaml

# Temporary files
*.tmp
*.bak
        """.strip()
        
        gitignore_path = os.path.join(self.repo_path, '.gitignore')
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        # Initial commit
        self.repo.index.add(['.gitignore', '.'])
        self.repo.index.commit("Initial configuration structure")
    
    def save_configuration(self, config_path: str, config_data: Dict[str, Any],
                          commit_message: str, author_name: str, author_email: str) -> str:
        """Save configuration and create Git commit"""
        
        full_path = os.path.join(self.repo_path, config_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write configuration file
        if config_path.endswith('.json'):
            with open(full_path, 'w') as f:
                json.dump(config_data, f, indent=2, sort_keys=True)
        else:
            with open(full_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=True)
        
        # Add to Git index
        self.repo.index.add([config_path])
        
        # Create commit
        actor = git.Actor(author_name, author_email)
        commit = self.repo.index.commit(commit_message, author=actor, committer=actor)
        
        return commit.hexsha
    
    def get_configuration(self, config_path: str, commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get configuration from specific commit or latest"""
        
        try:
            if commit_hash:
                # Get from specific commit
                commit = self.repo.commit(commit_hash)
                blob = commit.tree / config_path
                content = blob.data_stream.read().decode('utf-8')
            else:
                # Get from working directory
                full_path = os.path.join(self.repo_path, config_path)
                with open(full_path, 'r') as f:
                    content = f.read()
            
            # Parse configuration
            if config_path.endswith('.json'):
                return json.loads(content)
            else:
                return yaml.safe_load(content)
                
        except Exception as e:
            print(f"Error getting configuration: {e}")
            return None
    
    def get_configuration_history(self, config_path: str, limit: int = 50) -> List[ConfigurationCommit]:
        """Get configuration change history"""
        
        commits = []
        
        try:
            # Get commits that modified the file
            commit_iter = self.repo.iter_commits(paths=config_path, max_count=limit)
            
            for commit in commit_iter:
                # Get changed files in this commit
                changed_files = []
                if commit.parents:
                    for item in commit.diff(commit.parents[0]):
                        changed_files.append(item.a_path or item.b_path)
                
                commit_info = ConfigurationCommit(
                    commit_hash=commit.hexsha,
                    author=f"{commit.author.name} <{commit.author.email}>",
                    message=commit.message.strip(),
                    timestamp=datetime.fromtimestamp(commit.committed_date),
                    changed_files=changed_files,
                    environment=self._extract_environment_from_path(config_path)
                )
                
                commits.append(commit_info)
        
        except Exception as e:
            print(f"Error getting configuration history: {e}")
        
        return commits
    
    def create_branch(self, branch_name: str, base_branch: str = "main") -> bool:
        """Create new branch for configuration changes"""
        
        try:
            # Check out base branch
            if base_branch in [branch.name for branch in self.repo.branches]:
                self.repo.heads[base_branch].checkout()
            
            # Create new branch
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            
            return True
        except Exception as e:
            print(f"Error creating branch: {e}")
            return False
    
    def merge_branch(self, branch_name: str, target_branch: str = "main",
                    merge_message: str = None) -> bool:
        """Merge configuration branch"""
        
        try:
            # Check out target branch
            target = self.repo.heads[target_branch]
            target.checkout()
            
            # Merge source branch
            source = self.repo.heads[branch_name]
            merge_base = self.repo.merge_base(target, source)[0]
            
            # Perform merge
            self.repo.index.merge_tree(target.commit, base=merge_base)
            
            # Create merge commit
            if not merge_message:
                merge_message = f"Merge branch '{branch_name}' into {target_branch}"
            
            commit = self.repo.index.commit(
                merge_message,
                parent_commits=(target.commit, source.commit)
            )
            
            # Update branch reference
            target.commit = commit
            
            return True
        except Exception as e:
            print(f"Error merging branch: {e}")
            return False
    
    def create_tag(self, tag_name: str, commit_hash: Optional[str] = None,
                  message: str = None) -> bool:
        """Create tag for configuration release"""
        
        try:
            if commit_hash:
                commit = self.repo.commit(commit_hash)
            else:
                commit = self.repo.head.commit
            
            if message:
                # Annotated tag
                self.repo.create_tag(tag_name, ref=commit, message=message)
            else:
                # Lightweight tag
                self.repo.create_tag(tag_name, ref=commit)
            
            return True
        except Exception as e:
            print(f"Error creating tag: {e}")
            return False
    
    def rollback_to_commit(self, commit_hash: str, config_path: str) -> bool:
        """Rollback configuration to specific commit"""
        
        try:
            # Get configuration from target commit
            config_data = self.get_configuration(config_path, commit_hash)
            
            if config_data:
                # Save configuration with rollback message
                rollback_message = f"Rollback {config_path} to commit {commit_hash[:8]}"
                
                # Use system user for rollback
                author_name = "System"
                author_email = "system@chatbot-platform.com"
                
                self.save_configuration(
                    config_path, config_data, rollback_message,
                    author_name, author_email
                )
                
                return True
        except Exception as e:
            print(f"Error rolling back configuration: {e}")
        
        return False
    
    def sync_with_remote(self, branch: str = "main") -> Dict[str, Any]:
        """Sync with remote repository"""
        
        result = {
            'success': False,
            'pulled_commits': 0,
            'pushed_commits': 0,
            'conflicts': []
        }
        
        try:
            if 'origin' in [remote.name for remote in self.repo.remotes]:
                origin = self.repo.remotes.origin
                
                # Fetch latest changes
                origin.fetch()
                
                # Get current branch
                current_branch = self.repo.active_branch
                
                # Check for remote tracking branch
                remote_branch = f"origin/{branch}"
                if remote_branch in [ref.name for ref in self.repo.refs]:
                    # Count commits to pull
                    commits_to_pull = list(self.repo.iter_commits(f"{current_branch}..{remote_branch}"))
                    result['pulled_commits'] = len(commits_to_pull)
                    
                    # Count commits to push
                    commits_to_push = list(self.repo.iter_commits(f"{remote_branch}..{current_branch}"))
                    result['pushed_commits'] = len(commits_to_push)
                    
                    # Pull changes
                    if commits_to_pull:
                        origin.pull(branch)
                    
                    # Push changes
                    if commits_to_push:
                        origin.push(branch)
                    
                    result['success'] = True
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _extract_environment_from_path(self, config_path: str) -> str:
        """Extract environment from configuration path"""
        if 'development' in config_path:
            return 'development'
        elif 'staging' in config_path:
            return 'staging'
        elif 'production' in config_path:
            return 'production'
        else:
            return 'unknown'
    
    def get_diff(self, commit1: str, commit2: str, config_path: Optional[str] = None) -> str:
        """Get diff between two commits"""
        
        try:
            commit_obj1 = self.repo.commit(commit1)
            commit_obj2 = self.repo.commit(commit2)
            
            if config_path:
                # Diff specific file
                diff = commit_obj1.diff(commit_obj2, paths=config_path, create_patch=True)
            else:
                # Diff all files
                diff = commit_obj1.diff(commit_obj2, create_patch=True)
            
            return '\n'.join([str(d) for d in diff])
        except Exception as e:
            return f"Error generating diff: {e}"

class ConfigurationPromotion:
    """Manage configuration promotion across environments"""
    
    def __init__(self, git_manager: GitConfigurationManager,
                 configuration_manager: ConfigurationManager):
        self.git_manager = git_manager
        self.config_manager = configuration_manager
        self.promotion_policies = {}
    
    def set_promotion_policy(self, from_env: str, to_env: str, policy: Dict[str, Any]):
        """Set promotion policy between environments"""
        
        policy_key = f"{from_env}_{to_env}"
        self.promotion_policies[policy_key] = {
            'approval_required': policy.get('approval_required', True),
            'approvers': policy.get('approvers', []),
            'automatic_tests': policy.get('automatic_tests', []),
            'rollback_enabled': policy.get('rollback_enabled', True),
            'notification_channels': policy.get('notification_channels', [])
        }
    
    async def promote_configuration(self, config_paths: List[str], 
                                  from_env: str, to_env: str,
                                  promoted_by: str,
                                  approval_token: Optional[str] = None) -> Dict[str, Any]:
        """Promote configuration from one environment to another"""
        
        policy_key = f"{from_env}_{to_env}"
        policy = self.promotion_policies.get(policy_key, {})
        
        promotion_record = {
            'promotion_id': f"promo_{int(datetime.utcnow().timestamp())}",
            'config_paths': config_paths,
            'from_environment': from_env,
            'to_environment': to_env,
            'promoted_by': promoted_by,
            'started_at': datetime.utcnow(),
            'status': 'in_progress'
        }
        
        try:
            # Check approval requirements
            if policy.get('approval_required') and not approval_token:
                return {
                    'success': False,
                    'error': 'Approval required for this promotion',
                    'approval_required': True,
                    'approvers': policy.get('approvers', [])
                }
            
            # Run automatic tests
            test_results = await self._run_automatic_tests(
                config_paths, from_env, to_env, policy.get('automatic_tests', [])
            )
            
            if not test_results['passed']:
                promotion_record['status'] = 'failed'
                promotion_record['test_results'] = test_results
                return {
                    'success': False,
                    'error': 'Automatic tests failed',
                    'test_results': test_results
                }
            
            # Create promotion branch
            branch_name = f"promote-{from_env}-to-{to_env}-{promotion_record['promotion_id']}"
            self.git_manager.create_branch(branch_name, f"{to_env}")
            
            # Copy configurations
            promotion_results = []
            for config_path in config_paths:
                result = await self._promote_single_config(
                    config_path, from_env, to_env, promoted_by
                )
                promotion_results.append(result)
            
            # Commit changes
            commit_message = f"Promote configurations from {from_env} to {to_env}\n\n" + \
                           f"Promoted by: {promoted_by}\n" + \
                           f"Configs: {', '.join(config_paths)}"
            
            commit_hash = self.git_manager.save_configuration(
                f"promotion_{promotion_record['promotion_id']}.json",
                promotion_record,
                commit_message,
                promoted_by,
                f"{promoted_by}@chatbot-platform.com"
            )
            
            # Merge to target environment branch
            merge_success = self.git_manager.merge_branch(branch_name, to_env)
            
            if merge_success:
                # Deploy to target environment
                deployment_result = await self._deploy_promoted_configs(
                    config_paths, to_env, promoted_by
                )
                
                promotion_record['status'] = 'completed' if deployment_result['success'] else 'failed'
                promotion_record['deployment_result'] = deployment_result
                promotion_record['commit_hash'] = commit_hash
                promotion_record['completed_at'] = datetime.utcnow()
                
                # Send notifications
                await self._send_promotion_notifications(promotion_record, policy)
                
                return {
                    'success': deployment_result['success'],
                    'promotion_id': promotion_record['promotion_id'],
                    'commit_hash': commit_hash,
                    'deployment_result': deployment_result
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to merge promotion branch'
                }
        
        except Exception as e:
            promotion_record['status'] = 'error'
            promotion_record['error'] = str(e)
            promotion_record['completed_at'] = datetime.utcnow()
            
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _promote_single_config(self, config_path: str, from_env: str, 
                                   to_env: str, promoted_by: str) -> Dict[str, Any]:
        """Promote single configuration file"""
        
        try:
            # Get configuration from source environment
            from_config = self.git_manager.get_configuration(
                f"environments/{from_env}/{config_path}"
            )
            
            if from_config:
                # Apply environment-specific transformations
                to_config = self._transform_config_for_environment(from_config, to_env)
                
                # Save to target environment
                target_path = f"environments/{to_env}/{config_path}"
                self.git_manager.save_configuration(
                    target_path, to_config,
                    f"Promote {config_path} from {from_env} to {to_env}",
                    promoted_by,
                    f"{promoted_by}@chatbot-platform.com"
                )
                
                return {
                    'config_path': config_path,
                    'success': True,
                    'from_env': from_env,
                    'to_env': to_env
                }
        except Exception as e:
            return {
                'config_path': config_path,
                'success': False,
                'error': str(e)
            }
    
    def _transform_config_for_environment(self, config: Dict[str, Any], 
                                        target_env: str) -> Dict[str, Any]:
        """Transform configuration for target environment"""
        
        # Deep copy configuration
        import copy
        transformed_config = copy.deepcopy(config)
        
        # Environment-specific transformations
        if target_env == 'production':
            # Production-specific changes
            if 'debug' in transformed_config:
                transformed_config['debug'] = False
            
            if 'log_level' in transformed_config:
                transformed_config['log_level'] = 'INFO'
            
            # Update resource allocations for production
            if 'services' in transformed_config:
                for service_name, service_config in transformed_config['services'].items():
                    if 'replicas' in service_config:
                        # Scale up for production
                        service_config['replicas'] = max(service_config['replicas'] * 2, 3)
        
        elif target_env == 'staging':
            # Staging-specific changes
            if 'debug' in transformed_config:
                transformed_config['debug'] = True
            
            if 'log_level' in transformed_config:
                transformed_config['log_level'] = 'DEBUG'
        
        return transformed_config
    
    async def _run_automatic_tests(self, config_paths: List[str], from_env: str,
                                 to_env: str, tests: List[str]) -> Dict[str, Any]:
        """Run automatic tests for configuration promotion"""
        
        test_results = {
            'passed': True,
            'total_tests': len(tests),
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        for test_name in tests:
            test_result = await self._run_single_test(test_name, config_paths, from_env, to_env)
            test_results['test_details'].append(test_result)
            
            if test_result['passed']:
                test_results['passed_tests'] += 1
            else:
                test_results['failed_tests'] += 1
                test_results['passed'] = False
        
        return test_results
    
    async def _run_single_test(self, test_name: str, config_paths: List[str],
                             from_env: str, to_env: str) -> Dict[str, Any]:
        """Run single automatic test"""
        
        # Placeholder for test implementations
        if test_name == 'schema_validation':
            # Validate configurations against schemas
            return {'test': test_name, 'passed': True, 'message': 'Schema validation passed'}
        
        elif test_name == 'security_check':
            # Check for security issues
            return {'test': test_name, 'passed': True, 'message': 'Security check passed'}
        
        elif test_name == 'environment_compatibility':
            # Check environment compatibility
            return {'test': test_name, 'passed': True, 'message': 'Environment compatibility verified'}
        
        else:
            return {'test': test_name, 'passed': False, 'message': f'Unknown test: {test_name}'}
    
    async def _deploy_promoted_configs(self, config_paths: List[str], 
                                     to_env: str, promoted_by: str) -> Dict[str, Any]:
        """Deploy promoted configurations to target environment"""
        
        # Load and deploy configurations to the configuration manager
        deployed_configs = []
        failed_configs = []
        
        for config_path in config_paths:
            try:
                config_data = self.git_manager.get_configuration(
                    f"environments/{to_env}/{config_path}"
                )
                
                if config_data:
                    # Deploy each configuration item
                    for key, value in config_data.items():
                        success = await self.config_manager.set_configuration(
                            key=f"{config_path}.{key}",
                            value=value,
                            environment=Environment(to_env),
                            scope=ConfigurationScope.ENVIRONMENT,
                            user_id=promoted_by,
                            description=f"Promoted from Git: {config_path}"
                        )
                        
                        if success:
                            deployed_configs.append(f"{config_path}.{key}")
                        else:
                            failed_configs.append(f"{config_path}.{key}")
            except Exception as e:
                failed_configs.append(f"{config_path}: {str(e)}")
        
        return {
            'success': len(failed_configs) == 0,
            'deployed_configs': deployed_configs,
            'failed_configs': failed_configs
        }
    
    async def _send_promotion_notifications(self, promotion_record: Dict[str, Any],
                                          policy: Dict[str, Any]):
        """Send notifications about configuration promotion"""
        
        channels = policy.get('notification_channels', [])
        
        for channel in channels:
            if channel == 'slack':
                await self._send_slack_notification(promotion_record)
            elif channel == 'email':
                await self._send_email_notification(promotion_record)
    
    async def _send_slack_notification(self, promotion_record: Dict[str, Any]):
        """Send Slack notification"""
        # Placeholder for Slack integration
        print(f"Slack notification: Configuration promotion {promotion_record['promotion_id']} completed")
    
    async def _send_email_notification(self, promotion_record: Dict[str, Any]):
        """Send email notification"""
        # Placeholder for email integration
        print(f"Email notification: Configuration promotion {promotion_record['promotion_id']} completed")
```

---

## Infrastructure as Code

### Terraform Configuration Management

```yaml
# terraform/environments/production/main.tf
terraform {
  required_version = ">= 1.0"
  
  backend "s3" {
    bucket = "chatbot-platform-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
    
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = "production"
      Project     = "chatbot-platform"
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_eks_cluster" "cluster" {
  name = var.cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = var.cluster_name
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "chatbot-platform-prod"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

# Configuration Management
resource "kubernetes_namespace" "configuration" {
  metadata {
    name = "configuration-management"
    
    labels = {
      name = "configuration-management"
    }
  }
}

# HashiCorp Vault
resource "helm_release" "vault" {
  name       = "vault"
  repository = "https://helm.releases.hashicorp.com"
  chart      = "vault"
  version    = "0.25.0"
  namespace  = kubernetes_namespace.configuration.metadata[0].name
  
  values = [
    file("${path.module}/helm-values/vault-values.yaml")
  ]
  
  set {
    name  = "server.ha.enabled"
    value = "true"
  }
  
  set {
    name  = "server.ha.replicas"
    value = "3"
  }
  
  depends_on = [kubernetes_namespace.configuration]
}

# External Secrets Operator
resource "helm_release" "external_secrets" {
  name       = "external-secrets"
  repository = "https://charts.external-secrets.io"
  chart      = "external-secrets"
  version    = "0.9.9"
  namespace  = kubernetes_namespace.configuration.metadata[0].name
  
  values = [
    file("${path.module}/helm-values/external-secrets-values.yaml")
  ]
  
  depends_on = [kubernetes_namespace.configuration]
}

# Configuration Management Service
resource "kubernetes_deployment" "config_manager" {
  metadata {
    name      = "configuration-manager"
    namespace = kubernetes_namespace.configuration.metadata[0].name
    
    labels = {
      app = "configuration-manager"
    }
  }
  
  spec {
    replicas = 3
    
    selector {
      match_labels = {
        app = "configuration-manager"
      }
    }
    
    template {
      metadata {
        labels = {
          app = "configuration-manager"
        }
      }
      
      spec {
        container {
          name  = "config-manager"
          image = "chatbot-platform/configuration-manager:latest"
          
          port {
            container_port = 8080
          }
          
          env {
            name  = "ENVIRONMENT"
            value = var.environment
          }
          
          env {
            name = "DATABASE_URL"
            value_from {
              secret_key_ref {
                name = "database-credentials"
                key  = "url"
              }
            }
          }
          
          env {
            name = "VAULT_ADDR"
            value = "http://vault:8200"
          }
          
          env {
            name = "VAULT_TOKEN"
            value_from {
              secret_key_ref {
                name = "vault-credentials"
                key  = "token"
              }
            }
          }
          
          resources {
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "1Gi"
            }
          }
          
          liveness_probe {
            http_get {
              path = "/health"
              port = 8080
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }
          
          readiness_probe {
            http_get {
              path = "/ready"
              port = 8080
            }
            initial_delay_seconds = 5
            period_seconds        = 5
          }
        }
        
        service_account_name = kubernetes_service_account.config_manager.metadata[0].name
      }
    }
  }
}

resource "kubernetes_service_account" "config_manager" {
  metadata {
    name      = "configuration-manager"
    namespace = kubernetes_namespace.configuration.metadata[0].name
  }
}

resource "kubernetes_service" "config_manager" {
  metadata {
    name      = "configuration-manager"
    namespace = kubernetes_namespace.configuration.metadata[0].name
  }
  
  spec {
    selector = {
      app = "configuration-manager"
    }
    
    port {
      port        = 80
      target_port = 8080
      protocol    = "TCP"
    }
    
    type = "ClusterIP"
  }
}

# ConfigMaps for different environments
resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "app-configuration"
    namespace = kubernetes_namespace.configuration.metadata[0].name
  }
  
  data = {
    "app.yaml" = file("${path.module}/config/app-config-${var.environment}.yaml")
  }
}

resource "kubernetes_config_map" "database_config" {
  metadata {
    name      = "database-configuration"
    namespace = kubernetes_namespace.configuration.metadata[0].name
  }
  
  data = {
    "database.yaml" = file("${path.module}/config/database-config-${var.environment}.yaml")
  }
}

# Secrets (references to external secrets)
resource "kubernetes_secret" "database_credentials" {
  metadata {
    name      = "database-credentials"
    namespace = kubernetes_namespace.configuration.metadata[0].name
  }
  
  type = "Opaque"
  
  # These would be managed by External Secrets Operator
  data = {}
  
  lifecycle {
    ignore_changes = [data]
  }
}

# External Secret for Vault integration
resource "kubernetes_manifest" "database_external_secret" {
  manifest = {
    apiVersion = "external-secrets.io/v1beta1"
    kind       = "ExternalSecret"
    
    metadata = {
      name      = "database-credentials"
      namespace = kubernetes_namespace.configuration.metadata[0].name
    }
    
    spec = {
      refreshInterval = "1m"
      
      secretStoreRef = {
        name = "vault-secret-store"
        kind = "SecretStore"
      }
      
      target = {
        name           = "database-credentials"
        creationPolicy = "Owner"
        template = {
          type = "Opaque"
          data = {
            url = "postgresql://{{ .username }}:{{ .password }}@{{ .host }}:{{ .port }}/{{ .database }}"
          }
        }
      }
      
      data = [
        {
          secretKey = "username"
          remoteRef = {
            key      = "database/postgresql"
            property = "username"
          }
        },
        {
          secretKey = "password"
          remoteRef = {
            key      = "database/postgresql"
            property = "password"
          }
        },
        {
          secretKey = "host"
          remoteRef = {
            key      = "database/postgresql"
            property = "host"
          }
        },
        {
          secretKey = "port"
          remoteRef = {
            key      = "database/postgresql"
            property = "port"
          }
        },
        {
          secretKey = "database"
          remoteRef = {
            key      = "database/postgresql"
            property = "database"
          }
        }
      ]
    }
  }
  
  depends_on = [helm_release.external_secrets]
}

# Outputs
output "configuration_namespace" {
  value = kubernetes_namespace.configuration.metadata[0].name
}

output "vault_service_url" {
  value = "http://vault.${kubernetes_namespace.configuration.metadata[0].name}.svc.cluster.local:8200"
}

output "config_manager_service_url" {
  value = "http://configuration-manager.${kubernetes_namespace.configuration.metadata[0].name}.svc.cluster.local"
}
```

```yaml
# helm-values/vault-values.yaml
global:
  enabled: true
  
server:
  image:
    repository: "hashicorp/vault"
    tag: "1.15.2"
  
  resources:
    requests:
      memory: 256Mi
      cpu: 250m
    limits:
      memory: 512Mi
      cpu: 500m
  
  readinessProbe:
    enabled: true
    path: "/v1/sys/health?standbyok=true&sealedcode=204&uninitcode=204"
  
  livenessProbe:
    enabled: true
    path: "/v1/sys/health?standbyok=true"
    initialDelaySeconds: 60
  
  extraEnvironmentVars:
    VAULT_CACERT: /vault/userconfig/vault-ha-tls/vault.ca
    VAULT_TLSCERT: /vault/userconfig/vault-ha-tls/vault.crt
    VAULT_TLSKEY: /vault/userconfig/vault-ha-tls/vault.key
  
  volumes:
    - name: userconfig-vault-ha-tls
      secret:
        defaultMode: 420
        secretName: vault-ha-tls
  
  volumeMounts:
    - mountPath: /vault/userconfig/vault-ha-tls
      name: userconfig-vault-ha-tls
      readOnly: true
  
  standalone:
    enabled: false
  
  ha:
    enabled: true
    replicas: 3
    
    raft:
      enabled: true
      
      config: |
        ui = true
        
        listener "tcp" {
          tls_disable = 0
          address = "[::]:8200"
          cluster_address = "[::]:8201"
          tls_cert_file = "/vault/userconfig/vault-ha-tls/vault.crt"
          tls_key_file  = "/vault/userconfig/vault-ha-tls/vault.key"
          tls_client_ca_file = "/vault/userconfig/vault-ha-tls/vault.ca"
        }
        
        storage "raft" {
          path = "/vault/data"
          
          retry_join {
            leader_api_addr = "http://vault-0.vault-internal:8200"
          }
          
          retry_join {
            leader_api_addr = "http://vault-1.vault-internal:8200"
          }
          
          retry_join {
            leader_api_addr = "http://vault-2.vault-internal:8200"
          }
        }
        
        disable_mlock = true
        service_registration "kubernetes" {}

ui:
  enabled: true
  serviceType: "ClusterIP"
  
injector:
  enabled: false
```

```yaml
# helm-values/external-secrets-values.yaml
installCRDs: true

replicaCount: 2

resources:
  limits:
    cpu: 100m
    memory: 128Mi
  requests:
    cpu: 100m
    memory: 128Mi

serviceMonitor:
  enabled: true
  namespace: monitoring

webhook:
  replicaCount: 2
  
  resources:
    limits:
      cpu: 100m
      memory: 128Mi
    requests:
      cpu: 100m
      memory: 128Mi

certController:
  replicaCount: 2
  
  resources:
    limits:
      cpu: 100m
      memory: 128Mi
    requests:
      cpu: 100m
      memory: 128Mi
```

### Ansible Playbooks for Configuration Deployment

```yaml
# ansible/deploy-configuration.yml
---
- name: Deploy Configuration Management
  hosts: kubernetes_cluster
  gather_facts: false
  
  vars:
    environment: "{{ env | default('staging') }}"
    namespace: "configuration-management"
    vault_version: "1.15.2"
    
  tasks:
    - name: Create namespace
      kubernetes.core.k8s:
        name: "{{ namespace }}"
        api_version: v1
        kind: Namespace
        state: present
        
    - name: Deploy Vault using Helm
      kubernetes.core.helm:
        name: vault
        chart_ref: hashicorp/vault
        release_namespace: "{{ namespace }}"
        create_namespace: true
        values_files:
          - "helm-values/vault-values.yaml"
        set_values:
          - value: server.image.tag={{ vault_version }}
            value_type: string
        wait: true
        wait_condition:
          type: Ready
          status: "True"
        wait_timeout: 600
        
    - name: Deploy External Secrets Operator
      kubernetes.core.helm:
        name: external-secrets
        chart_ref: external-secrets/external-secrets
        release_namespace: "{{ namespace }}"
        values_files:
          - "helm-values/external-secrets-values.yaml"
        wait: true
        wait_timeout: 300
        
    - name: Deploy Configuration Manager
      kubernetes.core.k8s:
        state: present
        definition:
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: configuration-manager
            namespace: "{{ namespace }}"
            labels:
              app: configuration-manager
          spec:
            replicas: 3
            selector:
              matchLabels:
                app: configuration-manager
            template:
              metadata:
                labels:
                  app: configuration-manager
              spec:
                containers:
                - name: config-manager
                  image: "chatbot-platform/configuration-manager:{{ image_tag | default('latest') }}"
                  ports:
                  - containerPort: 8080
                  env:
                  - name: ENVIRONMENT
                    value: "{{ environment }}"
                  - name: VAULT_ADDR
                    value: "http://vault:8200"
                  resources:
                    requests:
                      cpu: 250m
                      memory: 512Mi
                    limits:
                      cpu: 500m
                      memory: 1Gi
                  livenessProbe:
                    httpGet:
                      path: /health
                      port: 8080
                    initialDelaySeconds: 30
                    periodSeconds: 10
                  readinessProbe:
                    httpGet:
                      path: /ready
                      port: 8080
                    initialDelaySeconds: 5
                    periodSeconds: 5
                    
    - name: Create Configuration Manager Service
      kubernetes.core.k8s:
        state: present
        definition:
          apiVersion: v1
          kind: Service
          metadata:
            name: configuration-manager
            namespace: "{{ namespace }}"
          spec:
            selector:
              app: configuration-manager
            ports:
            - port: 80
              targetPort: 8080
              protocol: TCP
            type: ClusterIP
            
    - name: Wait for Configuration Manager to be ready
      kubernetes.core.k8s_info:
        api_version: apps/v1
        kind: Deployment
        name: configuration-manager
        namespace: "{{ namespace }}"
        wait_condition:
          type: Available
          status: "True"
        wait_timeout: 300
        
    - name: Deploy configuration files
      kubernetes.core.k8s:
        state: present
        definition:
          apiVersion: v1
          kind: ConfigMap
          metadata:
            name: "{{ item.name }}"
            namespace: "{{ namespace }}"
          data: "{{ item.data }}"
      loop:
        - name: app-configuration
          data:
            app.yaml: "{{ lookup('file', 'config/app-config-' + environment + '.yaml') }}"
        - name: database-configuration
          data:
            database.yaml: "{{ lookup('file', 'config/database-config-' + environment + '.yaml') }}"
            
    - name: Verify deployment
      uri:
        url: "http://{{ ansible_host }}/api/health"
        method: GET
        status_code: 200
      retries: 5
      delay: 10
```

```yaml
# ansible/inventory/production.yml
---
kubernetes_cluster:
  hosts:
    k8s-master-1:
      ansible_host: 10.0.1.10
    k8s-master-2:
      ansible_host: 10.0.1.11
    k8s-master-3:
      ansible_host: 10.0.1.12
  vars:
    ansible_user: ubuntu
    ansible_ssh_private_key_file: ~/.ssh/kubernetes-key.pem
    kubeconfig_path: ~/.kube/config-production
```

---

## Configuration Security

### Security Best Practices

```python
import hashlib
import hmac
import secrets
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class ConfigurationSecurity:
    """Security controls for configuration management"""
    
    def __init__(self, encryption_key: bytes):
        self.fernet = Fernet(encryption_key)
        self.access_policies = {}
        self.audit_trail = []
        
    def encrypt_sensitive_value(self, value: str, context: Dict[str, Any] = None) -> str:
        """Encrypt sensitive configuration value"""
        
        # Add context to the encrypted data for additional security
        payload = {
            'value': value,
            'context': context or {},
            'encrypted_at': datetime.utcnow().isoformat(),
            'checksum': hashlib.sha256(value.encode()).hexdigest()
        }
        
        encrypted_data = self.fernet.encrypt(json.dumps(payload).encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_value(self, encrypted_value: str, 
                              expected_context: Dict[str, Any] = None) -> str:
        """Decrypt sensitive configuration value"""
        
        try:
            encrypted_data = base64.b64decode(encrypted_value.encode())
            decrypted_data = self.fernet.decrypt(encrypted_data)
            payload = json.loads(decrypted_data.decode())
            
            # Verify checksum
            value = payload['value']
            expected_checksum = hashlib.sha256(value.encode()).hexdigest()
            
            if payload['checksum'] != expected_checksum:
                raise ValueError("Checksum verification failed")
            
            # Verify context if provided
            if expected_context:
                if payload['context'] != expected_context:
                    raise ValueError("Context verification failed")
            
            return value
            
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
    
    def create_access_policy(self, policy_name: str, policy: Dict[str, Any]):
        """Create access policy for configuration management"""
        
        self.access_policies[policy_name] = {
            'users': policy.get('users', []),
            'roles': policy.get('roles', []),
            'permissions': policy.get('permissions', []),
            'environments': policy.get('environments', []),
            'config_patterns': policy.get('config_patterns', []),
            'time_restrictions': policy.get('time_restrictions', {}),
            'ip_restrictions': policy.get('ip_restrictions', []),
            'mfa_required': policy.get('mfa_required', False),
            'approval_required': policy.get('approval_required', False),
            'approvers': policy.get('approvers', []),
            'created_at': datetime.utcnow().isoformat()
        }
    
    def check_access_permission(self, user_id: str, action: str, 
                              config_key: str, environment: str,
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if user has permission for configuration action"""
        
        context = context or {}
        user_roles = context.get('user_roles', [])
        user_ip = context.get('user_ip')
        current_time = datetime.utcnow()
        
        access_result = {
            'allowed': False,
            'policy_matched': None,
            'restrictions': [],
            'approval_required': False,
            'mfa_required': False
        }
        
        # Check each access policy
        for policy_name, policy in self.access_policies.items():
            if self._policy_matches(policy, user_id, user_roles, action, 
                                  config_key, environment):
                
                # Check time restrictions
                if not self._check_time_restrictions(policy, current_time):
                    access_result['restrictions'].append('time_restriction')
                    continue
                
                # Check IP restrictions
                if not self._check_ip_restrictions(policy, user_ip):
                    access_result['restrictions'].append('ip_restriction')
                    continue
                
                # Policy matches and restrictions pass
                access_result['allowed'] = True
                access_result['policy_matched'] = policy_name
                access_result['mfa_required'] = policy.get('mfa_required', False)
                access_result['approval_required'] = policy.get('approval_required', False)
                break
        
        # Log access attempt
        self._log_access_attempt(user_id, action, config_key, environment, access_result)
        
        return access_result
    
    def _policy_matches(self, policy: Dict[str, Any], user_id: str, 
                       user_roles: List[str], action: str,
                       config_key: str, environment: str) -> bool:
        """Check if policy matches the access request"""
        
        # Check user/role match
        user_match = (
            user_id in policy['users'] or
            any(role in policy['roles'] for role in user_roles)
        )
        
        if not user_match:
            return False
        
        # Check permission match
        if action not in policy['permissions'] and '*' not in policy['permissions']:
            return False
        
        # Check environment match
        if environment not in policy['environments'] and '*' not in policy['environments']:
            return False
        
        # Check configuration pattern match
        if policy['config_patterns']:
            pattern_match = any(
                self._matches_pattern(config_key, pattern)
                for pattern in policy['config_patterns']
            )
            if not pattern_match:
                return False
        
        return True
    
    def _check_time_restrictions(self, policy: Dict[str, Any], 
                               current_time: datetime) -> bool:
        """Check time-based access restrictions"""
        
        time_restrictions = policy.get('time_restrictions', {})
        
        if not time_restrictions:
            return True
        
        # Check allowed hours
        allowed_hours = time_restrictions.get('allowed_hours')
        if allowed_hours:
            current_hour = current_time.hour
            if current_hour not in allowed_hours:
                return False
        
        # Check allowed days of week (0 = Monday)
        allowed_days = time_restrictions.get('allowed_days')
        if allowed_days:
            current_day = current_time.weekday()
            if current_day not in allowed_days:
                return False
        
        # Check blackout periods
        blackout_periods = time_restrictions.get('blackout_periods', [])
        for period in blackout_periods:
            start_time = datetime.fromisoformat(period['start'])
            end_time = datetime.fromisoformat(period['end'])
            
            if start_time <= current_time <= end_time:
                return False
        
        return True
    
    def _check_ip_restrictions(self, policy: Dict[str, Any], user_ip: str) -> bool:
        """Check IP-based access restrictions"""
        
        ip_restrictions = policy.get('ip_restrictions', [])
        
        if not ip_restrictions:
            return True
        
        if not user_ip:
            return False
        
        # Check if IP is in allowed list (supports CIDR notation)
        import ipaddress
        
        try:
            user_ip_obj = ipaddress.ip_address(user_ip)
            
            for allowed_ip in ip_restrictions:
                try:
                    if '/' in allowed_ip:
                        # CIDR notation
                        network = ipaddress.ip_network(allowed_ip, strict=False)
                        if user_ip_obj in network:
                            return True
                    else:
                        # Individual IP
                        if str(user_ip_obj) == allowed_ip:
                            return True
                except ValueError:
                    continue
        except ValueError:
            return False
        
        return False
    
    def _matches_pattern(self, config_key: str, pattern: str) -> bool:
        """Check if configuration key matches pattern"""
        import fnmatch
        return fnmatch.fnmatch(config_key, pattern)
    
    def _log_access_attempt(self, user_id: str, action: str, config_key: str,
                          environment: str, access_result: Dict[str, Any]):
        """Log configuration access attempt"""
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'config_key': config_key,
            'environment': environment,
            'allowed': access_result['allowed'],
            'policy_matched': access_result['policy_matched'],
            'restrictions': access_result['restrictions'],
            'audit_id': secrets.token_hex(16)
        }
        
        self.audit_trail.append(log_entry)
    
    def get_audit_trail(self, filters: Dict[str, Any] = None, 
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Get filtered audit trail"""
        
        filtered_logs = self.audit_trail
        
        if filters:
            # Apply filters
            if 'user_id' in filters:
                filtered_logs = [log for log in filtered_logs if log['user_id'] == filters['user_id']]
            
            if 'action' in filters:
                filtered_logs = [log for log in filtered_logs if log['action'] == filters['action']]
            
            if 'environment' in filters:
                filtered_logs = [log for log in filtered_logs if log['environment'] == filters['environment']]
            
            if 'allowed' in filters:
                filtered_logs = [log for log in filtered_logs if log['allowed'] == filters['allowed']]
            
            if 'start_time' in filters:
                start_time = datetime.fromisoformat(filters['start_time'])
                filtered_logs = [
                    log for log in filtered_logs 
                    if datetime.fromisoformat(log['timestamp']) >= start_time
                ]
            
            if 'end_time' in filters:
                end_time = datetime.fromisoformat(filters['end_time'])
                filtered_logs = [
                    log for log in filtered_logs 
                    if datetime.fromisoformat(log['timestamp']) <= end_time
                ]
        
        # Sort by timestamp (newest first) and limit
        filtered_logs.sort(key=lambda x: x['timestamp'], reverse=True)
        return filtered_logs[:limit]
    
    def create_configuration_signature(self, config_data: Dict[str, Any], 
                                     secret_key: str) -> str:
        """Create cryptographic signature for configuration integrity"""
        
        # Normalize configuration data
        normalized_data = json.dumps(config_data, sort_keys=True, separators=(',', ':'))
        
        # Create HMAC signature
        signature = hmac.new(
            secret_key.encode(),
            normalized_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_configuration_signature(self, config_data: Dict[str, Any],
                                     signature: str, secret_key: str) -> bool:
        """Verify configuration integrity using signature"""
        
        expected_signature = self.create_configuration_signature(config_data, secret_key)
        return hmac.compare_digest(signature, expected_signature)

# Example security policies
def create_example_security_policies(security_manager: ConfigurationSecurity):
    """Create example security policies"""
    
    # Production environment policy - strict access
    security_manager.create_access_policy("production_admin", {
        "users": ["admin1", "admin2"],
        "roles": ["platform_admin"],
        "permissions": ["read", "write", "delete"],
        "environments": ["production"],
        "config_patterns": ["*"],
        "time_restrictions": {
            "allowed_hours": list(range(9, 18)),  # 9 AM to 6 PM
            "allowed_days": [0, 1, 2, 3, 4],  # Monday to Friday
            "blackout_periods": [
                {
                    "start": "2025-12-24T00:00:00",
                    "end": "2025-12-26T23:59:59",
                    "reason": "Holiday freeze"
                }
            ]
        },
        "ip_restrictions": ["10.0.0.0/8", "192.168.1.0/24"],
        "mfa_required": True,
        "approval_required": True,
        "approvers": ["senior_admin1", "senior_admin2"]
    })
    
    # Developer policy - limited access
    security_manager.create_access_policy("developer_access", {
        "users": [],
        "roles": ["developer", "devops"],
        "permissions": ["read", "write"],
        "environments": ["development", "staging"],
        "config_patterns": ["services.*", "features.*"],
        "time_restrictions": {},
        "ip_restrictions": [],
        "mfa_required": False,
        "approval_required": False
    })
    
    # Read-only policy for monitoring
    security_manager.create_access_policy("monitoring_readonly", {
        "users": [],
        "roles": ["monitoring", "sre"],
        "permissions": ["read"],
        "environments": ["*"],
        "config_patterns": ["monitoring.*", "alerts.*", "metrics.*"],
        "time_restrictions": {},
        "ip_restrictions": [],
        "mfa_required": False,
        "approval_required": False
    })
    
    # Emergency access policy
    security_manager.create_access_policy("emergency_access", {
        "users": ["emergency_user"],
        "roles": ["incident_commander"],
        "permissions": ["read", "write"],
        "environments": ["*"],
        "config_patterns": ["*"],
        "time_restrictions": {},
        "ip_restrictions": [],
        "mfa_required": True,
        "approval_required": False  # Emergency access bypasses approval
    })
```

---

## Monitoring and Auditing

### Configuration Monitoring Dashboard

```python
import asyncio
import json
from typing import Dict, Any, List, Optional
from