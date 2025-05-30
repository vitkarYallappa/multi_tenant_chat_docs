# Security Implementation
## Multi-Tenant AI Chatbot Platform

**Document:** 04-Security-Implementation.md  
**Version:** 2.0  
**Last Updated:** May 30, 2025

---

## Table of Contents

1. [Security Architecture Overview](#security-architecture-overview)
2. [Authentication Implementation](#authentication-implementation)
3. [Authorization and RBAC](#authorization-and-rbac)
4. [Data Encryption](#data-encryption)
5. [API Security](#api-security)
6. [Tenant Isolation](#tenant-isolation)
7. [Compliance Implementation](#compliance-implementation)
8. [Security Monitoring](#security-monitoring)
9. [Incident Response](#incident-response)

---

## Security Architecture Overview

### Security Principles

1. **Zero Trust Architecture:** Never trust, always verify
2. **Defense in Depth:** Multiple layers of security controls
3. **Principle of Least Privilege:** Minimum necessary permissions
4. **Data Privacy by Design:** Privacy considerations in all design decisions
5. **Secure by Default:** Security configurations enabled by default
6. **Continuous Monitoring:** Real-time security event monitoring

### Security Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                        SECURITY LAYERS                         │
└─────────────────────────────────────────────────────────────────┘

External Protection:
├── WAF (Web Application Firewall)
├── DDoS Protection (CloudFlare)
├── Rate Limiting (API Gateway)
└── Geographic Filtering

Network Security:
├── VPC with Private Subnets
├── Security Groups & NACLs
├── Service Mesh (mTLS)
└── Network Segmentation

Application Security:
├── Authentication (JWT + MFA)
├── Authorization (RBAC)
├── Input Validation & Sanitization
├── Session Management
└── API Security (Rate Limiting, CORS)

Data Security:
├── Encryption at Rest (AES-256)
├── Encryption in Transit (TLS 1.3)
├── Key Management (HSM/KMS)
├── PII Detection & Masking
└── Secure Deletion

Infrastructure Security:
├── Container Security Scanning
├── Image Vulnerability Management
├── Secrets Management
├── Runtime Security Monitoring
└── Compliance Monitoring
```

---

## Authentication Implementation

### JWT-Based Authentication

#### Token Structure and Management

```python
import jwt
import secrets
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import redis
import json

class JWTAuthenticationService:
    def __init__(self, private_key_path: str, public_key_path: str, redis_client):
        # Load RSA keys for JWT signing
        with open(private_key_path, 'rb') as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)
        
        with open(public_key_path, 'rb') as f:
            self.public_key = serialization.load_pem_public_key(f.read())
        
        self.redis = redis_client
        self.token_blacklist_prefix = "blacklist:"
        self.session_prefix = "session:"
        
    def generate_access_token(self, user_data: dict) -> dict:
        """Generate JWT access token with comprehensive security claims"""
        now = datetime.utcnow()
        expires_in = timedelta(hours=1)  # Short-lived access tokens
        
        # Generate unique token ID for tracking and revocation
        jti = secrets.token_urlsafe(32)
        
        # Create comprehensive payload
        payload = {
            # Standard JWT claims
            "iss": "chatbot-platform",
            "sub": user_data["user_id"],
            "aud": ["chatbot-api", "admin-dashboard"],
            "iat": int(now.timestamp()),
            "exp": int((now + expires_in).timestamp()),
            "nbf": int(now.timestamp()),
            "jti": jti,
            
            # Custom security claims
            "tenant_id": user_data["tenant_id"],
            "user_role": user_data["role"],
            "permissions": user_data.get("permissions", []),
            "scopes": user_data.get("scopes", []),
            
            # Security context
            "session_id": user_data.get("session_id"),
            "device_id": user_data.get("device_id"),
            "ip_address": user_data.get("ip_address"),
            "user_agent_hash": self._hash_user_agent(user_data.get("user_agent", "")),
            
            # Rate limiting and quotas
            "rate_limit_tier": user_data.get("rate_limit_tier", "standard"),
            "daily_quota": user_data.get("daily_quota"),
            
            # Security flags
            "mfa_verified": user_data.get("mfa_verified", False),
            "password_change_required": user_data.get("password_change_required", False),
            "terms_accepted": user_data.get("terms_accepted", False),
            
            # Feature flags
            "feature_flags": user_data.get("feature_flags", []),
            
            # Compliance flags
            "data_residency": user_data.get("data_residency", "us"),
            "compliance_level": user_data.get("compliance_level", "standard")
        }
        
        # Sign token with RS256
        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        
        # Store session metadata in Redis
        session_data = {
            "user_id": user_data["user_id"],
            "tenant_id": user_data["tenant_id"],
            "created_at": now.isoformat(),
            "expires_at": (now + expires_in).isoformat(),
            "ip_address": user_data.get("ip_address"),
            "user_agent": user_data.get("user_agent"),
            "active": True,
            "mfa_verified": user_data.get("mfa_verified", False)
        }
        
        session_key = f"{self.session_prefix}{jti}"
        self.redis.setex(session_key, int(expires_in.total_seconds()), json.dumps(session_data))
        
        return {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": int(expires_in.total_seconds()),
            "expires_at": (now + expires_in).isoformat(),
            "scope": " ".join(payload.get("scopes", [])),
            "jti": jti
        }
    
    def generate_refresh_token(self, user_data: dict) -> dict:
        """Generate long-lived refresh token"""
        now = datetime.utcnow()
        expires_in = timedelta(days=30)  # Long-lived refresh tokens
        
        jti = secrets.token_urlsafe(32)
        
        payload = {
            "iss": "chatbot-platform",
            "sub": user_data["user_id"],
            "aud": "refresh-service",
            "iat": int(now.timestamp()),
            "exp": int((now + expires_in).timestamp()),
            "jti": jti,
            "token_type": "refresh",
            "tenant_id": user_data["tenant_id"],
            "device_id": user_data.get("device_id"),
            "ip_address": user_data.get("ip_address")
        }
        
        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        
        # Store refresh token metadata
        refresh_data = {
            "user_id": user_data["user_id"],
            "tenant_id": user_data["tenant_id"],
            "created_at": now.isoformat(),
            "expires_at": (now + expires_in).isoformat(),
            "device_id": user_data.get("device_id"),
            "active": True
        }
        
        refresh_key = f"refresh:{jti}"
        self.redis.setex(refresh_key, int(expires_in.total_seconds()), json.dumps(refresh_data))
        
        return {
            "refresh_token": token,
            "expires_in": int(expires_in.total_seconds()),
            "expires_at": (now + expires_in).isoformat(),
            "jti": jti
        }
    
    def verify_token(self, token: str) -> dict:
        """Comprehensive token verification with security checks"""
        try:
            # Decode and verify token signature
            payload = jwt.decode(token, self.public_key, algorithms=["RS256"])
            
            # Check if token is blacklisted
            blacklist_key = f"{self.blacklist_prefix}{payload['jti']}"
            if self.redis.exists(blacklist_key):
                raise jwt.InvalidTokenError("Token has been revoked")
            
            # Verify session is still active (for access tokens)
            if payload.get("token_type") != "refresh":
                session_key = f"{self.session_prefix}{payload['jti']}"
                session_data = self.redis.get(session_key)
                if not session_data:
                    raise jwt.InvalidTokenError("Session expired or invalid")
                
                session_info = json.loads(session_data)
                if not session_info.get("active", False):
                    raise jwt.InvalidTokenError("Session is not active")
                
                # Add session info to payload for use by application
                payload["session_info"] = session_info
            
            # Additional security validations
            self._validate_token_security(payload)
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise e
        except Exception as e:
            raise jwt.InvalidTokenError(f"Token validation failed: {str(e)}")
    
    def revoke_token(self, jti: str, expires_at: datetime):
        """Add token to blacklist and deactivate session"""
        # Add to blacklist
        blacklist_key = f"{self.blacklist_prefix}{jti}"
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        if ttl > 0:
            self.redis.setex(blacklist_key, ttl, "revoked")
        
        # Deactivate session
        session_key = f"{self.session_prefix}{jti}"
        session_data = self.redis.get(session_key)
        if session_data:
            session_info = json.loads(session_data)
            session_info["active"] = False
            session_info["revoked_at"] = datetime.utcnow().isoformat()
            self.redis.set(session_key, json.dumps(session_info))
    
    def _validate_token_security(self, payload: dict):
        """Additional security validations for tokens"""
        # Check for suspicious activity patterns
        if payload.get("mfa_verified") and payload.get("password_change_required"):
            # This combination might indicate account compromise
            pass
        
        # Validate audience
        expected_audiences = ["chatbot-api", "admin-dashboard"]
        if not any(aud in payload.get("aud", []) for aud in expected_audiences):
            raise jwt.InvalidTokenError("Invalid token audience")
    
    def _hash_user_agent(self, user_agent: str) -> str:
        """Hash user agent for privacy while maintaining security tracking"""
        import hashlib
        return hashlib.sha256(user_agent.encode()).hexdigest()[:16]
```

### Multi-Factor Authentication

#### TOTP Implementation

```python
import pyotp
import qrcode
import secrets
from io import BytesIO
import base64

class MFAService:
    def __init__(self, issuer_name: str = "Chatbot Platform"):
        self.issuer_name = issuer_name
    
    def generate_secret(self) -> str:
        """Generate a new TOTP secret for user"""
        return pyotp.random_base32()
    
    def generate_qr_code(self, secret: str, user_email: str, tenant_name: str) -> str:
        """Generate QR code for TOTP setup"""
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_email,
            issuer_name=f"{self.issuer_name} ({tenant_name})"
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 for frontend display
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def verify_totp(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token with time window tolerance"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    
    def generate_backup_codes(self, count: int = 10) -> list:
        """Generate backup codes for account recovery"""
        codes = []
        for _ in range(count):
            # Generate 8-character backup codes
            code = secrets.token_hex(4).upper()
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes
    
    def verify_backup_code(self, stored_codes: list, provided_code: str) -> tuple:
        """Verify backup code and return updated codes list"""
        # Remove any formatting from provided code
        clean_code = provided_code.replace("-", "").replace(" ", "").upper()
        
        for i, stored_code in enumerate(stored_codes):
            clean_stored = stored_code.replace("-", "").replace(" ", "").upper()
            if clean_code == clean_stored:
                # Remove used backup code
                updated_codes = stored_codes.copy()
                updated_codes.pop(i)
                return True, updated_codes
        
        return False, stored_codes
```

### Session Management

```python
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict

class SessionManager:
    def __init__(self, redis_client, session_timeout: int = 3600):
        self.redis = redis_client
        self.session_timeout = session_timeout
        self.session_prefix = "user_session:"
        self.device_prefix = "device_session:"
    
    def create_session(self, user_id: str, tenant_id: str, device_info: dict) -> str:
        """Create a new user session with device tracking"""
        session_id = secrets.token_urlsafe(32)
        
        # Create device fingerprint
        device_fingerprint = self._create_device_fingerprint(device_info)
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "device_fingerprint": device_fingerprint,
            "ip_address": device_info.get("ip_address"),
            "user_agent": device_info.get("user_agent"),
            "active": True,
            "mfa_verified": False,
            "login_method": device_info.get("login_method", "password")
        }
        
        # Store session data
        session_key = f"{self.session_prefix}{session_id}"
        self.redis.setex(session_key, self.session_timeout, json.dumps(session_data))
        
        # Track device sessions for security monitoring
        device_key = f"{self.device_prefix}{user_id}:{device_fingerprint}"
        device_sessions = self.redis.get(device_key)
        if device_sessions:
            sessions = json.loads(device_sessions)
        else:
            sessions = []
        
        sessions.append({
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "ip_address": device_info.get("ip_address")
        })
        
        # Keep only last 5 sessions per device
        sessions = sessions[-5:]
        self.redis.setex(device_key, 86400 * 30, json.dumps(sessions))  # 30 days
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str = None) -> Optional[Dict]:
        """Validate session and check for suspicious activity"""
        session_key = f"{self.session_prefix}{session_id}"
        session_data = self.redis.get(session_key)
        
        if not session_data:
            return None
        
        session = json.loads(session_data)
        
        # Check if session is active
        if not session.get("active", False):
            return None
        
        # Check for IP address changes (potential session hijacking)
        if ip_address and session.get("ip_address") != ip_address:
            # Log security event
            self._log_security_event("ip_change", session, {"new_ip": ip_address})
            
            # Optionally invalidate session based on security policy
            # For now, we'll allow but log the event
        
        # Update last activity
        session["last_activity"] = datetime.utcnow().isoformat()
        self.redis.setex(session_key, self.session_timeout, json.dumps(session))
        
        return session
    
    def invalidate_session(self, session_id: str):
        """Invalidate a specific session"""
        session_key = f"{self.session_prefix}{session_id}"
        session_data = self.redis.get(session_key)
        
        if session_data:
            session = json.loads(session_data)
            session["active"] = False
            session["invalidated_at"] = datetime.utcnow().isoformat()
            self.redis.setex(session_key, 300, json.dumps(session))  # Keep for 5 minutes for audit
    
    def invalidate_all_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user (useful for password changes)"""
        # This would require scanning Redis keys, which is expensive
        # In production, maintain a user->sessions mapping
        pattern = f"{self.session_prefix}*"
        for key in self.redis.scan_iter(match=pattern):
            session_data = self.redis.get(key)
            if session_data:
                session = json.loads(session_data)
                if session.get("user_id") == user_id:
                    self.invalidate_session(session["session_id"])
    
    def _create_device_fingerprint(self, device_info: dict) -> str:
        """Create a device fingerprint for tracking"""
        fingerprint_data = {
            "user_agent": device_info.get("user_agent", ""),
            "screen_resolution": device_info.get("screen_resolution", ""),
            "timezone": device_info.get("timezone", ""),
            "language": device_info.get("language", ""),
            "platform": device_info.get("platform", "")
        }
        
        fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]
    
    def _log_security_event(self, event_type: str, session: dict, additional_data: dict = None):
        """Log security events for monitoring"""
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session.get("session_id"),
            "user_id": session.get("user_id"),
            "tenant_id": session.get("tenant_id"),
            "ip_address": session.get("ip_address"),
            "additional_data": additional_data or {}
        }
        
        # Store in Redis for real-time monitoring
        event_key = f"security_event:{datetime.utcnow().strftime('%Y%m%d')}:{secrets.token_hex(8)}"
        self.redis.setex(event_key, 86400, json.dumps(event))  # Keep for 24 hours
```

---

## Authorization and RBAC

### Role-Based Access Control System

```python
from enum import Enum
from typing import List, Dict, Set
import fnmatch

class Permission(Enum):
    # Conversation permissions
    CONVERSATIONS_READ = "conversations:read"
    CONVERSATIONS_WRITE = "conversations:write"
    CONVERSATIONS_DELETE = "conversations:delete"
    CONVERSATIONS_EXPORT = "conversations:export"
    
    # Configuration permissions
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"
    CONFIG_FLOWS = "config:flows"
    CONFIG_INTEGRATIONS = "config:integrations"
    CONFIG_MODELS = "config:models"
    
    # User management permissions
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"
    USERS_INVITE = "users:invite"
    USERS_ROLES = "users:roles"
    
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"
    ANALYTICS_ADVANCED = "analytics:advanced"
    
    # Admin permissions
    ADMIN_BILLING = "admin:billing"
    ADMIN_AUDIT = "admin:audit"
    ADMIN_SECURITY = "admin:security"
    ADMIN_COMPLIANCE = "admin:compliance"
    
    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    API_KEYS = "api:keys"
    
    # Integration permissions
    INTEGRATIONS_READ = "integrations:read"
    INTEGRATIONS_WRITE = "integrations:write"
    INTEGRATIONS_EXECUTE = "integrations:execute"
    INTEGRATIONS_MARKETPLACE = "integrations:marketplace"

class Role(Enum):
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    MANAGER = "manager"
    MEMBER = "member"
    VIEWER = "viewer"
    API_USER = "api_user"

class RBACService:
    def __init__(self):
        # Define role hierarchies and permissions
        self.role_permissions = {
            Role.OWNER: ["*"],  # All permissions
            
            Role.ADMIN: [
                "conversations:*", "config:*", "users:*", 
                "analytics:*", "api:*", "integrations:*",
                "admin:billing", "admin:audit"
            ],
            
            Role.DEVELOPER: [
                "conversations:read", "config:read", "config:flows",
                "config:integrations", "config:models", "analytics:read",
                "api:*", "integrations:*"
            ],
            
            Role.MANAGER: [
                "conversations:read", "conversations:export",
                "config:read", "users:read", "users:invite",
                "analytics:*"
            ],
            
            Role.MEMBER: [
                "conversations:read", "conversations:write",
                "config:read", "analytics:read"
            ],
            
            Role.VIEWER: [
                "conversations:read", "config:read", "analytics:read"
            ],
            
            Role.API_USER: [
                "api:read", "api:write", "conversations:read", "conversations:write"
            ]
        }
        
        # Define resource-level permissions
        self.resource_permissions = {}
    
    def check_permission(self, user_role: str, user_permissions: List[str], 
                        required_permission: str, resource_id: str = None) -> bool:
        """Check if user has required permission"""
        
        # Check explicit permissions first
        if self._has_explicit_permission(user_permissions, required_permission):
            return True
        
        # Check role-based permissions
        try:
            role = Role(user_role)
            role_perms = self.role_permissions.get(role, [])
            if self._has_explicit_permission(role_perms, required_permission):
                return True
        except ValueError:
            # Invalid role
            pass
        
        # Check resource-level permissions if resource_id provided
        if resource_id:
            return self._check_resource_permission(user_permissions, required_permission, resource_id)
        
        return False
    
    def _has_explicit_permission(self, permissions: List[str], required: str) -> bool:
        """Check if permissions list contains required permission"""
        # Check for wildcard permissions
        if "*" in permissions:
            return True
        
        # Check exact match
        if required in permissions:
            return True
        
        # Check for category wildcards (e.g., "conversations:*")
        permission_category = required.split(":")[0]
        category_wildcard = f"{permission_category}:*"
        if category_wildcard in permissions:
            return True
        
        # Check for pattern matching
        for perm in permissions:
            if fnmatch.fnmatch(required, perm):
                return True
        
        return False
    
    def _check_resource_permission(self, user_permissions: List[str], 
                                 required_permission: str, resource_id: str) -> bool:
        """Check resource-level permissions"""
        # This would implement resource-specific permission checking
        # For example, checking if user can access specific conversations
        resource_perm = f"{required_permission}:{resource_id}"
        return resource_perm in user_permissions
    
    def get_user_permissions(self, role: str, custom_permissions: List[str] = None) -> Set[str]:
        """Get all permissions for a user including role and custom permissions"""
        permissions = set()
        
        # Add role-based permissions
        try:
            user_role = Role(role)
            role_perms = self.role_permissions.get(user_role, [])
            permissions.update(role_perms)
        except ValueError:
            pass
        
        # Add custom permissions
        if custom_permissions:
            permissions.update(custom_permissions)
        
        return permissions
    
    def can_assign_role(self, assigner_role: str, target_role: str) -> bool:
        """Check if a user can assign a specific role to another user"""
        role_hierarchy = {
            Role.OWNER: 6,
            Role.ADMIN: 5,
            Role.DEVELOPER: 4,
            Role.MANAGER: 3,
            Role.MEMBER: 2,
            Role.VIEWER: 1,
            Role.API_USER: 1
        }
        
        try:
            assigner_level = role_hierarchy.get(Role(assigner_role), 0)
            target_level = role_hierarchy.get(Role(target_role), 0)
            
            # Can only assign roles at same level or below
            return assigner_level >= target_level
        except ValueError:
            return False

class PermissionDecorator:
    """Decorator for enforcing permissions on API endpoints"""
    
    def __init__(self, required_permission: str, resource_id_param: str = None):
        self.required_permission = required_permission
        self.resource_id_param = resource_id_param
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Get current user from context (would be injected by middleware)
            current_user = get_current_user()  # This would be implemented
            
            if not current_user:
                raise UnauthorizedError("Authentication required")
            
            # Extract resource ID if specified
            resource_id = None
            if self.resource_id_param:
                resource_id = kwargs.get(self.resource_id_param)
            
            # Check permission
            rbac = RBACService()
            if not rbac.check_permission(
                current_user.role,
                current_user.permissions,
                self.required_permission,
                resource_id
            ):
                raise ForbiddenError(f"Permission denied: {self.required_permission}")
            
            return func(*args, **kwargs)
        
        return wrapper

# Usage example
@PermissionDecorator("conversations:read", resource_id_param="conversation_id")
def get_conversation(conversation_id: str):
    # API endpoint implementation
    pass
```

---

## Data Encryption

### Encryption Service Implementation

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64
import os
import hashlib
from typing import Dict, Any

class EncryptionService:
    def __init__(self, master_key: bytes, key_rotation_days: int = 90):
        self.master_key = master_key
        self.key_rotation_days = key_rotation_days
        self.fernet = Fernet(master_key)
        
        # Initialize field-level encryption keys
        self.field_encryption_keys = {}
        self._initialize_field_keys()
    
    def _initialize_field_keys(self):
        """Initialize different encryption keys for different data types"""
        sensitive_fields = [
            "email", "phone", "ssn", "credit_card", "api_keys", 
            "passwords", "tokens", "personal_data"
        ]
        
        for field_type in sensitive_fields:
            # Derive field-specific key from master key
            field_key = self._derive_field_key(field_type)
            self.field_encryption_keys[field_type] = Fernet(field_key)
    
    def _derive_field_key(self, field_type: str) -> bytes:
        """Derive field-specific encryption key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=field_type.encode(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return key
    
    def encrypt_field(self, data: str, field_type: str = "default") -> str:
        """Encrypt a single field with appropriate encryption key"""
        if not data:
            return data
        
        # Use field-specific encryption if available
        if field_type in self.field_encryption_keys:
            encrypted_bytes = self.field_encryption_keys[field_type].encrypt(data.encode())
        else:
            encrypted_bytes = self.fernet.encrypt(data.encode())
        
        return base64.b64encode(encrypted_bytes).decode()
    
    def decrypt_field(self, encrypted_data: str, field_type: str = "default") -> str:
        """Decrypt a single field"""
        if not encrypted_data:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            
            # Use field-specific decryption if available
            if field_type in self.field_encryption_keys:
                decrypted_bytes = self.field_encryption_keys[field_type].decrypt(encrypted_bytes)
            else:
                decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            
            return decrypted_bytes.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
    
    def encrypt_document(self, document: Dict[str, Any], encryption_config: Dict[str, str]) -> Dict[str, Any]:
        """Encrypt specified fields in a document according to configuration"""
        encrypted_doc = document.copy()
        
        for field_path, field_type in encryption_config.items():
            try:
                # Navigate to nested field
                current = encrypted_doc
                path_parts = field_path.split(".")
                
                for part in path_parts[:-1]:
                    if part not in current:
                        break
                    current = current[part]
                else:
                    field_name = path_parts[-1]
                    if field_name in current and current[field_name]:
                        # Encrypt the field
                        encrypted_value = self.encrypt_field(str(current[field_name]), field_type)
                        current[field_name] = encrypted_value
                        
                        # Add encryption metadata
                        metadata_field = f"{field_name}_encryption_meta"
                        current[metadata_field] = {
                            "encrypted": True,
                            "field_type": field_type,
                            "encrypted_at": datetime.utcnow().isoformat(),
                            "key_version": "v1"  # For key rotation tracking
                        }
            except Exception as e:
                # Log encryption error but don't fail entire operation
                logger.error(f"Failed to encrypt field {field_path}: {str(e)}")
        
        return encrypted_doc
    
    def decrypt_document(self, encrypted_doc: Dict[str, Any], encryption_config: Dict[str, str]) -> Dict[str, Any]:
        """Decrypt specified fields in a document"""
        decrypted_doc = encrypted_doc.copy()
        
        for field_path, field_type in encryption_config.items():
            try:
                # Navigate to nested field
                current = decrypted_doc
                path_parts = field_path.split(".")
                
                for part in path_parts[:-1]:
                    if part not in current:
                        break
                    current = current[part]
                else:
                    field_name = path_parts[-1]
                    metadata_field = f"{field_name}_encryption_meta"
                    
                    # Check if field is encrypted
                    if (field_name in current and 
                        metadata_field in current and 
                        current[metadata_field].get("encrypted", False)):
                        
                        # Decrypt the field
                        decrypted_value = self.decrypt_field(current[field_name], field_type)
                        current[field_name] = decrypted_value
                        
                        # Remove encryption metadata from output
                        del current[metadata_field]
            except Exception as e:
                logger.error(f"Failed to decrypt field {field_path}: {str(e)}")
        
        return decrypted_doc

class PIIDetectionService:
    """Service for detecting and masking personally identifiable information"""
    
    def __init__(self):
        # PII detection patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'(\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "url": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "address": r'\d+\s+[A-Za-z0-9\s,]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)',
            "date_of_birth": r'\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])[\/\-](\d{2}|\d{4})\b'
        }
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text and return found items by type"""
        import re
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def mask_pii(self, text: str, mask_char: str = "*") -> tuple:
        """Mask PII in text and return masked text with metadata"""
        import re
        masked_text = text
        pii_found = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = []
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
                
                # Apply appropriate masking strategy
                masked_value = self._mask_value(match.group(), pii_type, mask_char)
                masked_text = masked_text.replace(match.group(), masked_value)
            
            if matches:
                pii_found[pii_type] = matches
        
        return masked_text, pii_found
    
    def _mask_value(self, value: str, pii_type: str, mask_char: str) -> str:
        """Apply appropriate masking strategy based on PII type"""
        if pii_type == "email":
            # Keep domain for emails: user@domain.com -> ****@domain.com
            parts = value.split("@")
            if len(parts) == 2:
                return f"{mask_char * len(parts[0])}@{parts[1]}"
        elif pii_type == "credit_card":
            # Keep last 4 digits: 1234-5678-9012-3456 -> ****-****-****-3456
            clean_number = re.sub(r'[\s-]', '', value)
            if len(clean_number) >= 4:
                masked = mask_char * (len(clean_number) - 4) + clean_number[-4:]
                # Restore original formatting
                if '-' in value:
                    return f"{masked[:-4]}-{masked[-4:]}"
                return masked
        elif pii_type == "phone":
            # Keep last 4 digits: (555) 123-4567 -> (***) ***-4567
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                masked_digits = mask_char * (len(digits) - 4) + digits[-4:]
                # Try to preserve original formatting
                result = value
                for i, digit in enumerate(digits):
                    if i < len(digits) - 4:
                        result = result.replace(digit, mask_char, 1)
                return result
        
        # Default masking: replace all characters
        return mask_char * len(value)
```

---

## API Security

### Rate Limiting Implementation

```python
import time
import json
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

class RateLimitType(Enum):
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"

@dataclass
class RateLimitConfig:
    limit: int
    window_seconds: int
    rate_limit_type: RateLimitType
    burst_limit: Optional[int] = None
    
class RateLimitService:
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # Define rate limit tiers
        self.rate_limit_tiers = {
            "basic": {
                "api_calls": RateLimitConfig(100, 60, RateLimitType.PER_MINUTE),
                "model_requests": RateLimitConfig(50, 60, RateLimitType.PER_MINUTE),
                "daily_quota": RateLimitConfig(10000, 86400, RateLimitType.PER_DAY)
            },
            "standard": {
                "api_calls": RateLimitConfig(1000, 60, RateLimitType.PER_MINUTE),
                "model_requests": RateLimitConfig(500, 60, RateLimitType.PER_MINUTE),
                "daily_quota": RateLimitConfig(100000, 86400, RateLimitType.PER_DAY)
            },
            "premium": {
                "api_calls": RateLimitConfig(10000, 60, RateLimitType.PER_MINUTE),
                "model_requests": RateLimitConfig(5000, 60, RateLimitType.PER_MINUTE),
                "daily_quota": RateLimitConfig(1000000, 86400, RateLimitType.PER_DAY)
            },
            "enterprise": {
                "api_calls": RateLimitConfig(100000, 60, RateLimitType.PER_MINUTE),
                "model_requests": RateLimitConfig(50000, 60, RateLimitType.PER_MINUTE),
                "daily_quota": RateLimitConfig(10000000, 86400, RateLimitType.PER_DAY)
            }
        }
    
    def check_rate_limit(self, identifier: str, tier: str, operation: str) -> Dict:
        """Check if request is within rate limits"""
        config = self.rate_limit_tiers.get(tier, {}).get(operation)
        if not config:
            return {"allowed": True, "remaining": float('inf')}
        
        if config.rate_limit_type == RateLimitType.SLIDING_WINDOW:
            return self._check_sliding_window_rate_limit(identifier, operation, config)
        elif config.rate_limit_type == RateLimitType.TOKEN_BUCKET:
            return self._check_token_bucket_rate_limit(identifier, operation, config)
        else:
            return self._check_fixed_window_rate_limit(identifier, operation, config)
    
    def _check_sliding_window_rate_limit(self, identifier: str, operation: str, 
                                       config: RateLimitConfig) -> Dict:
        """Implement sliding window rate limiting"""
        now = time.time()
        window_start = now - config.window_seconds
        
        key = f"rate_limit:sliding:{identifier}:{operation}"
        
        # Use Redis sorted set to track requests with timestamps
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiration
        pipe.expire(key, config.window_seconds + 1)
        
        results = pipe.execute()
        current_count = results[1]
        
        if current_count >= config.limit:
            # Remove the request we just added since it's over limit
            self.redis.zrem(key, str(now))
            return {
                "allowed": False,
                "remaining": 0,
                "reset_time": window_start + config.window_seconds,
                "retry_after": config.window_seconds
            }
        
        return {
            "allowed": True,
            "remaining": config.limit - current_count - 1,
            "reset_time": now + config.window_seconds
        }
    
    def _check_token_bucket_rate_limit(self, identifier: str, operation: str,
                                     config: RateLimitConfig) -> Dict:
        """Implement token bucket rate limiting"""
        key = f"rate_limit:bucket:{identifier}:{operation}"
        now = time.time()
        
        # Get current bucket state
        bucket_data = self.redis.get(key)
        if bucket_data:
            bucket = json.loads(bucket_data)
        else:
            bucket = {
                "tokens": config.limit,
                "last_refill": now,
                "capacity": config.limit
            }
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - bucket["last_refill"]
        tokens_to_add = time_elapsed * (config.limit / config.window_seconds)
        
        # Update bucket
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        # Check if request can be allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            allowed = True
            remaining = int(bucket["tokens"])
        else:
            allowed = False
            remaining = 0
        
        # Store updated bucket state
        self.redis.setex(key, config.window_seconds * 2, json.dumps(bucket))
        
        return {
            "allowed": allowed,
            "remaining": remaining,
            "reset_time": now + (1 - bucket["tokens"]) * (config.window_seconds / config.limit)
        }
    
    def _check_fixed_window_rate_limit(self, identifier: str, operation: str,
                                     config: RateLimitConfig) -> Dict:
        """Implement fixed window rate limiting"""
        now = time.time()
        window = int(now // config.window_seconds)
        
        key = f"rate_limit:fixed:{identifier}:{operation}:{window}"
        
        # Get current count and increment
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, config.window_seconds)
        results = pipe.execute()
        
        current_count = results[0]
        
        if current_count > config.limit:
            return {
                "allowed": False,
                "remaining": 0,
                "reset_time": (window + 1) * config.window_seconds,
                "retry_after": ((window + 1) * config.window_seconds) - now
            }
        
        return {
            "allowed": True,
            "remaining": config.limit - current_count,
            "reset_time": (window + 1) * config.window_seconds
        }

# Middleware for FastAPI
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

class RateLimitMiddleware:
    def __init__(self, rate_limit_service: RateLimitService):
        self.rate_limit_service = rate_limit_service
    
    async def __call__(self, request: Request, call_next):
        # Extract rate limit identifier (API key, user ID, IP address)
        identifier = self._get_rate_limit_identifier(request)
        tier = self._get_rate_limit_tier(request)
        operation = self._get_operation_type(request)
        
        # Check rate limit
        result = self.rate_limit_service.check_rate_limit(identifier, tier, operation)
        
        if not result["allowed"]:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Rate limit exceeded",
                        "details": {
                            "retry_after": result.get("retry_after"),
                            "reset_time": result.get("reset_time")
                        }
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(result.get("limit", "")),
                    "X-RateLimit-Remaining": str(result.get("remaining", 0)),
                    "X-RateLimit-Reset": str(int(result.get("reset_time", 0))),
                    "Retry-After": str(int(result.get("retry_after", 60)))
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(result.get("limit", ""))
        response.headers["X-RateLimit-Remaining"] = str(result.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(int(result.get("reset_time", 0)))
        
        return response
    
    def _get_rate_limit_identifier(self, request: Request) -> str:
        """Extract identifier for rate limiting"""
        # Try API key first
        api_key = request.headers.get("Authorization", "").replace("ApiKey ", "")
        if api_key:
            return f"api_key:{api_key}"
        
        # Try user ID from JWT
        user = getattr(request.state, "user", None)
        if user:
            return f"user:{user.get('user_id')}"
        
        # Fall back to IP address
        client_ip = request.client.host
        return f"ip:{client_ip}"
    
    def _get_rate_limit_tier(self, request: Request) -> str:
        """Determine rate limit tier for request"""
        user = getattr(request.state, "user", None)
        if user:
            return user.get("rate_limit_tier", "standard")
        return "basic"
    
    def _get_operation_type(self, request: Request) -> str:
        """Determine operation type for rate limiting"""
        if "/api/v2/model/" in request.url.path:
            return "model_requests"
        elif "/api/v2/" in request.url.path:
            return "api_calls"
        return "general"
```


**Document Maintainer:** Security Engineering Team  
**Review Schedule:** Monthly security reviews, quarterly penetration testing  
**Related Documents:** System Architecture, API Specifications, Compliance Documentation