# Phase 06: External Service Clients & Integration
**Duration**: Week 11-12 (Days 51-60)  
**Team Size**: 3-4 developers  
**Complexity**: High  

## Overview
Implement external service clients for integrating with Model Orchestrator, Adaptor Service, Security Hub, and Analytics Engine. This phase establishes the communication layer between MCP Engine and other services in the platform, including gRPC clients, HTTP clients, circuit breakers, and retry mechanisms.

## Step 15: Client Infrastructure & Base Classes (Days 51-53)

### Files to Create
```
src/
├── clients/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── http_client.py
│   │   ├── grpc_client.py
│   │   ├── circuit_breaker.py
│   │   └── retry_handler.py
│   ├── model_orchestrator_client.py
│   ├── adaptor_service_client.py
│   ├── security_hub_client.py
│   ├── analytics_client.py
│   └── cache_client.py
```

### `/src/clients/base/http_client.py`
**Purpose**: Base HTTP client with retry logic, circuit breaker, and monitoring
```python
import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import httpx
from urllib.parse import urljoin
import uuid

from src.utils.logger import get_logger
from src.utils.metrics import MetricsCollector
from src.exceptions.base import MCPBaseException
from .circuit_breaker import CircuitBreaker
from .retry_handler import RetryHandler

logger = get_logger(__name__)

class HTTPClientError(MCPBaseException):
    """HTTP client specific errors"""
    pass

class BaseHTTPClient:
    """Base HTTP client with advanced features"""
    
    def __init__(
        self,
        service_name: str,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_enabled: bool = True,
        default_headers: Optional[Dict[str, str]] = None
    ):
        self.service_name = service_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Setup logging with service context
        self.logger = get_logger(f"client.{service_name}")
        
        # Initialize HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._default_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': f'MCP-Engine/{service_name}-client',
            **(default_headers or {})
        }
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            service_name=service_name,
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=HTTPClientError
        ) if circuit_breaker_enabled else None
        
        # Initialize retry handler
        self.retry_handler = RetryHandler(
            max_retries=max_retries,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0
        )
        
        # Metrics tracking
        self._request_count = 0
        self._error_count = 0
    
    async def initialize(self):
        """Initialize the HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=100,
                    keepalive_expiry=30.0
                ),
                headers=self._default_headers,
                follow_redirects=True
            )
            
            self.logger.info(
                "HTTP client initialized",
                service=self.service_name,
                base_url=self.base_url
            )
    
    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
            
            self.logger.info("HTTP client closed", service=self.service_name)
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform GET request
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout override
            tenant_id: Tenant context for metrics
            
        Returns:
            Response data
        """
        return await self._request(
            method="GET",
            endpoint=endpoint,
            params=params,
            headers=headers,
            timeout=timeout,
            tenant_id=tenant_id
        )
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform POST request
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            headers: Additional headers
            timeout: Request timeout override
            tenant_id: Tenant context for metrics
            
        Returns:
            Response data
        """
        return await self._request(
            method="POST",
            endpoint=endpoint,
            data=data,
            json_data=json_data,
            headers=headers,
            timeout=timeout,
            tenant_id=tenant_id
        )
    
    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform PUT request
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            headers: Additional headers
            timeout: Request timeout override
            tenant_id: Tenant context for metrics
            
        Returns:
            Response data
        """
        return await self._request(
            method="PUT",
            endpoint=endpoint,
            data=data,
            json_data=json_data,
            headers=headers,
            timeout=timeout,
            tenant_id=tenant_id
        )
    
    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform DELETE request
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout override
            tenant_id: Tenant context for metrics
            
        Returns:
            Response data
        """
        return await self._request(
            method="DELETE",
            endpoint=endpoint,
            params=params,
            headers=headers,
            timeout=timeout,
            tenant_id=tenant_id
        )
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Internal method to perform HTTP requests with circuit breaker and retry
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            json_data: JSON data
            headers: Additional headers
            timeout: Request timeout override
            tenant_id: Tenant context for metrics
            
        Returns:
            Response data
        """
        if not self._client:
            await self.initialize()
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Build URL
        url = urljoin(f"{self.base_url}/", endpoint.lstrip('/'))
        
        # Merge headers
        request_headers = {**self._default_headers}
        if headers:
            request_headers.update(headers)
        
        # Add request ID for tracing
        request_headers['X-Request-ID'] = request_id
        
        # Add tenant context if provided
        if tenant_id:
            request_headers['X-Tenant-ID'] = tenant_id
        
        # Setup request parameters
        request_kwargs = {
            'method': method,
            'url': url,
            'headers': request_headers,
            'timeout': timeout or self.timeout
        }
        
        if params:
            request_kwargs['params'] = params
        if data:
            request_kwargs['data'] = data
        if json_data:
            request_kwargs['json'] = json_data
        
        # Log request
        self.logger.debug(
            "HTTP request started",
            method=method,
            url=url,
            request_id=request_id,
            tenant_id=tenant_id
        )
        
        # Execute request with circuit breaker and retry
        try:
            if self.circuit_breaker:
                response_data = await self.circuit_breaker.call(
                    self._execute_request_with_retry,
                    request_kwargs,
                    request_id,
                    tenant_id
                )
            else:
                response_data = await self._execute_request_with_retry(
                    request_kwargs,
                    request_id,
                    tenant_id
                )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record success metrics
            self._record_request_metrics(
                method=method,
                endpoint=endpoint,
                status_code=200,  # Assume success if no exception
                execution_time_ms=execution_time,
                tenant_id=tenant_id,
                success=True
            )
            
            self.logger.debug(
                "HTTP request completed",
                method=method,
                url=url,
                request_id=request_id,
                execution_time_ms=int(execution_time)
            )
            
            return response_data
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record error metrics
            status_code = getattr(e, 'status_code', 0)
            self._record_request_metrics(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                execution_time_ms=execution_time,
                tenant_id=tenant_id,
                success=False
            )
            
            self.logger.error(
                "HTTP request failed",
                method=method,
                url=url,
                request_id=request_id,
                error=e,
                execution_time_ms=int(execution_time)
            )
            
            raise HTTPClientError(
                message=f"HTTP request failed: {str(e)}",
                error_code="HTTP_REQUEST_FAILED",
                details={
                    "service": self.service_name,
                    "method": method,
                    "url": url,
                    "request_id": request_id,
                    "status_code": status_code
                }
            )
    
    async def _execute_request_with_retry(
        self,
        request_kwargs: Dict[str, Any],
        request_id: str,
        tenant_id: Optional[str]
    ) -> Dict[str, Any]:
        """Execute HTTP request with retry logic"""
        
        async def _make_request():
            response = await self._client.request(**request_kwargs)
            
            # Check for HTTP errors
            if response.status_code >= 400:
                error_detail = f"HTTP {response.status_code}"
                try:
                    error_body = response.json()
                    error_detail = error_body.get('message', error_detail)
                except:
                    error_detail = response.text or error_detail
                
                # Create exception with status code for circuit breaker
                error = HTTPClientError(
                    message=error_detail,
                    error_code="HTTP_ERROR",
                    details={"status_code": response.status_code}
                )
                error.status_code = response.status_code
                raise error
            
            # Parse response
            try:
                return response.json()
            except json.JSONDecodeError:
                # Return text response if not JSON
                return {"text": response.text, "content_type": response.headers.get("content-type")}
        
        # Determine if error is retryable
        def _is_retryable_error(error: Exception) -> bool:
            if isinstance(error, HTTPClientError):
                status_code = getattr(error, 'status_code', 0)
                # Retry on server errors and rate limiting
                return status_code >= 500 or status_code == 429
            # Retry on network errors
            return isinstance(error, (httpx.TimeoutException, httpx.NetworkError))
        
        # Execute with retry
        return await self.retry_handler.execute(
            _make_request,
            is_retryable=_is_retryable_error,
            context={
                "service": self.service_name,
                "request_id": request_id,
                "tenant_id": tenant_id
            }
        )
    
    def _record_request_metrics(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        execution_time_ms: float,
        tenant_id: Optional[str],
        success: bool
    ):
        """Record request metrics"""
        self._request_count += 1
        if not success:
            self._error_count += 1
        
        # Record service-specific metrics
        MetricsCollector.record_external_service_call(
            service_name=self.service_name,
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration_seconds=execution_time_ms / 1000,
            tenant_id=tenant_id or "unknown",
            success=success
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the service
        
        Returns:
            Health check result
        """
        try:
            start_time = datetime.utcnow()
            
            # Perform simple health check request
            await self.get("/health", timeout=5.0)
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "service": self.service_name,
                "status": "healthy",
                "response_time_ms": int(response_time),
                "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else "disabled",
                "total_requests": self._request_count,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(self._request_count, 1)
            }
            
        except Exception as e:
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else "disabled",
                "total_requests": self._request_count,
                "error_count": self._error_count
            }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
```

### `/src/clients/base/grpc_client.py`
**Purpose**: Base gRPC client with connection management and error handling
```python
import asyncio
import grpc
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from datetime import datetime
import uuid

from src.utils.logger import get_logger
from src.utils.metrics import MetricsCollector
from src.exceptions.base import MCPBaseException
from .circuit_breaker import CircuitBreaker
from .retry_handler import RetryHandler

logger = get_logger(__name__)

T = TypeVar('T')

class GRPCClientError(MCPBaseException):
    """GRPC client specific errors"""
    pass

class BaseGRPCClient:
    """Base gRPC client with advanced features"""
    
    def __init__(
        self,
        service_name: str,
        server_address: str,
        stub_class: type,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_enabled: bool = True,
        max_message_length: int = 100 * 1024 * 1024  # 100MB
    ):
        self.service_name = service_name
        self.server_address = server_address
        self.stub_class = stub_class
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_message_length = max_message_length
        
        # Setup logging
        self.logger = get_logger(f"client.{service_name}")
        
        # gRPC components
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[Any] = None
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            service_name=service_name,
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=GRPCClientError
        ) if circuit_breaker_enabled else None
        
        # Initialize retry handler
        self.retry_handler = RetryHandler(
            max_retries=max_retries,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0
        )
        
        # Metrics tracking
        self._request_count = 0
        self._error_count = 0
    
    async def initialize(self):
        """Initialize the gRPC client"""
        if self._channel is None:
            # Configure channel options
            options = [
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000),
                ('grpc.max_receive_message_length', self.max_message_length),
                ('grpc.max_send_message_length', self.max_message_length),
            ]
            
            # Create channel
            self._channel = grpc.aio.insecure_channel(
                self.server_address,
                options=options
            )
            
            # Create stub
            self._stub = self.stub_class(self._channel)
            
            # Test connection
            try:
                await asyncio.wait_for(
                    self._channel.channel_ready(),
                    timeout=10.0
                )
                
                self.logger.info(
                    "gRPC client initialized",
                    service=self.service_name,
                    address=self.server_address
                )
                
            except asyncio.TimeoutError:
                self.logger.warning(
                    "gRPC channel not ready after timeout",
                    service=self.service_name,
                    address=self.server_address
                )
    
    async def close(self):
        """Close the gRPC client"""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            
            self.logger.info("gRPC client closed", service=self.service_name)
    
    async def call_method(
        self,
        method_name: str,
        request: Any,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, str]] = None,
        tenant_id: Optional[str] = None
    ) -> Any:
        """
        Call gRPC method with circuit breaker and retry
        
        Args:
            method_name: Method name to call
            request: Request object
            timeout: Request timeout override
            metadata: gRPC metadata
            tenant_id: Tenant context for metrics
            
        Returns:
            Response object
        """
        if not self._stub:
            await self.initialize()
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Prepare metadata
        grpc_metadata = []
        if metadata:
            grpc_metadata.extend([(k, v) for k, v in metadata.items()])
        
        # Add request ID and tenant context
        grpc_metadata.append(('x-request-id', request_id))
        if tenant_id:
            grpc_metadata.append(('x-tenant-id', tenant_id))
        
        self.logger.debug(
            "gRPC request started",
            method=method_name,
            service=self.service_name,
            request_id=request_id,
            tenant_id=tenant_id
        )
        
        try:
            # Execute with circuit breaker and retry
            if self.circuit_breaker:
                response = await self.circuit_breaker.call(
                    self._execute_grpc_call_with_retry,
                    method_name,
                    request,
                    grpc_metadata,
                    timeout,
                    request_id,
                    tenant_id
                )
            else:
                response = await self._execute_grpc_call_with_retry(
                    method_name,
                    request,
                    grpc_metadata,
                    timeout,
                    request_id,
                    tenant_id
                )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record success metrics
            self._record_request_metrics(
                method=method_name,
                execution_time_ms=execution_time,
                tenant_id=tenant_id,
                success=True
            )
            
            self.logger.debug(
                "gRPC request completed",
                method=method_name,
                service=self.service_name,
                request_id=request_id,
                execution_time_ms=int(execution_time)
            )
            
            return response
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record error metrics
            self._record_request_metrics(
                method=method_name,
                execution_time_ms=execution_time,
                tenant_id=tenant_id,
                success=False
            )
            
            self.logger.error(
                "gRPC request failed",
                method=method_name,
                service=self.service_name,
                request_id=request_id,
                error=e,
                execution_time_ms=int(execution_time)
            )
            
            raise GRPCClientError(
                message=f"gRPC call failed: {str(e)}",
                error_code="GRPC_CALL_FAILED",
                details={
                    "service": self.service_name,
                    "method": method_name,
                    "request_id": request_id
                }
            )
    
    async def _execute_grpc_call_with_retry(
        self,
        method_name: str,
        request: Any,
        metadata: list,
        timeout: Optional[float],
        request_id: str,
        tenant_id: Optional[str]
    ) -> Any:
        """Execute gRPC call with retry logic"""
        
        async def _make_call():
            # Get method from stub
            method = getattr(self._stub, method_name)
            if not method:
                raise GRPCClientError(
                    message=f"Method {method_name} not found",
                    error_code="METHOD_NOT_FOUND"
                )
            
            # Make the call
            try:
                response = await method(
                    request,
                    timeout=timeout or self.timeout,
                    metadata=metadata
                )
                return response
                
            except grpc.aio.AioRpcError as e:
                # Convert gRPC error to our exception
                error = GRPCClientError(
                    message=e.details(),
                    error_code="GRPC_ERROR",
                    details={
                        "grpc_code": e.code().name,
                        "grpc_details": e.details()
                    }
                )
                error.grpc_code = e.code()
                raise error
        
        # Determine if error is retryable
        def _is_retryable_error(error: Exception) -> bool:
            if isinstance(error, GRPCClientError):
                grpc_code = getattr(error, 'grpc_code', None)
                if grpc_code:
                    # Retry on transient errors
                    retryable_codes = {
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                        grpc.StatusCode.ABORTED,
                        grpc.StatusCode.INTERNAL
                    }
                    return grpc_code in retryable_codes
            return False
        
        # Execute with retry
        return await self.retry_handler.execute(
            _make_call,
            is_retryable=_is_retryable_error,
            context={
                "service": self.service_name,
                "method": method_name,
                "request_id": request_id,
                "tenant_id": tenant_id
            }
        )
    
    def _record_request_metrics(
        self,
        method: str,
        execution_time_ms: float,
        tenant_id: Optional[str],
        success: bool
    ):
        """Record request metrics"""
        self._request_count += 1
        if not success:
            self._error_count += 1
        
        # Record service-specific metrics
        MetricsCollector.record_grpc_call(
            service_name=self.service_name,
            method=method,
            duration_seconds=execution_time_ms / 1000,
            tenant_id=tenant_id or "unknown",
            success=success
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the gRPC service
        
        Returns:
            Health check result
        """
        try:
            if not self._channel:
                await self.initialize()
            
            start_time = datetime.utcnow()
            
            # Check channel state
            state = self._channel.get_state()
            
            # Try to ensure channel is ready
            await asyncio.wait_for(
                self._channel.channel_ready(),
                timeout=5.0
            )
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "service": self.service_name,
                "status": "healthy",
                "channel_state": state.name,
                "response_time_ms": int(response_time),
                "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else "disabled",
                "total_requests": self._request_count,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(self._request_count, 1)
            }
            
        except Exception as e:
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else "disabled",
                "total_requests": self._request_count,
                "error_count": self._error_count
            }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
```

### `/src/clients/base/circuit_breaker.py`
**Purpose**: Circuit breaker implementation for service resilience
```python
import asyncio
from typing import Any, Callable, Optional, Type
from datetime import datetime, timedelta
from enum import Enum
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)

class CircuitBreakerState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreakerError(Exception):
    """Circuit breaker specific error"""
    pass

class CircuitBreaker:
    """
    Circuit breaker implementation for service resilience
    
    The circuit breaker prevents cascading failures by failing fast
    when a service is experiencing issues.
    """
    
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception,
        success_threshold: int = 3
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        
        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Logging
        self.logger = get_logger(f"circuit_breaker.{service_name}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: When circuit is open
        """
        async with self._lock:
            # Check if circuit should be opened
            await self._check_state()
            
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.logger.info(
                        "Circuit breaker attempting reset",
                        service=self.service_name
                    )
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN for service {self.service_name}. "
                        f"Next attempt at {self.next_attempt_time}"
                    )
        
        # Execute function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except self.expected_exception as e:
            await self._on_failure(e)
            raise
        except Exception as e:
            # Unexpected exception - don't count as failure
            self.logger.warning(
                "Unexpected exception in circuit breaker",
                service=self.service_name,
                error=e
            )
            raise
    
    async def _check_state(self):
        """Check and update circuit breaker state"""
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.success_threshold:
                self._close_circuit()
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                self.logger.debug(
                    "Circuit breaker success",
                    service=self.service_name,
                    success_count=self.success_count,
                    success_threshold=self.success_threshold
                )
                
                if self.success_count >= self.success_threshold:
                    self._close_circuit()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    async def _on_failure(self, exception: Exception):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            self.logger.warning(
                "Circuit breaker failure",
                service=self.service_name,
                failure_count=self.failure_count,
                failure_threshold=self.failure_threshold,
                error=str(exception)
            )
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Failed during recovery attempt - reopen circuit
                self._open_circuit()
            elif self.failure_count >= self.failure_threshold:
                self._open_circuit()
    
    def _open_circuit(self):
        """Open the circuit breaker"""
        self.state = CircuitBreakerState.OPEN
        self.next_attempt_time = datetime.utcnow() + timedelta(seconds=self.recovery_timeout)
        
        self.logger.error(
            "Circuit breaker OPENED",
            service=self.service_name,
            failure_count=self.failure_count,
            next_attempt=self.next_attempt_time.isoformat()
        )
    
    def _close_circuit(self):
        """Close the circuit breaker"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.next_attempt_time = None
        
        self.logger.info(
            "Circuit breaker CLOSED",
            service=self.service_name
        )
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.next_attempt_time is None:
            return True
        return datetime.utcnow() >= self.next_attempt_time
    
    def get_state_info(self) -> dict:
        """Get current circuit breaker state information"""
        return {
            "service": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "success_count": self.success_count,
            "success_threshold": self.success_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "next_attempt_time": self.next_attempt_time.isoformat() if self.next_attempt_time else None,
            "recovery_timeout_seconds": self.recovery_timeout
        }
    
    async def reset(self):
        """Manually reset the circuit breaker"""
        async with self._lock:
            self._close_circuit()
            self.logger.info(
                "Circuit breaker manually reset",
                service=self.service_name
            )
    
    async def force_open(self):
        """Manually open the circuit breaker"""
        async with self._lock:
            self._open_circuit()
            self.logger.warning(
                "Circuit breaker manually opened",
                service=self.service_name
            )
```

## Step 16: Model Orchestrator & Adaptor Service Clients (Days 54-56)

### `/src/clients/model_orchestrator_client.py`
**Purpose**: Client for Model Orchestrator service integration
```python
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.clients.base.http_client import BaseHTTPClient
from src.models.domain.execution_context import ExecutionContext
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

class ModelOrchestratorClient(BaseHTTPClient):
    """Client for Model Orchestrator service"""
    
    def __init__(self):
        super().__init__(
            service_name="model_orchestrator",
            base_url=settings.external_services.model_orchestrator_url,
            timeout=30.0,
            max_retries=3,
            circuit_breaker_enabled=True
        )
    
    async def detect_intent(
        self,
        text: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        language: str = "en",
        tenant_config: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect intent from user text
        
        Args:
            text: User input text
            conversation_context: Conversation context
            language: Language code
            tenant_config: Tenant-specific model configuration
            tenant_id: Tenant identifier
            
        Returns:
            Intent detection result
        """
        try:
            request_data = {
                "operation": "intent_detection",
                "input": {
                    "text": text,
                    "language": language,
                    "conversation_context": conversation_context or {}
                },
                "model_preferences": tenant_config or {
                    "primary_provider": "openai",
                    "primary_model": "gpt-4-turbo",
                    "fallback_chain": ["anthropic:claude-3-sonnet"]
                },
                "output_requirements": {
                    "format": "structured",
                    "include_confidence": True,
                    "include_alternatives": True
                }
            }
            
            response = await self.post(
                "/api/v2/model/process",
                json_data=request_data,
                tenant_id=tenant_id,
                timeout=15.0
            )
            
            # Extract intent detection result
            result = response.get("data", {}).get("result", {})
            
            return {
                "intent": result.get("primary_intent"),
                "confidence": result.get("confidence", 0.0),
                "alternatives": result.get("alternatives", []),
                "entities": result.get("entities", []),
                "model_info": response.get("data", {}).get("model_info", {}),
                "processing_time_ms": response.get("data", {}).get("performance_metrics", {}).get("processing_time_ms", 0)
            }
            
        except Exception as e:
            self.logger.error(
                "Intent detection failed",
                text=text[:100],  # Log first 100 chars only
                tenant_id=tenant_id,
                error=e
            )
            
            # Return fallback result
            return {
                "intent": "general_inquiry",
                "confidence": 0.1,
                "alternatives": [],
                "entities": [],
                "model_info": {},
                "processing_time_ms": 0,
                "fallback": True,
                "error": str(e)
            }
    
    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        tenant_config: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text
        
        Args:
            text: Input text
            entity_types: Specific entity types to extract
            conversation_context: Conversation context
            tenant_config: Tenant-specific model configuration
            tenant_id: Tenant identifier
            
        Returns:
            List of extracted entities
        """
        try:
            request_data = {
                "operation": "entity_extraction",
                "input": {
                    "text": text,
                    "entity_types": entity_types or [],
                    "conversation_context": conversation_context or {}
                },
                "model_preferences": tenant_config or {
                    "primary_provider": "openai",
                    "primary_model": "gpt-4-turbo"
                }
            }
            
            response = await self.post(
                "/api/v2/model/process",
                json_data=request_data,
                tenant_id=tenant_id,
                timeout=10.0
            )
            
            result = response.get("data", {}).get("result", {})
            entities = result.get("entities", [])
            
            # Normalize entity format
            normalized_entities = []
            for entity in entities:
                normalized_entities.append({
                    "entity": entity.get("type", "unknown"),
                    "value": entity.get("value", ""),
                    "raw_value": entity.get("raw_value", entity.get("value", "")),
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "confidence": entity.get("confidence", 0.0),
                    "resolution": entity.get("resolution", {})
                })
            
            return normalized_entities
            
        except Exception as e:
            self.logger.error(
                "Entity extraction failed",
                text=text[:100],
                tenant_id=tenant_id,
                error=e
            )
            return []
    
    async def generate_response(
        self,
        intent: str,
        entities: List[Dict[str, Any]],
        conversation_context: Optional[Dict[str, Any]] = None,
        response_template: Optional[str] = None,
        personalization_data: Optional[Dict[str, Any]] = None,
        tenant_config: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate contextual response
        
        Args:
            intent: Detected intent
            entities: Extracted entities
            conversation_context: Conversation context
            response_template: Optional response template
            personalization_data: User personalization data
            tenant_config: Tenant-specific model configuration
            tenant_id: Tenant identifier
            
        Returns:
            Generated response
        """
        try:
            request_data = {
                "operation": "response_generation",
                "input": {
                    "intent": intent,
                    "entities": entities,
                    "conversation_context": conversation_context or {},
                    "response_template": response_template,
                    "personalization_data": personalization_data or {}
                },
                "model_preferences": tenant_config or {
                    "primary_provider": "anthropic",
                    "primary_model": "claude-3-sonnet",
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                "output_requirements": {
                    "format": "structured",
                    "include_reasoning": False,
                    "tone": "helpful"
                }
            }
            
            response = await self.post(
                "/api/v2/model/process",
                json_data=request_data,
                tenant_id=tenant_id,
                timeout=20.0
            )
            
            result = response.get("data", {}).get("result", {})
            
            return {
                "text": result.get("primary_output", "I'm here to help. How can I assist you?"),
                "type": "text",
                "confidence": result.get("confidence_score", 0.0),
                "reasoning": result.get("reasoning"),
                "alternatives": result.get("alternatives", []),
                "model_info": response.get("data", {}).get("model_info", {}),
                "processing_time_ms": response.get("data", {}).get("performance_metrics", {}).get("processing_time_ms", 0)
            }
            
        except Exception as e:
            self.logger.error(
                "Response generation failed",
                intent=intent,
                tenant_id=tenant_id,
                error=e
            )
            
            # Return fallback response
            return {
                "text": "I understand. How can I help you further?",
                "type": "text",
                "confidence": 0.1,
                "fallback": True,
                "error": str(e)
            }
    
    async def analyze_sentiment(
        self,
        text: str,
        tenant_config: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            tenant_config: Tenant-specific model configuration
            tenant_id: Tenant identifier
            
        Returns:
            Sentiment analysis result
        """
        try:
            request_data = {
                "operation": "sentiment_analysis",
                "input": {
                    "text": text
                },
                "model_preferences": tenant_config or {
                    "primary_provider": "huggingface",
                    "primary_model": "sentiment-roberta"
                }
            }
            
            response = await self.post(
                "/api/v2/model/process",
                json_data=request_data,
                tenant_id=tenant_id,
                timeout=5.0
            )
            
            result = response.get("data", {}).get("result", {})
            
            return {
                "sentiment": result.get("sentiment", "neutral"),
                "confidence": result.get("confidence", 0.0),
                "scores": result.get("scores", {}),
                "processing_time_ms": response.get("data", {}).get("performance_metrics", {}).get("processing_time_ms", 0)
            }
            
        except Exception as e:
            self.logger.error(
                "Sentiment analysis failed",
                text=text[:100],
                tenant_id=tenant_id,
                error=e
            )
            
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {},
                "fallback": True,
                "error": str(e)
            }
    
    async def get_model_status(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model orchestrator status
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Model status information
        """
        try:
            response = await self.get(
                "/api/v2/model/status",
                tenant_id=tenant_id,
                timeout=5.0
            )
            
            return response.get("data", {})
            
        except Exception as e:
            self.logger.error(
                "Failed to get model status",
                tenant_id=tenant_id,
                error=e
            )
            
            return {
                "status": "unknown",
                "error": str(e)
            }
```

### `/src/clients/adaptor_service_client.py`
**Purpose**: Client for Adaptor Service integration calls
```python
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.clients.base.http_client import BaseHTTPClient
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

class AdaptorServiceClient(BaseHTTPClient):
    """Client for Adaptor Service integration calls"""
    
    def __init__(self):
        super().__init__(
            service_name="adaptor_service",
            base_url=settings.external_services.adaptor_service_url,
            timeout=30.0,
            max_retries=3,
            circuit_breaker_enabled=True
        )
    
    async def execute_integration(
        self,
        integration_id: str,
        endpoint: str,
        method: str = "GET",
        request_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_ms: int = 5000,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute integration call via Adaptor Service
        
        Args:
            integration_id: Integration identifier
            endpoint: Integration endpoint
            method: HTTP method
            request_data: Request data
            headers: Additional headers
            timeout_ms: Request timeout in milliseconds
            tenant_id: Tenant identifier
            
        Returns:
            Integration execution result
        """
        try:
            execution_data = {
                "execution_id": f"exec_{datetime.utcnow().isoformat()}",
                "endpoint": endpoint,
                "method": method,
                "input_data": request_data or {},
                "headers": headers or {},
                "timeout_override_ms": timeout_ms,
                "metadata": {
                    "tenant_id": tenant_id,
                    "source": "mcp_engine",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            response = await self.post(
                f"/api/v2/integrations/{integration_id}/execute",
                json_data=execution_data,
                tenant_id=tenant_id,
                timeout=timeout_ms / 1000.0 + 5.0  # Add buffer for network overhead
            )
            
            execution_result = response.get("data", {})
            
            return {
                "success": execution_result.get("result", {}).get("status_code", 0) < 400,
                "status_code": execution_result.get("result", {}).get("status_code", 0),
                "data": execution_result.get("result", {}).get("response_data", {}),
                "error": execution_result.get("result", {}).get("error_message"),
                "execution_time_ms": execution_result.get("execution_metadata", {}).get("execution_time_ms", 0),
                "execution_id": execution_result.get("execution_id"),
                "retry_count": execution_result.get("execution_metadata", {}).get("retry_count", 0),
                "cache_hit": execution_result.get("execution_metadata", {}).get("cache_hit", False)
            }
            
        except Exception as e:
            self.logger.error(
                "Integration execution failed",
                integration_id=integration_id,
                endpoint=endpoint,
                method=method,
                tenant_id=tenant_id,
                error=e
            )
            
            return {
                "success": False,
                "status_code": 0,
                "data": {},
                "error": str(e),
                "execution_time_ms": 0,
                "execution_id": None,
                "retry_count": 0,
                "cache_hit": False,
                "client_error": True
            }
    
    async def test_integration(
        self,
        integration_id: str,
        test_case: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test integration configuration
        
        Args:
            integration_id: Integration identifier
            test_case: Optional specific test case
            tenant_id: Tenant identifier
            
        Returns:
            Test result
        """
        try:
            test_data = {
                "test_case": test_case
            } if test_case else {}
            
            response = await self.post(
                f"/api/v2/integrations/{integration_id}/test",
                json_data=test_data,
                tenant_id=tenant_id,
                timeout=10.0
            )
            
            return response.get("data", {})
            
        except Exception as e:
            self.logger.error(
                "Integration test failed",
                integration_id=integration_id,
                test_case=test_case,
                tenant_id=tenant_id,
                error=e
            )
            
            return {
                "success": False,
                "error": str(e),
                "test_results": []
            }
    
    async def get_integration_status(
        self,
        integration_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get integration status and health
        
        Args:
            integration_id: Integration identifier
            tenant_id: Tenant identifier
            
        Returns:
            Integration status
        """
        try:
            response = await self.get(
                f"/api/v2/integrations/{integration_id}/status",
                tenant_id=tenant_id,
                timeout=5.0
            )
            
            return response.get("data", {})
            
        except Exception as e:
            self.logger.error(
                "Failed to get integration status",
                integration_id=integration_id,
                tenant_id=tenant_id,
                error=e
            )
            
            return {
                "status": "unknown",
                "health": "unknown",
                "error": str(e)
            }
    
    async def list_integrations(
        self,
        category: Optional[str] = None,
        status_filter: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available integrations
        
        Args:
            category: Integration category filter
            status_filter: Status filter
            tenant_id: Tenant identifier
            
        Returns:
            List of integrations
        """
        try:
            params = {}
            if category:
                params["category"] = category
            if status_filter:
                params["status"] = status_filter
            
            response = await self.get(
                "/api/v2/integrations",
                params=params,
                tenant_id=tenant_id,
                timeout=10.0
            )
            
            return response.get("data", {}).get("integrations", [])
            
        except Exception as e:
            self.logger.error(
                "Failed to list integrations",
                category=category,
                status_filter=status_filter,
                tenant_id=tenant_id,
                error=e
            )
            
            return []
    
    async def create_integration(
        self,
        integration_config: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create new integration
        
        Args:
            integration_config: Integration configuration
            tenant_id: Tenant identifier
            
        Returns:
            Created integration details
        """
        try:
            response = await self.post(
                "/api/v2/integrations",
                json_data=integration_config,
                tenant_id=tenant_id,
                timeout=15.0
            )
            
            return response.get("data", {})
            
        except Exception as e:
            self.logger.error(
                "Failed to create integration",
                integration_name=integration_config.get("name"),
                tenant_id=tenant_id,
                error=e
            )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_integration(
        self,
        integration_id: str,
        updates: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update integration configuration
        
        Args:
            integration_id: Integration identifier
            updates: Configuration updates
            tenant_id: Tenant identifier
            
        Returns:
            Update result
        """
        try:
            response = await self.put(
                f"/api/v2/integrations/{integration_id}",
                json_data=updates,
                tenant_id=tenant_id,
                timeout=10.0
            )
            
            return response.get("data", {})
            
        except Exception as e:
            self.logger.error(
                "Failed to update integration",
                integration_id=integration_id,
                tenant_id=tenant_id,
                error=e
            )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_integration(
        self,
        integration_id: str,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Delete integration
        
        Args:
            integration_id: Integration identifier
            tenant_id: Tenant identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.delete(
                f"/api/v2/integrations/{integration_id}",
                tenant_id=tenant_id,
                timeout=10.0
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to delete integration",
                integration_id=integration_id,
                tenant_id=tenant_id,
                error=e
            )
            
            return False
```

## Success Criteria
- [x] Complete client infrastructure with HTTP and gRPC support
- [x] Circuit breaker implementation for service resilience
- [x] Retry mechanisms with exponential backoff
- [x] Model Orchestrator client for AI operations
- [x] Adaptor Service client for external integrations
- [x] Comprehensive error handling and fallback mechanisms
- [x] Performance monitoring and metrics collection
- [x] Connection management and resource cleanup

## Key Error Handling & Performance Considerations
1. **Circuit Breaker**: Prevents cascading failures with configurable thresholds
2. **Retry Logic**: Exponential backoff with jitter for transient failures
3. **Connection Pooling**: Efficient HTTP connection management
4. **Timeout Handling**: Configurable timeouts with fallback responses
5. **Error Classification**: Retryable vs non-retryable error detection
6. **Metrics Collection**: Detailed performance and error tracking
7. **Resource Management**: Proper cleanup of connections and resources

## Technologies Used
- **HTTP Client**: httpx with async support and connection pooling
- **gRPC Client**: grpc.aio with async streaming support
- **Circuit Breaker**: Custom implementation with state management
- **Retry Handler**: Exponential backoff with configurable policies
- **Monitoring**: Prometheus metrics integration
- **Error Handling**: Comprehensive exception hierarchy

## Cross-Service Integration
- **Model Orchestrator**: Intent detection, entity extraction, response generation
- **Adaptor Service**: External system integration execution
- **Security Hub**: Authentication and authorization (next phase)
- **Analytics Engine**: Event tracking and metrics
- **Circuit Breaker**: Service resilience and failure detection
- **Caching**: Response caching and performance optimization

## Next Phase Dependencies
Phase 7 will build upon:
- External service client infrastructure
- Circuit breaker and retry mechanisms
- Model Orchestrator integration capabilities
- Adaptor Service integration framework
- Error handling and monitoring systems