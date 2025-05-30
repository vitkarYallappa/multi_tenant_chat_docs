# Deployment and DevOps
## Multi-Tenant AI Chatbot Platform

**Document:** 06-Deployment-DevOps.md  
**Version:** 2.0  
**Last Updated:** May 30, 2025

---

## Table of Contents

1. [Infrastructure Overview](#infrastructure-overview)
2. [Container Strategy](#container-strategy)
3. [Kubernetes Configuration](#kubernetes-configuration)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Environment Management](#environment-management)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Security and Compliance](#security-and-compliance)
8. [Scaling and Performance](#scaling-and-performance)

---

## Infrastructure Overview

### Cloud Architecture Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-CLOUD ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

Primary Cloud (AWS):
├── Production Environment
├── Staging Environment  
├── Primary Data Storage
├── Main CI/CD Pipeline
└── Disaster Recovery Source

Secondary Cloud (Azure):
├── Disaster Recovery Target
├── Development Environment
├── Testing Infrastructure
└── Compliance Workloads

Edge Locations:
├── CloudFlare CDN
├── Regional Load Balancers
├── Edge Caching
└── DDoS Protection
```

### Infrastructure Components

| Component | Technology | Purpose | Scaling Strategy |
|-----------|------------|---------|------------------|
| **Container Orchestration** | Kubernetes 1.28+ | Service deployment and management | Horizontal pod autoscaling |
| **Service Mesh** | Istio 1.19+ | Service communication and security | Automatic sidecar injection |
| **Load Balancing** | AWS ALB / Nginx | Traffic distribution | Multiple availability zones |
| **Message Queue** | Apache Kafka | Async communication | Partition-based scaling |
| **Databases** | PostgreSQL, MongoDB, Redis | Data persistence | Read replicas and sharding |
| **Monitoring** | Prometheus + Grafana | Observability | Federated monitoring |
| **Logging** | ELK Stack | Centralized logging | Index lifecycle management |
| **Secrets Management** | HashiCorp Vault | Secret storage | High availability cluster |

---

## Container Strategy

### Docker Configuration

#### Base Images

```dockerfile
# Production Base Image
FROM python:3.11-slim-bullseye as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Development Image
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --user -r requirements-dev.txt

# Add development tools
USER root
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER app

# Production Image
FROM base as production

# Copy application code
COPY --chown=app:app . .

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Service-Specific Dockerfiles

```dockerfile
# Chat Service Dockerfile
FROM chatbot/base:production as chat-service

# Install service-specific dependencies
COPY chat-service/requirements.txt .
RUN pip install --user -r requirements.txt

# Copy service code
COPY --chown=app:app chat-service/ ./chat-service/
WORKDIR /home/app/chat-service

# Health check for chat service
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "chat_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

# MCP Service Dockerfile  
FROM chatbot/base:production as mcp-service

COPY mcp-service/requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=app:app mcp-service/ ./mcp-service/
WORKDIR /home/app/mcp-service

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "mcp_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Model Orchestrator Dockerfile
FROM chatbot/base:production as model-orchestrator

COPY model-orchestrator/requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=app:app model-orchestrator/ ./model-orchestrator/
WORKDIR /home/app/model-orchestrator

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "model_orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-Stage Build Optimization

```dockerfile
# Multi-stage build for optimal image size
FROM node:18-alpine as frontend-builder

WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci --only=production

COPY frontend/ .
RUN npm run build

FROM python:3.11-slim as backend-builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as production

# Copy Python dependencies
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy frontend build
COPY --from=frontend-builder /app/dist /app/static

# Copy application
COPY . /app
WORKDIR /app

# Security: Run as non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Kubernetes Configuration

### Namespace Organization

```yaml
# namespace-structure.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: chatbot-production
  labels:
    environment: production
    app: chatbot-platform
---
apiVersion: v1
kind: Namespace
metadata:
  name: chatbot-staging
  labels:
    environment: staging
    app: chatbot-platform
---
apiVersion: v1
kind: Namespace
metadata:
  name: chatbot-development
  labels:
    environment: development
    app: chatbot-platform
---
# Infrastructure namespaces
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
  labels:
    purpose: monitoring
---
apiVersion: v1
kind: Namespace
metadata:
  name: logging
  labels:
    purpose: logging
---
apiVersion: v1
kind: Namespace
metadata:
  name: security
  labels:
    purpose: security
```

### Service Deployments

```yaml
# chat-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chat-service
  namespace: chatbot-production
  labels:
    app: chat-service
    version: v1
    component: backend
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: chat-service
  template:
    metadata:
      labels:
        app: chat-service
        version: v1
        component: backend
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: chat-service
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
      - name: chat-service
        image: chatbot/chat-service:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: postgresql-url
        - name: MONGODB_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: mongodb-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: cache-credentials
              key: redis-url
        - name: KAFKA_BOOTSTRAP_SERVERS
          valueFrom:
            configMapKeyRef:
              name: kafka-config
              key: bootstrap-servers
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: tmp-volume
          mountPath: /tmp
      volumes:
      - name: config-volume
        configMap:
          name: chat-service-config
      - name: tmp-volume
        emptyDir: {}
      nodeSelector:
        workload-type: "application"
      tolerations:
      - key: "workload-type"
        operator: "Equal"
        value: "application"
        effect: "NoSchedule"
---
apiVersion: v1
kind: Service
metadata:
  name: chat-service
  namespace: chatbot-production
  labels:
    app: chat-service
spec:
  selector:
    app: chat-service
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chat-service-hpa
  namespace: chatbot-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chat-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### ConfigMaps and Secrets

```yaml
# config-management.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chat-service-config
  namespace: chatbot-production
data:
  app-config.yaml: |
    server:
      host: "0.0.0.0"
      port: 8000
      workers: 4
    
    logging:
      level: "INFO"
      format: "json"
      correlation_id: true
    
    monitoring:
      prometheus_enabled: true
      prometheus_port: 8000
      prometheus_path: "/metrics"
    
    features:
      rate_limiting_enabled: true
      caching_enabled: true
      circuit_breaker_enabled: true
    
    security:
      cors_origins: 
        - "https://app.chatbot-platform.com"
        - "https://admin.chatbot-platform.com"
      csrf_protection: true
      
---
apiVersion: v1
kind: Secret
metadata:
  name: database-credentials
  namespace: chatbot-production
type: Opaque
data:
  postgresql-url: <base64-encoded-connection-string>
  mongodb-url: <base64-encoded-connection-string>
  redis-url: <base64-encoded-connection-string>
---
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
  namespace: chatbot-production
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  anthropic-api-key: <base64-encoded-key>
  jwt-secret: <base64-encoded-secret>
  encryption-key: <base64-encoded-key>
```

### Ingress Configuration

```yaml
# ingress-configuration.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chatbot-platform-ingress
  namespace: chatbot-production
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.chatbot-platform.com"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    # Security headers
    nginx.ingress.kubernetes.io/server-snippet: |
      add_header X-Frame-Options "DENY" always;
      add_header X-Content-Type-Options "nosniff" always;
      add_header X-XSS-Protection "1; mode=block" always;
      add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
spec:
  tls:
  - hosts:
    - api.chatbot-platform.com
    - admin.chatbot-platform.com
    secretName: chatbot-platform-tls
  rules:
  - host: api.chatbot-platform.com
    http:
      paths:
      - path: /api/v2/chat
        pathType: Prefix
        backend:
          service:
            name: chat-service
            port:
              number: 80
      - path: /api/v2/mcp
        pathType: Prefix
        backend:
          service:
            name: mcp-service
            port:
              number: 80
      - path: /api/v2/model
        pathType: Prefix
        backend:
          service:
            name: model-orchestrator
            port:
              number: 80
      - path: /api/v2/integrations
        pathType: Prefix
        backend:
          service:
            name: adaptor-service
            port:
              number: 80
      - path: /api/v2/auth
        pathType: Prefix
        backend:
          service:
            name: security-hub
            port:
              number: 80
  - host: admin.chatbot-platform.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: admin-dashboard
            port:
              number: 80
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop, 'feature/*']
  pull_request:
    branches: [main, develop]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
          
      mongodb:
        image: mongo:7
        env:
          MONGO_INITDB_DATABASE: test
        ports:
          - 27017:27017
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for semantic versioning
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Cache Python dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
        
    - name: Run code quality checks
      run: |
        # Linting
        flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
        
        # Code formatting
        black --check src/ tests/
        
        # Import sorting
        isort --check-only src/ tests/
        
        # Type checking
        mypy src/
        
    - name: Run security checks
      run: |
        # Security vulnerability scanning
        bandit -r src/ -f json -o bandit-report.json
        
        # Dependency vulnerability checking
        safety check --json --output safety-report.json
        
        # License compliance
        pip-licenses --format=json --output-file=licenses-report.json
        
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v \
          --cov=src \
          --cov-report=xml \
          --cov-report=html \
          --junitxml=test-results.xml
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test
        MONGODB_URL: mongodb://localhost:27017/test
        REDIS_URL: redis://localhost:6379/0
        
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v \
          --junitxml=integration-test-results.xml
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test
        MONGODB_URL: mongodb://localhost:27017/test
        REDIS_URL: redis://localhost:6379/0
        
    - name: Upload test coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        
    - name: Upload test results
      uses: dorny/test-reporter@v1
      if: success() || failure()
      with:
        name: Test Results
        path: '*-test-results.xml'
        reporter: java-junit

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    strategy:
      matrix:
        service: [chat-service, mcp-service, model-orchestrator, adaptor-service, security-hub, analytics-engine]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.service }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: ./services/${{ matrix.service }}
        file: ./services/${{ matrix.service }}/Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64
        
    - name: Sign container image
      run: |
        cosign sign --yes ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.service }}@${{ steps.build.outputs.digest }}
      env:
        COSIGN_EXPERIMENTAL: 1

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
        
    - name: Set up Helm
      uses: azure/setup-helm@v3
      with:
        version: '3.12.0'
        
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
        
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --name chatbot-staging --region us-east-1
        
    - name: Deploy to staging
      run: |
        helm upgrade --install chatbot-platform ./helm/chatbot-platform \
          --namespace chatbot-staging \
          --create-namespace \
          --values ./helm/values-staging.yaml \
          --set image.tag=${{ github.sha }} \
          --wait \
          --timeout=10m
          
    - name: Run smoke tests
      run: |
        kubectl wait --for=condition=ready pod -l app=chat-service -n chatbot-staging --timeout=300s
        
        # Basic health checks
        kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -- \
          curl -f https://staging-api.chatbot-platform.com/health
        
        # API functionality tests
        python scripts/smoke-tests.py --environment=staging
        
    - name: Run integration tests
      run: |
        pytest tests/e2e/ -v \
          --base-url=https://staging-api.chatbot-platform.com \
          --junitxml=e2e-test-results.xml
      env:
        TEST_API_KEY: ${{ secrets.STAGING_API_KEY }}

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
        
    - name: Set up Helm
      uses: azure/setup-helm@v3
      with:
        version: '3.12.0'
        
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
        
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --name chatbot-production --region us-east-1
        
    - name: Blue-Green Deployment
      run: |
        # Deploy to green environment
        helm upgrade --install chatbot-platform-green ./helm/chatbot-platform \
          --namespace chatbot-green \
          --create-namespace \
          --values ./helm/values-production.yaml \
          --set image.tag=${{ github.sha }} \
          --set ingress.hosts[0].host=green-api.chatbot-platform.com \
          --wait \
          --timeout=15m
          
    - name: Production health checks
      run: |
        kubectl wait --for=condition=ready pod -l app=chat-service -n chatbot-green --timeout=600s
        
        # Comprehensive health validation
        python scripts/production-health-check.py --environment=green
        
        # Load testing
        python scripts/load-test.py --target=https://green-api.chatbot-platform.com --duration=300
        
    - name: Switch traffic to green
      run: |
        # Update main ingress to point to green
        helm upgrade chatbot-platform ./helm/chatbot-platform \
          --namespace chatbot-green \
          --values ./helm/values-production.yaml \
          --set image.tag=${{ github.sha }} \
          --set ingress.hosts[0].host=api.chatbot-platform.com \
          --wait
          
    - name: Monitor deployment
      run: |
        # Monitor for 10 minutes
        python scripts/deployment-monitor.py \
          --duration=600 \
          --error-threshold=0.01 \
          --response-time-threshold=1000
        
    - name: Cleanup blue environment
      run: |
        # Remove blue environment after successful green deployment
        helm uninstall chatbot-platform --namespace chatbot-blue || true
        kubectl delete namespace chatbot-blue --timeout=300s || true
        
    - name: Notify deployment success
      uses: 8398a7/action-slack@v3
      with:
        status: success
        channel: '#deployments'
        message: |
          ✅ Production deployment successful!
          Version: ${{ github.sha }}
          Environment: Production
          Services: All services updated
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
      if: success()
      
    - name: Notify deployment failure
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        channel: '#deployments'
        message: |
          ❌ Production deployment failed!
          Version: ${{ github.sha }}
          Environment: Production
          Check logs: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
      if: failure()
```

### Deployment Scripts

```bash
#!/bin/bash
# scripts/deploy.sh

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validation
validate_environment() {
    case $ENVIRONMENT in
        development|staging|production)
            log_info "Deploying to $ENVIRONMENT environment"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            echo "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Unable to connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Helm
    if ! helm version &> /dev/null; then
        log_error "Helm is not installed or not accessible"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "chatbot-$ENVIRONMENT" &> /dev/null; then
        log_warn "Namespace chatbot-$ENVIRONMENT does not exist, creating..."
        kubectl create namespace "chatbot-$ENVIRONMENT"
    fi
    
    log_info "Pre-deployment checks passed"
}

# Deploy application
deploy_application() {
    log_info "Deploying application version $VERSION to $ENVIRONMENT..."
    
    # Set values file based on environment
    VALUES_FILE="$PROJECT_ROOT/helm/values-$ENVIRONMENT.yaml"
    
    if [[ ! -f "$VALUES_FILE" ]]; then
        log_error "Values file not found: $VALUES_FILE"
        exit 1
    fi
    
    # Deploy with Helm
    helm upgrade --install "chatbot-platform" "$PROJECT_ROOT/helm/chatbot-platform" \
        --namespace "chatbot-$ENVIRONMENT" \
        --values "$VALUES_FILE" \
        --set image.tag="$VERSION" \
        --set deployment.timestamp="$(date +%s)" \
        --wait \
        --timeout=10m
    
    if [[ $? -eq 0 ]]; then
        log_info "Application deployed successfully"
    else
        log_error "Application deployment failed"
        exit 1
    fi
}

# Post-deployment verification
post_deployment_verification() {
    log_info "Running post-deployment verification..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/instance=chatbot-platform \
        -n "chatbot-$ENVIRONMENT" \
        --timeout=300s
    
    # Health checks
    log_info "Running health checks..."
    
    services=("chat-service" "mcp-service" "model-orchestrator" "adaptor-service")
    
    for service in "${services[@]}"; do
        log_info "Checking health of $service..."
        
        # Port forward to service
        kubectl port-forward "svc/$service" 8080:80 -n "chatbot-$ENVIRONMENT" &
        PF_PID=$!
        
        # Wait for port forward to establish
        sleep 5
        
        # Health check
        if curl -f http://localhost:8080/health --max-time 10 --retry 3 &> /dev/null; then
            log_info "$service is healthy"
        else
            log_error "$service health check failed"
            kill $PF_PID 2>/dev/null || true
            exit 1
        fi
        
        # Cleanup port forward
        kill $PF_PID 2>/dev/null || true
    done
    
    log_info "All health checks passed"
}

# Rollback function
rollback_deployment() {
    log_warn "Rolling back deployment..."
    
    helm rollback "chatbot-platform" -n "chatbot-$ENVIRONMENT"
    
    if [[ $? -eq 0 ]]; then
        log_info "Rollback completed successfully"
    else
        log_error "Rollback failed"
        exit 1
    fi
}

# Main execution
main() {
    log_info "Starting deployment process..."
    
    validate_environment
    pre_deployment_checks
    
    # Trap to rollback on failure
    trap rollback_deployment ERR
    
    deploy_application
    post_deployment_verification
    
    log_info "Deployment completed successfully!"
}

# Run main function
main "$@"
```

---

## Environment Management

### Helm Chart Structure

```yaml
# helm/chatbot-platform/Chart.yaml
apiVersion: v2
name: chatbot-platform
description: Multi-tenant AI chatbot platform
type: application
version: 1.0.0
appVersion: "1.0.0"

dependencies:
- name: postgresql
  version: 12.1.6
  repository: https://charts.bitnami.com/bitnami
  condition: postgresql.enabled
- name: mongodb
  version: 13.6.8
  repository: https://charts.bitnami.com/bitnami
  condition: mongodb.enabled
- name: redis
  version: 17.4.3
  repository: https://charts.bitnami.com/bitnami
  condition: redis.enabled
- name: kafka
  version: 20.0.6
  repository: https://charts.bitnami.com/bitnami
  condition: kafka.enabled
- name: prometheus
  version: 23.4.0
  repository: https://prometheus-community.github.io/helm-charts
  condition: monitoring.prometheus.enabled
- name: grafana
  version: 6.59.5
  repository: https://grafana.github.io/helm-charts
  condition: monitoring.grafana.enabled
```

### Environment-Specific Values

```yaml
# helm/values-production.yaml
global:
  imageRegistry: ghcr.io
  imagePullSecrets:
  - name: ghcr-secret
  storageClass: gp3

image:
  registry: ghcr.io
  repository: chatbot-platform
  tag: latest
  pullPolicy: Always

replicaCount:
  chatService: 5
  mcpService: 3
  modelOrchestrator: 3
  adaptorService: 3
  securityHub: 2
  analyticsEngine: 2

resources:
  chatService:
    requests:
      memory: 512Mi
      cpu: 500m
    limits:
      memory: 1Gi
      cpu: 1000m
  mcpService:
    requests:
      memory: 1Gi
      cpu: 500m
    limits:
      memory: 2Gi
      cpu: 1000m
  modelOrchestrator:
    requests:
      memory: 512Mi
      cpu: 500m
    limits:
      memory: 1Gi
      cpu: 1000m

autoscaling:
  enabled: true
  minReplicas:
    chatService: 5
    mcpService: 3
    modelOrchestrator: 3
  maxReplicas:
    chatService: 50
    mcpService: 30
    modelOrchestrator: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "2000"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/enable-modsecurity: "true"
    nginx.ingress.kubernetes.io/enable-owasp-core-rules: "true"
  hosts:
  - host: api.chatbot-platform.com
    paths:
    - path: /
      pathType: Prefix
  tls:
  - secretName: chatbot-platform-tls
    hosts:
    - api.chatbot-platform.com

# Database configurations
postgresql:
  enabled: false  # Use external managed database
  external:
    host: chatbot-prod-db.cluster-xyz.us-east-1.rds.amazonaws.com
    port: 5432
    database: chatbot_production
    existingSecret: database-credentials
    secretKeys:
      userPasswordKey: postgresql-password

mongodb:
  enabled: false  # Use external managed database
  external:
    host: chatbot-prod-mongo.cluster-xyz.us-east-1.docdb.amazonaws.com
    port: 27017
    database: chatbot_production
    existingSecret: database-credentials
    secretKeys:
      userPasswordKey: mongodb-password

redis:
  enabled: false  # Use external managed cache
  external:
    host: chatbot-prod-redis.xyz.cache.amazonaws.com
    port: 6379
    existingSecret: cache-credentials
    secretKeys:
      passwordKey: redis-password

# Monitoring
monitoring:
  prometheus:
    enabled: true
    retention: 30d
    storageSize: 100Gi
  grafana:
    enabled: true
    adminPassword: "" # Set via secret
    persistence:
      enabled: true
      size: 10Gi
  alerts:
    enabled: true
    slack:
      webhook: "" # Set via secret
      channel: "#alerts"

# Security
security:
  podSecurityPolicy:
    enabled: true
  networkPolicy:
    enabled: true
  rbac:
    create: true
  serviceAccount:
    create: true
    annotations:
      eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/chatbot-platform-role

# Backup and disaster recovery
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention: "30d"
  destinations:
  - s3://chatbot-backups/production/

# Feature flags
features:
  rateLimiting: true
  circuitBreaker: true
  distributedTracing: true
  auditLogging: true
  metricsCollection: true
```

```yaml
# helm/values-staging.yaml
global:
  imageRegistry: ghcr.io
  imagePullSecrets:
  - name: ghcr-secret

image:
  registry: ghcr.io
  repository: chatbot-platform
  tag: develop
  pullPolicy: Always

replicaCount:
  chatService: 2
  mcpService: 1
  modelOrchestrator: 1
  adaptorService: 1
  securityHub: 1
  analyticsEngine: 1

resources:
  chatService:
    requests:
      memory: 256Mi
      cpu: 250m
    limits:
      memory: 512Mi
      cpu: 500m
  mcpService:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 1Gi
      cpu: 500m

autoscaling:
  enabled: true
  minReplicas:
    chatService: 2
    mcpService: 1
  maxReplicas:
    chatService: 10
    mcpService: 5
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "500"
    cert-manager.io/cluster-issuer: letsencrypt-staging
  hosts:
  - host: staging-api.chatbot-platform.com
    paths:
    - path: /
      pathType: Prefix
  tls:
  - secretName: chatbot-platform-staging-tls
    hosts:
    - staging-api.chatbot-platform.com

# Use smaller managed databases for staging
postgresql:
  enabled: false
  external:
    host: chatbot-staging-db.cluster-xyz.us-east-1.rds.amazonaws.com
    port: 5432
    database: chatbot_staging

mongodb:
  enabled: false
  external:
    host: chatbot-staging-mongo.cluster-xyz.us-east-1.docdb.amazonaws.com
    port: 27017
    database: chatbot_staging

redis:
  enabled: false
  external:
    host: chatbot-staging-redis.xyz.cache.amazonaws.com
    port: 6379

monitoring:
  prometheus:
    enabled: true
    retention: 7d
    storageSize: 20Gi
  grafana:
    enabled: true
    persistence:
      enabled: true
      size: 5Gi

backup:
  enabled: false  # No backups for staging

features:
  rateLimiting: true
  circuitBreaker: true
  distributedTracing: true
  auditLogging: false
  metricsCollection: true
```


**Document Maintainer:** DevOps Team  
**Review Schedule:** Weekly during development, monthly in production  
**Related Documents:** System Architecture, Security Implementation, Performance Monitoring