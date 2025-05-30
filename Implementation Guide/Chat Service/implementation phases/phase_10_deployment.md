# Phase 10: Production Deployment & DevOps
**Duration:** Week 17-18  
**Steps:** 18 of 18

---

## 🎯 Objectives
- Containerize the application with Docker
- Create Kubernetes deployment manifests
- Implement comprehensive monitoring and observability
- Establish CI/CD pipelines for automated deployment
- Configure production security and performance optimizations
- Create operational runbooks and documentation

---

## 📋 Step 18: Production Deployment Pipeline

### What Will Be Implemented
- Docker containerization with multi-stage builds
- Kubernetes manifests for production deployment
- Helm charts for configuration management
- Monitoring stack with Prometheus and Grafana
- CI/CD pipeline with GitHub Actions
- Production security configurations

### Folders and Files Created

```
docker/
├── Dockerfile
├── Dockerfile.dev
├── docker-compose.yml
├── docker-compose.override.yml
└── .dockerignore

k8s/
├── namespace.yaml
├── configmap.yaml
├── secrets.yaml
├── deployment.yaml
├── service.yaml
├── ingress.yaml
├── hpa.yaml
├── pdb.yaml
└── rbac.yaml

helm/
├── Chart.yaml
├── values.yaml
├── values-staging.yaml
├── values-production.yaml
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    ├── configmap.yaml
    ├── secrets.yaml
    ├── ingress.yaml
    └── hpa.yaml

monitoring/
├── prometheus/
│   ├── prometheus.yml
│   ├── alert-rules.yml
│   └── recording-rules.yml
├── grafana/
│   ├── dashboards/
│   │   ├── chat-service-overview.json
│   │   ├── performance-metrics.json
│   │   └── business-metrics.json
│   └── datasources/
│       └── prometheus.yml
└── alertmanager/
    └── alertmanager.yml

scripts/
├── deploy.sh
├── rollback.sh
├── health-check.sh
├── backup.sh
└── maintenance.sh

docs/
├── deployment/
│   ├── README.md
│   ├── production-setup.md
│   ├── monitoring-guide.md
│   └── troubleshooting.md
└── runbooks/
    ├── incident-response.md
    ├── scaling-guide.md
    └── backup-recovery.md
```

### File Documentation

#### `docker/Dockerfile`
**Purpose:** Production Docker container with multi-stage build for optimization  
**Usage:** Build optimized container image for production deployment

```dockerfile
# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-root

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser \
    && useradd -r -g appuser appuser

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app directory and set ownership
WORKDIR /app
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/v2/health || exit 1

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

#### `k8s/deployment.yaml`
**Purpose:** Kubernetes deployment manifest for the chat service  
**Usage:** Deploy chat service pods with proper resource management and scaling

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chat-service
  namespace: chatbot-platform
  labels:
    app: chat-service
    version: v2.0.0
    component: backend
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: chat-service
  template:
    metadata:
      labels:
        app: chat-service
        version: v2.0.0
        component: backend
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: chat-service
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: chat-service
        image: chatbot-platform/chat-service:v2.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 8001
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: MONGODB_URI
          valueFrom:
            secretKeyRef:
              name: chat-service-secrets
              key: mongodb-uri
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: chat-service-secrets
              key: redis-url
        - name: KAFKA_BROKERS
          valueFrom:
            configMapKeyRef:
              name: chat-service-config
              key: kafka-brokers
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: chat-service-secrets
              key: jwt-secret
        - name: MCP_ENGINE_URL
          valueFrom:
            configMapKeyRef:
              name: chat-service-config
              key: mcp-engine-url
        - name: SECURITY_HUB_URL
          valueFrom:
            configMapKeyRef:
              name: chat-service-config
              key: security-hub-url
        envFrom:
        - configMapRef:
            name: chat-service-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v2/health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /api/v2/health
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 3
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 10"]
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
        kubernetes.io/arch: amd64
      tolerations:
      - key: "app"
        operator: "Equal"
        value: "chatbot"
        effect: "NoSchedule"
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - chat-service
              topologyKey: kubernetes.io/hostname
---
apiVersion: v1
kind: Service
metadata:
  name: chat-service
  namespace: chatbot-platform
  labels:
    app: chat-service
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: http
    protocol: TCP
    name: http
  selector:
    app: chat-service
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chat-service-ingress
  namespace: chatbot-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.chatbot-platform.com
    secretName: chat-service-tls
  rules:
  - host: api.chatbot-platform.com
    http:
      paths:
      - path: /chat(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: chat-service
            port:
              number: 80
```

#### `helm/values.yaml`
**Purpose:** Helm chart values for configuration management  
**Usage:** Manage environment-specific configurations and deployments

```yaml
# Default values for chat-service
replicaCount: 3

image:
  repository: chatbot-platform/chat-service
  tag: "v2.0.0"
  pullPolicy: IfNotPresent

nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  name: ""
  annotations: {}

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8001"
  prometheus.io/path: "/metrics"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: false
  runAsNonRoot: true
  runAsUser: 1000

service:
  type: ClusterIP
  port: 80
  targetPort: 8001

ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.chatbot-platform.com
      paths:
        - path: /chat(/|$)(.*)
          pathType: Prefix
  tls:
    - secretName: chat-service-tls
      hosts:
        - api.chatbot-platform.com

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - chat-service
        topologyKey: kubernetes.io/hostname

# Application configuration
config:
  environment: "production"
  logLevel: "INFO"
  debug: false
  
  # External services
  mcpEngineUrl: "mcp-engine-service:50051"
  securityHubUrl: "security-hub-service:50052"
  
  # Kafka configuration
  kafkaBrokers: "kafka-0.kafka-headless:9092,kafka-1.kafka-headless:9092,kafka-2.kafka-headless:9092"
  kafkaTopicPrefix: "chatbot.platform"
  
  # Performance settings
  maxConnectionsMongo: 100
  maxConnectionsRedis: 50
  requestTimeoutMs: 30000
  
  # CORS settings
  allowedOrigins: "https://app.chatbot-platform.com,https://admin.chatbot-platform.com"

# Secrets (to be provided via external secret management)
secrets:
  mongodbUri: ""
  redisUrl: ""
  jwtSecretKey: ""
  
# Database configuration
mongodb:
  enabled: false  # Use external MongoDB
  
redis:
  enabled: false  # Use external Redis

# Monitoring
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring
    interval: 30s
    path: /metrics

# Health checks
healthCheck:
  enabled: true
  liveness:
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  readiness:
    initialDelaySeconds: 5
    periodSeconds: 5
    timeoutSeconds: 5
    failureThreshold: 3

# Pod Disruption Budget
podDisruptionBudget:
  enabled: true
  minAvailable: 2

# Network policies
networkPolicy:
  enabled: true
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: ingress-nginx
      ports:
      - protocol: TCP
        port: 8001
    - from:
      - namespaceSelector:
          matchLabels:
            name: monitoring
      ports:
      - protocol: TCP
        port: 8001
  egress:
    - to:
      - namespaceSelector:
          matchLabels:
            name: database
      ports:
      - protocol: TCP
        port: 27017  # MongoDB
      - protocol: TCP
        port: 6379   # Redis
    - to:
      - namespaceSelector:
          matchLabels:
            name: messaging
      ports:
      - protocol: TCP
        port: 9092   # Kafka
```

#### `.github/workflows/ci-cd.yml`
**Purpose:** GitHub Actions CI/CD pipeline for automated testing and deployment  
**Usage:** Automated building, testing, and deployment on code changes

```yaml
name: Chat Service CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: chatbot-platform/chat-service

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      mongodb:
        image: mongo:7
        ports:
          - 27017:27017
        options: >-
          --health-cmd "mongosh --eval 'db.runCommand({ping: 1})'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U postgres"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt

    - name: Lint with flake8
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

    - name: Format check with black
      run: black --check src/

    - name: Import sort check with isort
      run: isort --check-only src/

    - name: Type check with mypy
      run: mypy src/

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src/ --cov-report=xml --cov-report=html
      env:
        MONGODB_URI: mongodb://localhost:27017/test_db
        REDIS_URL: redis://localhost:6379/1
        POSTGRESQL_URI: postgresql://postgres:postgres@localhost:5432/test_db

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        MONGODB_URI: mongodb://localhost:27017/test_db
        REDIS_URL: redis://localhost:6379/1
        POSTGRESQL_URI: postgresql://postgres:postgres@localhost:5432/test_db

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json

    - name: Run Safety check
      run: |
        pip install safety
        safety check --json

    - name: Run Semgrep
      uses: returntocorp/semgrep-action@v1
      with:
        config: >-
          p/security-audit
          p/secrets
          p/python

  build:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.event_name != 'pull_request'
    
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}

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
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./docker/Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}

    - name: Install Helm
      uses: azure/setup-helm@v3
      with:
        version: '3.12.0'

    - name: Deploy to staging
      run: |
        helm upgrade --install chat-service-staging ./helm \
          --namespace staging \
          --create-namespace \
          --values ./helm/values-staging.yaml \
          --set image.tag=${{ github.sha }} \
          --wait --timeout=10m

    - name: Run smoke tests
      run: |
        kubectl wait --for=condition=ready pod -l app=chat-service -n staging --timeout=300s
        curl -f https://staging-api.chatbot-platform.com/api/v2/health

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}

    - name: Install Helm
      uses: azure/setup-helm@v3
      with:
        version: '3.12.0'

    - name: Deploy to production
      run: |
        helm upgrade --install chat-service ./helm \
          --namespace production \
          --create-namespace \
          --values ./helm/values-production.yaml \
          --set image.tag=${{ github.event.release.tag_name }} \
          --wait --timeout=15m

    - name: Run production health checks
      run: |
        kubectl wait --for=condition=ready pod -l app=chat-service -n production --timeout=300s
        curl -f https://api.chatbot-platform.com/api/v2/health

    - name: Notify deployment success
      uses: 8398a7/action-slack@v3
      with:
        status: success
        text: "Chat Service ${{ github.event.release.tag_name }} deployed to production successfully!"
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

  performance-test:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Install k6
      run: |
        sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install k6

    - name: Run performance tests
      run: |
        k6 run tests/performance/load_test.js \
          --env BASE_URL=https://staging-api.chatbot-platform.com

    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: performance-results.json
```

#### `monitoring/prometheus/prometheus.yml`
**Purpose:** Prometheus configuration for monitoring the chat service  
**Usage:** Collect metrics from chat service and other components

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'chatbot-platform'
    environment: 'production'

rule_files:
  - "alert-rules.yml"
  - "recording-rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Chat Service metrics
  - job_name: 'chat-service'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - chatbot-platform
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
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name

  # Node exporter
  - job_name: 'node-exporter'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics

  # MongoDB metrics
  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb-exporter:9216']

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Kafka metrics
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka-exporter:9308']

  # PostgreSQL metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

#### `monitoring/prometheus/alert-rules.yml`
**Purpose:** Prometheus alerting rules for production monitoring  
**Usage:** Define alerts for system health, performance, and business metrics

```yaml
groups:
  - name: chat-service.rules
    rules:
      # High-level service health
      - alert: ChatServiceDown
        expr: up{job="chat-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Chat Service is down"
          description: "Chat Service has been down for more than 1 minute"

      - alert: ChatServiceHighErrorRate
        expr: rate(http_requests_total{job="chat-service",status=~"5.."}[5m]) / rate(http_requests_total{job="chat-service"}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in Chat Service"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      # Performance alerts
      - alert: ChatServiceHighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="chat-service"}[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency in Chat Service"
          description: "95th percentile latency is {{ $value }}s"

      - alert: ChatServiceHighMemoryUsage
        expr: container_memory_usage_bytes{pod=~"chat-service-.*"} / container_spec_memory_limit_bytes > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage in Chat Service"
          description: "Memory usage is {{ $value | humanizePercentage }} of limit"

      - alert: ChatServiceHighCPUUsage
        expr: rate(container_cpu_usage_seconds_total{pod=~"chat-service-.*"}[5m]) / container_spec_cpu_quota * container_spec_cpu_period > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage in Chat Service"
          description: "CPU usage is {{ $value | humanizePercentage }} of limit"

      # Business metrics
      - alert: MessageProcessingStalled
        expr: rate(chat_messages_processed_total[5m]) == 0 and rate(chat_messages_received_total[5m]) > 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Message processing appears to be stalled"
          description: "Messages are being received but not processed"

      - alert: ConversationResponseTimeHigh
        expr: histogram_quantile(0.95, rate(chat_conversation_response_time_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Conversation response time is high"
          description: "95th percentile response time is {{ $value }}s"

      # Database connectivity
      - alert: MongoDBConnectionsHigh
        expr: mongodb_connections{type="current"} / mongodb_connections{type="available"} > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "MongoDB connection usage is high"
          description: "{{ $value | humanizePercentage }} of available connections are in use"

      - alert: RedisConnectionsHigh
        expr: redis_connected_clients / redis_config_maxclients > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis connection usage is high"
          description: "{{ $value | humanizePercentage }} of max connections are in use"

      # External service dependencies
      - alert: MCPEngineUnreachable
        expr: up{job="mcp-engine"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "MCP Engine is unreachable"
          description: "Cannot reach MCP Engine service"

      - alert: SecurityHubUnreachable
        expr: up{job="security-hub"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Security Hub is unreachable"
          description: "Cannot reach Security Hub service"

  - name: infrastructure.rules
    rules:
      # Kubernetes cluster health
      - alert: KubernetesNodeNotReady
        expr: kube_node_status_condition{condition="Ready",status="true"} == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Kubernetes node is not ready"
          description: "Node {{ $labels.node }} has been not ready for more than 5 minutes"

      - alert: KubernetesPodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.namespace }}/{{ $labels.pod }} is restarting frequently"

      # Disk space
      - alert: DiskSpaceHigh
        expr: (node_filesystem_size_bytes{fstype!="tmpfs"} - node_filesystem_free_bytes{fstype!="tmpfs"}) / node_filesystem_size_bytes{fstype!="tmpfs"} > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Disk space usage is high"
          description: "Disk usage is {{ $value | humanizePercentage }} on {{ $labels.device }}"
```

#### `scripts/deploy.sh`
**Purpose:** Production deployment script with safety checks and rollback capability  
**Usage:** Automated deployment with pre/post deployment validation

```bash
#!/bin/bash

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-production}"
CHART_PATH="${CHART_PATH:-./helm}"
VALUES_FILE="${VALUES_FILE:-./helm/values-production.yaml}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
TIMEOUT="${TIMEOUT:-900s}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/deployment_status.json
}

trap cleanup EXIT

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if kubectl is available and configured
    if ! command -v kubectl &> /dev/null; then
        error_exit "kubectl is not installed or not in PATH"
    fi
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        error_exit "helm is not installed or not in PATH"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warn "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Verify chart and values files exist
    if [[ ! -f "$CHART_PATH/Chart.yaml" ]]; then
        error_exit "Chart.yaml not found in $CHART_PATH"
    fi
    
    if [[ ! -f "$VALUES_FILE" ]]; then
        error_exit "Values file not found: $VALUES_FILE"
    fi
    
    # Check if this is a production deployment
    if [[ "$NAMESPACE" == "production" ]]; then
        log_warn "This is a PRODUCTION deployment!"
        if [[ "${FORCE_PRODUCTION:-}" != "true" ]]; then
            read -p "Are you sure you want to proceed? (yes/no): " -r
            if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
                log_info "Deployment cancelled by user"
                exit 0
            fi
        fi
    fi
    
    log_success "Pre-deployment checks passed"
}

# Get current deployment info
get_current_deployment() {
    log_info "Getting current deployment information..."
    
    if kubectl get deployment chat-service -n "$NAMESPACE" &> /dev/null; then
        CURRENT_IMAGE=$(kubectl get deployment chat-service -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')
        CURRENT_REPLICAS=$(kubectl get deployment chat-service -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        
        log_info "Current image: $CURRENT_IMAGE"
        log_info "Current replicas: $CURRENT_REPLICAS"
        
        # Store current state for potential rollback
        kubectl get deployment chat-service -n "$NAMESPACE" -o json > /tmp/deployment_backup.json
    else
        log_info "No existing deployment found"
        CURRENT_IMAGE=""
        CURRENT_REPLICAS=""
    fi
}

# Deploy application
deploy_application() {
    log_info "Starting deployment..."
    
    # Build helm command
    HELM_CMD="helm upgrade --install chat-service $CHART_PATH"
    HELM_CMD="$HELM_CMD --namespace $NAMESPACE"
    HELM_CMD="$HELM_CMD --values $VALUES_FILE"
    HELM_CMD="$HELM_CMD --set image.tag=$IMAGE_TAG"
    HELM_CMD="$HELM_CMD --timeout $TIMEOUT"
    HELM_CMD="$HELM_CMD --wait"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        HELM_CMD="$HELM_CMD --dry-run"
        log_info "Running in dry-run mode"
    fi
    
    # Add debug output for non-production
    if [[ "$NAMESPACE" != "production" ]]; then
        HELM_CMD="$HELM_CMD --debug"
    fi
    
    log_info "Executing: $HELM_CMD"
    
    # Execute deployment
    if eval "$HELM_CMD"; then
        log_success "Helm deployment completed successfully"
    else
        error_exit "Helm deployment failed"
    fi
}

# Post-deployment validation
post_deployment_validation() {
    if [[ "$DRY_RUN" == "true" || "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping post-deployment validation"
        return 0
    fi
    
    log_info "Running post-deployment validation..."
    
    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    if ! kubectl wait --for=condition=ready pod -l app=chat-service -n "$NAMESPACE" --timeout=300s; then
        error_exit "Pods failed to become ready within timeout"
    fi
    
    # Check deployment status
    if ! kubectl rollout status deployment/chat-service -n "$NAMESPACE" --timeout=300s; then
        error_exit "Deployment rollout failed"
    fi
    
    # Health check
    log_info "Running health checks..."
    sleep 10  # Give the service time to fully start
    
    # Port forward for health check (if not using ingress)
    if [[ "$NAMESPACE" != "production" ]]; then
        kubectl port-forward svc/chat-service 8080:80 -n "$NAMESPACE" &
        PORT_FORWARD_PID=$!
        sleep 5
        
        if curl -f http://localhost:8080/api/v2/health &> /dev/null; then
            log_success "Health check passed"
        else
            log_error "Health check failed"
            kill $PORT_FORWARD_PID || true
            return 1
        fi
        
        kill $PORT_FORWARD_PID || true
    else
        # For production, use the actual ingress
        if curl -f https://api.chatbot-platform.com/api/v2/health &> /dev/null; then
            log_success "Production health check passed"
        else
            log_error "Production health check failed"
            return 1
        fi
    fi
    
    log_success "Post-deployment validation completed"
}

# Rollback function
rollback_deployment() {
    log_error "Deployment validation failed, initiating rollback..."
    
    if [[ -f /tmp/deployment_backup.json && "$CURRENT_IMAGE" != "" ]]; then
        log_info "Rolling back to previous deployment..."
        kubectl apply -f /tmp/deployment_backup.json
        kubectl rollout status deployment/chat-service -n "$NAMESPACE" --timeout=300s
        log_success "Rollback completed"
    else
        log_warn "No previous deployment found, cannot rollback automatically"
        log_warn "You may need to manually fix the deployment or delete it"
    fi
}

# Main deployment function
main() {
    log_info "Starting Chat Service deployment to $NAMESPACE"
    log_info "Image tag: $IMAGE_TAG"
    log_info "Values file: $VALUES_FILE"
    
    pre_deployment_checks
    get_current_deployment
    deploy_application
    
    if ! post_deployment_validation; then
        if [[ "$DRY_RUN" != "true" ]]; then
            rollback_deployment
        fi
        error_exit "Deployment failed validation"
    fi
    
    log_success "Deployment completed successfully!"
    
    # Display deployment information
    log_info "Deployment Summary:"
    echo "  Namespace: $NAMESPACE"
    echo "  Image: chatbot-platform/chat-service:$IMAGE_TAG"
    if [[ "$CURRENT_IMAGE" != "" ]]; then
        echo "  Previous Image: $CURRENT_IMAGE"
    fi
    
    # Show running pods
    log_info "Running pods:"
    kubectl get pods -l app=chat-service -n "$NAMESPACE"
    
    # Show service endpoints
    log_info "Service endpoints:"
    kubectl get svc chat-service -n "$NAMESPACE"
    
    if [[ "$NAMESPACE" == "production" ]]; then
        log_info "Production URL: https://api.chatbot-platform.com/api/v2/health"
    fi
}

# Run main function
main "$@"
```

#### `docs/runbooks/incident-response.md`
**Purpose:** Incident response procedures and troubleshooting guides  
**Usage:** Operations team reference for handling production incidents

```markdown
# Chat Service Incident Response Runbook

## 🚨 Emergency Contacts

- **On-Call Engineer**: [Pager/Phone]
- **Platform Team Lead**: [Contact]
- **DevOps Team**: [Slack Channel: #devops-alerts]
- **Management Escalation**: [Contact]

---

## 🔍 Initial Response (First 5 Minutes)

### 1. Acknowledge the Alert
- Acknowledge in monitoring system (PagerDuty/Slack)
- Join incident channel: `#incident-chat-service`
- Post initial status: "Investigating chat service issue"

### 2. Quick Health Assessment
```bash
# Check service status
kubectl get pods -n production -l app=chat-service

# Check recent deployments
helm history chat-service -n production

# Check basic health endpoint
curl -f https://api.chatbot-platform.com/api/v2/health
```

### 3. Initial Impact Assessment
- Check Grafana dashboard: "Chat Service Overview"
- Verify error rates and response times
- Check user-facing impact (customer reports)

---

## 🔧 Common Incident Scenarios

### Scenario 1: Service Completely Down

**Symptoms:**
- All health checks failing
- 0 ready pods
- 503 errors from load balancer

**Investigation Steps:**
```bash
# Check pod status
kubectl describe pods -n production -l app=chat-service

# Check recent events
kubectl get events -n production --sort-by='.lastTimestamp'

# Check deployment status
kubectl rollout status deployment/chat-service -n production
```

**Resolution Steps:**
1. If recent deployment: Rollback immediately
   ```bash
   helm rollback chat-service -n production
   ```

2. If infrastructure issue: Scale up manually
   ```bash
   kubectl scale deployment chat-service --replicas=5 -n production
   ```

3. If persistent: Check dependencies (MongoDB, Redis, Kafka)

### Scenario 2: High Error Rate (5xx Errors)

**Symptoms:**
- Error rate > 5%
- Pods running but returning errors
- Increased response times

**Investigation Steps:**
```bash
# Check application logs
kubectl logs -n production -l app=chat-service --tail=100

# Check resource usage
kubectl top pods -n production -l app=chat-service

# Check external dependencies
curl -f http://mongodb-service:27017/
curl -f http://redis-service:6379/ping
```

**Resolution Steps:**
1. Scale up if resource constrained
2. Restart pods if memory leaks suspected
3. Check and restart dependencies if needed

### Scenario 3: High Latency

**Symptoms:**
- P95 response time > 2s
- Timeout errors
- Queue backlog growing

**Investigation Steps:**
```bash
# Check database performance
# MongoDB slow queries
kubectl exec -it mongodb-0 -- mongosh --eval "db.currentOp()"

# Redis performance
kubectl exec -it redis-0 -- redis-cli --latency-history

# Check Kafka lag
kubectl exec -it kafka-0 -- kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group chat-service-consumers
```

### Scenario 4: Memory/CPU Issues

**Symptoms:**
- OOMKilled pods
- High CPU usage
- Performance degradation

**Investigation Steps:**
```bash
# Check resource usage
kubectl top pods -n production -l app=chat-service

# Check resource limits
kubectl describe deployment chat-service -n production

# Check for memory leaks
kubectl logs -n production -l app=chat-service | grep -i "memory\|oom"
```

---

## 📊 Monitoring & Diagnostics

### Key Metrics to Monitor
- **Request Rate**: `rate(http_requests_total[5m])`
- **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m])`
- **Response Time**: `histogram_quantile(0.95, http_request_duration_seconds_bucket)`
- **Pod Health**: `up{job="chat-service"}`

### Grafana Dashboards
- **Chat Service Overview**: Primary operational dashboard
- **Performance Metrics**: Detailed performance analysis
- **Business Metrics**: Message volume and conversation analytics

### Log Analysis
```bash
# Search for errors
kubectl logs -n production -l app=chat-service | grep -i error

# Search for specific patterns
kubectl logs -n production -l app=chat-service | grep -E "(timeout|connection.*failed|database.*error)"

# Check specific time range (if using log aggregation)
curl -X GET "http://elasticsearch:9200/chat-service-*/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        {"range": {"@timestamp": {"gte": "now-1h"}}},
        {"match": {"level": "ERROR"}}
      ]
    }
  }
}'
```

---

## 🔄 Recovery Procedures

### Immediate Recovery Actions

1. **Service Restart**
   ```bash
   kubectl rollout restart deployment/chat-service -n production
   ```

2. **Scale Up (if resource constrained)**
   ```bash
   kubectl scale deployment chat-service --replicas=6 -n production
   ```

3. **Emergency Rollback**
   ```bash
   helm rollback chat-service -n production
   kubectl rollout status deployment/chat-service -n production
   ```

### Database Recovery

1. **MongoDB Issues**
   ```bash
   # Check replica set status
   kubectl exec -it mongodb-0 -- mongosh --eval "rs.status()"
   
   # Force primary election if needed
   kubectl exec -it mongodb-0 -- mongosh --eval "rs.stepDown()"
   ```

2. **Redis Issues**
   ```bash
   # Check Redis status
   kubectl exec -it redis-0 -- redis-cli ping
   
   # Restart Redis if needed
   kubectl delete pod redis-0 -n database
   ```

### Data Consistency Checks

After recovery, verify data consistency:
```bash
# Check recent conversations
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
     "https://api.chatbot-platform.com/api/v2/chat/conversations?limit=10"

# Verify event processing
kubectl logs -n production -l app=chat-service | grep "Event processed successfully" | tail -10
```

---

## 📝 Communication Templates

### Initial Incident Notification
```
🚨 INCIDENT: Chat Service experiencing issues
Status: Investigating
Impact: [Customer-facing/Internal]
ETA: Investigating, updates every 15 minutes
Incident Channel: #incident-chat-service
```

### Status Update Template
```
📊 UPDATE: Chat Service Incident
Time: [HH:MM UTC]
Status: [Investigating/Mitigating/Resolved]
Actions Taken: [Brief description]
Next Update: [Time]
```

### Resolution Notification
```
✅ RESOLVED: Chat Service Incident
Duration: [X minutes]
Root Cause: [Brief description]
Actions Taken: [Summary]
Follow-up: Post-mortem scheduled for [date]
```

---

## 📋 Post-Incident Procedures

### Immediate (within 2 hours)
1. Ensure service is stable
2. Document timeline of events
3. Preserve relevant logs and metrics
4. Update incident channel with resolution

### Short-term (within 24 hours)
1. Schedule post-mortem meeting
2. Create incident report template
3. Identify immediate preventive measures
4. Update monitoring/alerting if needed

### Long-term (within 1 week)
1. Conduct blameless post-mortem
2. Create action items for prevention
3. Update runbooks with lessons learned
4. Share learnings with broader team

---

## 🔍 Escalation Procedures

### Level 1: On-Call Engineer (0-15 minutes)
- Initial response and basic troubleshooting
- Service restart and scaling actions
- Simple rollback procedures

### Level 2: Platform Team Lead (15-30 minutes)
- Complex troubleshooting
- Database/infrastructure issues
- Architectural decisions

### Level 3: Management/VP Engineering (30+ minutes)
- Customer communication required
- Extended outage (>1 hour)
- Security incidents
- Data loss scenarios

---

## 📞 External Dependencies

### MCP Engine Issues
- Contact: MCP Team (#mcp-engine-support)
- Escalation: MCP Team Lead
- Fallback: Enable rule-based responses

### Security Hub Issues
- Contact: Security Team (#security-support)
- Escalation: Security Team Lead
- Fallback: Disable authentication temporarily (emergency only)

### Infrastructure (MongoDB, Redis, Kafka)
- Contact: Infrastructure Team (#infra-support)
- Escalation: Infrastructure Team Lead
- Provider Support: AWS/GCP support ticket

---

## 🔐 Emergency Access

### Break-glass Procedures
```bash
# Emergency admin access
kubectl config use-context production-emergency
kubectl auth can-i "*" "*" --as=system:admin

# Emergency database access
kubectl port-forward svc/mongodb 27017:27017 -n database
mongosh "mongodb://localhost:27017/chatbot"
```

### Emergency Contacts
- **AWS Support**: [Phone/Email]
- **MongoDB Atlas**: [Phone/Email]
- **Confluent Support**: [Phone/Email]
```

---

## 🔧 Technologies Used
- **Docker**: Containerization platform
- **Kubernetes**: Container orchestration
- **Helm**: Package manager for Kubernetes
- **Prometheus**: Monitoring and alerting
- **Grafana**: Metrics visualization
- **GitHub Actions**: CI/CD automation

---

## ⚠️ Key Considerations

### Security
- Container image scanning and vulnerability management
- Kubernetes RBAC and network policies
- Secret management with external providers
- Runtime security monitoring

### Scalability
- Horizontal pod autoscaling based on metrics
- Resource limits and requests optimization
- Database connection pooling and optimization
- CDN and edge caching strategies

### Reliability
- Multi-AZ deployment for high availability
- Pod disruption budgets and graceful shutdowns
- Circuit breakers and retry logic
- Comprehensive health checks and readiness probes

### Observability
- Distributed tracing with Jaeger/Zipkin
- Structured logging with correlation IDs
- Business metrics and SLA monitoring
- Automated alerting and incident response

---

## 🎯 Success Criteria
- [ ] Application is containerized and optimized
- [ ] Kubernetes manifests deploy successfully
- [ ] CI/CD pipeline automates deployment process
- [ ] Monitoring and alerting are comprehensive
- [ ] Performance meets production requirements
- [ ] Security hardening is implemented
- [ ] Documentation and runbooks are complete
- [ ] Disaster recovery procedures are tested

---

## 🎉 Project Completion Summary

### 4-Month Implementation Roadmap Complete!

**Total Implementation:**
- **10 Phases** covering complete system architecture
- **18 Steps** from foundation to production deployment
- **~15,000 lines of code** across all components
- **Comprehensive testing** with >90% coverage target
- **Production-ready deployment** with monitoring and automation

**Key Achievements:**
1. ✅ **Robust Architecture**: Microservices with event-driven design
2. ✅ **Scalable Data Layer**: MongoDB, Redis, and PostgreSQL integration
3. ✅ **Multi-Channel Support**: WhatsApp, Slack, Web, and extensible framework
4. ✅ **Real-time Processing**: Kafka-based event streaming
5. ✅ **External Integrations**: gRPC clients and webhook handling
6. ✅ **Production Deployment**: Kubernetes with full DevOps pipeline
7. ✅ **Comprehensive Testing**: Unit, integration, and E2E tests
8. ✅ **Monitoring & Observability**: Prometheus, Grafana, and alerting
9. ✅ **Security & Compliance**: Authentication, authorization, and audit trails
10. ✅ **Documentation**: Complete technical and operational documentation

**Ready for Production:** The Chat Service is now ready to handle enterprise-scale conversations with high availability, security, and performance! 🚀

