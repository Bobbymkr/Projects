# 🚀 Deployment Guide - Adaptive Traffic Signal Control System

This comprehensive deployment guide covers all aspects of deploying the Adaptive Traffic Signal Control System in production environments.

## 📋 **Table of Contents**

- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Edge Computing Deployment](#edge-computing-deployment)
- [Configuration Management](#configuration-management)
- [Security Considerations](#security-considerations)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)

---

## 🏭 **Production Deployment**

### **Prerequisites**

#### **System Requirements**
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Windows Server 2019+
- **Python**: 3.9+ with pip
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 100GB+ available space
- **Network**: Stable internet connection for updates
- **GPU**: Optional but recommended for training workloads

#### **Hardware Specifications by Deployment Size**

| **Scale** | **CPU** | **RAM** | **Storage** | **Network** | **GPU** |
|-----------|---------|---------|-------------|-------------|---------|
| **Small (1-5 intersections)** | 4 cores | 16GB | 100GB SSD | 100 Mbps | Optional |
| **Medium (6-20 intersections)** | 8 cores | 32GB | 500GB SSD | 1 Gbps | Recommended |
| **Large (21+ intersections)** | 16+ cores | 64GB+ | 1TB+ SSD | 10 Gbps | Required |

### **Installation Steps**

#### **1. System Preparation**

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.9 python3.9-venv python3.9-dev \
    build-essential curl wget git htop tmux \
    postgresql-client redis-tools nginx

# Create application user
sudo useradd -m -s /bin/bash trafficctl
sudo usermod -aG sudo trafficctl

# Switch to application user
sudo su - trafficctl
```

#### **2. Application Installation**

```bash
# Clone production repository
git clone https://github.com/your-org/adaptive-traffic.git /opt/adaptive-traffic
cd /opt/adaptive-traffic

# Create production environment
python3.9 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install --upgrade pip
pip install -r requirements-prod.txt

# Set production environment variables
cat > .env << EOF
ENVIRONMENT=production
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@localhost:5432/traffic_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=$(openssl rand -hex 32)
EOF
```

#### **3. Database Setup**

```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE traffic_db;
CREATE USER traffic_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE traffic_db TO traffic_user;
\q
EOF

# Initialize database schema
source venv/bin/activate
python scripts/init_database.py
```

#### **4. Redis Configuration**

```bash
# Install and configure Redis
sudo apt install -y redis-server

# Configure Redis for production
sudo tee /etc/redis/redis.conf << EOF
bind 127.0.0.1
port 6379
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
EOF

# Restart Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server
```

### **Service Configuration**

#### **Systemd Service Files**

```bash
# Main application service
sudo tee /etc/systemd/system/adaptive-traffic.service << EOF
[Unit]
Description=Adaptive Traffic Control System
After=network.target postgresql.service redis.service

[Service]
Type=forking
User=trafficctl
Group=trafficctl
WorkingDirectory=/opt/adaptive-traffic
Environment=PATH=/opt/adaptive-traffic/venv/bin
ExecStart=/opt/adaptive-traffic/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# API service
sudo tee /etc/systemd/system/traffic-api.service << EOF
[Unit]
Description=Traffic Control API
After=network.target adaptive-traffic.service

[Service]
Type=exec
User=trafficctl
Group=trafficctl
WorkingDirectory=/opt/adaptive-traffic
Environment=PATH=/opt/adaptive-traffic/venv/bin
ExecStart=/opt/adaptive-traffic/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable adaptive-traffic traffic-api
sudo systemctl start adaptive-traffic traffic-api
```

---

## ☁️ **Cloud Deployment**

### **AWS Deployment**

#### **Infrastructure as Code (Terraform)**

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC Configuration
resource "aws_vpc" "traffic_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "adaptive-traffic-vpc"
    Environment = var.environment
  }
}

# Public Subnet
resource "aws_subnet" "public_subnet" {
  vpc_id                  = aws_vpc.traffic_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "adaptive-traffic-public-subnet"
  }
}

# EC2 Instance for Main Application
resource "aws_instance" "traffic_control" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_pair_name
  subnet_id     = aws_subnet.public_subnet.id
  
  vpc_security_group_ids = [aws_security_group.traffic_sg.id]

  user_data = base64encode(templatefile("user_data.sh", {
    db_endpoint = aws_db_instance.traffic_db.endpoint
    redis_endpoint = aws_elasticache_cluster.traffic_cache.cache_nodes[0].address
  }))

  tags = {
    Name = "adaptive-traffic-control"
    Environment = var.environment
  }
}

# RDS Database
resource "aws_db_instance" "traffic_db" {
  identifier = "adaptive-traffic-db"
  
  engine         = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  
  db_name  = "traffic_db"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.db_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.traffic_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  
  tags = {
    Name = "adaptive-traffic-database"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "traffic_cache" {
  cluster_id           = "adaptive-traffic-cache"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis6.x"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.traffic_cache_subnet_group.name
  security_group_ids   = [aws_security_group.cache_sg.id]
}
```

#### **Kubernetes Deployment**

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adaptive-traffic
  namespace: traffic-control
spec:
  replicas: 3
  selector:
    matchLabels:
      app: adaptive-traffic
  template:
    metadata:
      labels:
        app: adaptive-traffic
    spec:
      containers:
      - name: traffic-control
        image: adaptive-traffic:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: traffic-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: traffic-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: adaptive-traffic-service
  namespace: traffic-control
spec:
  selector:
    app: adaptive-traffic
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: traffic-control-ingress
  namespace: traffic-control
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - traffic-control.yourdomain.com
    secretName: traffic-control-tls
  rules:
  - host: traffic-control.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: adaptive-traffic-service
            port:
              number: 80
```

### **Docker Deployment**

#### **Production Dockerfile**

```dockerfile
# Dockerfile.prod
FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash trafficctl
USER trafficctl
WORKDIR /home/trafficctl

# Copy application code
COPY --chown=trafficctl:trafficctl . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "src/main.py"]
```

#### **Docker Compose for Production**

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://traffic_user:${DB_PASSWORD}@db:5432/traffic_db
      - REDIS_URL=redis://redis:6379/0
      - ENVIRONMENT=production
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/home/trafficctl/logs
      - ./models:/home/trafficctl/models
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=traffic_db
      - POSTGRES_USER=traffic_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

---

## 📱 **Edge Computing Deployment**

### **NVIDIA Jetson Deployment**

#### **Setup for Edge Devices**

```bash
# Install JetPack SDK
sudo apt update
sudo apt install -y nvidia-jetpack

# Install Python dependencies for edge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python-headless ultralytics

# Configure for low-power mode
sudo nvpmodel -m 0  # Maximum performance
sudo jetson_clocks   # Maximum clock speeds

# Setup edge-specific configuration
cat > edge_config.json << EOF
{
  "edge_mode": true,
  "inference_only": true,
  "model_path": "/opt/models/edge_optimized.onnx",
  "batch_size": 1,
  "max_queue_length": 20,
  "power_management": {
    "enable": true,
    "mode": "balanced"
  }
}
EOF
```

#### **Edge Optimization**

```python
# scripts/optimize_for_edge.py
import torch
import onnx
from src.rl.dqn_agent import DQNAgent

def optimize_model_for_edge(model_path, output_path):
    """Optimize trained model for edge deployment."""
    
    # Load trained model
    agent = DQNAgent.load(model_path)
    
    # Convert to ONNX format for faster inference
    dummy_input = torch.randn(1, agent.state_dim)
    torch.onnx.export(
        agent.q_network,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['q_values'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Edge-optimized model saved to {output_path}")

if __name__ == "__main__":
    optimize_model_for_edge(
        "models/production_dqn.pt",
        "models/edge_optimized.onnx"
    )
```

---

## ⚙️ **Configuration Management**

### **Environment-Specific Configurations**

#### **Development Configuration**

```yaml
# configs/dev.yaml
environment: development
debug: true
log_level: DEBUG

database:
  url: sqlite:///dev_traffic.db
  echo: true

redis:
  url: redis://localhost:6379/1

training:
  episodes: 10
  save_frequency: 5
  tensorboard_enabled: true

api:
  host: 127.0.0.1
  port: 8000
  reload: true

monitoring:
  enabled: false
```

#### **Production Configuration**

```yaml
# configs/prod.yaml
environment: production
debug: false
log_level: INFO

database:
  url: ${DATABASE_URL}
  pool_size: 20
  max_overflow: 30
  pool_pre_ping: true

redis:
  url: ${REDIS_URL}
  connection_pool_size: 50

training:
  episodes: 6000
  save_frequency: 100
  tensorboard_enabled: true
  
api:
  host: 0.0.0.0
  port: 8000
  workers: 4

monitoring:
  enabled: true
  prometheus_port: 9090
  grafana_enabled: true

security:
  secret_key: ${SECRET_KEY}
  cors_origins: ["https://yourdomain.com"]
  rate_limiting:
    enabled: true
    requests_per_minute: 1000
```

### **Dynamic Configuration**

```python
# src/config/dynamic_config.py
import os
import yaml
from typing import Dict, Any
from pydantic import BaseSettings

class DynamicConfig(BaseSettings):
    """Dynamic configuration management with environment override."""
    
    environment: str = "development"
    config_file: str = None
    
    class Config:
        env_file = ".env"
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration based on environment."""
        
        if self.config_file:
            config_path = self.config_file
        else:
            config_path = f"configs/{self.environment}.yaml"
        
        # Load base configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config = self._override_with_env(config)
        
        return config
    
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables."""
        
        for key, value in os.environ.items():
            if key.startswith('TRAFFIC_'):
                config_key = key[8:].lower()  # Remove TRAFFIC_ prefix
                config[config_key] = value
        
        return config

# Usage
config_manager = DynamicConfig()
app_config = config_manager.load_config()
```

---

## 🔒 **Security Considerations**

### **Network Security**

```bash
# Firewall configuration
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow specific ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 8000  # Application

# Database access (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 5432
```

### **SSL/TLS Configuration**

```nginx
# nginx/nginx.conf
server {
    listen 443 ssl http2;
    server_name traffic-control.yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name traffic-control.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

### **API Security**

```python
# src/api/security.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
```

---

## 📊 **Monitoring & Observability**

### **Prometheus Configuration**

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'adaptive-traffic'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### **Grafana Dashboard**

```json
{
  "dashboard": {
    "title": "Adaptive Traffic Control System",
    "panels": [
      {
        "title": "System Performance",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Queue Lengths",
        "targets": [
          {
            "expr": "avg_queue_length",
            "legendFormat": "Average Queue Length"
          }
        ]
      },
      {
        "title": "Response Times",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. High Memory Usage**
```bash
# Check memory usage
free -h
top -p $(pgrep -f adaptive-traffic)

# Optimize memory settings
export MALLOC_ARENA_MAX=2
ulimit -v 4194304  # Limit virtual memory to 4GB
```

#### **2. Database Connection Issues**
```python
# Database health check
from sqlalchemy import create_engine

def check_database_connection(database_url):
    try:
        engine = create_engine(database_url)
        connection = engine.connect()
        result = connection.execute("SELECT 1")
        connection.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
```

#### **3. Performance Issues**
```bash
# Profile application performance
python -m cProfile -o profile.out src/main.py

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.out')
p.sort_stats('cumulative').print_stats(20)
"
```

This deployment guide provides comprehensive instructions for deploying the Adaptive Traffic Signal Control System across various environments and scales.