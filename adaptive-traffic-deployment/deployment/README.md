# Deployment Configuration
## Phase 2: Horizontal Scaling Setup

This directory contains deployment configurations for horizontal scaling and production deployment.

---

## 🚀 Deployment Options

### 1. Docker Compose (Development/Testing)

**Single-node deployment with all services**

```bash
# Start all services
docker-compose -f deployment/docker/docker-compose.yml up -d

# Scale API instances
docker-compose -f deployment/docker/docker-compose.yml up -d --scale api=3

# View logs
docker-compose -f deployment/docker/docker-compose.yml logs -f api

# Stop services
docker-compose -f deployment/docker/docker-compose.yml down
```

**Services:**
- API server (scalable)
- Redis (caching/rate limiting)
- PostgreSQL (database)
- Prometheus (monitoring)
- Grafana (dashboards)
- Nginx (load balancer)

### 2. Kubernetes (Production)

**Multi-node, production-ready deployment**

```bash
# Deploy API
kubectl apply -f deployment/kubernetes/api-deployment.yaml

# Scale manually
kubectl scale deployment adaptive-traffic-api --replicas=5

# Check status
kubectl get pods -l app=adaptive-traffic-api

# View logs
kubectl logs -l app=adaptive-traffic-api -f
```

**Features:**
- Horizontal Pod Autoscaler (HPA)
- Auto-scaling based on CPU/memory
- Health checks and probes
- Resource limits and requests

### 3. Docker (Standalone)

**Simple container deployment**

```bash
# Build image
docker build -f deployment/docker/Dockerfile -t adaptive-traffic-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -e ENABLE_REDIS=true \
  -e REDIS_HOST=redis \
  -e ENABLE_CACHING=true \
  adaptive-traffic-api
```

---

## 📊 Scaling Strategies

### Horizontal Scaling

The API is designed to be stateless and horizontally scalable:

1. **Shared State**: Redis for caching and rate limiting
2. **Database**: PostgreSQL with connection pooling
3. **Load Balancer**: Nginx with health checks
4. **Auto-scaling**: Kubernetes HPA

### Scaling API Instances

**Docker Compose:**
```bash
docker-compose up -d --scale api=5
```

**Kubernetes:**
```bash
kubectl scale deployment adaptive-traffic-api --replicas=5
```

Or let HPA handle it automatically based on CPU/memory usage.

---

## 🔧 Configuration

### Environment Variables

Key environment variables for production:

```bash
ENVIRONMENT=production
ENABLE_REDIS=true
REDIS_HOST=redis
REDIS_PORT=6379
ENABLE_CACHING=true
ENABLE_RATE_LIMITING=true
DATABASE_URL=postgresql://user:pass@host:5432/db
ENABLE_PROMETHEUS=true
ENABLE_SENTRY=true
SENTRY_DSN=your-sentry-dsn
```

### Load Balancer Configuration

Nginx configuration supports:
- Least connections load balancing
- Health checks
- Rate limiting
- SSL termination (add certificates)
- Gzip compression

---

## 📈 Performance Considerations

### Resource Requirements

**Per API Instance:**
- CPU: 250m (request) - 500m (limit)
- Memory: 256Mi (request) - 512Mi (limit)

**Redis:**
- Memory: 256MB (configurable)
- Eviction policy: allkeys-lru

**PostgreSQL:**
- Connection pool: 10 per instance
- Max connections: 100

### Scaling Recommendations

- **Start with**: 3 instances
- **Scale to**: 5-10 instances for high load
- **Auto-scale**: Based on CPU (70%) and Memory (80%)
- **Load balancer**: Nginx with least_conn algorithm

---

## 🔍 Monitoring

### Health Checks

All deployments include health checks:
- Liveness probe: `/health`
- Readiness probe: `/health`
- Startup probe: `/health`

### Metrics

Prometheus scrapes metrics from:
- API instances: `/metrics`
- System metrics: node exporter
- Redis: redis exporter (optional)

### Dashboards

Grafana dashboards available:
- API Performance
- Traffic Metrics
- System Health

---

## 🚨 Production Checklist

Before deploying to production:

- [ ] Update environment variables
- [ ] Configure SSL/TLS certificates
- [ ] Set up database backups
- [ ] Configure Redis persistence
- [ ] Set up monitoring alerts
- [ ] Configure log aggregation
- [ ] Test load balancing
- [ ] Test auto-scaling
- [ ] Set resource limits appropriately
- [ ] Configure security policies

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Nginx Load Balancing](https://nginx.org/en/docs/http/load_balancing.html)

---

**Status**: Phase 2 - Horizontal Scaling Ready ✅

