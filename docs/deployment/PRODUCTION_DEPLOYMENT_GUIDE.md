# Production Deployment Guide

**Week 13: Complete Production Deployment Guide**

---

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- Docker registry access
- Helm 3.x (optional)

---

## Deployment Steps

### 1. Build and Push Docker Image

```bash
docker build -t adaptive-traffic-api:latest -f deployment/docker/Dockerfile .
docker tag adaptive-traffic-api:latest registry.example.com/adaptive-traffic-api:latest
docker push registry.example.com/adaptive-traffic-api:latest
```

### 2. Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f deployment/kubernetes/

# Wait for deployment
kubectl rollout status deployment/adaptive-traffic-api
```

### 3. Verify Deployment

```bash
# Check pods
kubectl get pods

# Check services
kubectl get services

# Check logs
kubectl logs -f deployment/adaptive-traffic-api
```

---

## Configuration

### Environment Variables

- `REDIS_HOST`: Redis host
- `REDIS_PORT`: Redis port
- `DATABASE_URL`: PostgreSQL connection string
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

### Resource Limits

- CPU: 2 cores per pod
- Memory: 4GB per pod
- Replicas: 3-20 (auto-scaled)

---

## Monitoring

### Access Dashboards

- Grafana: http://grafana.example.com
- Prometheus: http://prometheus.example.com
- Jaeger: http://jaeger.example.com

### Health Checks

```bash
curl http://api.adaptive-traffic.example.com/health
```

---

## Troubleshooting

### Pods Not Starting

1. Check pod logs: `kubectl logs <pod-name>`
2. Check events: `kubectl describe pod <pod-name>`
3. Check resource limits

### High Latency

1. Check HPA scaling
2. Review Prometheus metrics
3. Check database performance

---

*Last Updated: [Current Date]*

