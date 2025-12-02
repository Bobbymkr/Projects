# Kubernetes Deployment Guide
## Phase 4: Infrastructure Excellence

Complete Kubernetes deployment configuration for Adaptive Traffic Control System.

---

## 📋 Deployment Structure

### Core Components
- `api-deployment.yaml` - Main API deployment with HPA
- `configmap.yaml` - Application configuration
- `secrets.yaml.example` - Secrets template
- `namespace.yaml` - Namespace definitions
- `ingress.yaml` - Ingress configuration

### Supporting Services
- `redis-deployment.yaml` - Redis StatefulSet
- `postgres-deployment.yaml` - PostgreSQL StatefulSet

### Infrastructure
- `network-policy.yaml` - Network security policies
- `pod-disruption-budget.yaml` - High availability
- `resource-quotas.yaml` - Resource limits
- `service-monitor.yaml` - Prometheus monitoring
- `backup-cronjob.yaml` - Automated backups

---

## 🚀 Quick Start

### 1. Create Namespaces

```bash
kubectl apply -f namespace.yaml
```

### 2. Create Secrets

```bash
# Copy and edit secrets
cp secrets.yaml.example secrets.yaml
# Edit secrets.yaml with actual values
kubectl apply -f secrets.yaml
```

### 3. Deploy Database Services

```bash
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml
```

### 4. Deploy Application

```bash
kubectl apply -f configmap.yaml
kubectl apply -f api-deployment.yaml
```

### 5. Configure Ingress

```bash
kubectl apply -f ingress.yaml
```

### 6. Apply Security Policies

```bash
kubectl apply -f network-policy.yaml
kubectl apply -f pod-disruption-budget.yaml
kubectl apply -f resource-quotas.yaml
```

---

## 📊 Features

### High Availability
- ✅ Pod Disruption Budgets
- ✅ Multiple replicas (3+)
- ✅ Health checks (liveness, readiness, startup)
- ✅ Auto-scaling (HPA)

### Security
- ✅ Network policies
- ✅ Secrets management
- ✅ Resource quotas
- ✅ TLS/SSL support

### Monitoring
- ✅ Prometheus ServiceMonitor
- ✅ Metrics endpoints
- ✅ Health check endpoints

### Data Persistence
- ✅ Persistent volumes
- ✅ Automated backups
- ✅ Backup retention

---

## 🔧 Configuration

### Environment Variables
Set via ConfigMap or Secrets (see `configmap.yaml` and `secrets.yaml.example`)

### Resource Limits
Adjust in `api-deployment.yaml`:
- CPU: 250m-500m
- Memory: 256Mi-512Mi

### Scaling
Auto-scaling configured in HPA:
- Min replicas: 3
- Max replicas: 10
- CPU threshold: 70%
- Memory threshold: 80%

---

## 🔍 Monitoring

### View Pods
```bash
kubectl get pods -l app=adaptive-traffic-api
```

### View Logs
```bash
kubectl logs -l app=adaptive-traffic-api -f
```

### View Metrics
```bash
kubectl port-forward svc/adaptive-traffic-api-service 8000:80
curl http://localhost:8000/metrics
```

---

## 🔄 Updates and Rollouts

### Rolling Update
```bash
kubectl set image deployment/adaptive-traffic-api \
  api=adaptive-traffic-api:v2.0.0
```

### Rollback
```bash
kubectl rollout undo deployment/adaptive-traffic-api
```

### Status
```bash
kubectl rollout status deployment/adaptive-traffic-api
```

---

## 🛡️ Security Best Practices

1. **Secrets**: Never commit secrets.yaml to version control
2. **Network Policies**: Restrict pod-to-pod communication
3. **RBAC**: Implement role-based access control
4. **TLS**: Enable TLS/SSL for all ingress traffic
5. **Updates**: Keep images and dependencies updated

---

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Charts](https://helm.sh/)
- [Terraform Kubernetes Provider](https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs)

---

**Status**: Phase 4 - Infrastructure Excellence ✅

