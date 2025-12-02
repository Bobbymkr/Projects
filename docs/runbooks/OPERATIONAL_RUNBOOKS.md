# Operational Runbooks

**Week 6: Complete Operational Runbooks**

---

## High Latency Alert Response

### Symptoms
- 95th percentile latency >100ms
- User complaints about slow responses
- Alert: `HighAPILatency`

### Investigation Steps

1. **Check current latency**
   ```bash
   curl http://localhost:9090/api/v1/metrics/performance
   ```

2. **Check system resources**
   ```bash
   kubectl top pods
   ```

3. **Check logs for errors**
   ```bash
   kubectl logs -f deployment/adaptive-traffic-api
   ```

### Resolution Steps

1. **Scale up if needed**
   ```bash
   kubectl scale deployment adaptive-traffic-api --replicas=5
   ```

2. **Check for bottlenecks**
   - Review Prometheus dashboards
   - Check database connection pool
   - Review Redis performance

3. **Restart service if needed**
   ```bash
   kubectl rollout restart deployment/adaptive-traffic-api
   ```

---

## Service Failure Recovery

### Symptoms
- Service returns 5xx errors
- Alert: `ServiceDown` or `HighErrorRate`
- Health checks failing

### Investigation Steps

1. **Check service status**
   ```bash
   kubectl get pods
   kubectl describe pod <pod-name>
   ```

2. **Check logs**
   ```bash
   kubectl logs <pod-name> --tail=100
   ```

3. **Check resource limits**
   ```bash
   kubectl top pod <pod-name>
   ```

### Resolution Steps

1. **Restart failing pods**
   ```bash
   kubectl delete pod <pod-name>
   ```

2. **Check for resource exhaustion**
   - CPU/Memory limits
   - Disk space
   - Network issues

3. **Rollback if recent deployment**
   ```bash
   kubectl rollout undo deployment/adaptive-traffic-api
   ```

---

## Performance Degradation

### Symptoms
- Throughput decreasing
- Wait times increasing
- System metrics showing degradation

### Investigation Steps

1. **Check performance trends**
   - Review Grafana dashboards
   - Compare with baseline

2. **Check for resource constraints**
   ```bash
   kubectl top nodes
   kubectl top pods
   ```

3. **Check for bottlenecks**
   - Database queries
   - Redis performance
   - Network latency

### Resolution Steps

1. **Scale horizontally**
   ```bash
   kubectl scale deployment adaptive-traffic-api --replicas=10
   ```

2. **Optimize queries**
   - Review slow query logs
   - Add database indexes

3. **Clear caches if needed**
   ```bash
   redis-cli FLUSHALL
   ```

---

## Capacity Planning

### Current Capacity
- **API**: 500 req/s per pod
- **Database**: 1000 connections
- **Redis**: 10,000 ops/s

### Scaling Triggers
- CPU >70% for 10 minutes
- Memory >80% for 10 minutes
- Latency p95 >100ms

### Scaling Actions
1. **Horizontal scaling**: Add pods
2. **Vertical scaling**: Increase resources
3. **Database scaling**: Add read replicas

---

*Last Updated: [Current Date]*

