# Operational Runbooks
## Adaptive Traffic Signal Control System

**Version**: 1.0  
**Last Updated**: December 2025

---

## Table of Contents

1. [System Health Checks](#system-health-checks)
2. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring & Alerts](#monitoring--alerts)
5. [Emergency Procedures](#emergency-procedures)
6. [Performance Optimization](#performance-optimization)

---

## System Health Checks

### Daily Health Check

```bash
# Check system status
python scripts/health_check.py

# Verify all services are running
python scripts/verify_services.py

# Check database connectivity
python scripts/check_database.py
```

### Weekly Health Check

```bash
# Run comprehensive system test
pytest tests/system/ -v

# Check disk space
df -h

# Check memory usage
free -h

# Review logs for errors
grep -i error logs/*.log | tail -100
```

---

## Common Issues & Troubleshooting

### Issue 1: High Latency Alert

**Symptoms:**
- Agent inference latency > 50ms
- Dashboard shows red alert
- System performance degraded

**Diagnosis:**
1. Check Grafana dashboard: `agent_inference_latency_ms` metric
2. Identify which agent type is slow
3. Check resource usage: CPU/Memory/GPU
4. Review recent code deployments

**Resolution:**
```bash
# If CPU bound: Scale up replicas
kubectl scale deployment traffic-agent --replicas=10

# If memory bound: Increase limits
kubectl set resources deployment traffic-agent --limits=memory=8Gi

# If GPU bound: Add GPU nodes
kubectl label nodes <node-name> accelerator=gpu
```

**Prevention:**
- Enable auto-scaling with lower threshold
- Consider model compression
- Monitor resource usage trends

---

### Issue 2: Camera Feed Failure

**Symptoms:**
- No video frames received
- Detection accuracy drops to zero
- System falls back to fixed-time control

**Diagnosis:**
1. Check camera connectivity
2. Verify network connection
3. Check video stream URL
4. Review vision pipeline logs

**Resolution:**
```bash
# Test camera connectivity
python scripts/test_camera.py --camera_id <id>

# Restart vision service
kubectl restart deployment vision-service

# Verify fallback controller is active
python scripts/check_fallback.py
```

**Prevention:**
- Implement camera health monitoring
- Set up automatic failover
- Regular camera maintenance

---

### Issue 3: Model Performance Degradation

**Symptoms:**
- Wait times increasing
- Throughput decreasing
- Model accuracy dropping

**Diagnosis:**
1. Check model performance metrics
2. Review recent traffic patterns
3. Compare with baseline performance
4. Check for data drift

**Resolution:**
```bash
# Retrain model on recent data
python scripts/retrain_model.py --data recent_data/

# Switch to backup model
python scripts/switch_model.py --model backup_model.pth

# Enable fallback controller
python scripts/enable_fallback.py
```

**Prevention:**
- Regular model retraining
- Performance monitoring
- A/B testing for model updates

---

### Issue 4: Database Connection Issues

**Symptoms:**
- Cannot save metrics
- Historical data unavailable
- API errors

**Diagnosis:**
1. Check database status
2. Verify connection string
3. Check network connectivity
4. Review database logs

**Resolution:**
```bash
# Check database status
kubectl get pods -l app=database

# Restart database connection pool
python scripts/restart_db_pool.py

# Verify connectivity
python scripts/test_database.py
```

---

## Deployment Procedures

### Standard Deployment

```bash
# 1. Run tests
pytest tests/ -v

# 2. Build Docker image
docker build -t traffic-control:latest .

# 3. Tag and push
docker tag traffic-control:latest registry/traffic-control:v1.2.3
docker push registry/traffic-control:v1.2.3

# 4. Update Kubernetes deployment
kubectl set image deployment/traffic-agent \
  traffic-agent=registry/traffic-control:v1.2.3

# 5. Verify deployment
kubectl rollout status deployment/traffic-agent

# 6. Monitor for issues
kubectl logs -f deployment/traffic-agent
```

### Rollback Procedure

```bash
# Rollback to previous version
kubectl rollout undo deployment/traffic-agent

# Verify rollback
kubectl rollout status deployment/traffic-agent

# Check system health
python scripts/health_check.py
```

---

## Monitoring & Alerts

### Key Metrics to Monitor

1. **Performance Metrics**
   - Average wait time
   - Throughput (vehicles/hour)
   - Queue lengths
   - Inference latency

2. **System Metrics**
   - CPU usage
   - Memory usage
   - Network latency
   - Disk I/O

3. **Business Metrics**
   - Intersection efficiency
   - Fuel savings
   - Emission reduction

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Wait Time | > 15s | > 30s | Investigate & optimize |
| Inference Latency | > 20ms | > 50ms | Scale resources |
| CPU Usage | > 70% | > 90% | Scale horizontally |
| Memory Usage | > 80% | > 95% | Increase limits |
| Error Rate | > 1% | > 5% | Investigate immediately |

---

## Emergency Procedures

### Complete System Failure

1. **Immediate Actions:**
   - Switch to fixed-time control
   - Notify traffic management center
   - Activate backup systems

2. **Recovery Steps:**
   ```bash
   # Enable emergency mode
   python scripts/emergency_mode.py --enable
   
   # Restore from backup
   python scripts/restore_backup.py --backup latest
   
   # Verify system recovery
   python scripts/health_check.py
   ```

### Security Incident

1. **Immediate Actions:**
   - Isolate affected systems
   - Preserve logs
   - Notify security team

2. **Investigation:**
   ```bash
   # Collect logs
   kubectl logs --all-containers=true > incident_logs.txt
   
   # Check for unauthorized access
   python scripts/security_audit.py
   
   # Review access logs
   grep -i "unauthorized" logs/*.log
   ```

---

## Performance Optimization

### Model Optimization

```bash
# Quantize model
python scripts/quantize_model.py --model model.pth --output model_quantized.pth

# Prune model
python scripts/prune_model.py --model model.pth --sparsity 0.5

# Benchmark performance
python scripts/benchmark_model.py --model model.pth
```

### System Optimization

```bash
# Optimize database queries
python scripts/optimize_database.py

# Clear caches
python scripts/clear_caches.py

# Optimize resource allocation
kubectl autoscale deployment traffic-agent --cpu-percent=70 --min=3 --max=20
```

---

## Maintenance Windows

### Weekly Maintenance

- **Time**: Sunday 2:00 AM - 4:00 AM
- **Tasks**:
  - Database backup
  - Log rotation
  - System updates
  - Performance review

### Monthly Maintenance

- **Time**: First Sunday of month, 1:00 AM - 5:00 AM
- **Tasks**:
  - Full system backup
  - Security updates
  - Model retraining
  - Capacity planning review

---

## Contact Information

**On-Call Engineer**: [Contact Info]  
**Escalation Path**: Team Lead → Engineering Manager → CTO  
**Emergency Hotline**: [Phone Number]

---

**Document Status**: ✅ Active  
**Next Review**: Quarterly
