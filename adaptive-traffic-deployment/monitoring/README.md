# Monitoring & Observability Setup
## Phase 1.2: Production-Grade Monitoring

This directory contains all monitoring and observability configurations for the Adaptive Traffic Control System.

---

## 📊 Components

### 1. Prometheus
- **Location**: `monitoring/prometheus/`
- **Config**: `prometheus.yml`
- **Alerts**: `alerts/api_alerts.yml`

### 2. Grafana Dashboards
- **Location**: `monitoring/grafana/dashboards/`
- **Dashboards**:
  - API Performance Dashboard
  - Traffic Metrics Dashboard
  - System Health Dashboard

### 3. Alert Rules
- **Location**: `monitoring/prometheus/alerts/`
- **Alerts**: 10+ pre-configured alert rules

---

## 🚀 Quick Start

### 1. Start Prometheus

```bash
# Using Docker
docker run -d \
  --name=prometheus \
  -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus:/etc/prometheus \
  prom/prometheus

# Or download and run locally
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.windows-amd64.zip
unzip prometheus-2.45.0.windows-amd64.zip
cd prometheus-2.45.0.windows-amd64
.\prometheus.exe --config.file=../../monitoring/prometheus/prometheus.yml
```

### 2. Start Grafana

```bash
# Using Docker
docker run -d \
  --name=grafana \
  -p 3000:3000 \
  -v $(pwd)/monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards \
  grafana/grafana

# Access Grafana
# URL: http://localhost:3000
# Default credentials: admin/admin
```

### 3. Configure Grafana Data Source

1. Login to Grafana (http://localhost:3000)
2. Go to Configuration > Data Sources
3. Add Prometheus data source:
   - URL: `http://localhost:9090`
   - Access: Server (default)
   - Save & Test

### 4. Import Dashboards

1. Go to Dashboards > Import
2. Upload JSON files from `monitoring/grafana/dashboards/`
3. Select Prometheus data source
4. Import

---

## 📈 Available Metrics

### HTTP/API Metrics
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration histogram
- `api_response_time_seconds` - API response times
- `api_errors_total` - API error counts

### Traffic Metrics
- `traffic_decision_requests_total` - Traffic decision requests
- `traffic_decision_duration_seconds` - Decision processing time
- `total_intersections` - Total intersections
- `active_intersections` - Active intersections count
- `vehicles_processed_total` - Vehicles processed
- `average_wait_time_seconds` - Wait time histogram
- `queue_length` - Current queue lengths

### System Metrics
- `system_cpu_usage_percent` - CPU usage
- `system_memory_usage_percent` - Memory usage
- `active_websocket_connections` - WebSocket connections

### WebSocket Metrics
- `websocket_messages_sent_total` - Messages sent
- `websocket_messages_received_total` - Messages received

---

## 🚨 Alert Rules

### API Health Alerts
- **HighAPIErrorRate**: Error rate > 10 errors/second
- **SlowAPIResponse**: 99th percentile > 1 second
- **APIServerDown**: API server unavailable

### Traffic System Alerts
- **HighQueueLength**: Queue length > 50 vehicles
- **LongWaitTime**: 95th percentile wait time > 60 seconds
- **NoActiveIntersections**: No active intersections for 5 minutes

### System Resource Alerts
- **HighCPUUsage**: CPU usage > 80% for 10 minutes
- **HighMemoryUsage**: Memory usage > 85% for 10 minutes
- **TooManyWebSocketConnections**: Connections > 900

---

## 📝 Configuration

### Prometheus Configuration

Edit `monitoring/prometheus/prometheus.yml`:
- `scrape_interval`: How often to scrape metrics (default: 15s)
- `evaluation_interval`: Alert evaluation frequency (default: 15s)
- `targets`: Service endpoints to scrape

### Alertmanager Configuration

To enable alerting, configure Alertmanager:

```yaml
# alertmanager.yml
route:
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://your-webhook-url'
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'your-pagerduty-key'
```

---

## 🔧 Customization

### Adding New Metrics

1. Update `src/api/monitoring.py` to add new metrics
2. Update Prometheus scrape config if needed
3. Create Grafana panel for visualization

### Adding New Alerts

1. Edit `monitoring/prometheus/alerts/api_alerts.yml`
2. Define alert rule with PromQL expression
3. Reload Prometheus config

### Custom Dashboards

1. Create dashboard JSON in `monitoring/grafana/dashboards/`
2. Import via Grafana UI
3. Configure data sources and panels

---

## 📚 Documentation

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)

---

## 🎯 Best Practices

1. **Scrape Intervals**: Use 10-15s for critical metrics, 30s+ for less critical
2. **Alert Thresholds**: Set thresholds based on SLA requirements
3. **Dashboard Design**: Keep dashboards focused (max 10-12 panels)
4. **Retention**: Configure Prometheus retention (default: 15 days)
5. **Labeling**: Use consistent label naming across metrics

---

**Status**: Phase 1.2 Complete ✅  
**Next**: Phase 1.3 - Test Coverage Expansion

