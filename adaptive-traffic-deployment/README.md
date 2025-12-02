# Adaptive Traffic Deployment

Deployment infrastructure and configurations for the Adaptive Traffic system.

## Overview

This package contains deployment configurations and infrastructure:

- **Kubernetes**: K8s manifests for containerized deployment
- **Helm**: Helm charts for easy deployment
- **Docker**: Docker configurations
- **Monitoring**: Grafana dashboards and Prometheus configs

## Project Structure

```
adaptive-traffic-deployment/
├── deployment/
│   ├── kubernetes/       # K8s manifests
│   ├── helm/             # Helm charts
│   └── docker/           # Docker configs
├── monitoring/
│   ├── grafana/          # Grafana dashboards
│   └── prometheus/       # Prometheus configs
└── README.md
```

## Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Or use Helm
helm install adaptive-traffic deployment/helm/adaptive-traffic
```

## Docker Deployment

```bash
# Build images
docker build -t adaptive-traffic-core -f deployment/docker/core.Dockerfile .
docker build -t adaptive-traffic-api -f deployment/docker/api.Dockerfile .

# Run containers
docker-compose up -d
```

## Monitoring

Grafana dashboards are available in `monitoring/grafana/`. Import them into your Grafana instance.

## License

MIT License

