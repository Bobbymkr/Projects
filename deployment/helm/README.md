# Helm Chart for Adaptive Traffic Control API

## Installation

### Add the chart repository (if applicable)
```bash
helm repo add adaptive-traffic https://charts.adaptive-traffic.example.com
helm repo update
```

### Install with default values
```bash
helm install adaptive-traffic deployment/helm/adaptive-traffic \
  --namespace adaptive-traffic \
  --create-namespace
```

### Install with custom values
```bash
helm install adaptive-traffic deployment/helm/adaptive-traffic \
  --namespace adaptive-traffic \
  --set replicaCount=5 \
  --set image.tag=v1.0.0 \
  --set autoscaling.enabled=true
```

### Install from values file
```bash
helm install adaptive-traffic deployment/helm/adaptive-traffic \
  --namespace adaptive-traffic \
  -f my-values.yaml
```

## Configuration

Key configuration options in `values.yaml`:

```yaml
replicaCount: 3
image:
  repository: adaptive-traffic-api
  tag: latest

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10

redis:
  enabled: true
  
postgresql:
  enabled: true
```

## Upgrading

```bash
helm upgrade adaptive-traffic deployment/helm/adaptive-traffic \
  --namespace adaptive-traffic \
  --set image.tag=v1.1.0
```

## Uninstallation

```bash
helm uninstall adaptive-traffic --namespace adaptive-traffic
```

