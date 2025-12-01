#!/bin/bash
# Deployment Script for Adaptive Traffic Control System
# Phase 4: Infrastructure Excellence

set -euo pipefail

ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-adaptive-traffic}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=========================================="
echo "  Adaptive Traffic Control Deployment"
echo "=========================================="
echo "Environment: ${ENVIRONMENT}"
echo "Namespace: ${NAMESPACE}"
echo "Image Tag: ${IMAGE_TAG}"
echo "=========================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."
if ! command_exists kubectl; then
    echo "ERROR: kubectl is not installed"
    exit 1
fi

if ! command_exists helm; then
    echo "WARNING: Helm is not installed. Skipping Helm deployment."
    USE_HELM=false
else
    USE_HELM=true
fi

# Create namespace if it doesn't exist
echo "Creating namespace ${NAMESPACE}..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Deploy secrets
if [ -f "deployment/kubernetes/secrets.yaml" ]; then
    echo "Applying secrets..."
    kubectl apply -f deployment/kubernetes/secrets.yaml -n ${NAMESPACE}
else
    echo "WARNING: secrets.yaml not found. Please create it from secrets.yaml.example"
fi

# Deploy ConfigMaps
echo "Applying configuration..."
kubectl apply -f deployment/kubernetes/configmap.yaml -n ${NAMESPACE}

# Deploy database services
echo "Deploying PostgreSQL..."
kubectl apply -f deployment/kubernetes/postgres-deployment.yaml -n ${NAMESPACE}

echo "Deploying Redis..."
kubectl apply -f deployment/kubernetes/redis-deployment.yaml -n ${NAMESPACE}

# Wait for databases to be ready
echo "Waiting for databases to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n ${NAMESPACE} --timeout=300s

# Run database migrations
echo "Running database migrations..."
kubectl run migration-job --image=adaptive-traffic-api:${IMAGE_TAG} \
    --restart=Never \
    --env="DATABASE_URL=\$(kubectl get secret adaptive-traffic-secrets -n ${NAMESPACE} -o jsonpath='{.data.databaseUrl}' | base64 -d)" \
    -- python scripts/migrate_db.py upgrade head \
    -n ${NAMESPACE}

# Wait for migration to complete
kubectl wait --for=condition=complete job/migration-job -n ${NAMESPACE} --timeout=300s
kubectl delete job migration-job -n ${NAMESPACE}

# Deploy API
if [ "$USE_HELM" = true ]; then
    echo "Deploying with Helm..."
    helm upgrade --install adaptive-traffic \
        deployment/helm/adaptive-traffic \
        --namespace ${NAMESPACE} \
        --set image.tag=${IMAGE_TAG} \
        --set replicaCount=3 \
        --wait
else
    echo "Deploying with kubectl..."
    kubectl apply -f deployment/kubernetes/api-deployment.yaml -n ${NAMESPACE}
fi

# Deploy supporting resources
echo "Applying network policies..."
kubectl apply -f deployment/kubernetes/network-policy.yaml -n ${NAMESPACE}

echo "Applying pod disruption budgets..."
kubectl apply -f deployment/kubernetes/pod-disruption-budget.yaml -n ${NAMESPACE}

echo "Applying resource quotas..."
kubectl apply -f deployment/kubernetes/resource-quotas.yaml -n ${NAMESPACE}

# Deploy ingress if enabled
if [ "${DEPLOY_INGRESS:-true}" == "true" ]; then
    echo "Applying ingress configuration..."
    kubectl apply -f deployment/kubernetes/ingress.yaml -n ${NAMESPACE}
fi

# Wait for deployment to be ready
echo "Waiting for API deployment to be ready..."
kubectl wait --for=condition=available deployment/adaptive-traffic-api -n ${NAMESPACE} --timeout=300s

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "Status:"
kubectl get pods -n ${NAMESPACE}
echo ""
echo "Services:"
kubectl get services -n ${NAMESPACE}
echo ""

