@echo off
REM Deployment Script for Adaptive Traffic Control System (Windows)
REM Phase 4: Infrastructure Excellence

echo.
echo ==========================================
echo    ADAPTIVE TRAFFIC CONTROL DEPLOYMENT
echo ==========================================
echo.

REM Set environment variables
set ENVIRONMENT=production
set NAMESPACE=adaptive-traffic
set IMAGE_TAG=latest

if "%1" neq "" set ENVIRONMENT=%1
if "%2" neq "" set NAMESPACE=%2
if "%3" neq "" set IMAGE_TAG=%3

echo Environment: %ENVIRONMENT%
echo Namespace: %NAMESPACE%
echo Image Tag: %IMAGE_TAG%
echo.

REM Check if kubectl is available
kubectl version --client >nul 2>&1
if errorlevel 1 (
    echo ERROR: kubectl is not installed or not in PATH
    pause
    exit /b 1
)

echo Creating namespace %NAMESPACE%...
kubectl create namespace %NAMESPACE% --dry-run=client -o yaml | kubectl apply -f -

echo.
echo Applying configuration...
kubectl apply -f deployment\kubernetes\configmap.yaml -n %NAMESPACE%

echo.
echo Deploying services...
kubectl apply -f deployment\kubernetes\postgres-deployment.yaml -n %NAMESPACE%
kubectl apply -f deployment\kubernetes\redis-deployment.yaml -n %NAMESPACE%

echo.
echo Waiting for services to be ready...
kubectl wait --for=condition=ready pod -l app=postgres -n %NAMESPACE% --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n %NAMESPACE% --timeout=300s

echo.
echo Deploying API...
kubectl apply -f deployment\kubernetes\api-deployment.yaml -n %NAMESPACE%

echo.
echo Applying infrastructure configurations...
kubectl apply -f deployment\kubernetes\network-policy.yaml -n %NAMESPACE%
kubectl apply -f deployment\kubernetes\pod-disruption-budget.yaml -n %NAMESPACE%
kubectl apply -f deployment\kubernetes\resource-quotas.yaml -n %NAMESPACE%

echo.
echo Waiting for deployment...
kubectl wait --for=condition=available deployment/adaptive-traffic-api -n %NAMESPACE% --timeout=300s

echo.
echo ==========================================
echo    DEPLOYMENT COMPLETE!
echo ==========================================
echo.
echo Pod Status:
kubectl get pods -n %NAMESPACE%
echo.
echo Service Status:
kubectl get services -n %NAMESPACE%
echo.
pause

