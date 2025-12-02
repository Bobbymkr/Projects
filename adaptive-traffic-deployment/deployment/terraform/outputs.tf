# Terraform Outputs
# Phase 4: Infrastructure Excellence

output "namespace" {
  description = "Kubernetes namespace"
  value       = kubernetes_namespace.adaptive_traffic.metadata[0].name
}

output "api_service" {
  description = "API service endpoint"
  value       = "http://adaptive-traffic-api-service.default.svc.cluster.local"
}

output "ingress_url" {
  description = "Ingress URL"
  value       = var.environment == "production" ? "https://api.adaptive-traffic.example.com" : "https://staging-api.adaptive-traffic.example.com"
}

output "monitoring_namespace" {
  description = "Monitoring namespace"
  value       = kubernetes_namespace.monitoring.metadata[0].name
}

