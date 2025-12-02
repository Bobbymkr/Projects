# Terraform Infrastructure as Code
# Phase 4: Infrastructure Excellence

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket = "adaptive-traffic-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}

# Kubernetes Namespace
resource "kubernetes_namespace" "adaptive_traffic" {
  metadata {
    name = "adaptive-traffic"
    labels = {
      name        = "adaptive-traffic"
      environment = "production"
    }
  }
}

# Kubernetes Namespace for Monitoring
resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
    labels = {
      name   = "monitoring"
      purpose = "observability"
    }
  }
}

# ConfigMap
resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "adaptive-traffic-config"
    namespace = kubernetes_namespace.adaptive_traffic.metadata[0].name
  }
  
  data = {
    API_VERSION           = "v1"
    ENVIRONMENT          = "production"
    LOG_LEVEL            = "INFO"
    ENABLE_REDIS         = "true"
    ENABLE_CACHING       = "true"
    ENABLE_RATE_LIMITING = "true"
    DEFAULT_CACHE_TTL    = "300"
  }
}

# Secret (example - use proper secret management in production)
resource "kubernetes_secret" "app_secrets" {
  metadata {
    name      = "adaptive-traffic-secrets"
    namespace = kubernetes_namespace.adaptive_traffic.metadata[0].name
  }
  
  type = "Opaque"
  
  data = {
    SECRET_KEY = base64encode(var.secret_key)
    # Add other secrets from variables
  }
}

variable "secret_key" {
  description = "Application secret key"
  type        = string
  sensitive   = true
}

