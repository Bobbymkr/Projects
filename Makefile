# Makefile for Adaptive Traffic Signal Control System
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test test-cov lint format clean docs run-demo run-train run-inference setup-env create-configs

# Default target
help:
	@echo "Adaptive Traffic Signal Control System - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup-env        Create virtual environment and install dependencies"
	@echo "  install          Install the package in development mode"
	@echo "  install-dev      Install with development dependencies"
	@echo "  create-configs   Create default configuration files"
	@echo ""
	@echo "Development:"
	@echo "  format           Format code with black"
	@echo "  lint             Run linting with flake8"
	@echo "  test             Run tests"
	@echo "  test-cov         Run tests with coverage"
	@echo ""
	@echo "Running:"
	@echo "  run-demo         Run the complete demonstration"
	@echo "  run-train        Run training with default settings"
	@echo "  run-inference    Run inference with trained model"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Clean build artifacts and cache"
	@echo "  clean-all        Clean everything including virtual environment"

# Setup virtual environment
setup-env:
	@echo "Setting up virtual environment..."
	python -m venv .venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source .venv/bin/activate  # Linux/Mac"
	@echo "  .venv\\Scripts\\activate     # Windows"

# Install package
install:
	@echo "Installing package in development mode..."
	pip install -e .

# Install with development dependencies
install-dev:
	@echo "Installing package with development dependencies..."
	pip install -e ".[dev]"

# Create default configuration files
create-configs:
	@echo "Creating default configuration files..."
	python src/config.py

# Format code
format:
	@echo "Formatting code with black..."
	black src/ tests/ demo.py setup.py
	@echo "Code formatting complete!"

# Run linting
lint:
	@echo "Running linting with flake8..."
	flake8 src/ tests/ demo.py setup.py --max-line-length=88 --ignore=E203,W503
	@echo "Linting complete!"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run demo
run-demo:
	@echo "Running Adaptive Traffic demonstration..."
	python demo.py

# Run training
run-train:
	@echo "Running training with default settings..."
	python src/rl/train_dqn.py --episodes 10 --config configs/intersection.json

# Run inference
run-inference:
	@echo "Running inference with trained model..."
	python src/rl/inference.py sim --model runs/dqn_traffic.zip

# Build documentation
docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built in docs/_build/html/"

# Serve documentation locally
docs-serve:
	@echo "Serving documentation locally..."
	cd docs/_build/html && python -m http.server 8000

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

# Clean everything including virtual environment
clean-all: clean
	@echo "Cleaning virtual environment..."
	rm -rf .venv/
	@echo "Full cleanup complete!"

# Quick setup for new developers
quick-setup: setup-env
	@echo "Activating virtual environment and installing dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	@echo "Quick setup complete!"
	@echo "Activate the environment with: source .venv/bin/activate"

# Run all quality checks
quality: format lint test
	@echo "All quality checks passed!"

# Run full pipeline
pipeline: create-configs run-train run-inference
	@echo "Full pipeline completed!"

# Development workflow
dev: format lint test-cov
	@echo "Development workflow completed!"

# Production build
build: clean
	@echo "Building production package..."
	python setup.py sdist bdist_wheel
	@echo "Production build complete!"

# Install production package
install-prod:
	@echo "Installing production package..."
	pip install dist/adaptive_traffic-*.whl

# Run specific test file
test-file:
	@echo "Usage: make test-file FILE=tests/test_traffic_env.py"
	@if [ -z "$(FILE)" ]; then echo "Please specify FILE parameter"; exit 1; fi
	pytest $(FILE) -v

# Run specific training configuration
train-config:
	@echo "Usage: make train-config CONFIG=configs/sumo.json"
	@if [ -z "$(CONFIG)" ]; then echo "Please specify CONFIG parameter"; exit 1; fi
	python src/rl/train_dqn.py --config $(CONFIG) --episodes 10

# Generate test data
test-data:
	@echo "Generating test data..."
	python -c "import numpy as np; np.save('tests/test_data.npy', np.random.rand(100, 4))"
	@echo "Test data generated!"

# Check system requirements
check-system:
	@echo "Checking system requirements..."
	@python -c "import sys; print(f'Python version: {sys.version}')"
	@python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
	@python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
	@python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
	@echo "System check complete!"

# Performance benchmark
benchmark:
	@echo "Running performance benchmarks..."
	python -c "
import time
import numpy as np
from src.env.traffic_env import TrafficEnv

config = {'num_lanes': 4, 'phase_lanes': [[0,1],[2,3]], 'min_green': 5, 'max_green': 60}
env = TrafficEnv(config)

# Benchmark environment steps
start_time = time.time()
for _ in range(1000):
    obs, _ = env.reset()
    for _ in range(10):
        action = np.random.randint(0, env.action_space.n)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

end_time = time.time()
print(f'Environment benchmark: {end_time - start_time:.2f} seconds for 1000 episodes')
"
	@echo "Benchmark complete!"

# Docker commands (if using Docker)
docker-build:
	@echo "Building Docker image..."
	docker build -t adaptive-traffic .

docker-run:
	@echo "Running Docker container..."
	docker run -it --rm adaptive-traffic

# Git hooks setup
setup-hooks:
	@echo "Setting up Git hooks..."
	pre-commit install
	@echo "Git hooks installed!"

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt
	@echo "Dependencies updated!"

# Security check
security-check:
	@echo "Running security checks..."
	pip install safety
	safety check
	@echo "Security check complete!"

# Create release
release:
	@echo "Creating release..."
	@read -p "Enter version number (e.g., 1.0.1): " version; \
	sed -i "s/version=\"[^\"]*\"/version=\"$$version\"/" setup.py; \
	git add setup.py; \
	git commit -m "Bump version to $$version"; \
	git tag -a "v$$version" -m "Release version $$version"; \
	git push origin main --tags; \
	echo "Release v$$version created!"

# Help for specific commands
help-setup:
	@echo "Setup Commands:"
	@echo "  make setup-env      - Create virtual environment"
	@echo "  make install        - Install package"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo "  make quick-setup    - Complete setup for new developers"

help-dev:
	@echo "Development Commands:"
	@echo "  make format         - Format code"
	@echo "  make lint           - Run linting"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make quality        - Run all quality checks"

help-run:
	@echo "Running Commands:"
	@echo "  make run-demo       - Run demonstration"
	@echo "  make run-train      - Run training"
	@echo "  make run-inference  - Run inference"
	@echo "  make pipeline       - Run full pipeline"
