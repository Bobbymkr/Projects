#!/usr/bin/env python
"""
Script to install and configure RL, forecasting, optimization, and utility packages
for the Adaptive Traffic project.

This script installs additional libraries needed for the project beyond the core ML stack.
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    print(f"{Colors.HEADER}{Colors.BOLD}\n{message}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

def run_command(command, cwd=None):
    """Run a command and return its output."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            check=True,
            cwd=cwd
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {command}")
        print_error(f"Error: {e.stderr.strip()}")
        return None

def get_project_root():
    """Get the project root directory."""
    # Assuming this script is in the scripts directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    return script_dir.parent

def get_venv_python(venv_path):
    """Get the path to the Python executable in the virtual environment."""
    venv_path = Path(venv_path)
    
    if platform.system() == "Windows":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"
    
    if not python_path.exists():
        print_error(f"Python executable not found at {python_path}")
        return None
    
    return str(python_path)

def get_venv_pip(venv_path):
    """Get the path to the pip executable in the virtual environment."""
    venv_path = Path(venv_path)
    
    if platform.system() == "Windows":
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    if not pip_path.exists():
        print_error(f"Pip executable not found at {pip_path}")
        return None
    
    return str(pip_path)

def install_rl_packages(pip_path):
    """Install additional reinforcement learning packages."""
    print_header("Installing Additional RL Packages")
    
    packages = [
        "ray[rllib]",  # Ray RLlib for distributed RL
        "supersuit",  # Wrappers for multi-agent environments
        "pettingzoo"  # Multi-agent environments
    ]
    
    success = True
    for package in packages:
        command = f"\"{pip_path}\" install {package}"
        print_info(f"Installing {package}")
        result = run_command(command)
        
        if result is None:
            print_error(f"Failed to install {package}")
            success = False
        else:
            print_success(f"{package} installed successfully")
    
    return success

def install_forecasting_packages(pip_path):
    """Install packages for time series forecasting."""
    print_header("Installing Forecasting Packages")
    
    packages = [
        "statsmodels",  # Statistical models for time series
        "prophet",     # Facebook's time series forecasting tool
        "pmdarima",    # Auto ARIMA models
        "sktime"       # Scikit-learn compatible time series library
    ]
    
    success = True
    for package in packages:
        command = f"\"{pip_path}\" install {package}"
        print_info(f"Installing {package}")
        result = run_command(command)
        
        if result is None:
            print_error(f"Failed to install {package}")
            success = False
        else:
            print_success(f"{package} installed successfully")
    
    return success

def install_optimization_packages(pip_path):
    """Install packages for optimization."""
    print_header("Installing Optimization Packages")
    
    packages = [
        "scipy",       # Scientific computing
        "cvxpy",       # Convex optimization
        "pyomo",       # Optimization modeling
        "ortools"      # Google OR-Tools
    ]
    
    success = True
    for package in packages:
        command = f"\"{pip_path}\" install {package}"
        print_info(f"Installing {package}")
        result = run_command(command)
        
        if result is None:
            print_error(f"Failed to install {package}")
            success = False
        else:
            print_success(f"{package} installed successfully")
    
    return success

def install_utility_packages(pip_path):
    """Install utility packages for development and monitoring."""
    print_header("Installing Utility Packages")
    
    packages = [
        "pydantic",    # Data validation
        "tqdm",        # Progress bars
        "loguru",      # Logging
        "rich",        # Terminal formatting
        "typer",       # CLI tools
        "pytest",      # Testing
        "pytest-cov",  # Test coverage
        "black",       # Code formatting
        "flake8",      # Linting
        "isort",       # Import sorting
        "mypy",        # Type checking
        "pre-commit"   # Git hooks
    ]
    
    success = True
    for package in packages:
        command = f"\"{pip_path}\" install {package}"
        print_info(f"Installing {package}")
        result = run_command(command)
        
        if result is None:
            print_error(f"Failed to install {package}")
            success = False
        else:
            print_success(f"{package} installed successfully")
    
    return success

def install_monitoring_packages(pip_path):
    """Install packages for monitoring and observability."""
    print_header("Installing Monitoring Packages")
    
    packages = [
        "wandb",       # Weights & Biases for experiment tracking
        "mlflow",      # MLflow for experiment tracking
        "tensorboardX", # TensorBoard for PyTorch
        "prometheus-client", # Prometheus metrics
        "grafana-api"  # Grafana API client
    ]
    
    success = True
    for package in packages:
        command = f"\"{pip_path}\" install {package}"
        print_info(f"Installing {package}")
        result = run_command(command)
        
        if result is None:
            print_error(f"Failed to install {package}")
            success = False
        else:
            print_success(f"{package} installed successfully")
    
    return success

def install_project_in_dev_mode(pip_path):
    """Install the project in development mode."""
    print_header("Installing Project in Development Mode")
    
    project_root = get_project_root()
    command = f"\"{pip_path}\" install -e ."
    print_info("Installing project in development mode")
    result = run_command(command, cwd=str(project_root))
    
    if result is None:
        print_error("Failed to install project in development mode")
        return False
    else:
        print_success("Project installed in development mode")
        return True

def create_utilities_config():
    """Create a configuration file for utilities."""
    print_header("Creating Utilities Configuration")
    
    project_root = get_project_root()
    config_dir = os.path.join(project_root, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    utilities_config_path = os.path.join(config_dir, "utilities_config.json")
    
    config = {
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/adaptive_traffic.log"
        },
        "monitoring": {
            "enabled": True,
            "metrics_port": 8000,
            "use_tensorboard": True,
            "use_wandb": False,
            "use_mlflow": False
        },
        "development": {
            "debug": False,
            "test_mode": False,
            "profile_code": False
        }
    }
    
    with open(utilities_config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print_success(f"Utilities configuration created at {utilities_config_path}")

def setup_pre_commit_hooks():
    """Set up pre-commit hooks for the project."""
    print_header("Setting Up Pre-commit Hooks")
    
    project_root = get_project_root()
    pre_commit_config_path = os.path.join(project_root, ".pre-commit-config.yaml")
    
    # Create pre-commit config if it doesn't exist
    if not os.path.exists(pre_commit_config_path):
        pre_commit_config = """
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-docstrings]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-requests]
"""
        
        with open(pre_commit_config_path, "w") as f:
            f.write(pre_commit_config)
        
        print_success(f"Pre-commit config created at {pre_commit_config_path}")
    else:
        print_info(f"Pre-commit config already exists at {pre_commit_config_path}")
    
    # Install pre-commit hooks
    venv_path = os.path.join(project_root, ".venv")
    python_path = get_venv_python(venv_path)
    
    if python_path is None:
        print_error("Failed to get Python path")
        return False
    
    command = f"\"{python_path}\" -m pre_commit install"
    print_info("Installing pre-commit hooks")
    result = run_command(command, cwd=str(project_root))
    
    if result is None:
        print_error("Failed to install pre-commit hooks")
        return False
    else:
        print_success("Pre-commit hooks installed successfully")
        return True

def main():
    """Main function to install and configure utilities."""
    print_header("Adaptive Traffic Project - Utilities Setup")
    
    # Get project root directory
    project_root = get_project_root()
    print_info(f"Project root: {project_root}")
    
    # Check for virtual environment
    venv_path = os.path.join(project_root, ".venv")
    if not os.path.exists(venv_path):
        print_error(f"Virtual environment not found at {venv_path}")
        print_warning("Run setup_venv.py first to create a virtual environment")
        return
    
    # Get Python and pip paths
    python_path = get_venv_python(venv_path)
    pip_path = get_venv_pip(venv_path)
    
    if python_path is None or pip_path is None:
        print_error("Failed to get Python or pip path")
        return
    
    # Install RL packages
    install_rl_packages(pip_path)
    
    # Install forecasting packages
    install_forecasting_packages(pip_path)
    
    # Install optimization packages
    install_optimization_packages(pip_path)
    
    # Install utility packages
    install_utility_packages(pip_path)
    
    # Install monitoring packages
    install_monitoring_packages(pip_path)
    
    # Install project in development mode
    install_project_in_dev_mode(pip_path)
    
    # Create utilities configuration
    create_utilities_config()
    
    # Set up pre-commit hooks
    setup_pre_commit_hooks()
    
    print_header("Utilities Setup Complete")
    print_info("Additional packages installed and configured successfully")

if __name__ == "__main__":
    main()