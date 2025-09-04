#!/usr/bin/env python
"""
Script to create and configure an isolated Python virtual environment for the Adaptive Traffic project.

This script creates a virtual environment, installs required dependencies,
and configures the environment for the project.
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

def create_virtual_environment(venv_path):
    """Create a virtual environment at the specified path."""
    print_header("Creating Virtual Environment")
    
    venv_path = Path(venv_path)
    
    # Check if virtual environment already exists
    if venv_path.exists():
        print_warning(f"Virtual environment already exists at {venv_path}")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response != 'y':
            print_info("Using existing virtual environment")
            return True
        else:
            print_info("Recreating virtual environment...")
    
    # Create virtual environment
    print_info(f"Creating virtual environment at {venv_path}")
    
    try:
        # Use the built-in venv module
        import venv
        venv.create(venv_path, with_pip=True)
        print_success(f"Virtual environment created at {venv_path}")
        return True
    except Exception as e:
        print_error(f"Failed to create virtual environment: {str(e)}")
        
        # Try using the venv command directly
        result = run_command(f"{sys.executable} -m venv {venv_path}")
        if result is not None:
            print_success(f"Virtual environment created at {venv_path}")
            return True
        
        return False

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

def upgrade_pip(pip_path):
    """Upgrade pip to the latest version."""
    print_header("Upgrading pip")
    
    result = run_command(f"\"{pip_path}\" install --upgrade pip")
    if result is not None:
        print_success("Pip upgraded to the latest version")
        return True
    else:
        print_error("Failed to upgrade pip")
        return False

def install_requirements(pip_path, requirements_file):
    """Install requirements from a requirements file."""
    print_header(f"Installing requirements from {requirements_file}")
    
    if not os.path.exists(requirements_file):
        print_error(f"Requirements file not found: {requirements_file}")
        return False
    
    result = run_command(f"\"{pip_path}\" install -r {requirements_file}")
    if result is not None:
        print_success("Requirements installed successfully")
        return True
    else:
        print_error("Failed to install requirements")
        return False

def install_gpu_packages(pip_path, gpu_type):
    """Install GPU-specific packages based on the detected GPU type."""
    print_header(f"Installing GPU packages for {gpu_type}")
    
    if gpu_type == "NVIDIA":
        # Install NVIDIA-specific packages
        packages = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "tensorflow",
            "nvidia-ml-py3"
        ]
    elif gpu_type == "AMD":
        # Install AMD-specific packages
        packages = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6",
            "tensorflow-rocm"
        ]
    elif gpu_type == "Intel":
        # Install Intel-specific packages
        packages = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "tensorflow",
            "intel-extension-for-pytorch",
            "intel-extension-for-tensorflow"
        ]
    else:  # CPU
        # Install CPU-only packages
        packages = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "tensorflow"
        ]
    
    success = True
    for package in packages:
        print_info(f"Installing {package}")
        result = run_command(f"\"{pip_path}\" install {package}")
        if result is None:
            print_error(f"Failed to install {package}")
            success = False
    
    if success:
        print_success("GPU packages installed successfully")
    else:
        print_warning("Some GPU packages failed to install")
    
    return success

def install_development_packages(pip_path):
    """Install development packages."""
    print_header("Installing development packages")
    
    packages = [
        "pytest",
        "black",
        "flake8",
        "sphinx",
        "sphinx-rtd-theme"
    ]
    
    success = True
    for package in packages:
        print_info(f"Installing {package}")
        result = run_command(f"\"{pip_path}\" install {package}")
        if result is None:
            print_error(f"Failed to install {package}")
            success = False
    
    if success:
        print_success("Development packages installed successfully")
    else:
        print_warning("Some development packages failed to install")
    
    return success

def install_project(pip_path, project_root):
    """Install the project in development mode."""
    print_header("Installing project in development mode")
    
    setup_py = os.path.join(project_root, "setup.py")
    if not os.path.exists(setup_py):
        print_error(f"setup.py not found at {setup_py}")
        return False
    
    result = run_command(f"\"{pip_path}\" install -e .{project_root}")
    if result is not None:
        print_success("Project installed in development mode")
        return True
    else:
        print_error("Failed to install project")
        return False

def create_activation_script(venv_path, project_root):
    """Create an activation script for the virtual environment."""
    print_header("Creating activation script")
    
    venv_path = Path(venv_path)
    project_root = Path(project_root)
    
    if platform.system() == "Windows":
        activate_script = project_root / "activate.bat"
        with open(activate_script, "w") as f:
            f.write(f"@echo off\n")
            f.write(f"call \"{venv_path / 'Scripts' / 'activate.bat'}\"\n")
            f.write(f"echo Virtual environment activated for Adaptive Traffic project\n")
            f.write(f"echo Project root: {project_root}\n")
    else:
        activate_script = project_root / "activate.sh"
        with open(activate_script, "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"source \"{venv_path / 'bin' / 'activate'}\"\n")
            f.write(f"echo Virtual environment activated for Adaptive Traffic project\n")
            f.write(f"echo Project root: {project_root}\n")
        os.chmod(activate_script, 0o755)
    
    print_success(f"Activation script created at {activate_script}")
    return True

def load_system_config():
    """Load system configuration from the system_config.json file."""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    config_path = script_dir / "system_config.json"
    
    if not config_path.exists():
        print_error(f"System configuration file not found at {config_path}")
        print_warning("Run setup_prerequisites.py first to generate system configuration")
        return None
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print_error(f"Failed to load system configuration: {str(e)}")
        return None

def main():
    """Main function to create and configure the virtual environment."""
    print_header("Adaptive Traffic Project - Virtual Environment Setup")
    
    # Get project root directory
    project_root = get_project_root()
    print_info(f"Project root: {project_root}")
    
    # Load system configuration
    system_config = load_system_config()
    if system_config is None:
        gpu_type = "CPU"  # Default to CPU if no configuration is available
    else:
        gpu_type = system_config.get("gpu", {}).get("type", "CPU")
    
    # Define virtual environment path
    venv_path = os.path.join(project_root, ".venv")
    
    # Create virtual environment
    if not create_virtual_environment(venv_path):
        print_error("Failed to create virtual environment")
        return
    
    # Get Python and pip paths
    python_path = get_venv_python(venv_path)
    pip_path = get_venv_pip(venv_path)
    
    if python_path is None or pip_path is None:
        print_error("Failed to get Python or pip path")
        return
    
    # Upgrade pip
    upgrade_pip(pip_path)
    
    # Install requirements
    requirements_file = os.path.join(project_root, "requirements.txt")
    install_requirements(pip_path, requirements_file)
    
    # Install GPU-specific packages
    install_gpu_packages(pip_path, gpu_type)
    
    # Install development packages
    install_development_packages(pip_path)
    
    # Install project in development mode
    install_project(pip_path, project_root)
    
    # Create activation script
    create_activation_script(venv_path, project_root)
    
    print_header("Virtual Environment Setup Complete")
    print_info(f"Virtual environment created at {venv_path}")
    print_info(f"To activate the virtual environment:")
    
    if platform.system() == "Windows":
        print_info(f"  Run: {os.path.join(project_root, 'activate.bat')}")
    else:
        print_info(f"  Run: source {os.path.join(project_root, 'activate.sh')}")

if __name__ == "__main__":
    main()