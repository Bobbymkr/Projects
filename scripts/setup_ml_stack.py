#!/usr/bin/env python
"""
Script to install and configure GPU-accelerated ML stack for the Adaptive Traffic project.

This script installs and configures PyTorch, TensorFlow, and other ML libraries
with GPU acceleration based on the detected GPU type.
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

def detect_gpu_type():
    """Detect GPU type based on system information."""
    print_header("Detecting GPU Type")
    
    # Try to load system configuration
    system_config = load_system_config()
    if system_config and "gpu" in system_config:
        gpu_type = system_config["gpu"].get("type", "CPU")
        print_info(f"GPU type from system config: {gpu_type}")
        return gpu_type
    
    # If no system configuration, detect GPU manually
    if platform.system() == "Windows":
        try:
            # Use PowerShell to get GPU info
            gpu_info = run_command('powershell "Get-WmiObject -Class Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json"')
            gpu_data = json.loads(gpu_info)
            
            # Handle case where only one GPU is returned (not in a list)
            if not isinstance(gpu_data, list):
                gpu_data = [gpu_data]
                
            for gpu in gpu_data:
                gpu_name = gpu.get("Name", "Unknown")
                print_info(f"Detected GPU: {gpu_name}")
                
                # Check if it's NVIDIA
                if "NVIDIA" in gpu_name:
                    print_success("NVIDIA GPU detected, suitable for CUDA acceleration")
                    return "NVIDIA"
                # Check if it's AMD
                elif "AMD" in gpu_name or "Radeon" in gpu_name:
                    print_warning("AMD GPU detected, will use ROCm if compatible")
                    return "AMD"
                # Check if it's Intel
                elif "Intel" in gpu_name:
                    print_warning("Intel GPU detected, limited acceleration capabilities")
                    return "Intel"
        except Exception as e:
            print_error(f"Error detecting GPU: {str(e)}")
    
    # For non-Windows or if detection failed
    print_warning("No compatible GPU detected or unable to determine GPU type")
    print_info("Will configure for CPU-only operation")
    return "CPU"

def install_pytorch(pip_path, gpu_type):
    """Install PyTorch with GPU acceleration based on GPU type."""
    print_header("Installing PyTorch")
    
    if gpu_type == "NVIDIA":
        # Install PyTorch with CUDA support
        command = f"\"{pip_path}\" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    elif gpu_type == "AMD":
        # Install PyTorch with ROCm support
        command = f"\"{pip_path}\" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6"
    elif gpu_type == "Intel":
        # Install PyTorch with Intel support
        command = f"\"{pip_path}\" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        command_intel = f"\"{pip_path}\" install intel-extension-for-pytorch"
    else:  # CPU
        # Install PyTorch CPU version
        command = f"\"{pip_path}\" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    print_info(f"Installing PyTorch for {gpu_type}")
    result = run_command(command)
    
    if result is None:
        print_error("Failed to install PyTorch")
        return False
    
    # Install Intel extension if needed
    if gpu_type == "Intel":
        print_info("Installing Intel Extension for PyTorch")
        result_intel = run_command(command_intel)
        if result_intel is None:
            print_warning("Failed to install Intel Extension for PyTorch")
    
    # Verify PyTorch installation
    verify_command = f"\"{get_venv_python(os.path.dirname(os.path.dirname(pip_path)))}\" -c \"import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())\""
    verify_result = run_command(verify_command)
    
    if verify_result and "PyTorch version" in verify_result:
        print_success("PyTorch installed successfully")
        print_info(verify_result)
        return True
    else:
        print_error("Failed to verify PyTorch installation")
        return False

def install_tensorflow(pip_path, gpu_type):
    """Install TensorFlow with GPU acceleration based on GPU type."""
    print_header("Installing TensorFlow")
    
    if gpu_type == "NVIDIA":
        # Install TensorFlow with CUDA support
        command = f"\"{pip_path}\" install tensorflow"
    elif gpu_type == "AMD":
        # Install TensorFlow with ROCm support (if available)
        command = f"\"{pip_path}\" install tensorflow-rocm"
    elif gpu_type == "Intel":
        # Install TensorFlow with Intel support
        command = f"\"{pip_path}\" install tensorflow"
        command_intel = f"\"{pip_path}\" install intel-extension-for-tensorflow"
    else:  # CPU
        # Install TensorFlow CPU version
        command = f"\"{pip_path}\" install tensorflow"
    
    print_info(f"Installing TensorFlow for {gpu_type}")
    result = run_command(command)
    
    if result is None:
        print_error("Failed to install TensorFlow")
        return False
    
    # Install Intel extension if needed
    if gpu_type == "Intel":
        print_info("Installing Intel Extension for TensorFlow")
        result_intel = run_command(command_intel)
        if result_intel is None:
            print_warning("Failed to install Intel Extension for TensorFlow")
    
    # Verify TensorFlow installation
    verify_command = f"\"{get_venv_python(os.path.dirname(os.path.dirname(pip_path)))}\" -c \"import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices(\'GPU\')))\""
    verify_result = run_command(verify_command)
    
    if verify_result and "TensorFlow version" in verify_result:
        print_success("TensorFlow installed successfully")
        print_info(verify_result)
        return True
    else:
        print_error("Failed to verify TensorFlow installation")
        return False

def install_ultralytics(pip_path):
    """Install Ultralytics for computer vision tasks."""
    print_header("Installing Ultralytics")
    
    command = f"\"{pip_path}\" install ultralytics"
    print_info("Installing Ultralytics")
    result = run_command(command)
    
    if result is None:
        print_error("Failed to install Ultralytics")
        return False
    
    # Verify Ultralytics installation
    verify_command = f"\"{get_venv_python(os.path.dirname(os.path.dirname(pip_path)))}\" -c \"from ultralytics import YOLO; print('Ultralytics imported successfully')\""
    verify_result = run_command(verify_command)
    
    if verify_result and "Ultralytics imported successfully" in verify_result:
        print_success("Ultralytics installed successfully")
        return True
    else:
        print_error("Failed to verify Ultralytics installation")
        return False

def install_gymnasium(pip_path):
    """Install Gymnasium for reinforcement learning environments."""
    print_header("Installing Gymnasium")
    
    command = f"\"{pip_path}\" install gymnasium"
    print_info("Installing Gymnasium")
    result = run_command(command)
    
    if result is None:
        print_error("Failed to install Gymnasium")
        return False
    
    # Verify Gymnasium installation
    verify_command = f"\"{get_venv_python(os.path.dirname(os.path.dirname(pip_path)))}\" -c \"import gymnasium as gym; print('Gymnasium version:', gym.__version__)\""
    verify_result = run_command(verify_command)
    
    if verify_result and "Gymnasium version" in verify_result:
        print_success("Gymnasium installed successfully")
        print_info(verify_result)
        return True
    else:
        print_error("Failed to verify Gymnasium installation")
        return False

def install_stable_baselines3(pip_path):
    """Install Stable Baselines3 for reinforcement learning algorithms."""
    print_header("Installing Stable Baselines3")
    
    command = f"\"{pip_path}\" install stable-baselines3[extra]"
    print_info("Installing Stable Baselines3")
    result = run_command(command)
    
    if result is None:
        print_error("Failed to install Stable Baselines3")
        return False
    
    # Verify Stable Baselines3 installation
    verify_command = f"\"{get_venv_python(os.path.dirname(os.path.dirname(pip_path)))}\" -c \"import stable_baselines3; print('Stable Baselines3 version:', stable_baselines3.__version__)\""
    verify_result = run_command(verify_command)
    
    if verify_result and "Stable Baselines3 version" in verify_result:
        print_success("Stable Baselines3 installed successfully")
        print_info(verify_result)
        return True
    else:
        print_error("Failed to verify Stable Baselines3 installation")
        return False

def install_optuna(pip_path):
    """Install Optuna for hyperparameter optimization."""
    print_header("Installing Optuna")
    
    command = f"\"{pip_path}\" install optuna"
    print_info("Installing Optuna")
    result = run_command(command)
    
    if result is None:
        print_error("Failed to install Optuna")
        return False
    
    # Verify Optuna installation
    verify_command = f"\"{get_venv_python(os.path.dirname(os.path.dirname(pip_path)))}\" -c \"import optuna; print('Optuna version:', optuna.__version__)\""
    verify_result = run_command(verify_command)
    
    if verify_result and "Optuna version" in verify_result:
        print_success("Optuna installed successfully")
        print_info(verify_result)
        return True
    else:
        print_error("Failed to verify Optuna installation")
        return False

def install_visualization_tools(pip_path):
    """Install visualization tools for ML experiments."""
    print_header("Installing Visualization Tools")
    
    tools = [
        "tensorboard",
        "matplotlib",
        "seaborn",
        "plotly"
    ]
    
    success = True
    for tool in tools:
        command = f"\"{pip_path}\" install {tool}"
        print_info(f"Installing {tool}")
        result = run_command(command)
        
        if result is None:
            print_error(f"Failed to install {tool}")
            success = False
    
    if success:
        print_success("Visualization tools installed successfully")
    else:
        print_warning("Some visualization tools failed to install")
    
    return success

def install_opencv(pip_path):
    """Install OpenCV for computer vision tasks."""
    print_header("Installing OpenCV")
    
    command = f"\"{pip_path}\" install opencv-python opencv-contrib-python"
    print_info("Installing OpenCV")
    result = run_command(command)
    
    if result is None:
        print_error("Failed to install OpenCV")
        return False
    
    # Verify OpenCV installation
    verify_command = f"\"{get_venv_python(os.path.dirname(os.path.dirname(pip_path)))}\" -c \"import cv2; print('OpenCV version:', cv2.__version__)\""
    verify_result = run_command(verify_command)
    
    if verify_result and "OpenCV version" in verify_result:
        print_success("OpenCV installed successfully")
        print_info(verify_result)
        return True
    else:
        print_error("Failed to verify OpenCV installation")
        return False

def create_ml_config(gpu_type):
    """Create a configuration file for ML stack."""
    print_header("Creating ML Configuration")
    
    project_root = get_project_root()
    config_dir = os.path.join(project_root, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    ml_config_path = os.path.join(config_dir, "ml_config.json")
    
    config = {
        "gpu": {
            "type": gpu_type,
            "use_gpu": gpu_type != "CPU"
        },
        "pytorch": {
            "version": run_command(f"\"{get_venv_python(os.path.join(project_root, '.venv'))}\" -c \"import torch; print(torch.__version__)\"") or "unknown",
            "cuda_available": run_command(f"\"{get_venv_python(os.path.join(project_root, '.venv'))}\" -c \"import torch; print(torch.cuda.is_available())\"") == "True"
        },
        "tensorflow": {
            "version": run_command(f"\"{get_venv_python(os.path.join(project_root, '.venv'))}\" -c \"import tensorflow as tf; print(tf.__version__)\"") or "unknown",
            "gpu_available": run_command(f"\"{get_venv_python(os.path.join(project_root, '.venv'))}\" -c \"import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')) > 0)\"") == "True"
        },
        "vision": {
            "opencv_version": run_command(f"\"{get_venv_python(os.path.join(project_root, '.venv'))}\" -c \"import cv2; print(cv2.__version__)\"") or "unknown",
            "ultralytics_available": run_command(f"\"{get_venv_python(os.path.join(project_root, '.venv'))}\" -c \"try: from ultralytics import YOLO; print(True); except: print(False)\"") == "True"
        },
        "rl": {
            "gymnasium_version": run_command(f"\"{get_venv_python(os.path.join(project_root, '.venv'))}\" -c \"import gymnasium as gym; print(gym.__version__)\"") or "unknown",
            "sb3_version": run_command(f"\"{get_venv_python(os.path.join(project_root, '.venv'))}\" -c \"import stable_baselines3; print(stable_baselines3.__version__)\"") or "unknown"
        },
        "optimization": {
            "optuna_version": run_command(f"\"{get_venv_python(os.path.join(project_root, '.venv'))}\" -c \"import optuna; print(optuna.__version__)\"") or "unknown"
        }
    }
    
    with open(ml_config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print_success(f"ML configuration created at {ml_config_path}")

def main():
    """Main function to install and configure GPU-accelerated ML stack."""
    print_header("Adaptive Traffic Project - ML Stack Setup")
    
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
    
    # Detect GPU type
    gpu_type = detect_gpu_type()
    
    # Install PyTorch
    install_pytorch(pip_path, gpu_type)
    
    # Install TensorFlow
    install_tensorflow(pip_path, gpu_type)
    
    # Install Ultralytics
    install_ultralytics(pip_path)
    
    # Install Gymnasium
    install_gymnasium(pip_path)
    
    # Install Stable Baselines3
    install_stable_baselines3(pip_path)
    
    # Install Optuna
    install_optuna(pip_path)
    
    # Install visualization tools
    install_visualization_tools(pip_path)
    
    # Install OpenCV
    install_opencv(pip_path)
    
    # Create ML configuration
    create_ml_config(gpu_type)
    
    print_header("ML Stack Setup Complete")
    print_info(f"GPU type: {gpu_type}")
    print_info("ML libraries installed and configured successfully")

if __name__ == "__main__":
    main()