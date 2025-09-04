#!/usr/bin/env python
"""
Setup script for GPU and system prerequisites for the Adaptive Traffic project.

This script checks and sets up the necessary prerequisites for the project,
including GPU detection, CUDA configuration, and system requirements.
"""

import os
import sys
import subprocess
import platform
import shutil
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

def run_command(command):
    """Run a command and return its output."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return e.stderr.strip()

def check_python_version():
    """Check if Python version meets requirements."""
    print_header("Checking Python Version")
    python_version = platform.python_version()
    print_info(f"Detected Python version: {python_version}")
    
    major, minor, _ = map(int, python_version.split('.'))
    if major >= 3 and minor >= 8:
        print_success(f"Python version {python_version} meets requirements (3.8+)")
        return True
    else:
        print_error(f"Python version {python_version} does not meet requirements (3.8+)")
        return False

def check_gpu():
    """Check for GPU and CUDA compatibility."""
    print_header("Checking GPU and CUDA Compatibility")
    
    # Check if we're on Windows
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
                gpu_ram = int(gpu.get("AdapterRAM", 0)) / (1024 * 1024)  # Convert to MB
                driver_version = gpu.get("DriverVersion", "Unknown")
                
                print_info(f"Detected GPU: {gpu_name}")
                print_info(f"GPU RAM: {gpu_ram:.0f} MB")
                print_info(f"Driver Version: {driver_version}")
                
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

def check_cuda():
    """Check for CUDA installation."""
    print_header("Checking CUDA Installation")
    
    # Try to get CUDA version using nvcc
    nvcc_output = run_command("nvcc --version")
    if "release" in nvcc_output.lower():
        cuda_version = nvcc_output.split("release")[1].split(",")[0].strip()
        print_success(f"CUDA Toolkit found: {cuda_version}")
        return True
    
    # Check if CUDA_PATH environment variable is set
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        print_success(f"CUDA found at {cuda_path}")
        return True
    
    print_warning("CUDA not found or not properly installed")
    return False

def check_pytorch_cuda():
    """Check if PyTorch is installed with CUDA support."""
    print_header("Checking PyTorch CUDA Support")
    
    try:
        # Try to import torch and check CUDA availability
        import torch
        print_info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_success(f"PyTorch CUDA is available: {torch.version.cuda}")
            print_info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print_info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print_warning("PyTorch CUDA is not available")
            return False
    except ImportError:
        print_error("PyTorch is not installed")
        return False
    except Exception as e:
        print_error(f"Error checking PyTorch CUDA support: {str(e)}")
        return False

def check_tensorflow_gpu():
    """Check if TensorFlow is installed with GPU support."""
    print_header("Checking TensorFlow GPU Support")
    
    try:
        # Try to import tensorflow and check GPU availability
        import tensorflow as tf
        print_info(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print_success(f"TensorFlow GPU is available: {len(gpus)} device(s)")
            for gpu in gpus:
                print_info(f"Device: {gpu}")
            return True
        else:
            print_warning("TensorFlow GPU is not available")
            return False
    except ImportError:
        print_error("TensorFlow is not installed")
        return False
    except Exception as e:
        print_error(f"Error checking TensorFlow GPU support: {str(e)}")
        return False

def check_sumo():
    """Check for SUMO installation."""
    print_header("Checking SUMO Installation")
    
    # Check if SUMO_HOME environment variable is set
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home and os.path.exists(sumo_home):
        print_success(f"SUMO_HOME is set to {sumo_home}")
    else:
        print_warning("SUMO_HOME environment variable is not set or path does not exist")
    
    # Check if sumo is in PATH
    sumo_path = shutil.which("sumo")
    if sumo_path:
        print_success(f"SUMO found in PATH: {sumo_path}")
        try:
            sumo_version = run_command("sumo --version")
            print_info(f"SUMO version: {sumo_version.splitlines()[0] if sumo_version else 'Unknown'}")
            return True
        except Exception:
            pass
    
    print_warning("SUMO not found in PATH")
    return False

def check_system_requirements():
    """Check if system meets minimum requirements."""
    print_header("Checking System Requirements")
    
    # Check OS
    os_name = platform.system()
    os_version = platform.version()
    print_info(f"Operating System: {os_name} {os_version}")
    
    # Check CPU
    if platform.system() == "Windows":
        cpu_info = run_command("wmic cpu get name")
        cpu_name = cpu_info.splitlines()[1] if len(cpu_info.splitlines()) > 1 else "Unknown"
    else:
        cpu_info = run_command("cat /proc/cpuinfo | grep 'model name' | uniq")
        cpu_name = cpu_info.split(":")[1].strip() if ":" in cpu_info else "Unknown"
    
    print_info(f"CPU: {cpu_name}")
    
    # Check RAM
    if platform.system() == "Windows":
        ram_info = run_command("wmic ComputerSystem get TotalPhysicalMemory")
        try:
            ram_bytes = int(ram_info.splitlines()[1])
            ram_gb = ram_bytes / (1024**3)
        except (IndexError, ValueError):
            ram_gb = 0
    else:
        ram_info = run_command("free -m | grep Mem")
        try:
            ram_mb = int(ram_info.split()[1])
            ram_gb = ram_mb / 1024
        except (IndexError, ValueError):
            ram_gb = 0
    
    print_info(f"RAM: {ram_gb:.2f} GB")
    
    # Check disk space
    if platform.system() == "Windows":
        disk_info = run_command("wmic logicaldisk where DeviceID='C:' get Size,FreeSpace")
        try:
            lines = disk_info.splitlines()
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 2:
                    free_bytes = int(parts[0])
                    total_bytes = int(parts[1])
                    free_gb = free_bytes / (1024**3)
                    total_gb = total_bytes / (1024**3)
                    print_info(f"Disk Space: {free_gb:.2f} GB free of {total_gb:.2f} GB total")
        except (IndexError, ValueError):
            print_info("Could not determine disk space")
    else:
        disk_info = run_command("df -h / | tail -1")
        print_info(f"Disk Space: {disk_info.split()[3]} free of {disk_info.split()[1]} total")
    
    # Evaluate if system meets requirements
    meets_requirements = True
    
    # Check RAM (minimum 8GB recommended)
    if ram_gb < 8:
        print_warning(f"RAM is below recommended minimum of 8GB ({ram_gb:.2f}GB detected)")
        meets_requirements = False
    else:
        print_success(f"RAM meets recommended minimum ({ram_gb:.2f}GB detected)")
    
    return meets_requirements

def create_config_file(gpu_type):
    """Create a configuration file with system information."""
    print_header("Creating Configuration File")
    
    config = {
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
        },
        "gpu": {
            "type": gpu_type,
            "cuda_available": check_cuda(),
        },
        "ml_frameworks": {
            "pytorch_cuda": check_pytorch_cuda(),
            "tensorflow_gpu": check_tensorflow_gpu(),
        },
        "sumo": {
            "installed": check_sumo(),
            "sumo_home": os.environ.get("SUMO_HOME", ""),
        }
    }
    
    # Create scripts directory if it doesn't exist
    scripts_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    config_path = scripts_dir / "system_config.json"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print_success(f"Configuration file created at {config_path}")
    return config

def main():
    """Main function to check and set up prerequisites."""
    print_header("Adaptive Traffic Project - System Prerequisites Setup")
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check system requirements
    system_ok = check_system_requirements()
    
    # Check GPU and CUDA
    gpu_type = check_gpu()
    cuda_ok = check_cuda() if gpu_type in ["NVIDIA", "AMD"] else False
    
    # Check ML frameworks
    pytorch_ok = check_pytorch_cuda()
    tensorflow_ok = check_tensorflow_gpu()
    
    # Check SUMO
    sumo_ok = check_sumo()
    
    # Create configuration file
    config = create_config_file(gpu_type)
    
    # Summary
    print_header("Setup Summary")
    print_info(f"Python: {'✓' if python_ok else '✗'}")
    print_info(f"System Requirements: {'✓' if system_ok else '✗'}")
    print_info(f"GPU Type: {gpu_type}")
    print_info(f"CUDA Available: {'✓' if cuda_ok else '✗'}")
    print_info(f"PyTorch CUDA: {'✓' if pytorch_ok else '✗'}")
    print_info(f"TensorFlow GPU: {'✓' if tensorflow_ok else '✗'}")
    print_info(f"SUMO Installed: {'✓' if sumo_ok else '✗'}")
    
    # Recommendations
    print_header("Recommendations")
    
    if not python_ok:
        print_warning("Install Python 3.8 or higher")
    
    if not system_ok:
        print_warning("Upgrade system resources to meet minimum requirements")
    
    if gpu_type == "NVIDIA" and not cuda_ok:
        print_warning("Install CUDA Toolkit for GPU acceleration")
    
    if gpu_type == "AMD" and not cuda_ok:
        print_warning("Install ROCm for GPU acceleration")
    
    if gpu_type == "Intel":
        print_warning("Intel GPU has limited acceleration capabilities, consider using CPU mode")
    
    if not pytorch_ok:
        print_warning("Install PyTorch with appropriate GPU support")
    
    if not tensorflow_ok:
        print_warning("Install TensorFlow with appropriate GPU support")
    
    if not sumo_ok:
        print_warning("Install SUMO and set SUMO_HOME environment variable")
    
    print_header("Next Steps")
    print_info("1. Address any warnings above")
    print_info("2. Create or update Python virtual environment")
    print_info("3. Install required dependencies")
    print_info("4. Configure environment variables")

if __name__ == "__main__":
    main()