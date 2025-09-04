#!/usr/bin/env python
"""
Script to install and configure SUMO (Simulation of Urban MObility) for the Adaptive Traffic project.

This script downloads, installs, and configures SUMO, and sets the necessary environment variables.
"""

import os
import sys
import subprocess
import platform
import json
import zipfile
import shutil
import urllib.request
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

def check_sumo_installed():
    """Check if SUMO is already installed."""
    print_header("Checking SUMO Installation")
    
    # Check if SUMO_HOME environment variable is set
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home and os.path.exists(sumo_home):
        print_success(f"SUMO_HOME is set to {sumo_home}")
        
        # Check if sumo is in PATH
        sumo_path = shutil.which("sumo")
        if sumo_path:
            print_success(f"SUMO found in PATH: {sumo_path}")
            try:
                sumo_version = run_command("sumo --version")
                print_info(f"SUMO version: {sumo_version.splitlines()[0] if sumo_version else 'Unknown'}")
                return True, sumo_home
            except Exception:
                pass
    
    print_warning("SUMO not found or not properly installed")
    return False, None

def download_sumo(download_dir, version="1.18.0"):
    """Download SUMO installer."""
    print_header(f"Downloading SUMO {version}")
    
    os_name = platform.system()
    if os_name == "Windows":
        url = f"https://sumo.dlr.de/releases/{version}/sumo-win64-{version}.zip"
        installer_path = os.path.join(download_dir, f"sumo-win64-{version}.zip")
    elif os_name == "Linux":
        print_warning("For Linux, it's recommended to install SUMO using your package manager")
        print_info("For Ubuntu: sudo apt-get install sumo sumo-tools sumo-doc")
        print_info("For other distributions, please refer to SUMO documentation")
        return None
    elif os_name == "Darwin":  # macOS
        print_warning("For macOS, it's recommended to install SUMO using Homebrew")
        print_info("brew install sumo")
        return None
    else:
        print_error(f"Unsupported operating system: {os_name}")
        return None
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Download the installer
    print_info(f"Downloading from {url}")
    try:
        urllib.request.urlretrieve(url, installer_path)
        print_success(f"Downloaded to {installer_path}")
        return installer_path
    except Exception as e:
        print_error(f"Failed to download SUMO: {str(e)}")
        return None

def install_sumo_windows(installer_path, install_dir):
    """Install SUMO on Windows."""
    print_header("Installing SUMO on Windows")
    
    # Create installation directory if it doesn't exist
    os.makedirs(install_dir, exist_ok=True)
    
    # Extract the zip file
    print_info(f"Extracting {installer_path} to {install_dir}")
    try:
        with zipfile.ZipFile(installer_path, 'r') as zip_ref:
            zip_ref.extractall(install_dir)
        
        # The zip file contains a directory named 'sumo-<version>'
        # Find this directory
        sumo_dirs = [d for d in os.listdir(install_dir) if d.startswith("sumo-")]
        if not sumo_dirs:
            print_error("SUMO directory not found in the extracted files")
            return None
        
        sumo_dir = os.path.join(install_dir, sumo_dirs[0])
        print_success(f"SUMO extracted to {sumo_dir}")
        return sumo_dir
    except Exception as e:
        print_error(f"Failed to extract SUMO: {str(e)}")
        return None

def set_environment_variables(sumo_home):
    """Set SUMO_HOME environment variable."""
    print_header("Setting Environment Variables")
    
    os_name = platform.system()
    if os_name == "Windows":
        # Set environment variable for the current process
        os.environ["SUMO_HOME"] = sumo_home
        
        # Set environment variable permanently using PowerShell
        print_info("Setting SUMO_HOME environment variable permanently")
        command = f'powershell -Command "[Environment]::SetEnvironmentVariable(\'SUMO_HOME\', \'%s\', \'User\')"; exit $LASTEXITCODE' % sumo_home
        result = run_command(command)
        
        # Add SUMO bin directory to PATH
        sumo_bin = os.path.join(sumo_home, "bin")
        print_info(f"Adding {sumo_bin} to PATH")
        command = f'powershell -Command "$path = [Environment]::GetEnvironmentVariable(\'Path\', \'User\'); if ($path -notlike \'*%s*\') { [Environment]::SetEnvironmentVariable(\'Path\', $path + \';%s\', \'User\') }"; exit $LASTEXITCODE' % (sumo_bin.replace("\\", "\\\\"), sumo_bin.replace("\\", "\\\\"))
        result = run_command(command)
        
        print_success("Environment variables set")
        print_warning("You may need to restart your terminal or IDE for the changes to take effect")
    elif os_name in ["Linux", "Darwin"]:
        # Set environment variable for the current process
        os.environ["SUMO_HOME"] = sumo_home
        
        # Add to .bashrc or .zshrc
        shell = os.environ.get("SHELL", "")
        if "zsh" in shell:
            rc_file = os.path.expanduser("~/.zshrc")
        else:
            rc_file = os.path.expanduser("~/.bashrc")
        
        print_info(f"Adding SUMO_HOME to {rc_file}")
        with open(rc_file, "a") as f:
            f.write(f"\n# SUMO environment variables\nexport SUMO_HOME={sumo_home}\nexport PATH=$PATH:{os.path.join(sumo_home, 'bin')}\n")
        
        print_success("Environment variables set")
        print_warning(f"You need to run 'source {rc_file}' or restart your terminal for the changes to take effect")
    else:
        print_error(f"Unsupported operating system: {os_name}")

def verify_sumo_installation(sumo_home):
    """Verify SUMO installation."""
    print_header("Verifying SUMO Installation")
    
    # Check if SUMO_HOME exists
    if not os.path.exists(sumo_home):
        print_error(f"SUMO_HOME directory does not exist: {sumo_home}")
        return False
    
    # Check if SUMO binaries exist
    sumo_bin = os.path.join(sumo_home, "bin", "sumo.exe" if platform.system() == "Windows" else "sumo")
    if not os.path.exists(sumo_bin):
        print_error(f"SUMO binary not found: {sumo_bin}")
        return False
    
    # Try running SUMO
    try:
        # Add SUMO bin to PATH for this process
        os.environ["PATH"] = os.path.join(sumo_home, "bin") + os.pathsep + os.environ.get("PATH", "")
        
        # Run SUMO version command
        sumo_version = run_command("sumo --version")
        if sumo_version:
            print_success(f"SUMO version: {sumo_version.splitlines()[0]}")
            return True
        else:
            print_error("Failed to run SUMO")
            return False
    except Exception as e:
        print_error(f"Error running SUMO: {str(e)}")
        return False

def create_sumo_config():
    """Create a SUMO configuration file for the project."""
    print_header("Creating SUMO Configuration")
    
    project_root = get_project_root()
    config_dir = os.path.join(project_root, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    sumo_config_path = os.path.join(config_dir, "sumo_config.json")
    
    config = {
        "sumo_home": os.environ.get("SUMO_HOME", ""),
        "gui_enabled": True,
        "simulation": {
            "step_length": 1.0,
            "begin": 0,
            "end": 3600,  # 1 hour simulation by default
            "route_files": "routes.xml",
            "net_file": "network.net.xml"
        },
        "output": {
            "tripinfo": "tripinfo.xml",
            "summary": "summary.xml"
        }
    }
    
    with open(sumo_config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print_success(f"SUMO configuration created at {sumo_config_path}")

def create_example_network():
    """Create an example SUMO network for testing."""
    print_header("Creating Example SUMO Network")
    
    project_root = get_project_root()
    examples_dir = os.path.join(project_root, "examples", "sumo")
    os.makedirs(examples_dir, exist_ok=True)
    
    # Create a simple network using SUMO's netgenerate
    net_file = os.path.join(examples_dir, "example.net.xml")
    command = f"netgenerate --grid --grid.x-number=3 --grid.y-number=3 --output-file={net_file}"
    result = run_command(command)
    
    if result is None:
        print_error("Failed to create example network")
        return
    
    # Create a simple route file
    route_file = os.path.join(examples_dir, "example.rou.xml")
    with open(route_file, "w") as f:
        f.write("""<routes>
    <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" />
    <flow id="flow0" type="car" from="0/0to1/0" to="2/2to2/1" begin="0" end="3600" period="3" />
    <flow id="flow1" type="car" from="2/0to2/1" to="0/2to0/1" begin="0" end="3600" period="4" />
</routes>""")
    
    # Create a SUMO configuration file
    config_file = os.path.join(examples_dir, "example.sumocfg")
    with open(config_file, "w") as f:
        f.write("""<configuration>
    <input>
        <net-file value="example.net.xml"/>
        <route-files value="example.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
    </time>
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>""")
    
    print_success(f"Example SUMO network created at {examples_dir}")
    print_info(f"To run the example: sumo-gui -c {config_file}")

def main():
    """Main function to install and configure SUMO."""
    print_header("Adaptive Traffic Project - SUMO Setup")
    
    # Check if SUMO is already installed
    installed, sumo_home = check_sumo_installed()
    
    if not installed:
        # Define download and installation directories
        project_root = get_project_root()
        download_dir = os.path.join(project_root, "downloads")
        install_dir = os.path.join(project_root, "tools", "sumo")
        
        # Download SUMO
        installer_path = download_sumo(download_dir)
        if installer_path is None:
            print_error("Failed to download SUMO")
            return
        
        # Install SUMO
        if platform.system() == "Windows":
            sumo_home = install_sumo_windows(installer_path, install_dir)
            if sumo_home is None:
                print_error("Failed to install SUMO")
                return
        else:
            print_warning("Please install SUMO manually for your operating system")
            return
    
    # Set environment variables
    set_environment_variables(sumo_home)
    
    # Verify installation
    if not verify_sumo_installation(sumo_home):
        print_error("SUMO installation verification failed")
        return
    
    # Create SUMO configuration
    create_sumo_config()
    
    # Create example network
    create_example_network()
    
    print_header("SUMO Setup Complete")
    print_info(f"SUMO_HOME: {sumo_home}")
    print_info("You can now use SUMO with the Adaptive Traffic project")

if __name__ == "__main__":
    main()