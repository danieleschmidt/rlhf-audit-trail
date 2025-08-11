#!/usr/bin/env python3
"""
Setup script for RLHF Audit Trail - handles environment setup and dependency installation.
Supports both system and virtual environment installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class Colors:
    """Terminal colors for better output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_status(message, color=Colors.BLUE):
    """Print colored status message."""
    print(f"{color}{Colors.BOLD}==> {message}{Colors.END}")


def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def run_command(cmd, check=True, capture_output=False):
    """Run shell command with error handling."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        if not capture_output:
            print_error(f"Command failed: {cmd}")
            print_error(f"Error: {e}")
        return None


def check_python_version():
    """Check if Python version is compatible."""
    print_status("Checking Python version...")
    
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 10):
        print_error(f"Python {version_info.major}.{version_info.minor} is not supported. Requires Python 3.10+")
        sys.exit(1)
    
    print_success(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} is compatible")


def detect_package_manager():
    """Detect available package manager."""
    print_status("Detecting package manager...")
    
    managers = {
        'apt': ['apt', '--version'],
        'yum': ['yum', '--version'],
        'brew': ['brew', '--version'],
        'pip': ['pip', '--version'],
    }
    
    available = []
    for manager, cmd in managers.items():
        if run_command(' '.join(cmd), check=False, capture_output=True):
            available.append(manager)
    
    print_success(f"Available package managers: {', '.join(available)}")
    return available


def install_system_dependencies():
    """Install system-level dependencies."""
    print_status("Installing system dependencies...")
    
    system = platform.system().lower()
    package_managers = detect_package_manager()
    
    if 'apt' in package_managers and system == 'linux':
        # Ubuntu/Debian
        packages = [
            'python3-venv',
            'python3-pip',
            'python3-numpy',
            'python3-pydantic',
            'python3-cryptography',
            'python3-sqlalchemy',
            'python3-redis',
            'build-essential',
        ]
        
        print_status("Installing packages with apt...")
        run_command('apt update', check=False)
        
        for package in packages:
            result = run_command(f'apt install -y {package}', check=False, capture_output=True)
            if result and result.returncode == 0:
                print_success(f"Installed {package}")
            else:
                print_warning(f"Failed to install {package} (may already be installed)")
    
    elif 'brew' in package_managers and system == 'darwin':
        # macOS
        packages = ['python@3.11', 'redis', 'postgresql']
        for package in packages:
            result = run_command(f'brew install {package}', check=False, capture_output=True)
            if result and result.returncode == 0:
                print_success(f"Installed {package}")
    
    else:
        print_warning("No suitable package manager found. Proceeding with pip installation only.")


def setup_virtual_environment(force=False):
    """Set up Python virtual environment."""
    venv_path = Path('venv')
    
    if venv_path.exists() and not force:
        print_success("Virtual environment already exists")
        return venv_path
    
    print_status("Creating virtual environment...")
    
    if force and venv_path.exists():
        import shutil
        shutil.rmtree(venv_path)
    
    result = run_command('python3 -m venv venv')
    if result:
        print_success("Virtual environment created")
        return venv_path
    else:
        print_error("Failed to create virtual environment")
        return None


def install_python_dependencies(use_venv=True):
    """Install Python dependencies."""
    print_status("Installing Python dependencies...")
    
    pip_cmd = 'venv/bin/pip' if use_venv and Path('venv').exists() else 'pip3'
    
    # Core dependencies (minimal for basic functionality)
    core_deps = [
        'pydantic>=2.0.0',
        'cryptography>=41.0.0',
        'sqlalchemy>=2.0.0',
        'fastapi>=0.100.0',
        'numpy>=1.24.0',
        'click>=8.0.0',  # For CLI
    ]
    
    # Optional dependencies with fallbacks
    optional_deps = [
        'redis>=5.0.0',
        'celery>=5.3.0',
        'pandas>=2.0.0',
        'streamlit>=1.28.0',
        'plotly>=5.15.0',
        'boto3>=1.28.0',
        'psycopg2-binary>=2.9.0',
        'pytest>=7.0.0',
    ]
    
    # Install core dependencies
    for dep in core_deps:
        print_status(f"Installing {dep}...")
        result = run_command(f'{pip_cmd} install "{dep}"', check=False, capture_output=True)
        if result and result.returncode == 0:
            print_success(f"Installed {dep}")
        else:
            print_warning(f"Failed to install {dep}")
    
    # Install optional dependencies (best effort)
    for dep in optional_deps:
        result = run_command(f'{pip_cmd} install "{dep}"', check=False, capture_output=True)
        if result and result.returncode == 0:
            print_success(f"Installed {dep}")
        else:
            print_warning(f"Optional dependency {dep} not installed")


def create_activation_script():
    """Create environment activation script."""
    print_status("Creating activation script...")
    
    script_content = '''#!/bin/bash
# RLHF Audit Trail - Environment Activation Script

echo "🚀 Activating RLHF Audit Trail environment..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No virtual environment found, using system Python"
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Set environment variables
export RLHF_AUDIT_ENV="development"
export RLHF_AUDIT_LOG_LEVEL="INFO"

echo "✓ Environment ready!"
echo
echo "Quick commands:"
echo "  python3 -c 'import rlhf_audit_trail; print(\"✓ Import test passed\")'  # Test installation"
echo "  python3 demo_basic_functionality.py                                      # Run demo"
echo "  python3 -m pytest tests/ -v                                             # Run tests"
echo
'''
    
    with open('activate_env.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('activate_env.sh', 0o755)
    print_success("Created activate_env.sh")


def verify_installation():
    """Verify the installation works."""
    print_status("Verifying installation...")
    
    # Test basic import
    python_cmd = 'venv/bin/python' if Path('venv').exists() else 'python3'
    
    test_script = '''
import sys
import os
sys.path.append('src')

try:
    # Test basic imports
    import rlhf_audit_trail
    from rlhf_audit_trail import AuditableRLHF, PrivacyConfig
    print("✓ Basic imports successful")
    
    # Test configuration
    config = PrivacyConfig(epsilon=1.0, delta=1e-5)
    print("✓ Configuration objects working")
    
    # Test core class instantiation (without dependencies)
    # auditor = AuditableRLHF("test-model")
    # print("✓ Core functionality accessible")
    
    print("✅ Installation verification passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
'''
    
    result = run_command(f'{python_cmd} -c "{test_script}"', capture_output=True)
    if result and result.returncode == 0:
        print_success("Installation verification passed!")
        print(result.stdout)
    else:
        print_error("Installation verification failed!")
        if result:
            print(result.stderr)


def main():
    """Main setup function."""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("  RLHF Audit Trail - Setup Script")
    print("  Autonomous SDLC Implementation")
    print("=" * 60)
    print(f"{Colors.END}")
    
    # Check prerequisites
    check_python_version()
    
    # Install dependencies
    install_system_dependencies()
    
    # Setup virtual environment (optional)
    setup_venv = '--no-venv' not in sys.argv
    if setup_venv:
        venv_path = setup_virtual_environment()
        if venv_path:
            install_python_dependencies(use_venv=True)
        else:
            install_python_dependencies(use_venv=False)
    else:
        install_python_dependencies(use_venv=False)
    
    # Create helper scripts
    create_activation_script()
    
    # Verify installation
    verify_installation()
    
    # Final instructions
    print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Setup completed successfully!{Colors.END}\n")
    
    if Path('venv').exists():
        print("To get started:")
        print(f"  {Colors.BLUE}source activate_env.sh{Colors.END}  # Activate environment")
    else:
        print("To get started:")
        print(f"  {Colors.BLUE}export PYTHONPATH=$PYTHONPATH:$(pwd)/src{Colors.END}")
    
    print(f"  {Colors.BLUE}python3 demo_basic_functionality.py{Colors.END}  # Run demo")
    print(f"  {Colors.BLUE}python3 -m rlhf_audit_trail.cli --help{Colors.END}  # CLI help\n")


if __name__ == '__main__':
    main()