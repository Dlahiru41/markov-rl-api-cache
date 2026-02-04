#!/usr/bin/env python3
"""
Setup script for Enterprise Demo dependencies.

This script checks and installs all required dependencies for running
the ENTERPRISE_LIVE_DEMO.py presentation script.
"""

import sys
import subprocess
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    print(f"  Installing {package_name}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ⚠️  Warning: Failed to install {package_name}")
        return False

def main():
    print("=" * 80)
    print("ENTERPRISE DEMO - DEPENDENCY SETUP")
    print("=" * 80)
    print()
    
    # Define required packages
    packages = [
        ('gymnasium', 'gymnasium'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('torch', 'torch'),
        ('scikit-learn', 'sklearn'),
    ]
    
    print("Checking dependencies...")
    print()
    
    missing = []
    installed = []
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            print(f"  ✓ {package_name:<20} already installed")
            installed.append(package_name)
        else:
            print(f"  ✗ {package_name:<20} not found")
            missing.append(package_name)
    
    print()
    
    if not missing:
        print("=" * 80)
        print("✅ ALL DEPENDENCIES INSTALLED!")
        print("=" * 80)
        print()
        print("You can now run the demo:")
        print("  python ENTERPRISE_LIVE_DEMO.py")
        print()
        return 0
    
    print(f"Found {len(missing)} missing package(s).")
    print()
    
    # Ask for confirmation
    response = input("Install missing packages now? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        print()
        print("Installing missing packages...")
        print()
        
        failed = []
        for package in missing:
            if install_package(package):
                print(f"    ✓ {package} installed")
            else:
                failed.append(package)
        
        print()
        
        if failed:
            print("=" * 80)
            print("⚠️  PARTIAL INSTALLATION")
            print("=" * 80)
            print()
            print("The following packages failed to install:")
            for pkg in failed:
                print(f"  • {pkg}")
            print()
            print("Please install them manually:")
            print(f"  pip install {' '.join(failed)}")
            print()
            return 1
        else:
            print("=" * 80)
            print("✅ INSTALLATION COMPLETE!")
            print("=" * 80)
            print()
            print("You can now run the demo:")
            print("  python ENTERPRISE_LIVE_DEMO.py")
            print()
            return 0
    else:
        print()
        print("Installation cancelled.")
        print()
        print("To install manually:")
        print(f"  pip install {' '.join(missing)}")
        print()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print("Installation cancelled by user.")
        sys.exit(1)
