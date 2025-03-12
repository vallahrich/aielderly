"""
setup.py

Simple script to install all required packages for the elderly-focused voice assistant.
"""

import sys
import subprocess
import os
from pathlib import Path

def install_requirements():
    """Install packages from requirements.txt"""
    print("Installing required packages...")
    requirements_path = os.path.join(Path(__file__).parent, "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print(f"Error: requirements.txt not found at {requirements_path}")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def main():
    if install_requirements():
        print("\nSetup completed successfully!")
        print("You can now run the voice assistant using: python main.py")
    else:
        print("\nSetup failed. Please try installing packages manually:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 