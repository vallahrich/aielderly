"""
check_dependencies.py

This script checks if all the required dependencies are installed 
and provides guidance on installing missing packages.
"""

import importlib
import subprocess
import sys

# List of required packages with friendly names
REQUIRED_PACKAGES = [
    ("pygame", "pygame"),
    ("gtts", "gTTS"),
    ("pyttsx3", "pyttsx3"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("torch", "torch"),
    ("rasa", "rasa"),
    ("numpy", "numpy"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
]

def check_package(package_info):
    """Check if a package is installed using pip."""
    import_name, display_name = package_info
    try:
        # First try importing
        importlib.import_module(import_name)
        return True
    except ImportError:
        try:
            # Check if it's installed with pip but just not importable
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", display_name.lower()],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

def main():
    """Check all dependencies and report status."""
    missing_packages = []
    
    print("Checking dependencies...")
    
    for package_info in REQUIRED_PACKAGES:
        import_name, display_name = package_info
        if check_package(package_info):
            print(f"✓ {display_name} is installed")
        else:
            print(f"✗ {display_name} is NOT installed")
            missing_packages.append(display_name)
    
    if missing_packages:
        print("\nThe following packages are missing:")
        for package in missing_packages:
            print(f"- {package}")
        
        print("\nYou can install them using:")
        print("python setup.py")
        print("\nOr manually with:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\nAll required packages are installed!")
        print("You should be able to run the application with: python main.py")

if __name__ == "__main__":
    main()
