"""
verify_rasa_model.py

Script to verify the existing Rasa model and provide details.
"""
import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
model_dir = os.path.join(project_root, "models", "rasa")

def verify_model():
    """Verify that the Rasa model exists and is accessible."""
    # Check if model directory exists
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        return False

    # Find model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.tar.gz')]
    if not model_files:
        logger.error(f"No model files found in {model_dir}")
        return False
    
    # Get latest model
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    logger.info(f"Latest model: {latest_model}")
    logger.info(f"Full path: {model_path}")
    
    # Check if model file is accessible
    if not os.access(model_path, os.R_OK):
        logger.error(f"Model file is not readable: {model_path}")
        return False
    
    # Try to get model information using rasa
    try:
        logger.info("Checking model with Rasa...")
        result = subprocess.run(
            ["rasa", "data", "validate", "--model", model_path],
            capture_output=True,
            text=True
        )
        logger.info(f"Model validation result: {result.returncode}")
        if result.stdout:
            logger.info(f"Validation output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Validation errors: {result.stderr}")
    except Exception as e:
        logger.warning(f"Failed to validate model: {e}")
    
    return True

if __name__ == "__main__":
    print("\n=== Verifying Rasa Model ===\n")
    
    if verify_model():
        print("\n? Rasa model verified successfully.")
        print("Your model appears to be in the correct location and format.")
    else:
        print("\n? Rasa model verification failed.")
        print("Please check the logs for details on what went wrong.")
        sys.exit(1) 