"""
train_rasa.py

Script to train the Rasa NLU model for the elderly voice assistant.
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
project_root = Path(__file__).parent.parent.parent
rasa_dir = os.path.join(project_root, "nlp", "rasa")
model_dir = os.path.join(project_root, "models", "rasa")

def train_rasa_model():
    """Train the Rasa model and save it to the models directory."""
    logger.info(f"Starting Rasa model training in {rasa_dir}")
    
    # Make sure the model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Change to the Rasa directory
    os.chdir(rasa_dir)
    
    try:
        # Run Rasa train command
        logger.info("Running 'rasa train'...")
        result = subprocess.run(
            ["rasa", "train", "--out", model_dir], 
            check=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info("Rasa model training completed successfully")
        logger.info(f"Model saved to {model_dir}")
        
        # Check if model was created
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.tar.gz')]
        if model_files:
            logger.info(f"Created model files: {', '.join(model_files)}")
        else:
            logger.warning("No model files found in output directory")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to train Rasa model: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

if __name__ == "__main__":
    if train_rasa_model():
        logger.info("Rasa model training completed successfully")
    else:
        logger.error("Rasa model training failed")
        sys.exit(1) 