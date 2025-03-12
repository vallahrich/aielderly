"""
initialize_rasa.py

Script to set up Rasa for the elderly voice assistant.
This creates the necessary Rasa files if they don't exist.
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
rasa_dir = os.path.join(project_root, "nlp", "rasa")

def initialize_rasa():
    """Initialize Rasa project structure if it doesn't exist."""
    logger.info("Initializing Rasa project structure")
    
    # Check if Rasa directory exists
    if not os.path.exists(rasa_dir):
        os.makedirs(rasa_dir, exist_ok=True)
        logger.info(f"Created Rasa directory at {rasa_dir}")
    
    # Check if Rasa project is already initialized
    if os.path.exists(os.path.join(rasa_dir, "config.yml")):
        logger.info("Rasa project already initialized")
        return True
    
    # Change to the Rasa directory
    os.chdir(rasa_dir)
    
    try:
        # Initialize Rasa project
        logger.info("Running 'rasa init'...")
        process = subprocess.Popen(
            ["rasa", "init", "--no-prompt"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Process output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Check for any errors
        stderr = process.stderr.read()
        if stderr:
            logger.warning(f"Stderr output: {stderr}")
        
        # Check return code
        if process.returncode != 0:
            logger.error(f"Rasa init failed with return code {process.returncode}")
            return False
        
        logger.info("Rasa project initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Rasa project: {e}")
        return False

if __name__ == "__main__":
    if initialize_rasa():
        logger.info("Rasa initialization completed successfully")
    else:
        logger.error("Rasa initialization failed")
        sys.exit(1) 