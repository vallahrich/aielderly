"""
update_intents.py

Script to update old intent names to new ones in Rasa training data.
"""
import os
import glob
import yaml
import re
from pathlib import Path

# Define the intent mapping
INTENT_MAPPING = {
    "Greeting": "greet",
    "Goodbye": "goodbye",
    "ThankYou": "thankyou",
    "Help": "help",
    "Time": "ask_time"
}

def update_yaml_file(file_path):
    """Update intent names in a YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original file
    with open(f"{file_path}.bak", 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Replace intent names
    for old_intent, new_intent in INTENT_MAPPING.items():
        # Replace intent declarations
        content = re.sub(r'intent:\s+' + old_intent, f'intent: {new_intent}', content)
        # Replace intent references
        content = re.sub(r'- intent:\s+' + old_intent, f'- intent: {new_intent}', content)
    
    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated {file_path}")

def main():
    """Update intent names in all Rasa training files."""
    # Get the directory containing this script
    base_dir = Path(__file__).parent
    
    # Find all YAML files in the data directory
    yaml_files = glob.glob(str(base_dir / "data" / "*.yml"))
    
    for file_path in yaml_files:
        update_yaml_file(file_path)
    
    print("All files updated successfully!")

if __name__ == "__main__":
    main() 