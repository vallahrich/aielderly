"""
main.py

Main application module that integrates speech recognition (ASR), natural language
understanding (NLU) using Rasa, and text-to-speech (TTS) components for an elderly-focused
voice assistant.
"""

import os
import sys
import json
import random
import time
import datetime
import logging
from pathlib import Path
import argparse

# Configure logging BEFORE imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("senior_voice_assistant.log")
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import components with error handling
try:
    from asr.speech_recognition import AudioRecorder
    from tts.tts_engine import TTSEngine
    
    # Import Rasa components with multiple fallbacks for different versions
    # Define a variable to hold which interpreter we'll use
    RASA_INTERPRETER = None
    INTERPRETER_PATH = None
    
    # Try all known locations for Rasa interpreters across different versions
    possible_rasa_imports = [
        # Rasa 3.x
        ("rasa.core.interpreter", "RasaNLUInterpreter"),
        # Rasa 2.x
        ("rasa.nlu.model", "Interpreter"),
        # Rasa 1.x legacy
        ("rasa.model", "get_model"),
        # Other possible locations
        ("rasa.shared.nlu.interpreter", "Interpreter"),
        ("rasa.shared.core.interpreter", "NaturalLanguageInterpreter"),
        # Last resort - use regex interpreter
        ("rasa.shared.nlu.interpreter", "RegexInterpreter")
    ]
    
    for module_path, class_name in possible_rasa_imports:
        try:
            logger.info(f"Trying to import {class_name} from {module_path}")
            module = __import__(module_path, fromlist=[class_name])
            RASA_INTERPRETER = getattr(module, class_name)
            INTERPRETER_PATH = f"{module_path}.{class_name}"
            logger.info(f"Successfully imported {INTERPRETER_PATH}")
            break
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to import {module_path}.{class_name}: {e}")
    
    if RASA_INTERPRETER is None:
        logger.error("Could not import any Rasa interpreter classes")
        raise ImportError("No compatible Rasa interpreter found")
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.info("Please run 'python setup.py' to install all required packages.")
    logger.info("Or run 'python check_dependencies.py' to check which packages are missing.")
    sys.exit(1)

class InterpreterWrapper:
    """
    A wrapper class for Rasa interpreters to ensure consistent interface
    regardless of the specific interpreter class used.
    """
    def __init__(self, interpreter, interpreter_path):
        self.interpreter = interpreter
        self.interpreter_path = interpreter_path
    
    def parse(self, text):
        """
        Parse text to determine intent and entities.
        Handles different interpreter interfaces.
        
        Args:
            text (str): The text to parse
            
        Returns:
            dict: Intent and entity information
        """
        logger.debug(f"Parsing text with {self.interpreter_path}: {text}")
        
        # Handle RegexInterpreter specifically
        if "RegexInterpreter" in self.interpreter_path:
            # Simple regex-based intent detection
            if any(word in text.lower() for word in ["hallo", "hi", "tag", "moin", "grüß", "gruss"]):
                intent_name = "greeting"
                confidence = 1.0
            elif any(word in text.lower() for word in ["tschüss", "wiedersehen", "bye", "ciao"]):
                intent_name = "goodbye"
                confidence = 1.0
            elif any(word in text.lower() for word in ["danke", "dank"]):
                intent_name = "thankyou"
                confidence = 1.0
            elif any(word in text.lower() for word in ["hilfe", "helfen", "hilf"]):
                intent_name = "help"
                confidence = 1.0
            elif any(word in text.lower() for word in ["zeit", "uhr", "spät"]):
                intent_name = "time"
                confidence = 1.0
            elif any(word in text.lower() for word in ["wetter", "temperatur", "regen", "sonne"]):
                intent_name = "weather"
                confidence = 1.0
            elif any(word in text.lower() for word in ["erinner", "vergiss nicht", "termin"]):
                intent_name = "reminder_set"
                confidence = 1.0
            else:
                intent_name = "unknown"
                confidence = 0.3
            
            return {
                "intent": {"name": intent_name, "confidence": confidence},
                "entities": [],
                "text": text
            }
        
        # Handle other Rasa interpreters
        elif hasattr(self.interpreter, "parse"):
            # Standard parse method
            return self.interpreter.parse(text)
        
        elif hasattr(self.interpreter, "interpret"):
            # Some interpreters use interpret instead of parse
            return self.interpreter.interpret(text)
        
        else:
            # Last resort - return unknown intent
            logger.warning(f"Unknown interpreter interface for {self.interpreter_path}")
            return {
                "intent": {"name": "unknown", "confidence": 0.0},
                "entities": [],
                "text": text
            }

class VoiceAssistant:
    def __init__(self, config_path):
        """
        Initialize the voice assistant with all components.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file {config_path}")
            raise
        
        # Initialize components
        logger.info("Initializing ASR module")
        self.asr = AudioRecorder(config_path)
        
        # Flag to indicate if we use fallback
        self.use_fallback = False
        
        logger.info("Initializing NLU module")
        try:
            # Try to load the Rasa NLU model
            model_path = os.path.join(project_root, "models", "rasa")
            logger.info(f"Looking for Rasa models in: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Rasa model directory not found at {model_path}")
                raise FileNotFoundError(f"Rasa model not found at {model_path}")
            
            # Find the latest model in the directory
            model_files = [f for f in os.listdir(model_path) if f.endswith('.tar.gz')]
            if not model_files:
                logger.error(f"No Rasa model files found in {model_path}")
                raise FileNotFoundError(f"No Rasa model files found in {model_path}")
            
            # Use the latest model
            latest_model = sorted(model_files)[-1]
            model_file = os.path.join(model_path, latest_model)
            logger.info(f"Found Rasa model: {latest_model}")
            logger.info(f"Full path: {model_file}")
            
            # Use Agent instead of interpreters
            from rasa.core.agent import Agent
            logger.info("Loading model with Rasa Agent")
            agent = Agent.load(model_file)
            
            # Create a wrapper for the agent to provide consistent interface with async support
            class AgentWrapper:
                def __init__(self, agent):
                    self.agent = agent
                
                def parse(self, text):
                    """Synchronous wrapper for the async parse_message method."""
                    try:
                        # Import asyncio at the top of the method
                        import asyncio
                        
                        # Check if we're already in an event loop
                        try:
                            loop = asyncio.get_running_loop()
                            # We're in an event loop, use run_coroutine_threadsafe
                            return asyncio.run_coroutine_threadsafe(
                                self.agent.parse_message(text),
                                loop
                            ).result()
                        except RuntimeError:
                            # We're not in an event loop, use asyncio.run
                            return asyncio.run(self.agent.parse_message(text))
                    except Exception as e:
                        logger.error(f"Error in parse method: {e}")
                        # Fallback to a simple structure
                        return {
                            "intent": {"name": "unknown", "confidence": 0.0},
                            "entities": [],
                            "text": text
                        }
            
            self.nlu = AgentWrapper(agent)
            logger.info("Rasa model loaded successfully using Agent")
            self.use_fallback = False
        except Exception as e:
            logger.error(f"Failed to load Rasa model: {e}")
            logger.warning("Falling back to SimpleInterpreter")
            
            # Import and use the SimpleInterpreter as fallback
            from nlp.fallback_nlu import SimpleInterpreter
            self.nlu = SimpleInterpreter()
            self.use_fallback = True
            logger.info("Using SimpleInterpreter as fallback")
        
        logger.info("Initializing TTS module")
        self.tts = TTSEngine(config_path)
        
        # Set default language
        self.language = self.config["app"]["default_language"]
        
        # Flag to track if the assistant is active
        self.is_active = True
        
        # Create required directories
        required_dirs = ["temp", os.path.join("temp", "asr"), os.path.join("temp", "tts"), "models", os.path.join("models", "rasa")]
        for directory in required_dirs:
            dir_path = os.path.join(project_root, directory)
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info("Voice Assistant initialized successfully")
    
    def get_response(self, intent_data):
        """
        Generate a response based on the detected intent and entities.
        """
        try:
            # Extract intent name and confidence
            intent = intent_data.get("intent", {})
            intent_name = intent.get("name", "unknown")
            confidence = intent.get("confidence", 0.0)
            
            logger.info(f"Generating response for intent: {intent_name} (confidence: {confidence:.2f})")
            
            # Set a default confidence threshold if not in config
            confidence_threshold = 0.6  # Default value
            
            # Try to get from config, but use default if not available
            if "app" in self.config and "confidence_threshold" in self.config["app"]:
                confidence_threshold = self.config["app"]["confidence_threshold"]
            elif "nlu" in self.config and "confidence_threshold" in self.config["nlu"]:
                confidence_threshold = self.config["nlu"]["confidence_threshold"]
            
            # Check if confidence is below threshold
            if confidence < confidence_threshold:
                logger.info(f"Intent confidence too low: {confidence:.2f} < {confidence_threshold}")
                return random.choice(self.config["responses"].get("unknown", ["Das habe ich nicht verstanden."]))
            
            # Map Rasa intents to response keys
            intent_mapping = {
                "ask_time": "time",
                "greet": "greeting",
                "goodbye": "goodbye",
                "ask_weather": "weather",
                "thankyou": "thankyou",
                "help": "help"
            }
            
            # Map the intent if needed
            response_key = intent_mapping.get(intent_name, intent_name)
            logger.info(f"Mapped intent '{intent_name}' to response key '{response_key}'")
            
            # Verify that responses exists in the config
            if "responses" not in self.config:
                logger.error("No 'responses' section found in configuration")
                return "Es tut mir leid, ich kann momentan nicht antworten."
            
            # Handle special cases
            if response_key == "time":
                # Get current time
                now = datetime.datetime.now()
                time_str = now.strftime("%H:%M")
                
                # Check if time templates exist
                if "time" in self.config["responses"] and self.config["responses"]["time"]:
                    response_template = random.choice(self.config["responses"]["time"])
                    return response_template.replace("{time}", time_str)
                else:
                    # Fallback time response
                    return f"Es ist jetzt {time_str} Uhr."
            
            # Get response from config for the intent
            if response_key in self.config["responses"]:
                responses = self.config["responses"][response_key]
                if responses:
                    return random.choice(responses)
                else:
                    logger.warning(f"Empty response list for intent: {response_key}")
                    return "Ich verstehe, was Sie meinen, aber ich weiß nicht, wie ich antworten soll."
            else:
                logger.warning(f"No response configured for intent: {response_key}. Available keys: {list(self.config['responses'].keys())}")
                # Try to use the original intent name
                if intent_name in self.config["responses"]:
                    responses = self.config["responses"][intent_name]
                    if responses:
                        return random.choice(responses)
                
                # Last resort fallback
                return "Ich verstehe Ihre Anfrage, aber ich weiß nicht, wie ich darauf antworten soll."
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error details
            return "Es tut mir leid, es gab einen Fehler bei der Verarbeitung Ihrer Anfrage."
    
    def confirm_critical_action(self, action):
        """
        Ask for confirmation before performing critical actions.
        
        Args:
            action (str): Description of the action to confirm
            
        Returns:
            bool: True if confirmed, False otherwise
        """
        # Only confirm if enabled in config
        if not self.config["app"]["confirm_critical_commands"]:
            return True
        
        # Generate confirmation question
        confirmation_templates = self.config["responses"]["confirm"]
        confirmation_question = random.choice(confirmation_templates).format(action=action)
        
        # Ask for confirmation
        self.tts.speak(confirmation_question)
        
        # Listen for confirmation
        confirmation_text = self.asr.listen_and_transcribe()
        
        # Check if confirmation is positive
        confirmation_keywords = ["ja", "okay", "sicher", "natürlich", "bitte", "gerne"]
        return any(keyword in confirmation_text.lower() for keyword in confirmation_keywords)
    
    def handle_conversation_loop(self):
        """
        Main conversation loop for the voice assistant.
        """
        # Introduce the assistant
        intro_message = f"Hallo! Ich bin {self.config['app']['name']}, Ihr persönlicher Sprachassistent. Wie kann ich Ihnen heute helfen?"
        logger.info("Starting conversation")
        self.tts.speak(intro_message)
        
        # Check for text input mode
        text_input_mode = self.config.get("app", {}).get("text_input_mode", False)
        
        # Main conversation loop
        while self.is_active:
            try:
                # Get input (either voice or text)
                if text_input_mode:
                    # Text input mode for testing
                    logger.info("Text input mode active. Type your message:")
                    transcription = input("> ")
                else:
                    # Voice input mode
                    logger.info("Listening for user input...")
                    transcription = self.asr.listen_and_transcribe()
                
                if not transcription or transcription.isspace():
                    logger.info("No input detected, continuing...")
                    # Add a small pause to prevent CPU overuse in the loop
                    time.sleep(0.5)
                    continue
                
                logger.info(f"Received input: {transcription}")
                
                # Parse intent using appropriate method
                if self.use_fallback:
                    # Using SimpleInterpreter
                    intent_data = self.nlu.parse(transcription)
                elif hasattr(self.nlu, 'parse'):
                    # Standard parse method
                    intent_data = self.nlu.parse(transcription)
                elif hasattr(self.nlu, 'interpret'):
                    # Some interpreters use interpret instead of parse
                    intent_data = self.nlu.interpret(transcription)
                else:
                    # Last resort - handle as unknown
                    logger.warning(f"Unknown interpreter interface")
                    intent_data = {
                        "intent": {"name": "unknown", "confidence": 0.0},
                        "entities": [],
                        "text": transcription
                    }
                
                intent_name = intent_data.get("intent", {}).get("name", "")
                logger.info(f"Detected intent: {intent_name}")
                
                # Check for exit command
                if intent_name == "goodbye":
                    logger.info("Goodbye intent detected, exiting...")
                    response = self.get_response(intent_data)
                    self.tts.speak(response)
                    break
                
                # Generate and speak response
                response = self.get_response(intent_data)
                logger.info(f"Response: {response}")
                self.tts.speak(response)
                
                # Small pause between interactions
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected, exiting...")
                self.tts.speak("Auf Wiedersehen!")
                break
            
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}")
                error_msg = "Es tut mir leid, es gab einen Fehler. Bitte versuchen Sie es erneut."
                self.tts.speak(error_msg)
                # Add a pause after error
                time.sleep(1)
    
    def run(self):
        """
        Run the voice assistant.
        """
        try:
            self.handle_conversation_loop()
        except Exception as e:
            logger.error(f"Error running voice assistant: {str(e)}")
        finally:
            logger.info("Voice assistant shutting down")
            self.is_active = False

def main():
    """
    Main entry point for the application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Elderly-focused voice assistant")
    parser.add_argument("--text-mode", action="store_true", help="Run in text input mode (for testing)")
    parser.add_argument("--voice-mode", action="store_true", help="Run in voice input mode (default)")
    args = parser.parse_args()
    
    # Set up configuration path
    config_dir = os.path.join(project_root, "config")
    config_path = os.path.join(config_dir, "settings.json")
    
    # Check if config directory and file exist
    if not os.path.exists(config_dir):
        logger.error(f"Configuration directory not found at {config_dir}")
        os.makedirs(config_dir, exist_ok=True)
        logger.info(f"Created configuration directory at {config_dir}")
        
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found at {config_path}")
        logger.info("Creating default configuration file")
        
        # Create a default configuration
        default_config = {
            "app": {
                "name": "Senior Voice Assistant",
                "default_language": "de",
                "confirm_critical_commands": True,
                "text_input_mode": True,  # Start in text input mode for testing
                "confidence_threshold": 0.6
            },
            "asr": {
                "model_name": "facebook/wav2vec2-large-xlsr-53-german",
                "fine_tuned_model_path": "",
                "sample_rate": 16000,
                "record_seconds": 5
            },
            "nlu": {
                "confidence_threshold": 0.6
            },
            "tts": {
                "engine": "pyttsx3",
                "language": "de",
                "voice_id": None,
                "rate": 150,
                "volume": 1.0,
                "temp_dir": "temp/tts"
            },
            "responses": {
                "greeting": [
                    "Hallo! Wie kann ich Ihnen helfen?",
                    "Guten Tag! Womit kann ich Ihnen behilflich sein?",
                    "Grüß Gott! Was kann ich für Sie tun?"
                ],
                "goodbye": [
                    "Auf Wiedersehen! Haben Sie einen schönen Tag.",
                    "Bis bald! Passen Sie auf sich auf.",
                    "Tschüss! Ich freue mich auf unser nächstes Gespräch."
                ],
                "thankyou": [
                    "Gern geschehen!",
                    "Keine Ursache.",
                    "Bitte sehr!"
                ],
                "help": [
                    "Ich kann Ihnen mit verschiedenen Dingen helfen. Sie können nach dem Wetter fragen, Erinnerungen setzen oder die Uhrzeit erfragen.",
                    "Ich bin Ihr persönlicher Assistent. Ich kann Wetter, Uhrzeit und Erinnerungen für Sie verwalten."
                ],
                "unknown": [
                    "Entschuldigung, das habe ich nicht verstanden.",
                    "Könnten Sie das bitte wiederholen?",
                    "Tut mir leid, ich verstehe nicht, was Sie meinen."
                ],
                "weather": [
                    "Heute scheint die Sonne bei angenehmen 22 Grad.",
                    "Es ist leicht bewölkt mit Temperaturen um 18 Grad."
                ],
                "reminder_set": [
                    "Ich habe eine Erinnerung für Sie gesetzt.",
                    "Ihre Erinnerung wurde eingerichtet."
                ],
                "time": [
                    "Es ist jetzt {time} Uhr."
                ],
                "confirm": [
                    "Möchten Sie wirklich {action}?",
                    "Sind Sie sicher, dass Sie {action} möchten?"
                ]
            }
        }
        
        # Save default configuration
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Created default configuration file at {config_path}")
    
    # Update text input mode based on command line args
    if args.text_mode or args.voice_mode:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if args.text_mode:
            config["app"]["text_input_mode"] = True
            logger.info("Running in text input mode")
        elif args.voice_mode:
            config["app"]["text_input_mode"] = False
            logger.info("Running in voice input mode")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    
    # Create and run the voice assistant
    try:
        assistant = VoiceAssistant(config_path)
        assistant.run()
    except Exception as e:
        logger.error(f"Failed to initialize or run voice assistant: {str(e)}")

if __name__ == "__main__":
    main()
