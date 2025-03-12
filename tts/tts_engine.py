"""
tts_engine.py

This module handles text-to-speech conversion for the elderly-focused
voice assistant, using a German TTS system.
"""

import os
import json
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies with error handling
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    logger.warning("pyttsx3 not available. Some TTS features will be limited.")
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    logger.warning("gTTS not available. Some TTS features will be limited.")
    GTTS_AVAILABLE = False

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except (ImportError, pygame.error):
    logger.warning("pygame not available or failed to initialize mixer. Audio playback will be limited.")
    PYGAME_AVAILABLE = False

import time

class TTSEngine:
    def __init__(self, config_path):
        """
        Initialize the Text-to-Speech engine.
        
        Args:
            config_path (str): Path to the configuration file containing TTS settings
        """
        # Load configuration
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {config_path}")
            # Use default configuration
            self.config = {
                "tts": {
                    "engine": "pyttsx3" if PYTTSX3_AVAILABLE else "gtts" if GTTS_AVAILABLE else "none",
                    "language": "de",
                    "voice_id": None,
                    "rate": 150,
                    "volume": 1.0,
                    "temp_dir": "temp/tts"
                }
            }
        
        # TTS settings
        self.engine_type = self.config["tts"]["engine"]
        self.language = self.config["tts"]["language"]
        self.voice_id = self.config["tts"].get("voice_id")
        self.rate = self.config["tts"].get("rate", 150)
        self.volume = self.config["tts"].get("volume", 1.0)
        
        # Create temp directory for TTS output if it doesn't exist
        self.temp_dir = Path(self.config["tts"].get("temp_dir", "temp/tts"))
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize the TTS engine
        self.engine = None
        if self.engine_type == "pyttsx3" and PYTTSX3_AVAILABLE:
            self._init_pyttsx3()
        elif self.engine_type == "gtts" and not GTTS_AVAILABLE:
            logger.warning("gTTS engine selected but not available. Trying pyttsx3 instead.")
            self.engine_type = "pyttsx3" if PYTTSX3_AVAILABLE else "none"
            if PYTTSX3_AVAILABLE:
                self._init_pyttsx3()
    
    def _init_pyttsx3(self):
        """
        Initialize the pyttsx3 TTS engine.
        """
        try:
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Set voice (if specified)
            if self.voice_id:
                self.engine.setProperty('voice', self.voice_id)
            else:
                # Try to find a German voice
                voices = self.engine.getProperty('voices')
                german_voices = [v for v in voices if "german" in v.name.lower() or "deutsch" in v.name.lower()]
                if german_voices:
                    self.engine.setProperty('voice', german_voices[0].id)
                    logger.info(f"Using German voice: {german_voices[0].name}")
                else:
                    logger.warning("No German voice found. Using default voice.")
            
            logger.info("pyttsx3 TTS engine initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing pyttsx3: {str(e)}")
            self.engine = None
    
    def get_available_voices(self):
        """
        Get a list of available voices for the TTS engine.
        
        Returns:
            list: Available voice information
        """
        voices = []
        
        if self.engine_type == "pyttsx3" and self.engine:
            pyttsx3_voices = self.engine.getProperty('voices')
            for voice in pyttsx3_voices:
                voices.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages,
                    'gender': voice.gender,
                    'age': voice.age
                })
        
        return voices
    
    def speak_pyttsx3(self, text):
        """
        Use pyttsx3 to convert text to speech and play it.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.engine:
            logger.error("pyttsx3 engine not initialized.")
            return False
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Error with pyttsx3 speech: {str(e)}")
            return False
    
    def speak_gtts(self, text):
        """
        Use Google Text-to-Speech (gTTS) to convert text to speech and play it.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a temporary file
            temp_file = self.temp_dir / f"tts_output_{int(time.time())}.mp3"
            
            # Generate speech with gTTS
            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(str(temp_file))
            
            # Play the audio
            pygame.mixer.music.load(str(temp_file))
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
                
            return True
        except Exception as e:
            logger.error(f"Error with gTTS speech: {str(e)}")
            return False
    
    def speak(self, text):
        """
        Convert text to speech and play it using the configured TTS engine.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Speaking: {text}")
        
        if self.engine_type == "pyttsx3":
            return self.speak_pyttsx3(text)
        elif self.engine_type == "gtts":
            return self.speak_gtts(text)
        else:
            logger.error(f"Unsupported TTS engine: {self.engine_type}")
            # Fall back to pyttsx3
            return self.speak_pyttsx3(text)
    
    def save_to_file(self, text, filename=None):
        """
        Save the speech to an audio file without playing it.
        
        Args:
            text (str): Text to convert to speech
            filename (str, optional): Filename to save to (default: timestamp-based)
            
        Returns:
            str: Path to the saved audio file
        """
        if filename is None:
            filename = f"speech_{int(time.time())}.mp3"
        
        filepath = self.temp_dir / filename
        
        try:
            if self.engine_type == "pyttsx3" and self.engine:
                # Save as WAV for pyttsx3
                wav_path = str(filepath).replace('.mp3', '.wav')
                self.engine.save_to_file(text, wav_path)
                self.engine.runAndWait()
                return wav_path
            else:
                # Save as MP3 for gTTS
                tts = gTTS(text=text, lang=self.language, slow=False)
                tts.save(str(filepath))
                return str(filepath)
        except Exception as e:
            logger.error(f"Error saving speech to file: {str(e)}")
            return None