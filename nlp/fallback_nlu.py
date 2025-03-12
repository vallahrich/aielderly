"""
fallback_nlu.py

A simple fallback NLU module that can be used when Rasa isn't available.
This provides basic intent recognition capabilities.
"""

import re
import random
import logging

logger = logging.getLogger(__name__)

class SimpleInterpreter:
    """
    A very basic interpreter that uses regex patterns to detect intents.
    For use when Rasa is not available.
    """
    
    def __init__(self, model_directory=None):
        """Initialize the simple interpreter."""
        logger.warning("Using fallback SimpleInterpreter instead of Rasa")
        
        # Define patterns for basic intents with ASCII equivalents for special characters
        self.intent_patterns = {
            "greeting": [
                r"hallo", r"guten\s+(morgen|tag|abend)", r"hi", r"servus", 
                r"gr(u|ü)(ss|ß)\s+dich", r"gr(u|ü)(ss|ß)\s+gott", r"hey",
                r"moin", r"gru(ß|ss)", r"tag", r"morgen", r"abend"
            ],
            "goodbye": [
                r"tsch(u|ü)ss", r"auf\s+wiedersehen", r"bis\s+sp(a|ä)ter", 
                r"bis\s+bald", r"ciao", r"mach's\s+gut", r"bye"
            ],
            "thankyou": [
                r"danke", r"vielen\s+dank", r"herzlichen\s+dank", 
                r"besten\s+dank", r"ich\s+bedanke\s+mich"
            ],
            "help": [
                r"hilfe", r"hilf\s+mir", r"kannst\s+du\s+(mir\s+)?helfen", 
                r"ich\s+brauche\s+hilfe", r"unterst(u|ü)tzung", r"wie\s+funktionierst\s+du",
                r"was\s+kannst\s+du(\s+tun)?", r"zeig\s+mir", r"erkl(a|ä)r", r"anleitung"
            ],
            "weather": [
                r"wetter", r"wie\s+ist\s+das\s+wetter", r"temperatur",
                r"regnet\s+es", r"scheint\s+die\s+sonne", r"wie\s+warm", r"wie\s+kalt",
                r"(wird\s+es|ist\s+es)\s+(heute|morgen|jetzt)", r"vorhersage", r"regen"
            ],
            "reminder_set": [
                r"erinner(e|ung)", r"erinnerst\s+du\s+mich", r"termin",
                r"vergiss\s+nicht", r"merken"
            ],
            "time": [
                r"(wie\s+sp(a|ä)t|uhrzeit|wie\s+viel\s+uhr)", r"datum", 
                r"welcher\s+tag", r"wochentag"
            ]
        }
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def parse(self, text):
        """
        Parse text to identify intent and entities.
        
        Args:
            text (str): The text to parse
            
        Returns:
            dict: A dict with intent and entities, mimicking Rasa's format
        """
        if not text:
            return {
                "intent": {"name": "", "confidence": 0.0},
                "entities": [],
                "text": text
            }
        
        print(f"SimpleInterpreter parsing: '{text}'")
        
        # Check each intent's patterns
        matched_intents = []
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text.lower()):
                    print(f"  Pattern '{pattern.pattern}' matched intent: {intent}")
                    matched_intents.append(intent)
                    break
        
        # Determine the most likely intent
        if matched_intents:
            # If multiple intents match, just pick the first one for simplicity
            selected_intent = matched_intents[0]
            confidence = 0.8  # Arbitrary high confidence for matched patterns
            print(f"Selected intent: {selected_intent} (confidence: {confidence})")
        else:
            selected_intent = "unknown"
            confidence = 0.3  # Low confidence for no matches
            print(f"No intent matched, using: {selected_intent} (confidence: {confidence})")
        
        # Return in a format similar to Rasa
        return {
            "intent": {
                "name": selected_intent,
                "confidence": confidence
            },
            "entities": [],  # We're not extracting entities in this simple version
            "text": text
        } 