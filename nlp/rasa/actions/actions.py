"""
actions.py

This module contains custom action handlers for the Rasa assistant.
These actions are executed when specific intents are triggered.
"""

from typing import Any, Text, Dict, List
from datetime import datetime, time
import logging
import requests

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionCheckWeather(Action):
    """Custom action to check weather for a specific location."""
    
    def name(self) -> Text:
        return "action_check_weather"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        location = tracker.get_slot("location")
        
        if location:
            # In a real implementation, you would call a weather API for the specific location
            dispatcher.utter_message(text=f"In {location} ist es heute sonnig bei 22 Grad.")
        else:
            dispatcher.utter_message(response="utter_default_weather")
        
        return []


class ActionSetReminder(Action):
    """Custom action to set a reminder."""
    
    def name(self) -> Text:
        return "action_set_reminder"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        time = tracker.get_slot("time")
        
        if time:
            dispatcher.utter_message(text=f"Ich habe eine Erinnerung für {time} Uhr eingerichtet.")
            return [SlotSet("time", time)]
        else:
            dispatcher.utter_message(response="utter_ask_reminder_time")
            return []


class ActionGetTime(Action):
    """Custom action to get the current time."""
    
    def name(self) -> Text:
        return "action_get_time"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        current_time = datetime.datetime.now().strftime("%H:%M")
        dispatcher.utter_message(text=f"Es ist jetzt {current_time} Uhr.")
        
        return []


class ActionReadNews(Action):
    """Custom action to read the news."""
    
    def name(self) -> Text:
        return "action_read_news"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="Die heutigen Nachrichten: In Berlin wurde ein neues Gesetz verabschiedet. Die Wettervorhersage für morgen ist sonnig.")
        
        return []


class ActionMakeCall(Action):
    """Custom action to initiate a call."""
    
    def name(self) -> Text:
        return "action_make_call"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        contact = tracker.get_slot("contact")
        
        if not contact:
            dispatcher.utter_message(response="utter_ask_call_who")
            return []
        
        dispatcher.utter_message(text=f"Ich stelle eine Verbindung zu {contact} her. Bitte warten Sie einen Moment.")
        return [SlotSet("contact", contact)]


class ActionTellTime(Action):
    def name(self) -> Text:
        return "action_tell_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        current_time = datetime.datetime.now().strftime("%H:%M")
        dispatcher.utter_message(text=f"Es ist jetzt {current_time} Uhr.")
        return []


class ActionGreet(Action):
    def name(self) -> Text:
        return "action_greet"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_greet")
        return []


class ActionGoodbye(Action):
    def name(self) -> Text:
        return "action_goodbye"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_goodbye")
        return []


class ActionTellWeather(Action):
    def name(self) -> Text:
        return "action_tell_weather"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Normally you would call a weather API here
        # This is just a placeholder response
        dispatcher.utter_message(text="Heute ist es sonnig mit 22 Grad.")
        return []


class ActionThankYou(Action):
    def name(self) -> Text:
        return "action_thank_you"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_youre_welcome")
        return []


class ActionHelp(Action):
    def name(self) -> Text:
        return "action_help"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_help")
        return []