# Elderly-Focused Voice Assistant

A voice assistant designed specifically for elderly users, with a focus on German language support.

## Features

- Speech recognition optimized for elderly voices
- Natural language understanding using Rasa
- Text-to-speech with clear, adjustable speech
- Simple, conversational interface

## Setup

1. Install Python 3.8 or higher
2. Clone this repository
3. Install dependencies:
   ```
   python setup.py
   ```
4. Initialize Rasa (if not already done):
   ```
   python initialize_rasa.py
   ```
5. Train the Rasa model:
   ```
   python nlp/rasa/train_rasa.py
   ```

## Running the Application

Start the voice assistant:
```
python main.py
```

For testing, you can run in text input mode:
```
python main.py --text-mode
```

## Rasa Development

To modify the voice assistant's understanding capabilities:

1. Edit the NLU data in `nlp/rasa/data/nlu.yml`
2. Edit the conversation flow in `nlp/rasa/data/stories.yml`
3. Retrain the model:
   ```
   python nlp/rasa/train_rasa.py
   ```

## Configuration

The application uses a configuration file at `config/settings.json`. A default configuration will be created on first run if the file doesn't exist. 