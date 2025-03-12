"""
speech_recognition.py

This module handles audio recording and processing for the elderly-focused 
voice assistant, using the Wav2Vec 2.0 model for transcription.
"""

import pyaudio
import numpy as np
import wave
import os
import json
import time
from pathlib import Path
from .wav2vec_model import Wav2VecASR

class AudioRecorder:
    def __init__(self, config_path):
        """
        Initialize the audio recorder with the specified configuration.
        
        Args:
            config_path (str): Path to the configuration file containing audio settings
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = self.config["asr"]["sample_rate"]
        self.chunk_size = self.config["asr"]["chunk_size"]
        self.record_seconds = self.config["asr"]["record_seconds"]
        self.silence_threshold = self.config["asr"]["silence_threshold"]
        self.silence_duration = self.config["asr"]["silence_duration"]
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Initialize Wav2Vec ASR model
        self.asr_model = Wav2VecASR(config_path)
        
        # Create temp directory for audio recordings if it doesn't exist
        self.temp_dir = Path(self.config["asr"]["temp_dir"])
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def __del__(self):
        """Clean up PyAudio resources when the object is destroyed."""
        self.audio.terminate()
    
    def record_audio(self, timeout=None, voice_activity_detection=True):
        """
        Record audio from the microphone, optionally using voice activity detection
        to automatically stop recording after silence is detected.
        
        Args:
            timeout (float, optional): Maximum recording time in seconds
            voice_activity_detection (bool): Whether to use VAD to stop recording
            
        Returns:
            numpy.ndarray: Recorded audio as a numpy array
            int: Sample rate of the recorded audio
        """
        print("Listening... (Speak now)")
        
        # Open microphone stream
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        silent_chunks = 0
        start_time = time.time()
        
        # Record audio
        try:
            # Initial waiting period to avoid cutting off the beginning
            time.sleep(0.2)
            
            while True:
                # Check timeout
                if timeout and (time.time() - start_time > timeout):
                    break
                
                # Read audio chunk
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Voice activity detection (simple amplitude-based)
                if voice_activity_detection:
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    volume = np.abs(audio_chunk).mean()
                    
                    # Check if the volume is below the silence threshold
                    if volume < self.silence_threshold:
                        silent_chunks += 1
                        # Stop recording after silence_duration seconds of silence
                        if silent_chunks > self.sample_rate / self.chunk_size * self.silence_duration:
                            break
                    else:
                        silent_chunks = 0
                
                # If not using VAD, record for a fixed duration
                elif len(frames) > int(self.sample_rate / self.chunk_size * self.record_seconds):
                    break
        
        finally:
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
        
        # Convert frames to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        
        print("Recording finished.")
        return audio_data, self.sample_rate
    
    def save_audio(self, audio_data, sample_rate, filename=None):
        """
        Save the recorded audio to a WAV file.
        
        Args:
            audio_data (numpy.ndarray): Audio data as numpy array
            sample_rate (int): Sample rate of the audio
            filename (str, optional): Filename to save to (default: timestamp-based)
            
        Returns:
            str: Path to the saved audio file
        """
        if filename is None:
            filename = f"recording_{int(time.time())}.wav"
        
        filepath = self.temp_dir / filename
        
        # Convert float32 back to int16 for saving
        audio_int16 = (audio_data * 32768.0).astype(np.int16)
        
        # Save as WAV file
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return str(filepath)
    
    def transcribe_audio(self, audio_data=None, audio_path=None):
        """
        Transcribe audio using the Wav2Vec 2.0 model.
        
        Args:
            audio_data (numpy.ndarray, optional): Audio data as numpy array
            audio_path (str, optional): Path to an audio file to transcribe
            
        Returns:
            str: Transcribed text in German
        
        Note: Either audio_data or audio_path must be provided
        """
        if audio_data is None and audio_path is None:
            raise ValueError("Either audio_data or audio_path must be provided")
        
        # If audio path is provided, load the audio file
        if audio_path:
            with wave.open(audio_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe the audio
        transcription = self.asr_model.transcribe(audio_data, self.sample_rate)
        
        return transcription
    
    def listen_and_transcribe(self):
        """
        Record audio from the microphone and transcribe it using the ASR model.
        
        Returns:
            str: Transcribed text
        """
        try:
            print("Listening... (Speak now)")
            
            # Record audio
            audio_array = self.record_audio()
            
            # Check if audio contains actual speech
            if self._is_silent(audio_array):
                print("No speech detected (silence)")
                return ""
            
            print("Processing speech...")
            # Transcribe audio
            transcription = self.asr_model.transcribe(audio_array)
            
            print(f"Transcribed: '{transcription}'")
            return transcription
        
        except Exception as e:
            print(f"Error in speech recognition: {str(e)}")
            return ""

    def _is_silent(self, audio_array, threshold=0.01):
        """
        Check if the audio is mostly silence.
        
        Args:
            audio_array: The audio data
            threshold: The amplitude threshold for considering as speech
            
        Returns:
            bool: True if audio is silent, False otherwise
        """
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(np.square(audio_array)))
        print(f"Audio RMS: {rms} (threshold: {threshold})")
        return rms < threshold