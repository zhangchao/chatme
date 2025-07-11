"""
Audio processing module for speech-to-text and text-to-speech
"""
import threading
import time
import subprocess
import tempfile
import os
from typing import Optional, Callable
import numpy as np
import logging
import config
from utils import AudioBuffer

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available - speech recognition disabled")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available - audio recording disabled")

class SpeechToText:
    """Handles speech-to-text conversion using Whisper"""
    
    def __init__(self, model_size: str = config.WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load Whisper model"""
        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available")
            return False

        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            self.is_loaded = True
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            return False
    
    def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio data to text"""
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Transcribe audio
            result = self.model.transcribe(temp_path, language=config.WHISPER_LANGUAGE)
            text = result["text"].strip()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return text if text else None
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

class TextToSpeech:
    """Handles text-to-speech conversion"""
    
    def __init__(self, engine: str = config.TTS_ENGINE):
        self.engine = engine
        self.is_speaking = False
        
    def speak(self, text: str, callback: Optional[Callable] = None):
        """Convert text to speech"""
        if not text.strip():
            return
            
        self.is_speaking = True
        
        def _speak():
            try:
                if self.engine == "macos":
                    self._speak_macos(text)
                else:
                    self._speak_pyttsx3(text)
            except Exception as e:
                logger.error(f"Error in text-to-speech: {e}")
            finally:
                self.is_speaking = False
                if callback:
                    callback()
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def _speak_macos(self, text: str):
        """Use macOS built-in say command"""
        try:
            subprocess.run([
                "say", 
                "-v", config.TTS_VOICE,
                "-r", str(config.TTS_RATE),
                text
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error with macOS say command: {e}")
            # Fallback to simple say
            subprocess.run(["say", text])
    
    def _speak_pyttsx3(self, text: str):
        """Use pyttsx3 for text-to-speech"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', config.TTS_RATE)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"Error with pyttsx3: {e}")

class AudioRecorder:
    """Handles audio recording with voice activity detection"""
    
    def __init__(self):
        self.audio = None
        if PYAUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
            except Exception as e:
                logger.error(f"Failed to initialize PyAudio: {e}")
        self.is_recording = False
        self.audio_buffer = AudioBuffer()
        self.recording_thread = None
        
    def start_recording(self, callback: Optional[Callable[[str], None]] = None):
        """Start continuous audio recording"""
        if not PYAUDIO_AVAILABLE or not self.audio:
            logger.error("PyAudio not available - cannot start recording")
            return False

        if self.is_recording:
            return True

        self.is_recording = True
        self.audio_buffer.clear()
        
        def _record():
            try:
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=config.AUDIO_CHANNELS,
                    rate=config.AUDIO_SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=config.AUDIO_CHUNK_SIZE
                )
                
                logger.info("Started audio recording")
                silence_start = None
                
                while self.is_recording:
                    try:
                        data = stream.read(config.AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                        
                        # Simple voice activity detection
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        volume = np.sqrt(np.mean(audio_array**2))
                        
                        if volume > config.VOICE_ACTIVATION_THRESHOLD * 1000:
                            # Voice detected
                            silence_start = None
                            self.audio_buffer.put(data)
                        else:
                            # Silence detected
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > config.SILENCE_DURATION:
                                # Process accumulated audio
                                if callback:
                                    audio_data = self.audio_buffer.get_all()
                                    if audio_data:
                                        self._process_audio_data(audio_data, callback)
                                silence_start = None
                        
                        time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                        
                    except Exception as e:
                        logger.error(f"Error in recording loop: {e}")
                        break
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                logger.error(f"Error in audio recording: {e}")
        
        self.recording_thread = threading.Thread(target=_record, daemon=True)
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        logger.info("Stopped audio recording")
    
    def _process_audio_data(self, audio_data: bytes, callback: Callable[[str], None]):
        """Process recorded audio data"""
        def _process():
            try:
                # Convert raw audio to WAV format
                wav_data = self._raw_to_wav(audio_data)
                
                # Transcribe using Whisper
                stt = SpeechToText()
                text = stt.transcribe_audio(wav_data)
                
                if text and callback:
                    callback(text)
                    
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
        
        thread = threading.Thread(target=_process, daemon=True)
        thread.start()
    
    def _raw_to_wav(self, raw_data: bytes) -> bytes:
        """Convert raw audio data to WAV format"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(config.AUDIO_CHANNELS)
                wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wav_file.setframerate(config.AUDIO_SAMPLE_RATE)
                wav_file.writeframes(raw_data)
            
            with open(temp_file.name, 'rb') as f:
                wav_data = f.read()
            
            os.unlink(temp_file.name)
            return wav_data
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()
