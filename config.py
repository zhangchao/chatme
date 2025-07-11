"""
Configuration settings for the Multimodal AI Assistant
"""
import os
from pathlib import Path

# Application Settings
APP_TITLE = "Multimodal AI Assistant"
APP_ICON = "🤖"

# Camera Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Audio Settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
AUDIO_FORMAT = "int16"
AUDIO_CHANNELS = 1
VOICE_ACTIVATION_THRESHOLD = 0.5
SILENCE_DURATION = 2.0  # seconds of silence before stopping recording

# MLX-VLM Settings
VLM_MODEL_NAME = "mlx-community/llava-1.5-7b-4bit"
VLM_MAX_TOKENS = 512
VLM_TEMPERATURE = 0.7

# Whisper Settings
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large
WHISPER_LANGUAGE = "en"

# Text-to-Speech Settings
TTS_ENGINE = "macos"  # "macos" or "pyttsx3"
TTS_RATE = 200
TTS_VOICE = "com.apple.speech.synthesis.voice.samantha"

# UI Settings
MAX_CONVERSATION_HISTORY = 50
PROCESSING_DELAY = 0.1
UI_UPDATE_INTERVAL = 0.1

# File Paths
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [TEMP_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Model Prompts
SYSTEM_PROMPT = """You are a helpful multimodal AI assistant that can see and hear. 
You receive both visual information from a camera and audio input from the user's speech.
Respond naturally and helpfully based on what you see and hear. Be conversational and engaging.
If you see the user making gestures or expressions, acknowledge them appropriately."""

VISION_PROMPT_TEMPLATE = """Based on this image, describe what you see. 
Focus on:
- People and their expressions/gestures
- Objects and their context
- Actions being performed
- Overall scene and environment

Image context: {context}"""

MULTIMODAL_PROMPT_TEMPLATE = """Visual context: {visual_context}
User said: "{audio_text}"
Previous conversation: {conversation_history}

Respond naturally to the user, taking into account both what you see and what they said."""
