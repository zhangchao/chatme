"""
Utility functions for camera, audio, and system operations
"""
import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Tuple, Any
import logging
from pathlib import Path
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraManager:
    """Manages camera operations and frame capture"""
    
    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def start(self) -> bool:
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            self.is_running = True
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Camera stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        if not self.is_running or not self.cap:
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                return frame
            return None
            
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    def capture_snapshot(self) -> Optional[np.ndarray]:
        """Capture a snapshot for processing"""
        frame = self.get_frame()
        if frame is not None:
            # Convert BGR to RGB for processing
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

class AudioBuffer:
    """Thread-safe audio buffer for recording"""
    
    def __init__(self, maxsize: int = 1000):
        self.buffer = queue.Queue(maxsize=maxsize)
        self.is_recording = False
        
    def put(self, data: bytes):
        """Add audio data to buffer"""
        if not self.buffer.full():
            self.buffer.put(data)
    
    def get_all(self) -> bytes:
        """Get all audio data from buffer"""
        audio_data = b""
        while not self.buffer.empty():
            try:
                audio_data += self.buffer.get_nowait()
            except queue.Empty:
                break
        return audio_data
    
    def clear(self):
        """Clear the buffer"""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break

def save_temp_image(image: np.ndarray, prefix: str = "snapshot") -> str:
    """Save image to temporary file and return path"""
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = config.TEMP_DIR / filename
    
    try:
        cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return ""

def cleanup_temp_files(max_age_hours: int = 24):
    """Clean up old temporary files"""
    try:
        current_time = time.time()
        for file_path in config.TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > (max_age_hours * 3600):
                    file_path.unlink()
                    logger.info(f"Cleaned up old temp file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")

def format_conversation_history(history: list, max_entries: int = 5) -> str:
    """Format conversation history for context"""
    if not history:
        return "No previous conversation."
    
    recent_history = history[-max_entries:]
    formatted = []
    
    for entry in recent_history:
        if entry.get("type") == "user":
            formatted.append(f"User: {entry.get('text', '')}")
        elif entry.get("type") == "assistant":
            formatted.append(f"Assistant: {entry.get('text', '')}")
    
    return "\n".join(formatted)

def check_system_requirements() -> dict:
    """Check if system meets requirements"""
    requirements = {
        "camera": False,
        "audio": False,
        "mlx": False,
        "whisper": False
    }
    
    # Check camera
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            requirements["camera"] = True
            cap.release()
    except:
        pass
    
    # Check audio
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        if pa.get_device_count() > 0:
            requirements["audio"] = True
        pa.terminate()
    except Exception as e:
        logger.info(f"Audio not available: {e}")
        pass
    
    # Check MLX
    try:
        import mlx.core as mx
        requirements["mlx"] = True
    except:
        pass
    
    # Check Whisper
    try:
        import whisper
        requirements["whisper"] = True
    except Exception as e:
        logger.info(f"Whisper not available: {e}")
        pass
    
    return requirements
