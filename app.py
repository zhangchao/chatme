"""
Multimodal AI Assistant - Main Streamlit Application
"""
import streamlit as st
import cv2
import numpy as np
import time
import threading
from PIL import Image
import logging
from typing import Optional

# Import our modules
import config
from utils import CameraManager, check_system_requirements, save_temp_image, cleanup_temp_files
from audio_processor import AudioRecorder, SpeechToText, TextToSpeech
from vision_processor import MultimodalProcessor
from conversation_manager import ConversationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

class MultimodalAssistant:
    """Main application class"""
    
    def __init__(self):
        self.camera_manager = None
        self.audio_recorder = None
        self.multimodal_processor = None
        self.conversation_manager = None
        self.tts = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize all components"""
        if self.is_initialized:
            return True
            
        try:
            # Initialize components
            self.camera_manager = CameraManager()
            self.audio_recorder = AudioRecorder()
            self.multimodal_processor = MultimodalProcessor()
            self.conversation_manager = ConversationManager()
            self.tts = TextToSpeech()
            
            # Start camera
            if not self.camera_manager.start():
                st.error("Failed to start camera")
                return False
            
            self.is_initialized = True
            logger.info("Multimodal assistant initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing assistant: {e}")
            st.error(f"Initialization error: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.camera_manager:
                self.camera_manager.stop()
            if self.audio_recorder:
                self.audio_recorder.cleanup()
            cleanup_temp_files()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def process_user_input(self, audio_text: str, image: np.ndarray):
        """Process user input and generate response"""
        try:
            # Save image snapshot
            image_path = save_temp_image(image, "conversation")
            
            # Add user message to conversation
            self.conversation_manager.add_user_message(
                text=audio_text,
                image_path=image_path,
                metadata={"timestamp": time.time()}
            )
            
            # Process multimodal input
            response = self.multimodal_processor.process_multimodal_input(
                image=image,
                audio_text=audio_text,
                conversation_history=self.conversation_manager.get_recent_history()
            )
            
            # Add assistant response to conversation
            self.conversation_manager.add_assistant_message(
                text=response,
                metadata={"timestamp": time.time()}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error processing your request."

def main():
    """Main application function"""
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = MultimodalAssistant()
    
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False
    
    if 'last_response' not in st.session_state:
        st.session_state.last_response = ""
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # System status
        st.subheader("System Status")
        requirements = check_system_requirements()
        
        for component, status in requirements.items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {component.title()}")
        
        # Initialize button
        if st.button("🚀 Initialize Assistant", type="primary"):
            with st.spinner("Initializing..."):
                if st.session_state.assistant.initialize():
                    st.success("Assistant initialized!")
                else:
                    st.error("Failed to initialize assistant")
        
        # Audio controls
        st.subheader("🎤 Audio Controls")
        
        if st.button("🎤 Start Listening"):
            if st.session_state.assistant.is_initialized:
                st.session_state.is_listening = True
                st.session_state.assistant.audio_recorder.start_recording(
                    callback=lambda text: handle_speech_input(text)
                )
                st.success("Listening started...")
            else:
                st.warning("Please initialize the assistant first")
        
        if st.button("🛑 Stop Listening"):
            st.session_state.is_listening = False
            if st.session_state.assistant.audio_recorder:
                st.session_state.assistant.audio_recorder.stop_recording()
            st.info("Listening stopped")
        
        # Conversation controls
        st.subheader("💬 Conversation")
        
        if st.button("🗑️ Clear History"):
            if st.session_state.assistant.conversation_manager:
                st.session_state.assistant.conversation_manager.clear_conversation()
                st.success("Conversation cleared")
        
        if st.button("💾 Save Conversation"):
            if st.session_state.assistant.conversation_manager:
                if st.session_state.assistant.conversation_manager.save_conversation():
                    st.success("Conversation saved")
                else:
                    st.error("Failed to save conversation")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        camera_placeholder = st.empty()
        
        # Display camera feed
        if st.session_state.assistant.is_initialized:
            display_camera_feed(camera_placeholder)
        else:
            camera_placeholder.info("Initialize the assistant to start camera feed")
    
    with col2:
        st.subheader("💬 Conversation")
        conversation_placeholder = st.empty()
        
        # Display conversation history
        if st.session_state.assistant.conversation_manager:
            display_conversation_history(conversation_placeholder)
    
    # Status bar
    st.subheader("📊 Status")
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        listening_status = "🎤 Listening" if st.session_state.is_listening else "🔇 Not Listening"
        st.write(listening_status)
    
    with status_col2:
        if st.session_state.assistant.conversation_manager:
            stats = st.session_state.assistant.conversation_manager.get_session_stats()
            st.write(f"💬 Messages: {stats['total_messages']}")
    
    with status_col3:
        if st.session_state.assistant.tts and st.session_state.assistant.tts.is_speaking:
            st.write("🔊 Speaking")
        else:
            st.write("🔇 Silent")
    
    # Text input for manual testing
    st.subheader("⌨️ Manual Input")
    manual_input = st.text_input("Type a message to test the assistant:")
    
    if st.button("Send") and manual_input:
        if st.session_state.assistant.is_initialized:
            # Get current camera frame
            frame = st.session_state.assistant.camera_manager.capture_snapshot()
            if frame is not None:
                response = st.session_state.assistant.process_user_input(manual_input, frame)
                st.session_state.last_response = response
                
                # Speak the response
                st.session_state.assistant.tts.speak(response)
                
                st.success(f"Response: {response}")
            else:
                st.warning("No camera frame available")
        else:
            st.warning("Please initialize the assistant first")

def display_camera_feed(placeholder):
    """Display real-time camera feed"""
    try:
        frame = st.session_state.assistant.camera_manager.get_frame()
        if frame is not None:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        else:
            placeholder.warning("No camera frame available")
    except Exception as e:
        placeholder.error(f"Camera error: {e}")

def display_conversation_history(placeholder):
    """Display conversation history"""
    try:
        history = st.session_state.assistant.conversation_manager.get_recent_history(10)
        
        if not history:
            placeholder.info("No conversation yet. Start talking to the assistant!")
            return
        
        conversation_html = "<div style='max-height: 400px; overflow-y: auto;'>"
        
        for entry in history:
            timestamp = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            
            if entry.type == "user":
                conversation_html += f"""
                <div style='margin-bottom: 10px; padding: 10px; background-color: #e3f2fd; border-radius: 10px;'>
                    <strong>👤 User ({timestamp}):</strong><br>
                    {entry.text}
                </div>
                """
            else:
                conversation_html += f"""
                <div style='margin-bottom: 10px; padding: 10px; background-color: #f3e5f5; border-radius: 10px;'>
                    <strong>🤖 Assistant ({timestamp}):</strong><br>
                    {entry.text}
                </div>
                """
        
        conversation_html += "</div>"
        placeholder.markdown(conversation_html, unsafe_allow_html=True)
        
    except Exception as e:
        placeholder.error(f"Error displaying conversation: {e}")

def handle_speech_input(text: str):
    """Handle speech input from audio recorder"""
    try:
        if text and st.session_state.assistant.is_initialized:
            # Get current camera frame
            frame = st.session_state.assistant.camera_manager.capture_snapshot()
            if frame is not None:
                # Process the input
                response = st.session_state.assistant.process_user_input(text, frame)
                st.session_state.last_response = response
                
                # Speak the response
                st.session_state.assistant.tts.speak(response)
                
                # Force UI update
                st.rerun()
                
    except Exception as e:
        logger.error(f"Error handling speech input: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Application error: {e}")
    finally:
        # Cleanup
        if 'assistant' in st.session_state:
            st.session_state.assistant.cleanup()
