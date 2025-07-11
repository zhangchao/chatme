"""
Simplified Multimodal AI Assistant Demo - Basic Version
"""
import streamlit as st
import cv2
import numpy as np
import time
import subprocess
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🤖 Multimodal AI Assistant Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleCameraManager:
    """Simple camera manager for demo"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        
    def start(self) -> bool:
        """Start camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            self.is_running = True
            return True
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def get_frame(self):
        """Get current frame"""
        if not self.is_running or not self.cap:
            return None
        try:
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    def stop(self):
        """Stop camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()

class SimpleVisionProcessor:
    """Simple vision processor for demo"""
    
    def analyze_image(self, image: np.ndarray) -> str:
        """Simple image analysis"""
        if image is None:
            return "No image available"
        
        height, width = image.shape[:2]
        brightness = np.mean(image)
        
        # Simple analysis
        analysis = []
        analysis.append(f"Image dimensions: {width}x{height}")
        
        if brightness > 150:
            analysis.append("Scene appears bright and well-lit")
        elif brightness < 80:
            analysis.append("Scene appears dark or dimly lit")
        else:
            analysis.append("Scene has moderate lighting")
        
        # Color analysis
        avg_color = np.mean(image, axis=(0, 1))
        dominant_color = "blue" if avg_color[2] > max(avg_color[0], avg_color[1]) else \
                        "green" if avg_color[1] > avg_color[0] else "red"
        analysis.append(f"Dominant color tone: {dominant_color}")
        
        return ". ".join(analysis) + "."

class SimpleTextToSpeech:
    """Simple TTS using macOS say command"""
    
    def speak(self, text: str):
        """Speak text"""
        if not text.strip():
            return
        try:
            subprocess.run(["say", text], check=True)
        except Exception as e:
            logger.error(f"TTS error: {e}")

def main():
    """Main demo application"""
    st.title("🤖 Multimodal AI Assistant Demo")
    st.markdown("*A simplified version demonstrating core functionality*")
    
    # Initialize session state
    if 'camera_manager' not in st.session_state:
        st.session_state.camera_manager = SimpleCameraManager()
    
    if 'vision_processor' not in st.session_state:
        st.session_state.vision_processor = SimpleVisionProcessor()
    
    if 'tts' not in st.session_state:
        st.session_state.tts = SimpleTextToSpeech()
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Camera controls
        st.subheader("📹 Camera")
        if st.button("🚀 Start Camera", type="primary"):
            if st.session_state.camera_manager.start():
                st.success("Camera started!")
            else:
                st.error("Failed to start camera. Please check permissions.")
        
        if st.button("🛑 Stop Camera"):
            st.session_state.camera_manager.stop()
            st.info("Camera stopped")
        
        # Conversation controls
        st.subheader("💬 Conversation")
        if st.button("🗑️ Clear History"):
            st.session_state.conversation = []
            st.success("Conversation cleared")
        
        # System info
        st.subheader("ℹ️ System Status")
        st.write("📹 Camera:", "✅ Available" if st.session_state.camera_manager.is_running else "❌ Not running")
        st.write("🔊 TTS:", "✅ Available")
        st.write("👁️ Vision:", "✅ Basic analysis")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        camera_placeholder = st.empty()
        
        # Display camera feed
        if st.session_state.camera_manager.is_running:
            frame = st.session_state.camera_manager.get_frame()
            if frame is not None:
                camera_placeholder.image(frame, channels="RGB", use_column_width=True)
            else:
                camera_placeholder.warning("No camera frame available")
        else:
            camera_placeholder.info("Start the camera to see the feed")
    
    with col2:
        st.subheader("💬 Conversation")
        
        # Display conversation history
        conversation_container = st.container()
        with conversation_container:
            if st.session_state.conversation:
                for i, entry in enumerate(st.session_state.conversation[-10:]):  # Show last 10
                    if entry['type'] == 'user':
                        st.markdown(f"**👤 You:** {entry['text']}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {entry['text']}")
                    st.markdown("---")
            else:
                st.info("No conversation yet. Try the buttons below!")
    
    # Interaction buttons
    st.subheader("🎯 Try These Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👁️ Analyze Current View", type="primary"):
            if st.session_state.camera_manager.is_running:
                frame = st.session_state.camera_manager.get_frame()
                if frame is not None:
                    analysis = st.session_state.vision_processor.analyze_image(frame)
                    
                    # Add to conversation
                    st.session_state.conversation.append({
                        'type': 'user',
                        'text': 'Analyze what you see',
                        'timestamp': time.time()
                    })
                    st.session_state.conversation.append({
                        'type': 'assistant',
                        'text': f"I can see: {analysis}",
                        'timestamp': time.time()
                    })
                    
                    # Speak the response
                    st.session_state.tts.speak(f"I can see: {analysis}")
                    
                    st.success("Analysis complete! Check the conversation.")
                    st.rerun()
                else:
                    st.warning("No camera frame available")
            else:
                st.warning("Please start the camera first")
    
    with col2:
        if st.button("👋 Say Hello"):
            greeting = "Hello! I'm your multimodal AI assistant. I can see through the camera and respond with both text and speech."
            
            # Add to conversation
            st.session_state.conversation.append({
                'type': 'user',
                'text': 'Say hello',
                'timestamp': time.time()
            })
            st.session_state.conversation.append({
                'type': 'assistant',
                'text': greeting,
                'timestamp': time.time()
            })
            
            # Speak the greeting
            st.session_state.tts.speak(greeting)
            
            st.success("Hello spoken! Check the conversation.")
            st.rerun()
    
    with col3:
        if st.button("📊 System Status"):
            status = "System status: Camera is running, vision analysis is active, and text-to-speech is working properly." if st.session_state.camera_manager.is_running else "System status: Camera is not running. Please start the camera for full functionality."
            
            # Add to conversation
            st.session_state.conversation.append({
                'type': 'user',
                'text': 'Check system status',
                'timestamp': time.time()
            })
            st.session_state.conversation.append({
                'type': 'assistant',
                'text': status,
                'timestamp': time.time()
            })
            
            # Speak the status
            st.session_state.tts.speak(status)
            
            st.success("Status reported! Check the conversation.")
            st.rerun()
    
    # Manual text input
    st.subheader("⌨️ Manual Input")
    user_input = st.text_input("Type a message to the assistant:")
    
    if st.button("Send Message") and user_input:
        # Simple response generation
        response = f"I received your message: '{user_input}'. "
        
        if st.session_state.camera_manager.is_running:
            frame = st.session_state.camera_manager.get_frame()
            if frame is not None:
                analysis = st.session_state.vision_processor.analyze_image(frame)
                response += f"Based on what I can see: {analysis}"
            else:
                response += "I can't see anything right now as the camera isn't providing a clear image."
        else:
            response += "I can't see anything right now as the camera isn't running."
        
        # Add to conversation
        st.session_state.conversation.append({
            'type': 'user',
            'text': user_input,
            'timestamp': time.time()
        })
        st.session_state.conversation.append({
            'type': 'assistant',
            'text': response,
            'timestamp': time.time()
        })
        
        # Speak the response
        st.session_state.tts.speak(response)
        
        st.success("Message processed! Check the conversation.")
        st.rerun()
    
    # Instructions
    st.subheader("📋 Instructions")
    st.markdown("""
    **How to use this demo:**
    1. **Start the Camera**: Click "Start Camera" in the sidebar
    2. **Grant Permissions**: Allow camera access when prompted by your browser/system
    3. **Try Actions**: Use the buttons above to interact with the assistant
    4. **Listen**: The assistant will speak responses using your system's text-to-speech
    5. **View History**: Check the conversation panel to see the interaction history
    
    **Features demonstrated:**
    - 📹 Real-time camera feed
    - 👁️ Basic image analysis (brightness, colors, dimensions)
    - 🔊 Text-to-speech responses
    - 💬 Conversation history
    - 🎯 Interactive buttons for common actions
    
    **Note**: This is a simplified demo. The full version includes advanced AI models, 
    speech recognition, and more sophisticated multimodal understanding.
    """)

if __name__ == "__main__":
    main()
