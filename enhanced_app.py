"""
Enhanced Multimodal AI Assistant with Image Upload and Voice Input
"""
import streamlit as st
import numpy as np
import time
import io
import base64
import tempfile
import os
from PIL import Image
import logging
from typing import Optional

# Import our modules
import config
from utils import save_temp_image, cleanup_temp_files, check_system_requirements
from audio_processor import SpeechToText, TextToSpeech, WHISPER_AVAILABLE, PYAUDIO_AVAILABLE
from vision_processor import MultimodalProcessor
from conversation_manager import ConversationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🤖 Enhanced Multimodal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnhancedMultimodalAssistant:
    """Enhanced application class with image upload and voice input"""
    
    def __init__(self):
        self.multimodal_processor = None
        self.conversation_manager = None
        self.tts = None
        self.stt = None
        self.is_initialized = False
        self.current_image = None
        
    def initialize(self):
        """Initialize all components"""
        if self.is_initialized:
            return True
            
        try:
            # Initialize components
            self.multimodal_processor = MultimodalProcessor()
            self.conversation_manager = ConversationManager()
            self.tts = TextToSpeech()
            
            # Initialize speech-to-text if available
            if WHISPER_AVAILABLE:
                self.stt = SpeechToText()
            
            self.is_initialized = True
            logger.info("Enhanced multimodal assistant initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing assistant: {e}")
            st.error(f"Initialization error: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            cleanup_temp_files()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def process_image_and_text(self, image: np.ndarray, text: str) -> str:
        """Process image and text input to generate response"""
        try:
            # Save image snapshot
            image_path = save_temp_image(image, "uploaded")
            
            # Add user message to conversation
            self.conversation_manager.add_user_message(
                text=text,
                image_path=image_path,
                metadata={"timestamp": time.time(), "input_type": "text_and_image"}
            )
            
            # Process multimodal input
            response = self.multimodal_processor.process_multimodal_input(
                image=image,
                audio_text=text,
                conversation_history=self.conversation_manager.get_recent_history()
            )
            
            # Add assistant response to conversation
            self.conversation_manager.add_assistant_message(
                text=response,
                metadata={"timestamp": time.time()}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return "I'm sorry, I encountered an error processing your request."

def record_audio():
    """Record audio using browser's audio recording capabilities"""
    # This is a placeholder for browser-based audio recording
    # In a real implementation, you would use streamlit-webrtc or similar
    st.info("🎤 Audio recording feature requires additional setup. Please use text input for now.")
    return None

def process_uploaded_image(uploaded_file):
    """Process uploaded image file"""
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array, image
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}")
        return None, None

def main():
    """Main application function"""
    st.title("🤖 Enhanced Multimodal AI Assistant")
    st.markdown("*Upload images and interact with voice or text input*")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = EnhancedMultimodalAssistant()

    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

    if 'last_response' not in st.session_state:
        st.session_state.last_response = ""

    if 'voice_input_text' not in st.session_state:
        st.session_state.voice_input_text = ""
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # System status
        st.subheader("System Status")
        requirements = check_system_requirements()
        
        st.write("🧠 MLX:", "✅" if requirements.get("mlx", False) else "❌")
        st.write("🗣️ Whisper:", "✅" if WHISPER_AVAILABLE else "❌")
        st.write("🎤 Audio:", "✅" if PYAUDIO_AVAILABLE else "❌")
        st.write("🔊 TTS:", "✅")
        
        # Initialize button
        if st.button("🚀 Initialize Assistant", type="primary"):
            with st.spinner("Initializing..."):
                if st.session_state.assistant.initialize():
                    st.success("Assistant initialized!")
                else:
                    st.error("Failed to initialize assistant")
        
        # Image upload section
        st.subheader("📷 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'gif'],
            help="Upload an image to analyze"
        )
        
        if uploaded_file is not None:
            # Process uploaded image
            image_array, pil_image = process_uploaded_image(uploaded_file)
            if image_array is not None:
                st.session_state.current_image = image_array
                st.success("✅ Image uploaded successfully!")
                
                # Show image info
                st.write(f"📏 Size: {pil_image.size}")
                st.write(f"🎨 Mode: {pil_image.mode}")
            else:
                st.error("❌ Failed to process image")
        
        # Voice input section
        st.subheader("🎤 Voice Input")

        if WHISPER_AVAILABLE:
            # Audio file upload for voice input
            audio_file = st.file_uploader(
                "Upload audio file (WAV, MP3, M4A)",
                type=['wav', 'mp3', 'm4a'],
                help="Record audio on your device and upload it here"
            )

            if audio_file is not None:
                st.audio(audio_file, format='audio/wav')

                if st.button("🔄 Transcribe Audio"):
                    with st.spinner("Transcribing audio..."):
                        try:
                            # Save uploaded audio to temporary file
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                tmp_file.write(audio_file.read())
                                tmp_path = tmp_file.name

                            # Transcribe using Whisper
                            if st.session_state.assistant.stt:
                                if st.session_state.assistant.stt.load_model():
                                    with open(tmp_path, 'rb') as f:
                                        audio_bytes = f.read()

                                    transcription = st.session_state.assistant.stt.transcribe_audio(audio_bytes)

                                    if transcription:
                                        st.success("✅ Transcription complete!")
                                        st.session_state.voice_input_text = transcription
                                        st.write(f"**Transcribed:** *\"{transcription}\"*")
                                    else:
                                        st.error("❌ Could not transcribe audio")
                                else:
                                    st.error("❌ Failed to load speech model")

                            # Clean up
                            os.unlink(tmp_path)

                        except Exception as e:
                            st.error(f"❌ Transcription error: {e}")
        else:
            st.warning("Whisper not available - voice input disabled")

        # Voice input text display
        if 'voice_input_text' in st.session_state and st.session_state.voice_input_text:
            st.write("**Voice Input:**")
            st.info(f"🗣️ \"{st.session_state.voice_input_text}\"")

            if st.button("🤖 Process Voice Input"):
                if st.session_state.current_image is not None and st.session_state.assistant.is_initialized:
                    with st.spinner("Processing voice input..."):
                        response = st.session_state.assistant.process_image_and_text(
                            st.session_state.current_image,
                            st.session_state.voice_input_text
                        )
                        st.session_state.last_response = response

                        # Speak response
                        if st.session_state.assistant.tts:
                            st.session_state.assistant.tts.speak(response)

                        # Clear voice input
                        st.session_state.voice_input_text = ""
                        st.success("Voice input processed!")
                        st.rerun()
                else:
                    st.warning("Please upload an image and initialize the assistant first")
        
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
        st.subheader("📷 Current Image")
        
        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image, caption="Uploaded Image", use_column_width=True)
            
            # Quick analysis button
            if st.button("🔍 Quick Analysis", type="secondary"):
                if st.session_state.assistant.is_initialized:
                    with st.spinner("Analyzing image..."):
                        response = st.session_state.assistant.process_image_and_text(
                            st.session_state.current_image, 
                            "Please analyze this image and describe what you see."
                        )
                        st.session_state.last_response = response
                        
                        # Speak the response
                        if st.session_state.assistant.tts:
                            st.session_state.assistant.tts.speak(response)
                        
                        st.success("Analysis complete!")
                        st.rerun()
                else:
                    st.warning("Please initialize the assistant first")
        else:
            st.info("📤 Upload an image using the sidebar to get started")
    
    with col2:
        st.subheader("💬 Conversation")
        conversation_placeholder = st.empty()
        
        # Display conversation history
        if st.session_state.assistant.conversation_manager:
            display_conversation_history(conversation_placeholder)
    
    # Text input section
    st.subheader("💭 Ask About the Image")
    
    if st.session_state.current_image is not None:
        user_question = st.text_area(
            "What would you like to know about this image?",
            placeholder="e.g., 'What objects do you see?', 'Describe the colors and mood', 'What's happening in this scene?'",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🤔 Ask Question", type="primary") and user_question:
                if st.session_state.assistant.is_initialized:
                    with st.spinner("Processing your question..."):
                        response = st.session_state.assistant.process_image_and_text(
                            st.session_state.current_image, 
                            user_question
                        )
                        st.session_state.last_response = response
                        
                        # Speak the response
                        if st.session_state.assistant.tts:
                            st.session_state.assistant.tts.speak(response)
                        
                        st.success("Response generated!")
                        st.rerun()
                else:
                    st.warning("Please initialize the assistant first")
        
        with col2:
            if st.button("🔊 Repeat Last Response"):
                if st.session_state.last_response and st.session_state.assistant.tts:
                    st.session_state.assistant.tts.speak(st.session_state.last_response)
                    st.success("Speaking response...")
                else:
                    st.warning("No response to repeat")
    else:
        st.info("📷 Please upload an image first to ask questions about it")
    
    # Quick action buttons
    if st.session_state.current_image is not None:
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🏷️ Identify Objects"):
                if st.session_state.assistant.is_initialized:
                    response = st.session_state.assistant.process_image_and_text(
                        st.session_state.current_image,
                        "What objects can you identify in this image? List them."
                    )
                    st.session_state.assistant.tts.speak(response)
                    st.rerun()
        
        with col2:
            if st.button("🎨 Describe Colors"):
                if st.session_state.assistant.is_initialized:
                    response = st.session_state.assistant.process_image_and_text(
                        st.session_state.current_image,
                        "Describe the colors, lighting, and visual mood of this image."
                    )
                    st.session_state.assistant.tts.speak(response)
                    st.rerun()
        
        with col3:
            if st.button("📖 Tell a Story"):
                if st.session_state.assistant.is_initialized:
                    response = st.session_state.assistant.process_image_and_text(
                        st.session_state.current_image,
                        "Create a short story or narrative based on what you see in this image."
                    )
                    st.session_state.assistant.tts.speak(response)
                    st.rerun()
        
        with col4:
            if st.button("🔍 Detailed Analysis"):
                if st.session_state.assistant.is_initialized:
                    response = st.session_state.assistant.process_image_and_text(
                        st.session_state.current_image,
                        "Provide a comprehensive analysis of this image including composition, style, content, and any notable details."
                    )
                    st.session_state.assistant.tts.speak(response)
                    st.rerun()

    # Instructions and help section
    st.subheader("📋 How to Use This Application")

    with st.expander("📖 Complete User Guide", expanded=False):
        st.markdown("""
        ### 🚀 Getting Started

        1. **Initialize the Assistant**
           - Click "🚀 Initialize Assistant" in the sidebar
           - Wait for successful initialization

        2. **Upload an Image**
           - Use the "📷 Image Upload" section in the sidebar
           - Supported formats: JPG, JPEG, PNG, GIF
           - The image will appear in the main area

        ### 🎤 Voice Input Options

        **Method 1: Upload Audio File**
        - Record audio on your device using:
          - iPhone: Voice Memos app
          - Android: Voice Recorder
          - Mac: QuickTime Player
          - Windows: Voice Recorder
        - Upload the audio file in the sidebar
        - Click "🔄 Transcribe Audio" to convert speech to text
        - Click "🤖 Process Voice Input" to analyze with the image

        **Method 2: Direct Text Input**
        - Type your question in the text area
        - Click "🤔 Ask Question" to get a response

        ### ⚡ Quick Actions

        - **🔍 Quick Analysis**: Get immediate image analysis
        - **🏷️ Identify Objects**: List objects in the image
        - **🎨 Describe Colors**: Analyze colors and mood
        - **📖 Tell a Story**: Create narrative from the image
        - **🔍 Detailed Analysis**: Comprehensive image breakdown

        ### 🔊 Audio Responses

        - All responses are automatically spoken using text-to-speech
        - Use "🔊 Repeat Last Response" to hear the response again
        - Adjust your system volume for comfortable listening

        ### 💬 Conversation Features

        - View conversation history in the right panel
        - Save conversations using "💾 Save Conversation"
        - Clear history with "🗑️ Clear History"

        ### 🎯 Example Questions to Try

        - "What objects do you see in this image?"
        - "Describe the colors and lighting in this scene"
        - "What's the mood or atmosphere of this image?"
        - "Are there any people? What are they doing?"
        - "What story does this image tell?"
        - "What's the setting or location shown here?"
        - "What details stand out to you?"

        ### 🔧 Troubleshooting

        **Audio Issues:**
        - Ensure audio files are in WAV, MP3, or M4A format
        - Check that Whisper is properly installed
        - Try shorter audio recordings (under 1 minute)

        **Image Issues:**
        - Use common image formats (JPG, PNG, etc.)
        - Ensure images are not corrupted
        - Try smaller file sizes if upload fails

        **Performance:**
        - Initialize the assistant before use
        - Wait for processing to complete
        - Check system requirements in the sidebar
        """)

def display_conversation_history(placeholder):
    """Display conversation history"""
    try:
        history = st.session_state.assistant.conversation_manager.get_recent_history(10)
        
        if not history:
            placeholder.info("No conversation yet. Upload an image and ask questions!")
            return
        
        conversation_html = "<div style='max-height: 400px; overflow-y: auto;'>"
        
        for entry in history:
            timestamp = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            
            if entry.type == "user":
                conversation_html += f"""
                <div style='margin-bottom: 10px; padding: 10px; background-color: #e3f2fd; border-radius: 10px;'>
                    <strong>👤 You ({timestamp}):</strong><br>
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
