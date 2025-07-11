"""
Voice Input Component for Browser-based Audio Recording
"""
import streamlit as st
import numpy as np
import tempfile
import os
import wave
import io
from audio_processor import SpeechToText, WHISPER_AVAILABLE
import logging

logger = logging.getLogger(__name__)

def create_voice_input_component():
    """Create a voice input component using browser audio recording"""
    
    # JavaScript code for audio recording
    audio_recorder_js = """
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    
    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.onload = function() {
                        const base64Audio = reader.result.split(',')[1];
                        // Send audio data to Streamlit
                        window.parent.postMessage({
                            type: 'audio_recorded',
                            data: base64Audio
                        }, '*');
                    };
                    reader.readAsDataURL(audioBlob);
                    
                    // Stop all tracks
                    stream.getTracks().forEach(track => track.stop());
                };
                
                mediaRecorder.start();
                isRecording = true;
                document.getElementById('recordBtn').textContent = '🛑 Stop Recording';
                document.getElementById('status').textContent = '🎤 Recording...';
            })
            .catch(err => {
                console.error('Error accessing microphone:', err);
                document.getElementById('status').textContent = '❌ Microphone access denied';
            });
    }
    
    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            document.getElementById('recordBtn').textContent = '🎤 Start Recording';
            document.getElementById('status').textContent = '⏳ Processing audio...';
        }
    }
    
    function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }
    </script>
    
    <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin: 10px 0;">
        <h4>🎤 Voice Input</h4>
        <button id="recordBtn" onclick="toggleRecording()" 
                style="padding: 10px 20px; font-size: 16px; border: none; border-radius: 5px; 
                       background-color: #ff4b4b; color: white; cursor: pointer;">
            🎤 Start Recording
        </button>
        <p id="status" style="margin-top: 10px; font-weight: bold;">Ready to record</p>
        <p style="font-size: 12px; color: #666;">
            Click to start recording your question. Click again to stop and process.
        </p>
    </div>
    """
    
    return audio_recorder_js

def create_simple_voice_input():
    """Create a simplified voice input using file upload"""
    st.subheader("🎤 Voice Input Alternative")
    
    # Option 1: File upload for audio
    st.write("**Option 1: Upload Audio File**")
    audio_file = st.file_uploader(
        "Upload an audio file (WAV, MP3, M4A)",
        type=['wav', 'mp3', 'm4a'],
        help="Record audio on your device and upload it here"
    )
    
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        
        if st.button("🔄 Transcribe Audio"):
            if WHISPER_AVAILABLE:
                with st.spinner("Transcribing audio..."):
                    try:
                        # Save uploaded audio to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_file.read())
                            tmp_path = tmp_file.name
                        
                        # Transcribe using Whisper
                        stt = SpeechToText()
                        if stt.load_model():
                            # Read the audio file as bytes
                            with open(tmp_path, 'rb') as f:
                                audio_bytes = f.read()
                            
                            transcription = stt.transcribe_audio(audio_bytes)
                            
                            if transcription:
                                st.success("✅ Transcription complete!")
                                st.write("**Transcribed text:**")
                                st.write(f"*\"{transcription}\"*")
                                
                                # Store in session state for use
                                st.session_state.voice_input_text = transcription
                                return transcription
                            else:
                                st.error("❌ Could not transcribe audio")
                        else:
                            st.error("❌ Failed to load speech recognition model")
                        
                        # Clean up temporary file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error transcribing audio: {e}")
                        logger.error(f"Audio transcription error: {e}")
            else:
                st.error("❌ Whisper not available for speech recognition")
    
    # Option 2: Text input as fallback
    st.write("**Option 2: Type Your Question**")
    text_input = st.text_area(
        "Or type your question here:",
        placeholder="What would you like to know about the image?",
        height=100
    )
    
    if text_input:
        st.session_state.voice_input_text = text_input
        return text_input
    
    return None

def process_voice_input(voice_text: str, image: np.ndarray, assistant) -> str:
    """Process voice input with image context"""
    if not voice_text or image is None:
        return "Please provide both voice input and an image."
    
    try:
        # Process the voice input with image context
        response = assistant.process_image_and_text(image, voice_text)
        return response
    except Exception as e:
        logger.error(f"Error processing voice input: {e}")
        return "I'm sorry, I encountered an error processing your voice input."

def create_voice_interaction_demo():
    """Create a complete voice interaction demo"""
    st.subheader("🗣️ Voice + Vision Interaction")
    
    if 'voice_input_text' not in st.session_state:
        st.session_state.voice_input_text = ""
    
    # Voice input section
    voice_text = create_simple_voice_input()
    
    # Display current voice input
    if st.session_state.voice_input_text:
        st.write("**Current voice input:**")
        st.info(f"🗣️ \"{st.session_state.voice_input_text}\"")
        
        # Process with image if available
        if st.session_state.get('current_image') is not None:
            if st.button("🤖 Process Voice + Image", type="primary"):
                if st.session_state.assistant.is_initialized:
                    with st.spinner("Processing voice input with image..."):
                        response = process_voice_input(
                            st.session_state.voice_input_text,
                            st.session_state.current_image,
                            st.session_state.assistant
                        )
                        
                        # Display response
                        st.success("✅ Response generated!")
                        st.write("**Assistant's response:**")
                        st.write(response)
                        
                        # Speak the response
                        if st.session_state.assistant.tts:
                            st.session_state.assistant.tts.speak(response)
                        
                        # Clear voice input
                        st.session_state.voice_input_text = ""
                        st.rerun()
                else:
                    st.warning("Please initialize the assistant first")
        else:
            st.warning("Please upload an image first to use voice interaction")

def create_recording_instructions():
    """Create instructions for voice recording"""
    st.subheader("📋 Voice Recording Instructions")
    
    with st.expander("How to use voice input"):
        st.markdown("""
        ### 🎤 Recording Audio on Your Device
        
        **For the best experience:**
        
        1. **Use your device's voice recorder:**
           - iPhone: Voice Memos app
           - Android: Voice Recorder or Google Recorder
           - Mac: QuickTime Player or Voice Memos
           - Windows: Voice Recorder app
        
        2. **Recording tips:**
           - Speak clearly and at normal pace
           - Record in a quiet environment
           - Keep recordings under 1 minute for best results
           - Save as WAV or MP3 format
        
        3. **Upload and transcribe:**
           - Upload your audio file using the file uploader
           - Click "Transcribe Audio" to convert speech to text
           - Review the transcription before processing
        
        ### 🔄 Alternative: Direct Text Input
        
        If voice recording isn't convenient, you can:
        - Type your question directly in the text area
        - Use the same multimodal processing capabilities
        - Get spoken responses via text-to-speech
        
        ### 🎯 Example Questions to Try
        
        - "What objects do you see in this image?"
        - "Describe the colors and mood of this scene"
        - "What story does this image tell?"
        - "Are there any people in this image? What are they doing?"
        - "What's the setting or location shown here?"
        """)

# Example usage function
def demo_voice_features():
    """Demonstrate voice features"""
    st.title("🎤 Voice Input Demo")
    
    create_recording_instructions()
    create_voice_interaction_demo()
    
    # Show system capabilities
    st.subheader("🔧 System Capabilities")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Speech Recognition:**")
        if WHISPER_AVAILABLE:
            st.success("✅ Whisper available")
        else:
            st.error("❌ Whisper not installed")
    
    with col2:
        st.write("**Text-to-Speech:**")
        st.success("✅ System TTS available")

if __name__ == "__main__":
    demo_voice_features()
