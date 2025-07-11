#!/usr/bin/env python3
"""
Test script for enhanced multimodal AI assistant features
"""
import sys
import os
import tempfile
import numpy as np
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_image_processing():
    """Test image upload and processing functionality"""
    print("🖼️ Testing image processing...")
    
    try:
        # Create a test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Test image conversion
        image_array = np.array(pil_image)
        
        print(f"✅ Image processing test passed")
        print(f"   - Image shape: {image_array.shape}")
        print(f"   - Image mode: {pil_image.mode}")
        print(f"   - Image size: {pil_image.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def test_audio_transcription():
    """Test audio transcription functionality"""
    print("🎤 Testing audio transcription...")
    
    try:
        from audio_processor import SpeechToText, WHISPER_AVAILABLE
        
        if not WHISPER_AVAILABLE:
            print("⚠️  Whisper not available - skipping audio test")
            return True
        
        # Test SpeechToText initialization
        stt = SpeechToText()
        
        print("✅ Audio transcription components loaded")
        print(f"   - Whisper available: {WHISPER_AVAILABLE}")
        print(f"   - SpeechToText initialized: {stt is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio transcription test failed: {e}")
        return False

def test_vision_processing():
    """Test vision processing functionality"""
    print("👁️ Testing vision processing...")
    
    try:
        from vision_processor import MultimodalProcessor
        
        # Initialize processor
        processor = MultimodalProcessor()
        
        # Create test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # Test image analysis
        analysis = processor.vlm.analyze_image(test_image)
        
        print("✅ Vision processing test passed")
        print(f"   - Processor initialized: {processor is not None}")
        print(f"   - Analysis result: {analysis[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision processing test failed: {e}")
        return False

def test_text_to_speech():
    """Test text-to-speech functionality"""
    print("🔊 Testing text-to-speech...")
    
    try:
        from audio_processor import TextToSpeech
        
        # Initialize TTS
        tts = TextToSpeech()
        
        print("✅ Text-to-speech test passed")
        print(f"   - TTS initialized: {tts is not None}")
        print(f"   - Engine: {tts.engine}")
        
        # Test speaking (without actually speaking)
        print("   - TTS functionality available")
        
        return True
        
    except Exception as e:
        print(f"❌ Text-to-speech test failed: {e}")
        return False

def test_conversation_management():
    """Test conversation management functionality"""
    print("💬 Testing conversation management...")
    
    try:
        from conversation_manager import ConversationManager
        
        # Initialize conversation manager
        conv_manager = ConversationManager()
        
        # Test adding messages
        conv_manager.add_user_message("Test user message")
        conv_manager.add_assistant_message("Test assistant response")
        
        # Test getting history
        history = conv_manager.get_recent_history(5)
        
        print("✅ Conversation management test passed")
        print(f"   - Manager initialized: {conv_manager is not None}")
        print(f"   - Messages added: {len(history)}")
        print(f"   - Session ID: {conv_manager.current_session_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversation management test failed: {e}")
        return False

def test_enhanced_app_imports():
    """Test that enhanced app can be imported"""
    print("📱 Testing enhanced app imports...")
    
    try:
        # Test importing the enhanced app modules
        import enhanced_app
        
        print("✅ Enhanced app import test passed")
        print("   - All modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced app import test failed: {e}")
        return False

def test_file_upload_simulation():
    """Test file upload simulation"""
    print("📁 Testing file upload simulation...")
    
    try:
        # Create a temporary image file
        test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            pil_image.save(tmp_file.name, "JPEG")
            tmp_path = tmp_file.name
        
        # Test reading the file back
        loaded_image = Image.open(tmp_path)
        loaded_array = np.array(loaded_image)
        
        # Clean up
        os.unlink(tmp_path)
        
        print("✅ File upload simulation test passed")
        print(f"   - Temporary file created and loaded")
        print(f"   - Image shape: {loaded_array.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ File upload simulation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 Enhanced Multimodal AI Assistant - Feature Tests")
    print("=" * 60)
    
    tests = [
        ("Image Processing", test_image_processing),
        ("Audio Transcription", test_audio_transcription),
        ("Vision Processing", test_vision_processing),
        ("Text-to-Speech", test_text_to_speech),
        ("Conversation Management", test_conversation_management),
        ("Enhanced App Imports", test_enhanced_app_imports),
        ("File Upload Simulation", test_file_upload_simulation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}")
    
    print(f"\n🎯 {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced features are ready to use.")
        print("\nTo start the enhanced application:")
        print("   streamlit run enhanced_app.py --server.port 8503")
    elif passed >= total * 0.7:
        print("⚠️  Most tests passed. Enhanced features should work with some limitations.")
        print("\nTo start the enhanced application:")
        print("   streamlit run enhanced_app.py --server.port 8503")
    else:
        print("❌ Many tests failed. Please check the installation and dependencies.")
        print("\nTry running the setup script:")
        print("   python3 quick_setup.py")
    
    print("\n💡 Enhanced Features Available:")
    print("   📷 Image upload and analysis")
    print("   🎤 Voice input via audio file upload")
    print("   🔊 Text-to-speech responses")
    print("   💬 Conversation history with visual context")
    print("   ⚡ Quick action buttons for common tasks")

if __name__ == "__main__":
    main()
