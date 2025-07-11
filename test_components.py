#!/usr/bin/env python3
"""
Test script to verify all components are working
"""
import sys
import time
import numpy as np
from PIL import Image

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ NumPy imported")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import whisper
        print("✅ Whisper imported")
    except ImportError as e:
        print(f"❌ Whisper import failed: {e}")
        return False
    
    try:
        import mlx.core as mx
        print("✅ MLX imported")
    except ImportError as e:
        print(f"❌ MLX import failed: {e}")
        return False
    
    try:
        import pyaudio
        print("✅ PyAudio imported")
    except ImportError as e:
        print(f"❌ PyAudio import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test camera functionality"""
    print("\n📹 Testing camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera")
            cap.release()
            return False
        
        print(f"✅ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_audio():
    """Test audio functionality"""
    print("\n🎤 Testing audio...")
    
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        
        device_count = pa.get_device_count()
        print(f"✅ Found {device_count} audio devices")
        
        # Test default input device
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            stream.close()
            print("✅ Audio input device accessible")
        except Exception as e:
            print(f"❌ Audio input test failed: {e}")
            pa.terminate()
            return False
        
        pa.terminate()
        return True
        
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False

def test_whisper():
    """Test Whisper model loading"""
    print("\n🗣️ Testing Whisper...")
    
    try:
        import whisper
        print("Loading Whisper model (this may take a moment)...")
        model = whisper.load_model("tiny")  # Use tiny model for quick test
        print("✅ Whisper model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        return False

def test_mlx_vlm():
    """Test MLX-VLM functionality"""
    print("\n👁️ Testing MLX-VLM...")
    
    try:
        from mlx_vlm import load
        print("Testing MLX-VLM import...")
        print("✅ MLX-VLM imported successfully")
        print("ℹ️  Model loading will be tested during first use")
        return True
        
    except Exception as e:
        print(f"❌ MLX-VLM test failed: {e}")
        print("ℹ️  This is expected if MLX-VLM is not installed")
        return False

def test_tts():
    """Test text-to-speech"""
    print("\n🔊 Testing text-to-speech...")
    
    try:
        import subprocess
        # Test macOS say command
        result = subprocess.run(["say", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ macOS TTS (say command) available")
            return True
        else:
            print("❌ macOS TTS not available")
            return False
            
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

def test_our_modules():
    """Test our custom modules"""
    print("\n🔧 Testing custom modules...")
    
    try:
        import config
        print("✅ Config module imported")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        import utils
        print("✅ Utils module imported")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    try:
        import audio_processor
        print("✅ Audio processor module imported")
    except ImportError as e:
        print(f"❌ Audio processor import failed: {e}")
        return False
    
    try:
        import vision_processor
        print("✅ Vision processor module imported")
    except ImportError as e:
        print(f"❌ Vision processor import failed: {e}")
        return False
    
    try:
        import conversation_manager
        print("✅ Conversation manager module imported")
    except ImportError as e:
        print(f"❌ Conversation manager import failed: {e}")
        return False
    
    return True

def test_system_requirements():
    """Test system requirements"""
    print("\n💻 Testing system requirements...")
    
    from utils import check_system_requirements
    requirements = check_system_requirements()
    
    all_good = True
    for component, status in requirements.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {component.title()}: {'Available' if status else 'Not available'}")
        if not status:
            all_good = False
    
    return all_good

def main():
    """Main test function"""
    print("🤖 Multimodal AI Assistant - Component Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Camera", test_camera),
        ("Audio", test_audio),
        ("Whisper", test_whisper),
        ("MLX-VLM", test_mlx_vlm),
        ("Text-to-Speech", test_tts),
        ("Custom Modules", test_our_modules),
        ("System Requirements", test_system_requirements),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}")
    
    print(f"\n🎯 {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
    elif passed >= total * 0.7:
        print("⚠️  Most tests passed. The application should work with some limitations.")
    else:
        print("❌ Many tests failed. Please check the installation.")
    
    print("\nTo start the application:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
