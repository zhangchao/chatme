#!/usr/bin/env python3
"""
Quick setup script for Multimodal AI Assistant Demo
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🤖 Multimodal AI Assistant - Quick Setup")
    print("=" * 50)
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    
    # Install core dependencies
    core_deps = [
        "streamlit",
        "opencv-python", 
        "numpy<2.0",
        "pillow"
    ]
    
    print("\n📦 Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}, but continuing...")
    
    # Optional: Install MLX for Apple Silicon
    if sys.platform == "darwin":
        print("\n🍎 Installing MLX for Apple Silicon (optional)...")
        run_command("pip install mlx mlx-vlm", "Installing MLX and MLX-VLM")
    
    # Create directories
    print("\n📁 Creating directories...")
    for directory in ["temp", "models", "logs"]:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Streamlit import failed")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError:
        print("❌ OpenCV import failed")
    
    try:
        import numpy
        print("✅ NumPy imported successfully")
    except ImportError:
        print("❌ NumPy import failed")
    
    # Test camera
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is accessible")
            cap.release()
        else:
            print("⚠️  Camera not accessible (may need permissions)")
    except Exception as e:
        print(f"⚠️  Camera test failed: {e}")
    
    # Test TTS
    try:
        result = subprocess.run(["say", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Text-to-speech available")
        else:
            print("⚠️  Text-to-speech not available")
    except Exception:
        print("⚠️  Text-to-speech test failed")
    
    print("\n" + "=" * 50)
    print("🎉 Quick setup completed!")
    print("\nTo start the demo application:")
    print("   streamlit run demo_app.py --server.port 8502")
    print("\nThen open: http://localhost:8502")
    print("\n📋 Next steps:")
    print("1. Grant camera permissions when prompted")
    print("2. Click 'Start Camera' in the sidebar")
    print("3. Try the interaction buttons")
    print("4. Enjoy your multimodal AI assistant!")
    
    print("\n💡 For advanced features:")
    print("   pip install openai-whisper  # Speech recognition")
    print("   brew install portaudio && pip install pyaudio  # Audio recording")

if __name__ == "__main__":
    main()
