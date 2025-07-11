#!/usr/bin/env python3
"""
Setup script for Multimodal AI Assistant
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_macos():
    """Check if running on macOS"""
    if sys.platform != "darwin":
        print("⚠️  This application is optimized for macOS")
        return False
    print("✅ Running on macOS")
    return True

def install_homebrew_dependencies():
    """Install Homebrew dependencies"""
    print("🍺 Installing Homebrew dependencies...")
    
    # Check if Homebrew is installed
    try:
        subprocess.run(["brew", "--version"], check=True, capture_output=True)
        print("✅ Homebrew is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Homebrew not found. Please install Homebrew first:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return False
    
    # Install PortAudio for audio processing
    if not run_command("brew install portaudio", "Installing PortAudio"):
        print("⚠️  PortAudio installation failed, but continuing...")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["temp", "models", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_python_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def download_models():
    """Download required models"""
    print("📥 Downloading AI models...")
    
    # Download Whisper model
    download_whisper = input("Download Whisper model now? (y/n): ").lower().strip() == 'y'
    if download_whisper:
        if not run_command('python -c "import whisper; whisper.load_model(\'base\')"', "Downloading Whisper model"):
            print("⚠️  Whisper model download failed, but will be downloaded on first use")
    
    print("ℹ️  MLX-VLM models will be downloaded automatically on first use")

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    test_script = """
import sys
try:
    import streamlit
    import cv2
    import numpy
    import whisper
    import mlx.core
    print("✅ All core dependencies imported successfully")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🤖 Multimodal AI Assistant Setup")
    print("=" * 40)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_macos():
        print("⚠️  Continuing anyway, but some features may not work optimally")
    
    # Create directories
    create_directories()
    
    # Install system dependencies
    if not install_homebrew_dependencies():
        print("⚠️  Some system dependencies may be missing")
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Download models
    download_models()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
        print("\nFor more information, see README.md")
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
