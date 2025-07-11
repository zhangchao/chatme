# 🤖 Multimodal AI Assistant

A local multimodal AI assistant application that combines vision, speech, and language understanding to create an interactive AI companion that can see, hear, and respond naturally.

## 🚀 Quick Start - Demo Version

**Try the working demo first:**
```bash
streamlit run demo_app.py --server.port 8502
```
Then open http://localhost:8502 in your browser.

The demo includes:
- 📹 Real-time camera feed
- 👁️ Basic image analysis
- 🔊 Text-to-speech responses
- 💬 Conversation history

## ✨ Features

- **👁️ Visual Understanding**: Real-time camera feed analysis using MLX-VLM
- **🎤 Speech Recognition**: Local speech-to-text using OpenAI Whisper
- **🔊 Text-to-Speech**: Natural voice responses using macOS built-in TTS
- **💬 Conversation Memory**: Persistent conversation history with visual context
- **🎯 Multimodal Processing**: Combines visual and audio inputs for contextual responses
- **🖥️ User-Friendly Interface**: Clean Streamlit web interface
- **🔒 Privacy-First**: Runs entirely locally on your Mac - no external API calls

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Vision**: MLX-VLM (optimized for Apple Silicon)
- **Speech-to-Text**: OpenAI Whisper
- **Text-to-Speech**: macOS `say` command / pyttsx3
- **Computer Vision**: OpenCV
- **Audio Processing**: PyAudio, librosa
- **Language Processing**: Local language models via MLX

## 📋 Requirements

### System Requirements
- **macOS** (Apple Silicon recommended for MLX optimization)
- **Python 3.8+**
- **Camera** (built-in or external webcam)
- **Microphone** (built-in or external)
- **Speakers/Headphones** for audio output

### Hardware Recommendations
- **Apple Silicon Mac** (M1/M2/M3) for optimal MLX performance
- **8GB+ RAM** (16GB+ recommended)
- **5GB+ free disk space** for models and dependencies

## 🚀 Installation

### Quick Setup (Recommended)
```bash
# 1. Install core dependencies
pip install streamlit opencv-python numpy pillow

# 2. Fix NumPy compatibility
pip install "numpy<2.0"

# 3. Install MLX for Apple Silicon (optional)
pip install mlx mlx-vlm

# 4. Run the demo
streamlit run demo_app.py --server.port 8502
```

### Full Installation (Advanced)

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd chatme
```

#### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

#### 3. Install Dependencies
```bash
# Core dependencies
pip install streamlit opencv-python numpy pillow

# Fix NumPy compatibility issues
pip install "numpy<2.0"

# Optional: Advanced AI features
pip install mlx mlx-vlm openai-whisper

# Optional: Audio support (requires system dependencies)
# brew install portaudio  # Install first
# pip install pyaudio
```

#### 4. Download Models (Optional)
```bash
python -c "import whisper; whisper.load_model('base')"  # If whisper installed
```

## 🎯 Usage

### Demo Version (Recommended)
```bash
streamlit run demo_app.py --server.port 8502
```
Open http://localhost:8502 in your browser.

**Demo Features:**
1. **Start Camera**: Click "🚀 Start Camera" in the sidebar
2. **Grant Permissions**: Allow camera access when prompted
3. **Try Actions**: Use the interaction buttons to test features
4. **Listen**: Responses are spoken using system text-to-speech
5. **View History**: Check the conversation panel

### Full Application (Advanced)
```bash
streamlit run app.py
```

**Full Features:**
1. **Initialize**: Click "🚀 Initialize Assistant" in the sidebar
2. **Start Listening**: Click "🎤 Start Listening" for voice interaction
3. **Interact**: Speak naturally while camera captures video
4. **AI Processing**: Advanced multimodal understanding with MLX-VLM
5. **Review History**: Complete conversation history with visual context

## 🎛️ Configuration

Edit `config.py` to customize:

- **Camera settings**: Resolution, FPS, camera index
- **Audio settings**: Sample rate, chunk size, voice activation threshold
- **Model settings**: MLX-VLM model, Whisper model size
- **UI settings**: Max conversation history, update intervals

## 📁 Project Structure

```
chatme/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── utils.py                  # Utility functions
├── audio_processor.py        # Speech-to-text and TTS
├── vision_processor.py       # Vision and multimodal processing
├── conversation_manager.py   # Conversation history management
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── temp/                    # Temporary files (auto-created)
├── models/                  # Model cache (auto-created)
└── logs/                    # Conversation logs (auto-created)
```

## 🔧 Troubleshooting

### Common Issues

**Camera not working:**
- Check camera permissions in System Preferences > Security & Privacy > Camera
- Try different camera index in `config.py`
- Ensure no other applications are using the camera

**Audio not working:**
- Check microphone permissions in System Preferences > Security & Privacy > Microphone
- Install PortAudio: `brew install portaudio`
- Try different audio device settings

**MLX-VLM not loading:**
- Ensure you're on Apple Silicon Mac for optimal performance
- Check available disk space for model downloads
- Try smaller model variants in `config.py`

**Performance issues:**
- Close other resource-intensive applications
- Reduce camera resolution in `config.py`
- Use smaller AI models

### Logs and Debugging

- Check console output for error messages
- Conversation logs are saved in the `logs/` directory
- Enable debug logging by modifying the logging level in `config.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MLX Team** for the excellent Apple Silicon optimization
- **OpenAI** for Whisper speech recognition
- **Streamlit** for the amazing web framework
- **OpenCV** for computer vision capabilities

## 🔮 Future Enhancements

- [ ] Support for multiple languages
- [ ] Advanced gesture recognition
- [ ] Integration with more vision models
- [ ] Voice cloning capabilities
- [ ] Mobile app companion
- [ ] Plugin system for extensions

## 📞 Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information
4. Include system information and error logs

---

**Happy chatting with your AI assistant! 🤖✨**
