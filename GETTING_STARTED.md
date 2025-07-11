# 🚀 Getting Started with Multimodal AI Assistant

## 🎯 What You Have

A complete multimodal AI assistant application with two versions:

### 📱 Demo Version (Ready to Use)
- **File**: `demo_app.py`
- **Features**: Camera feed, basic image analysis, text-to-speech, conversation history
- **Status**: ✅ Working and ready to use
- **Requirements**: Basic Python packages (streamlit, opencv-python, numpy, pillow)

### 🧠 Full Version (Advanced)
- **File**: `app.py`
- **Features**: Advanced AI models, speech recognition, MLX-VLM integration
- **Status**: ⚠️ Requires additional setup and dependencies
- **Requirements**: MLX, Whisper, PyAudio, system permissions

## 🚀 Quick Start (Recommended)

### Option 1: Automatic Setup
```bash
python3 quick_setup.py
./run_demo.sh
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install streamlit opencv-python "numpy<2.0" pillow

# Run demo
streamlit run demo_app.py --server.port 8502
```

Then open: http://localhost:8502

## 🎮 How to Use the Demo

1. **Start the Application**
   - Run the demo using one of the methods above
   - Your browser will open to http://localhost:8502

2. **Enable Camera**
   - Click "🚀 Start Camera" in the sidebar
   - Grant camera permissions when prompted by your browser/system

3. **Try the Features**
   - **👁️ Analyze Current View**: AI analyzes what the camera sees
   - **👋 Say Hello**: Assistant introduces itself
   - **📊 System Status**: Reports current system state
   - **⌨️ Manual Input**: Type messages to the assistant

4. **Listen to Responses**
   - All responses are spoken using your system's text-to-speech
   - Check the conversation panel to see the history

## 🔧 Troubleshooting

### Camera Issues
- **Permission Denied**: Grant camera access in browser settings
- **No Camera Found**: Check if camera is connected and not used by other apps
- **Black Screen**: Try refreshing the page or restarting the application

### Audio Issues
- **No Speech**: Check system volume and text-to-speech settings
- **macOS**: Ensure "say" command works: `say "test"`

### Installation Issues
- **Import Errors**: Run `python3 quick_setup.py` to install dependencies
- **NumPy Conflicts**: Install compatible version: `pip install "numpy<2.0"`
- **Port Conflicts**: Use different port: `--server.port 8503`

## 🎯 Next Steps

### For Basic Users
- Use the demo version as-is
- Experiment with different camera angles and lighting
- Try various text inputs to see how the assistant responds

### For Advanced Users
- Install additional dependencies for full features:
  ```bash
  pip install mlx mlx-vlm openai-whisper
  brew install portaudio && pip install pyaudio
  ```
- Try the full application: `streamlit run app.py`
- Customize the AI models in `config.py`

## 📁 Project Structure

```
chatme/
├── demo_app.py              # 📱 Working demo application
├── app.py                   # 🧠 Full-featured application
├── quick_setup.py           # 🛠️ Automatic setup script
├── run_demo.sh             # 🚀 Demo launcher script
├── config.py               # ⚙️ Configuration settings
├── utils.py                # 🔧 Utility functions
├── audio_processor.py      # 🎤 Audio processing
├── vision_processor.py     # 👁️ Vision processing
├── conversation_manager.py # 💬 Conversation management
├── requirements.txt        # 📦 Python dependencies
├── README.md              # 📖 Full documentation
└── GETTING_STARTED.md     # 🚀 This file
```

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ Camera feed appears in the browser
- ✅ "Analyze Current View" button provides image analysis
- ✅ Assistant speaks responses out loud
- ✅ Conversation history updates in real-time
- ✅ Manual text input generates contextual responses

## 💡 Tips for Best Experience

1. **Good Lighting**: Ensure adequate lighting for better image analysis
2. **Stable Camera**: Keep camera steady for consistent analysis
3. **Clear Audio**: Ensure system volume is up for speech responses
4. **Browser Permissions**: Grant all requested permissions for full functionality
5. **Patience**: First-time model loading may take a moment

## 🆘 Need Help?

1. **Check the logs**: Look at the terminal output for error messages
2. **Test components**: Run `python3 test_components.py` to check system status
3. **Restart**: Try stopping and restarting the application
4. **Fresh start**: Clear browser cache and restart

## 🎊 Enjoy Your AI Assistant!

You now have a working multimodal AI assistant that can:
- See through your camera
- Analyze visual scenes
- Respond with both text and speech
- Maintain conversation history
- Interact naturally through multiple modalities

Have fun exploring the capabilities and experimenting with different interactions!
