#!/bin/bash

# Enhanced Multimodal AI Assistant Launcher

echo "🤖 Starting Enhanced Multimodal AI Assistant..."
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not found. Please run setup first:"
    echo "   python3 quick_setup.py"
    exit 1
fi

# Check if enhanced app exists
if [ ! -f "enhanced_app.py" ]; then
    echo "❌ enhanced_app.py not found in current directory"
    exit 1
fi

echo "✅ All checks passed"
echo ""
echo "🧪 Running feature tests..."
python3 test_enhanced_features.py | tail -10

echo ""
echo "🚀 Starting enhanced application..."
echo "📱 The app will open in your default browser"
echo "🌐 URL: http://localhost:8503"
echo ""
echo "✨ Enhanced Features Available:"
echo "   📷 Image upload and analysis"
echo "   🎤 Voice input via audio file upload"
echo "   🔊 Text-to-speech responses"
echo "   💬 Conversation history with visual context"
echo "   ⚡ Quick action buttons for common tasks"
echo ""
echo "💡 Instructions:"
echo "   1. Initialize the assistant in the sidebar"
echo "   2. Upload an image using the file uploader"
echo "   3. Ask questions via text or upload audio files"
echo "   4. Listen to spoken responses"
echo "   5. Press Ctrl+C here to stop the application"
echo ""

# Start the application
streamlit run enhanced_app.py --server.port 8503

echo ""
echo "👋 Enhanced application stopped. Goodbye!"
