#!/bin/bash

# Multimodal AI Assistant Demo Launcher

echo "🤖 Starting Multimodal AI Assistant Demo..."
echo "=========================================="

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

# Check if demo app exists
if [ ! -f "demo_app.py" ]; then
    echo "❌ demo_app.py not found in current directory"
    exit 1
fi

echo "✅ All checks passed"
echo ""
echo "🚀 Starting demo application..."
echo "📱 The app will open in your default browser"
echo "🌐 URL: http://localhost:8502"
echo ""
echo "💡 Instructions:"
echo "   1. Grant camera permissions when prompted"
echo "   2. Click 'Start Camera' in the sidebar"
echo "   3. Try the interaction buttons"
echo "   4. Press Ctrl+C here to stop the application"
echo ""

# Start the application
streamlit run demo_app.py --server.port 8502

echo ""
echo "👋 Demo application stopped. Goodbye!"
