#!/bin/bash

# Multimodal AI Assistant Launcher Script

echo "🤖 Starting Multimodal AI Assistant..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first:"
    echo "   python3 setup.py"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ Dependencies not installed!"
    echo "Please run setup first:"
    echo "   python3 setup.py"
    exit 1
fi

# Clean up old temporary files
echo "🧹 Cleaning up temporary files..."
python -c "from utils import cleanup_temp_files; cleanup_temp_files()" 2>/dev/null || true

# Start the application
echo "🚀 Starting Streamlit application..."
echo "The application will open in your default browser"
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app.py

echo ""
echo "👋 Application stopped. Goodbye!"
