#!/bin/bash

# Startup script for Humanoid Navigation Demo

echo "🤖 Starting Humanoid Navigation Demo..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "ui/frontend/node_modules" ]; then
    echo "❌ Node modules not found. Please run:"
    echo "   cd ui/frontend"
    echo "   npm install"
    exit 1
fi

# Start backend
echo "🚀 Starting backend server..."
source venv/bin/activate
python api/server.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend..."
cd ui/frontend
./start.sh > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

echo ""
echo "✅ Demo is running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "💡 Logs: backend.log and ui/frontend/frontend.log"
echo "Press Ctrl+C to stop"

# Wait for user interrupt
wait
