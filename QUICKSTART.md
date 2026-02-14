# Quick Start Guide

## First Time Setup

### 1. Install Python Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install packages
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd ui/frontend
npm install
cd ../..
```

## Running the Demo

### Easy Way (Recommended for macOS/Linux)

```bash
./start_demo.sh
```

### Manual Way (Works on all platforms)

**Terminal 1 - Backend:**
```bash
source venv/bin/activate  # Or venv\Scripts\activate on Windows
python api/server.py
```

**Terminal 2 - Frontend:**
```bash
cd ui/frontend
npm start
```

## Access the Demo

- **Main App**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Quick Test

1. Open http://localhost:3000
2. Select a humanoid model (e.g., "MuJoCo Humanoid")
3. Click "Initialize Environment"
4. Set target coordinates (try X: 5.0, Y: 0.0)
5. Click "Start"
6. Watch the humanoid walk!

## Troubleshooting

### MuJoCo won't install
```bash
# Try installing with specific flags
pip install --upgrade pip
pip install mujoco --no-cache-dir
```

### Port 8000 or 3000 already in use
```bash
# Find and kill the process
# On macOS/Linux:
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### WebSocket connection fails
- Make sure backend is running first
- Check there are no firewall issues
- Verify both servers are on the correct ports

## Next Steps

- Try different humanoid models
- Experiment with various target positions
- Modify locomotion controller for better walking
- Add your own custom humanoid models
- Implement obstacle avoidance

## Development

### Adding New Features

- **Backend**: Edit files in `api/`, `environments/`, `controllers/`
- **Frontend**: Edit files in `ui/frontend/src/`
- **Config**: Modify `config/humanoid_configs.py`

### Testing Changes

Backend changes take effect immediately (server auto-reloads)
Frontend changes auto-refresh in the browser

## Project Structure Overview

```
mujoco-demo/
├── api/                    # FastAPI backend
├── config/                 # Configuration files
├── controllers/            # Walking & navigation logic
├── environments/           # MuJoCo environment
├── ui/frontend/           # React application
├── utils/                  # Helper functions
├── requirements.txt        # Python packages
└── start_demo.sh          # Startup script
```

Enjoy exploring your humanoid navigation demo! 🤖
