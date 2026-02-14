# Humanoid Navigation Demo

An interactive portfolio demo showcasing a humanoid robot navigating through a virtual environment using MuJoCo physics simulation. Users can select different humanoid models and set navigation targets in real-time.

## Features

- 🤖 Multiple humanoid robot models
- 🎯 Interactive target selection
- 📡 Real-time WebSocket communication
- 🎨 Modern, intuitive UI
- 🚀 MuJoCo physics simulation
- 🎮 Simple locomotion controller

## Project Structure

```
mujoco-demo/
├── api/                    # FastAPI backend
│   └── server.py          # Main API server with WebSocket
├── config/                 # Configuration files
│   └── humanoid_configs.py # Humanoid model configurations
├── controllers/            # Control algorithms
│   └── locomotion_controller.py  # Locomotion and navigation
├── environments/           # Simulation environments
│   └── humanoid_env.py    # Gymnasium environment wrapper
├── models/                 # Custom MuJoCo models (if any)
├── ui/frontend/           # React frontend
│   ├── public/
│   └── src/
│       ├── App.js         # Main React component
│       ├── App.css        # Styling
│       └── index.js       # Entry point
├── utils/                  # Utility functions
├── requirements.txt        # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- MuJoCo (will be installed via pip)

## Installation

### 1. Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd ui/frontend

# Install Node dependencies
npm install
```

## Running the Demo

### Option 1: Run Separately

**Terminal 1 - Backend:**
```bash
source venv/bin/activate
python api/server.py
```

**Terminal 2 - Frontend:**
```bash
cd ui/frontend
npm start
```

### Option 2: Run with Script

```bash
# Make script executable
chmod +x start_demo.sh

# Run the demo
./start_demo.sh
```

The demo will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Usage

1. **Select a Humanoid**: Choose from available humanoid models
2. **Initialize**: Click "Initialize Environment" to load the simulation
3. **Set Target**: Enter X and Y coordinates for the navigation target
4. **Start**: Click "Start" to begin the simulation
5. **Observe**: Watch the humanoid navigate to the target
6. **Experiment**: Try different targets and humanoid models!

## API Endpoints

- `GET /humanoids` - List available humanoid models
- `POST /initialize` - Initialize environment with selected humanoid
- `POST /target` - Set navigation target
- `POST /reset` - Reset simulation
- `GET /state` - Get current simulation state
- `WebSocket /ws/simulation` - Real-time simulation updates

## Configuration

Edit `config/humanoid_configs.py` to:
- Add custom humanoid models
- Adjust control parameters
- Modify environment settings
- Configure obstacle generation

## Customization

### Adding New Humanoid Models

1. Add model XML to `models/` directory
2. Register in `config/humanoid_configs.py`:

```python
HUMANOID_MODELS = {
    "my_humanoid": {
        "name": "My Custom Humanoid",
        "description": "Description here",
        "xml_path": "models/my_humanoid.xml",
        "action_dim": 17,
        "observation_dim": 376,
        "height": 1.5,
        "mass": 50.0,
        "builtin": False
    }
}
```

### Improving Locomotion

The current controller in `controllers/locomotion_controller.py` uses a simple sinusoidal gait pattern. For better performance:

1. Train an RL policy using Stable-Baselines3
2. Implement trajectory optimization
3. Use pre-trained locomotion policies

## Technical Stack

**Backend:**
- MuJoCo 3.0+ - Physics simulation
- FastAPI - Web framework
- WebSockets - Real-time communication
- Gymnasium - RL environment wrapper

**Frontend:**
- React 18 - UI framework
- WebSocket API - Real-time updates
- CSS3 - Styling

## Troubleshooting

**MuJoCo Installation Issues:**
```bash
# On macOS with Apple Silicon
pip install mujoco --no-binary mujoco

# Verify installation
python -c "import mujoco; print(mujoco.__version__)"
```

**Port Already in Use:**
```bash
# Change backend port in api/server.py
uvicorn.run(app, host="0.0.0.0", port=8001)

# Update frontend proxy in ui/frontend/package.json
"proxy": "http://localhost:8001"
```

**WebSocket Connection Failed:**
- Ensure backend is running
- Check firewall settings
- Verify CORS configuration

## Performance Optimization

For better performance:
- Reduce frame rate in WebSocket loop
- Lower image quality in rendering
- Use simpler humanoid models
- Decrease simulation timestep

## Future Enhancements

- [ ] Add obstacle avoidance
- [ ] Implement path planning visualization
- [ ] Train RL policy for better locomotion
- [ ] Add multiple camera angles
- [ ] Support for custom environments
- [ ] Record and replay trajectories
- [ ] Add terrain variations
- [ ] Multiplayer mode with multiple humanoids

## License

MIT License - feel free to use for your portfolio!

## Credits

- MuJoCo physics engine by DeepMind
- Humanoid model based on MuJoCo's built-in models

## Contact

Built for portfolio demonstration purposes.
