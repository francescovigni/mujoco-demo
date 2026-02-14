"""FastAPI backend for rover navigation demo."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import asyncio
import json
import sys
import os
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.rover_env import RoverNavigationEnv
from controllers.rover_controller import RoverController
from config.robot_configs import ROBOT_MODELS

app = FastAPI(title="Rover Navigation API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
current_env: Optional[RoverNavigationEnv] = None
current_controller: Optional[RoverController] = None
simulation_running = False


class TargetPosition(BaseModel):
    """Target position model."""
    x: float
    y: float


class RobotConfig(BaseModel):
    """Robot configuration model."""
    model_name: str


class SimulationState(BaseModel):
    """Simulation state model."""
    running: bool
    position: List[float]
    target: List[float]
    distance: float


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"message": "Rover Navigation API", "status": "running"}


@app.get("/humanoids")
async def get_robots():
    """Get available robot models."""
    return {
        "humanoids": [
            {
                "id": key,
                "name": value["name"],
                "description": value["description"],
                "height": value.get("height", 0.12),
                "mass": value.get("mass", 5.0),
            }
            for key, value in ROBOT_MODELS.items()
        ]
    }


@app.post("/initialize")
async def initialize_environment(config: RobotConfig):
    """Initialize environment with selected robot."""
    global current_env, current_controller, simulation_running
    
    try:
        # Clean up existing environment
        if current_env is not None:
            current_env.close()
        
        # Get model configuration
        if config.model_name not in ROBOT_MODELS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown robot model: {config.model_name}"}
            )
        
        model_config = ROBOT_MODELS[config.model_name]
        
        # Create environment (no rendering needed - Three.js handles it)
        current_env = RoverNavigationEnv(
            render_mode=None,
            target_pos=np.array([10.0, 5.0])
        )
        
        # Create controller
        current_controller = RoverController()
        current_controller.set_target(np.array([10.0, 5.0]))
        current_controller.set_obstacles(current_env.obstacles)
        
        # Reset environment
        obs, info = current_env.reset()
        simulation_running = False
        
        # Return scene data for Three.js
        return {
            "status": "initialized",
            "model": config.model_name,
            "scene": {
                "obstacles": [
                    {
                        "position": obs_data["position"],
                        "radius": obs_data["radius"],
                        "height": obs_data["height"],
                        "color": obs_data["color"][:3],  # RGB only
                    }
                    for obs_data in current_env.obstacles
                ],
                "target": current_env.target_pos.tolist(),
                "roverPosition": current_env.data.qpos[:3].tolist(),  # x, y, z
                "roverQuaternion": current_env.data.qpos[3:7].tolist(),  # w, x, y, z
                "groundSize": 40.0,
            }
        }
    
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/target")
async def set_target(target: TargetPosition):
    """Set navigation target."""
    global current_controller
    
    if current_controller is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized"}
        )
    
    current_controller.set_target(np.array([target.x, target.y]))
    
    if current_env is not None:
        current_env.target_pos = np.array([target.x, target.y])
        current_env._update_target_marker()  # Update visual marker
    
    return {
        "status": "target_set",
        "target": [target.x, target.y]
    }


@app.post("/reset")
async def reset_simulation():
    """Reset the simulation."""
    global current_env, current_controller, simulation_running
    
    if current_env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized"}
        )
    
    obs, info = current_env.reset()
    if current_controller is not None:
        current_controller.reset()
    
    simulation_running = False
    
    return {
        "status": "reset",
        "info": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in info.items()}
    }


@app.get("/state")
async def get_state():
    """Get current simulation state."""
    global current_env, simulation_running
    
    if current_env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized"}
        )
    
    position = current_env.data.qpos[:2].tolist()
    target = current_env.target_pos.tolist()
    distance = float(np.linalg.norm(current_env.data.qpos[:2] - current_env.target_pos))
    
    return {
        "running": simulation_running,
        "position": position,
        "target": target,
        "distance": distance,
        "height": float(np.linalg.norm(current_env.data.qvel[:2])),
    }


@app.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation.
    
    Sends lightweight state data for Three.js client-side rendering.
    No server-side frame rendering - much faster!
    """
    global current_env, current_controller, simulation_running
    
    await websocket.accept()
    print("WebSocket accepted, connection established")
    
    # Send initial state if environment exists
    if current_env is not None:
        try:
            planned_path = None
            if current_controller is not None:
                planned_path = current_controller.compute_path(current_env.data.qpos[:2])
            
            initial_state = {
                "type": "state",
                "position": current_env.data.qpos[:3].tolist(),  # x, y, z
                "quaternion": current_env.data.qpos[3:7].tolist(),  # w, x, y, z
                "target": current_env.target_pos.tolist(),
                "distance": float(np.linalg.norm(current_env.data.qpos[:2] - current_env.target_pos)),
                "speed": float(np.linalg.norm(current_env.data.qvel[:2])),
                "path": planned_path.tolist() if planned_path is not None else [],
                "terminated": False,
                "truncated": False,
            }
            await websocket.send_json(initial_state)
            print("Initial state sent")
        except Exception as e:
            print(f"Error sending initial state: {e}")
            traceback.print_exc()
    else:
        print("Warning: No environment initialized")
    
    try:
        print("Entering WebSocket main loop...")
        path_dirty = False  # Track if path needs recompute
        
        while True:
            # Drain ALL pending messages
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.005)
                    message = json.loads(data)
                    
                    if message["type"] == "start":
                        simulation_running = True
                        print("Simulation started")
                    elif message["type"] == "stop":
                        simulation_running = False
                        print("Simulation stopped")
                    elif message["type"] == "set_target":
                        if current_controller and current_env:
                            target = np.array([message["x"], message["y"]])
                            current_controller.set_target(target)
                            current_env.target_pos = target
                            current_env._update_target_marker()
                            path_dirty = True
                            # Send target update immediately
                            await websocket.send_json({
                                "type": "target_update",
                                "target": target.tolist(),
                            })
                            print(f"Target set to: {target}")
                    elif message["type"] == "repulsive_params":
                        if current_controller:
                            if "repulse_range" in message:
                                current_controller.repulse_range = float(message["repulse_range"])
                            if "repulse_gain" in message:
                                current_controller.repulse_gain = float(message["repulse_gain"])
                            if "obstacle_margin" in message:
                                current_controller.obstacle_margin = float(message["obstacle_margin"])
                            path_dirty = True
                    elif message["type"] == "rover_params":
                        if current_controller:
                            if "speed_factor" in message:
                                current_controller.speed_factor = float(message["speed_factor"])
                            if "num_path_points" in message:
                                current_controller.num_path_points = int(message["num_path_points"])
                            path_dirty = True
                    elif message["type"] == "reset":
                        if current_env:
                            current_env.reset()
                            if current_controller:
                                current_controller.reset()
                            # Send reset state
                            planned_path = current_controller.compute_path(current_env.data.qpos[:2]) if current_controller else None
                            await websocket.send_json({
                                "type": "state",
                                "position": current_env.data.qpos[:3].tolist(),
                                "quaternion": current_env.data.qpos[3:7].tolist(),
                                "target": current_env.target_pos.tolist(),
                                "distance": float(np.linalg.norm(current_env.data.qpos[:2] - current_env.target_pos)),
                                "speed": 0.0,
                                "path": planned_path.tolist() if planned_path is not None else [],
                                "terminated": False,
                                "truncated": False,
                            })
                        simulation_running = False
                        print("Environment reset")
                except asyncio.TimeoutError:
                    break
                except RuntimeError as e:
                    if "disconnect" in str(e).lower():
                        raise
                    print(f"Runtime error: {e}")
                    break
                except Exception as e:
                    print(f"Error processing command: {e}")
                    traceback.print_exc()
                    break

            # Send path update if parameters changed (and not running)
            if path_dirty and not simulation_running and current_env is not None and current_controller is not None:
                try:
                    planned_path = current_controller.compute_path(current_env.data.qpos[:2])
                    await websocket.send_json({
                        "type": "path_update",
                        "path": planned_path.tolist(),
                    })
                    path_dirty = False
                except Exception as e:
                    print(f"Error sending path update: {e}")
            
            # Run simulation step if active
            if simulation_running and current_env and current_controller:
                try:
                    # Get current state
                    current_pos = current_env.data.qpos[:2]
                    current_vel = current_env.data.qvel[:2]
                    body_quat = current_env.data.qpos[3:7].copy()
                    body_angvel = current_env.data.qvel[3:6].copy()
                    
                    # Get action from controller
                    action = current_controller.get_action(current_pos, current_vel, body_quat, body_angvel)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = current_env.step(action)
                    
                    # Compute path (less frequently during simulation for performance)
                    planned_path = current_controller.compute_path(current_env.data.qpos[:2])
                    
                    # Send lightweight state update (no frame!)
                    state_update = {
                        "type": "state",
                        "position": current_env.data.qpos[:3].tolist(),  # x, y, z
                        "quaternion": current_env.data.qpos[3:7].tolist(),  # w, x, y, z
                        "target": current_env.target_pos.tolist(),
                        "distance": float(info["distance_to_target"]),
                        "speed": float(info.get("speed", 0.0)),
                        "path": planned_path.tolist(),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                    
                    await websocket.send_json(state_update)
                    
                    # Reset if episode ended
                    if terminated or truncated:
                        simulation_running = False
                        await websocket.send_json({
                            "type": "episode_end",
                            "reason": "terminated" if terminated else "truncated"
                        })
                        print(f"Episode ended: {('terminated' if terminated else 'truncated')}")
                except Exception as e:
                    print(f"Error in simulation step: {e}")
                    traceback.print_exc()
            
            await asyncio.sleep(0.016)  # ~60 FPS target
            
    except WebSocketDisconnect:
        print("Client disconnected normally")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
    finally:
        print("WebSocket connection closed")


# ---------- Serve React static build (production) ----------
_static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(_static_dir):
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(_static_dir, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
