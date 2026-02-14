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
import base64
from PIL import Image
import io
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

# Camera state
camera_distance = 5.4
camera_azimuth = 26.0
camera_elevation = -19.0
camera_lookat = [0.0, 0.0, 0.3]


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
        
        # Create environment
        current_env = RoverNavigationEnv(
            render_mode="rgb_array",
            target_pos=np.array([10.0, 5.0])
        )
        
        # Create controller
        current_controller = RoverController()
        current_controller.set_target(np.array([10.0, 5.0]))
        current_controller.set_obstacles(current_env.obstacles)
        
        # Reset environment
        obs, info = current_env.reset()
        simulation_running = False
        
        return {
            "status": "initialized",
            "model": config.model_name,
            "observation_shape": obs.shape,
            "action_shape": current_env.action_space.shape,
        }
    
    except Exception as e:
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
    """WebSocket endpoint for real-time simulation."""
    global current_env, current_controller, simulation_running
    global camera_distance, camera_azimuth, camera_elevation, camera_lookat
    
    await websocket.accept()
    print("WebSocket accepted, connection established")
    
    # Send initial frame if environment exists
    if current_env is not None:
        try:
            print("Sending initial frame...")
            # Update camera lookat to rover position
            camera_lookat = [current_env.data.qpos[0], current_env.data.qpos[1], 0.3]
            
            # Compute planned path for visualisation
            planned_path = None
            if current_controller is not None:
                planned_path = current_controller.compute_path(current_env.data.qpos[:2])
            
            frame = current_env.render(
                camera_distance=camera_distance,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                camera_lookat=camera_lookat,
                path=planned_path,
            )
            if frame is not None:
                img = Image.fromarray(frame)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=65)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                initial_state = {
                    "type": "state",
                    "position": current_env.data.qpos[:2].tolist(),
                    "target": current_env.target_pos.tolist(),
                    "distance": float(np.linalg.norm(current_env.data.qpos[:2] - current_env.target_pos)),
                    "height": float(np.linalg.norm(current_env.data.qvel[:2])),
                    "frame": img_str,
                    "terminated": False,
                    "truncated": False,
                    "reward": 0.0,
                }
                await websocket.send_json(initial_state)
                print("Initial frame sent successfully")
            else:
                print("Warning: render() returned None")
        except Exception as e:
            print(f"Error sending initial frame: {e}")
            traceback.print_exc()
    else:
        print("Warning: No environment initialized")
    
    try:
        print("Entering WebSocket main loop...")
        while True:
            # Drain ALL pending messages before rendering
            camera_changed = False
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
                            print(f"✅ Target set to: {target} and marker updated")
                    elif message["type"] == "camera_update":
                        if "distance" in message:
                            camera_distance = float(message["distance"])
                        if "azimuth" in message:
                            camera_azimuth = float(message["azimuth"])
                        if "elevation" in message:
                            camera_elevation = float(message["elevation"])
                        if "lookat" in message:
                            camera_lookat = message["lookat"]
                        camera_changed = True
                    elif message["type"] == "repulsive_params":
                        if current_controller:
                            if "repulse_range" in message:
                                current_controller.repulse_range = float(message["repulse_range"])
                            if "repulse_gain" in message:
                                current_controller.repulse_gain = float(message["repulse_gain"])
                            if "obstacle_margin" in message:
                                current_controller.obstacle_margin = float(message["obstacle_margin"])
                            camera_changed = True   # re-render to update the path
                    elif message["type"] == "rover_params":
                        if current_controller:
                            if "speed_factor" in message:
                                current_controller.speed_factor = float(message["speed_factor"])
                            if "num_path_points" in message:
                                current_controller.num_path_points = int(message["num_path_points"])
                            camera_changed = True   # re-render to update the path
                    elif message["type"] == "reset":
                        if current_env:
                            current_env.reset()
                            if current_controller:
                                current_controller.reset()
                        simulation_running = False
                        print("Environment reset")
                except asyncio.TimeoutError:
                    break  # No more messages pending
                except RuntimeError as e:
                    if "disconnect" in str(e).lower():
                        raise  # Re-raise to exit outer loop
                    print(f"Runtime error: {e}")
                    break
                except Exception as e:
                    print(f"Error processing command: {e}")
                    traceback.print_exc()
                    break

            # Send camera frame if camera changed and simulation is NOT running
            # (if simulation is running, the sim loop will handle rendering)
            if camera_changed and not simulation_running and current_env is not None:
                try:
                    camera_lookat = [current_env.data.qpos[0], current_env.data.qpos[1], 0.3]
                    planned_path = None
                    if current_controller is not None:
                        planned_path = current_controller.compute_path(current_env.data.qpos[:2])
                    frame = current_env.render(
                        camera_distance=camera_distance,
                        camera_azimuth=camera_azimuth,
                        camera_elevation=camera_elevation,
                        camera_lookat=camera_lookat,
                        path=planned_path,
                    )
                    if frame is not None:
                        img = Image.fromarray(frame)
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=65)
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                        await websocket.send_json({
                            "type": "state",
                            "position": current_env.data.qpos[:2].tolist(),
                            "target": current_env.target_pos.tolist(),
                            "distance": float(np.linalg.norm(current_env.data.qpos[:2] - current_env.target_pos)),
                            "height": float(np.linalg.norm(current_env.data.qvel[:2])),
                            "frame": img_str,
                            "terminated": False,
                            "truncated": False,
                            "reward": 0.0,
                        })
                except Exception as e:
                    print(f"Error rendering camera update: {e}")
                    traceback.print_exc()
            
            # Run simulation step if active
            if simulation_running and current_env and current_controller:
                try:
                    # Get current state
                    current_pos = current_env.data.qpos[:2]
                    current_vel = current_env.data.qvel[:2]
                    body_quat = current_env.data.qpos[3:7].copy()  # chassis quaternion
                    body_angvel = current_env.data.qvel[3:6].copy()  # chassis angular velocity
                    
                    # Get action from controller
                    action = current_controller.get_action(current_pos, current_vel, body_quat, body_angvel)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = current_env.step(action)
                    
                    # Update camera lookat to follow rover
                    camera_lookat = [current_env.data.qpos[0], current_env.data.qpos[1], 0.3]
                    
                    # Compute planned path for visualisation
                    planned_path = current_controller.compute_path(current_env.data.qpos[:2])
                    
                    # Render with camera parameters and path overlay
                    frame = current_env.render(
                        camera_distance=camera_distance,
                        camera_azimuth=camera_azimuth,
                        camera_elevation=camera_elevation,
                        camera_lookat=camera_lookat,
                        path=planned_path,
                    )
                    
                    # Convert frame to base64
                    if frame is not None:
                        img = Image.fromarray(frame)
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=65)
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                    else:
                        img_str = ""
                    
                    # Send state update
                    state_update = {
                        "type": "state",
                        "position": current_pos.tolist(),
                        "target": current_env.target_pos.tolist(),
                        "distance": float(info["distance_to_target"]),
                        "height": float(info.get("speed", 0.0)),
                        "frame": img_str,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "reward": float(reward),
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
            
            await asyncio.sleep(0.01)  # ~100 FPS max
            
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
