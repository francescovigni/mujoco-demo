"""Configuration for different humanoid models."""

HUMANOID_MODELS = {
    "mujoco_humanoid": {
        "name": "MuJoCo Humanoid",
        "description": "Default MuJoCo humanoid model with 17 DoF",
        "xml_path": "humanoid",  # Built-in MuJoCo model
        "action_dim": 17,
        "observation_dim": 376,
        "height": 1.4,
        "mass": 45.0,
        "builtin": True
    },
    "mujoco_humanoid_simple": {
        "name": "Simple Humanoid",
        "description": "Simplified humanoid for faster simulation",
        "xml_path": "humanoidstandup",  # Built-in MuJoCo model
        "action_dim": 17,
        "observation_dim": 376,
        "height": 1.4,
        "mass": 45.0,
        "builtin": True
    }
}

# Environment configuration
ENV_CONFIG = {
    "timestep": 0.002,
    "frame_skip": 5,
    "max_episode_steps": 1000,
    "target_tolerance": 0.5,  # Distance to target in meters
}

# Control configuration
CONTROL_CONFIG = {
    "action_scale": 0.4,
    "forward_speed": 1.0,
    "turn_speed": 0.5,
}

# Obstacle configuration
OBSTACLE_CONFIG = {
    "num_obstacles": 5,
    "min_radius": 0.2,
    "max_radius": 0.5,
    "min_height": 0.5,
    "max_height": 2.0,
}
