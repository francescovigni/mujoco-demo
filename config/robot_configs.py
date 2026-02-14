"""Configuration for robot models."""

ROBOT_MODELS = {
    "rover_4wd": {
        "name": "4WD Rover",
        "description": "Four-wheeled differential-drive rover with LIDAR dome",
        "action_dim": 4,
        "height": 0.12,
        "mass": 5.0,
    },
}

# Environment configuration
ENV_CONFIG = {
    "timestep": 0.005,
    "frame_skip": 4,
    "max_episode_steps": 2000,
    "target_tolerance": 0.5,
}

# Control configuration
CONTROL_CONFIG = {
    "max_speed": 15.0,
    "turn_speed": 2.5,
}

# Obstacle configuration
OBSTACLE_CONFIG = {
    "num_obstacles": 8,
    "min_radius": 0.3,
    "max_radius": 0.6,
    "min_height": 0.5,
    "max_height": 1.5,
}
