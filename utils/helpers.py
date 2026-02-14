"""Utility functions for the humanoid navigation demo."""

import numpy as np
from typing import Tuple, List


def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Euclidean distance between two positions."""
    return float(np.linalg.norm(pos1 - pos2))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def quaternion_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quat: Quaternion [w, x, y, z]
    
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def generate_random_obstacles(
    num_obstacles: int,
    workspace_size: Tuple[float, float],
    min_radius: float = 0.2,
    max_radius: float = 0.5,
) -> List[dict]:
    """
    Generate random obstacles in the workspace.
    
    Args:
        num_obstacles: Number of obstacles to generate
        workspace_size: (width, height) of workspace
        min_radius: Minimum obstacle radius
        max_radius: Maximum obstacle radius
    
    Returns:
        List of obstacle dictionaries with position and size
    """
    obstacles = []
    
    for _ in range(num_obstacles):
        x = np.random.uniform(-workspace_size[0]/2, workspace_size[0]/2)
        y = np.random.uniform(-workspace_size[1]/2, workspace_size[1]/2)
        radius = np.random.uniform(min_radius, max_radius)
        
        obstacles.append({
            'position': [x, y],
            'radius': radius,
        })
    
    return obstacles


def check_collision(
    pos: np.ndarray,
    obstacles: List[dict],
    robot_radius: float = 0.3,
) -> bool:
    """
    Check if position collides with any obstacle.
    
    Args:
        pos: Position to check [x, y]
        obstacles: List of obstacle dictionaries
        robot_radius: Radius of the robot
    
    Returns:
        True if collision detected
    """
    for obstacle in obstacles:
        obs_pos = np.array(obstacle['position'])
        obs_radius = obstacle['radius']
        
        dist = calculate_distance(pos, obs_pos)
        if dist < (robot_radius + obs_radius):
            return True
    
    return False


def smooth_trajectory(
    waypoints: List[np.ndarray],
    num_points: int = 50,
) -> np.ndarray:
    """
    Create smooth trajectory through waypoints using spline interpolation.
    
    Args:
        waypoints: List of waypoint positions
        num_points: Number of interpolation points
    
    Returns:
        Smooth trajectory as numpy array
    """
    from scipy.interpolate import CubicSpline
    
    if len(waypoints) < 2:
        return np.array(waypoints)
    
    waypoints = np.array(waypoints)
    t = np.linspace(0, 1, len(waypoints))
    t_smooth = np.linspace(0, 1, num_points)
    
    cs = CubicSpline(t, waypoints, axis=0)
    smooth_path = cs(t_smooth)
    
    return smooth_path
