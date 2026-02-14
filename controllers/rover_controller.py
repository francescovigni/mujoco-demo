"""Differential-drive navigation controller for a 4-wheeled rover."""

import numpy as np
from typing import Optional, List, Dict


def quat_to_yaw(quat):
    """Extract yaw angle from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny, cosy)


class RoverController:
    """Differential-drive controller with potential-field obstacle avoidance.

    Uses attractive gradient toward target + repulsive gradient from obstacles.

    Actuators (velocity motors):
      0: front-left   1: front-right
      2: rear-left    3: rear-right
    """

    def __init__(self):
        self.target_pos = np.array([10.0, 5.0])
        self.obstacles: List[Dict] = []
        self.max_speed = 1.0       # max normalised speed command
        self.approach_dist = 0.5   # stop within this distance
        self.speed_factor = 1.0    # user-adjustable speed multiplier (0.1–1.0)
        self.num_path_points = 40  # user-adjustable path resolution

        # Tuning
        self.kp_heading = 1.8      # proportional gain on heading error
        self.kp_speed = 1.0        # proportional gain on forward speed
        self.max_turn_rate = 0.9   # max differential (normalised)

        # Obstacle avoidance
        self.obstacle_margin = 0.45  # rover effective radius
        self.repulse_range = 1.2     # start repelling within this distance
        self.repulse_gain = 2.5      # strength of repulsive field

    def set_target(self, target_pos):
        self.target_pos = np.asarray(target_pos, dtype=float)

    def set_obstacles(self, obstacles: List[Dict]):
        """Set obstacle list.  Each dict needs 'position' [x,y] and 'radius'."""
        self.obstacles = obstacles

    def _repulsive_gradient(self, pos: np.ndarray) -> np.ndarray:
        """Compute repulsive gradient pointing away from nearby obstacles."""
        grad = np.zeros(2)
        for obs in self.obstacles:
            obs_pos = np.array(obs['position'])
            obs_rad = obs['radius']
            diff = pos - obs_pos
            dist = np.linalg.norm(diff)
            clearance = dist - obs_rad - self.obstacle_margin
            if clearance < self.repulse_range and dist > 0.01:
                strength = self.repulse_gain * (
                    1.0 / max(clearance, 0.05) - 1.0 / self.repulse_range
                )
                grad += strength * diff / dist
        return grad

    def get_action(self, current_pos, current_vel,
                   body_quat=None, body_angvel=None):
        """Return 4-element action array (normalised −1 … 1)."""
        to_target = self.target_pos - current_pos
        dist = np.linalg.norm(to_target)

        if dist < self.approach_dist:
            return np.zeros(4)

        # Attractive gradient (toward target)
        attract = to_target / (dist + 1e-8)

        # Repulsive gradient (away from obstacles) – fade near target
        repulse = self._repulsive_gradient(current_pos)
        # Dampen repulsion when close to target so we don't oscillate
        repulse_weight = min(1.0, dist / 2.0)
        repulse *= repulse_weight

        # Combined desired direction
        desired_dir = attract + repulse
        desired_mag = np.linalg.norm(desired_dir)
        if desired_mag > 1e-6:
            desired_dir /= desired_mag

        desired_yaw = np.arctan2(desired_dir[1], desired_dir[0])

        # Current heading from body quaternion
        if body_quat is not None:
            current_yaw = quat_to_yaw(body_quat)
        else:
            current_yaw = 0.0

        # Heading error (wrap to −π … π)
        heading_err = desired_yaw - current_yaw
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi

        # Turn rate (differential)
        turn = np.clip(self.kp_heading * heading_err,
                       -self.max_turn_rate, self.max_turn_rate)

        # Forward speed – reduced when heading is far off, ramps down near target
        heading_factor = max(0.2, 1.0 - abs(heading_err) / (np.pi / 2))
        dist_factor = min(1.0, dist / 1.5)
        forward = self.kp_speed * heading_factor * dist_factor * self.speed_factor

        # Differential drive: to turn LEFT (positive yaw) → right faster
        limit = self.max_speed * self.speed_factor
        left  = np.clip(forward - turn, -limit, limit)
        right = np.clip(forward + turn, -limit, limit)

        # All 4 wheels: [fl, fr, rl, rr]
        return np.array([left, right, left, right], dtype=np.float32)

    # ------------------------------------------------------------------
    def compute_path(self, start_pos: np.ndarray,
                     num_points: int = None, step_size: float = 0.25
                     ) -> np.ndarray:
        """Forward-simulate the potential field to predict the path.

        Returns an (N, 2) array of 2-D waypoints from *start_pos* toward
        the target, steering around obstacles.
        """
        if num_points is None:
            num_points = self.num_path_points
        path = [start_pos.copy()]
        pos = start_pos.copy()

        for _ in range(num_points):
            to_target = self.target_pos - pos
            dist = np.linalg.norm(to_target)
            if dist < self.approach_dist:
                break

            attract = to_target / (dist + 1e-8)
            repulse = self._repulsive_gradient(pos)
            repulse *= min(1.0, dist / 2.0)

            desired = attract + repulse
            mag = np.linalg.norm(desired)
            if mag > 1e-6:
                desired /= mag

            pos = pos + desired * step_size
            path.append(pos.copy())

        # Always include the target as the last point
        path.append(self.target_pos.copy())
        return np.array(path)

    def reset(self):
        pass
