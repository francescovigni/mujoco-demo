"""Humanoid navigation environment using MuJoCo."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from typing import Optional, Tuple, Dict, Any
import os


class HumanoidNavigationEnv(gym.Env):
    """Custom environment for humanoid navigation."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        model_name: str = "humanoid",
        render_mode: Optional[str] = None,
        target_pos: Optional[np.ndarray] = None,
        obstacles: Optional[list] = None,
    ):
        """
        Initialize humanoid navigation environment.
        
        Args:
            model_name: Name of the MuJoCo model to use
            render_mode: Rendering mode ('human' or 'rgb_array')
            target_pos: Target position [x, y] for navigation
            obstacles: List of obstacle positions and sizes
        """
        super().__init__()
        
        self.model_name = model_name
        self.render_mode = render_mode
        
        # Navigation target (must be set before calling _get_obs)
        self.target_pos = target_pos if target_pos is not None else np.array([5.0, 0.0])
        
        # Generate random obstacles if not provided
        if obstacles is None:
            self.obstacles = self._generate_obstacles(num_obstacles=8)
        else:
            self.obstacles = obstacles
        
        # Load MuJoCo model with obstacles
        if model_name in ["humanoid", "humanoidstandup"]:
            # Use built-in MuJoCo models with obstacles added
            xml_string = self._get_builtin_xml(model_name, self.obstacles, self.target_pos)
            self.model = mujoco.MjModel.from_xml_string(xml_string)
        else:
            # Load custom model
            self.model = mujoco.MjModel.from_xml_path(model_name)
        
        self.data = mujoco.MjData(self.model)
        
        # Episode tracking
        self.steps = 0
        self.max_steps = 1000
        
        # Set up rendering
        if render_mode == "human" or render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            # Set camera to tracking camera for better view
            self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
            # Create a free camera for custom views
            self.free_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self.free_camera)
        else:
            self.renderer = None
            self.camera_id = -1
            self.free_camera = None
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )
        
        obs_size = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def _generate_obstacles(self, num_obstacles: int = 8) -> list:
        """Generate random obstacles in the environment."""
        obstacles = []
        np.random.seed(42)  # For reproducibility
        for i in range(num_obstacles):
            # Random position (avoiding spawn area around origin)
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(2.5, 8.0)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            
            # Random size
            radius = np.random.uniform(0.3, 0.6)
            height = np.random.uniform(0.8, 2.5)
            
            # Random color
            color = [
                np.random.uniform(0.3, 0.7),
                np.random.uniform(0.3, 0.7),
                np.random.uniform(0.3, 0.7),
                1.0
            ]
            
            obstacles.append({
                'position': [x, y],
                'radius': radius,
                'height': height,
                'color': color,
                'id': i
            })
        return obstacles
        
    def _get_builtin_xml(self, model_name: str, obstacles: list = None, target_pos: np.ndarray = None) -> str:
        """Get XML string for built-in MuJoCo models with obstacles."""
        obstacles = obstacles or []
        target_pos = target_pos if target_pos is not None else np.array([5.0, 0.0])
        
        if model_name == "humanoid":
            # Based on Gymnasium humanoid XML with modifications:
            # - position actuators (servos) instead of torque motors
            # - condim=3 with friction for stable contact
            # - box feet for better stability (instead of sphere)
            # - per-joint damping/stiffness from Gymnasium defaults
            xml_str = """
<mujoco model="humanoid">
  <compiler angle="degree" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="3" contype="1" margin="0.001" rgba="0.8 0.6 .4 1" friction="1.0 0.005 0.0001"/>
    <position ctrllimited="true" ctrlrange="-1.57 1.57" kp="200"/>
  </default>
  <option integrator="RK4" iterations="50" solver="PGS" timestep="0.003"/>
  <size nkey="5" nuser_geom="1"/>
  <visual>
    <map fogend="5" fogstart="3" znear="0.02"/>
    <quality shadowsize="2048"/>
  </visual>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom condim="3" friction="1 .1 .1" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 0.125" type="plane"/>
    <body name="torso" pos="0 0 1.4">
      <camera name="track" mode="trackcom" pos="0 -4 0" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>
      <geom fromto="0 -.07 0 0 .07 0" name="torso1" size="0.07" type="capsule"/>
      <geom name="head" pos="0 0 .19" size=".09" type="sphere" user="258"/>
      <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist" size="0.06" type="capsule"/>
      <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
        <geom fromto="0 -.06 0 0 .06 0" name="lwaist" size="0.06" type="capsule"/>
        <joint armature="0.02" axis="0 0 1" damping="5" name="abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
        <joint armature="0.02" axis="0 1 0" damping="5" name="abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
        <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
          <joint armature="0.02" axis="1 0 0" damping="5" name="abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
          <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt" size="0.09" type="capsule"/>
          <body name="right_thigh" pos="0 -0.1 -0.04">
            <joint armature="0.01" axis="1 0 0" damping="5" name="right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
            <joint armature="0.01" axis="0 0 1" damping="5" name="right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
            <joint armature="0.0080" axis="0 1 0" damping="5" name="right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
            <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1" size="0.06" type="capsule"/>
            <body name="right_shin" pos="0 0.01 -0.403">
              <joint armature="0.0060" axis="0 -1 0" name="right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
              <geom fromto="0 0 0 0 0 -.3" name="right_shin1" size="0.049" type="capsule"/>
              <body name="right_foot" pos="0 0 -0.35">
                <geom name="right_foot" pos="0.04 0 -0.03" size="0.08 0.06 0.03" type="box" friction="1.5 0.1 0.1"/>
              </body>
            </body>
          </body>
          <body name="left_thigh" pos="0 0.1 -0.04">
            <joint armature="0.01" axis="-1 0 0" damping="5" name="left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
            <joint armature="0.01" axis="0 0 -1" damping="5" name="left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
            <joint armature="0.01" axis="0 1 0" damping="5" name="left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
            <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size="0.06" type="capsule"/>
            <body name="left_shin" pos="0 -0.01 -0.403">
              <joint armature="0.0060" axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
              <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size="0.049" type="capsule"/>
              <body name="left_foot" pos="0 0 -0.35">
                <geom name="left_foot" pos="0.04 0 -0.03" size="0.08 0.06 0.03" type="box" friction="1.5 0.1 0.1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="right_upper_arm" pos="0 -0.17 0.06">
        <joint armature="0.0068" axis="2 1 1" name="right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
        <joint armature="0.0051" axis="0 -1 1" name="right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
        <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1" size="0.04 0.16" type="capsule"/>
        <body name="right_lower_arm" pos=".18 -.18 -.18">
          <joint armature="0.0028" axis="0 -1 1" name="right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
          <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="right_larm" size="0.031" type="capsule"/>
          <geom name="right_hand" pos=".18 .18 .18" size="0.04" type="sphere"/>
        </body>
      </body>
      <body name="left_upper_arm" pos="0 0.17 0.06">
        <joint armature="0.0068" axis="2 -1 1" name="left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
        <joint armature="0.0051" axis="0 1 1" name="left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
        <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1" size="0.04 0.16" type="capsule"/>
        <body name="left_lower_arm" pos=".18 .18 -.18">
          <joint armature="0.0028" axis="0 -1 -1" name="left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
          <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="left_larm" size="0.031" type="capsule"/>
          <geom name="left_hand" pos=".18 -.18 .18" size="0.04" type="sphere"/>
        </body>
      </body>
    </body>
"""
            
            # Add obstacles
            for obs in obstacles:
                x, y = obs['position']
                r, g, b, a = obs['color']
                radius = obs['radius']
                height = obs['height']
                obs_id = obs['id']
                
                xml_str += f"""
    <body name="obstacle_{obs_id}" pos="{x} {y} {height/2}">
      <geom name="obstacle_{obs_id}" type="cylinder" size="{radius} {height/2}" rgba="{r} {g} {b} {a}" contype="1" conaffinity="1"/>
    </body>
"""
            
            # Add target marker
            xml_str += f"""
    <body name="target_marker" pos="{target_pos[0]} {target_pos[1]} 0.1">
      <geom name="target_base" type="cylinder" size="0.3 0.02" rgba="0.2 0.8 0.2 0.8" contype="0" conaffinity="0"/>
      <geom name="target_marker" type="sphere" size="0.15" rgba="0.2 1.0 0.2 0.9" contype="0" conaffinity="0" pos="0 0 0.5"/>
    </body>
"""
            
            xml_str += """
  </worldbody>
  <tendon>
    <fixed name="left_hipknee">
      <joint coef="-1" joint="left_hip_y"/>
      <joint coef="1" joint="left_knee"/>
    </fixed>
    <fixed name="right_hipknee">
      <joint coef="-1" joint="right_hip_y"/>
      <joint coef="1" joint="right_knee"/>
    </fixed>
  </tendon>
  <actuator>
    <position joint="abdomen_y" name="abdomen_y" kp="200"/>
    <position joint="abdomen_z" name="abdomen_z" kp="200"/>
    <position joint="abdomen_x" name="abdomen_x" kp="200"/>
    <position joint="right_hip_x" name="right_hip_x" kp="300"/>
    <position joint="right_hip_z" name="right_hip_z" kp="200"/>
    <position joint="right_hip_y" name="right_hip_y" kp="400"/>
    <position joint="right_knee" name="right_knee" kp="300"/>
    <position joint="left_hip_x" name="left_hip_x" kp="300"/>
    <position joint="left_hip_z" name="left_hip_z" kp="200"/>
    <position joint="left_hip_y" name="left_hip_y" kp="400"/>
    <position joint="left_knee" name="left_knee" kp="300"/>
    <position joint="right_shoulder1" name="right_shoulder1" kp="50"/>
    <position joint="right_shoulder2" name="right_shoulder2" kp="50"/>
    <position joint="right_elbow" name="right_elbow" kp="50"/>
    <position joint="left_shoulder1" name="left_shoulder1" kp="50"/>
    <position joint="left_shoulder2" name="left_shoulder2" kp="50"/>
    <position joint="left_elbow" name="left_elbow" kp="50"/>
  </actuator>
</mujoco>
"""
            return xml_str
        return ""
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Position and orientation of torso
        torso_pos = self.data.qpos[:3].copy()
        torso_quat = self.data.qpos[3:7].copy()
        
        # Joint positions and velocities
        qpos = self.data.qpos[7:].copy()
        qvel = self.data.qvel.copy()
        
        # Direction and distance to target
        target_dir = self.target_pos - torso_pos[:2]
        target_dist = np.linalg.norm(target_dir)
        target_dir_norm = target_dir / (target_dist + 1e-8)
        
        # Concatenate observation
        obs = np.concatenate([
            torso_pos,
            torso_quat,
            qpos,
            qvel,
            target_dir_norm,
            [target_dist],
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Apply action - position actuators with ctrlrange [-1.57, 1.57]
        ctrl = np.clip(action, -1.57, 1.57)
        self.data.ctrl[:] = ctrl
        
        # Step simulation - fewer substeps for more responsive control
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check termination - very forgiving bounds
        torso_height = self.data.qpos[2]
        terminated = torso_height < 0.4 or torso_height > 3.0  # Very forgiving
        truncated = self.steps >= self.max_steps
        
        if terminated:
            print(f"⚠️ Robot terminated! Height={torso_height:.3f}")
        
        # Check if goal reached
        torso_pos = self.data.qpos[:2]
        dist_to_target = np.linalg.norm(torso_pos - self.target_pos)
        if dist_to_target < 0.5:
            reward += 100.0
            terminated = True
        
        info = {
            "distance_to_target": dist_to_target,
            "torso_height": torso_height,
            "position": torso_pos.copy(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        # Get humanoid position
        torso_pos = self.data.qpos[:2]
        
        # Distance to target - reduced weight for exploration
        dist_to_target = np.linalg.norm(torso_pos - self.target_pos)
        reward_dist = -0.1 * dist_to_target
        
        # Strong reward for staying upright
        torso_height = self.data.qpos[2]
        reward_upright = 5.0 if 1.0 < torso_height < 1.7 else -2.0
        
        # Penalty for high action - encourage smooth movements
        action_penalty = -0.005 * np.sum(np.square(self.data.ctrl))
        
        # Forward progress reward - reduced to prevent rushing
        forward_vel = self.data.qvel[0]
        reward_forward = 0.05 * forward_vel if forward_vel > 0 else 0.0
        
        # Stability bonus - penalize high angular velocities
        angular_penalty = -0.5 * np.sum(np.square(self.data.qvel[3:6]))
        
        total_reward = reward_dist + reward_upright + action_penalty + reward_forward + angular_penalty
        
        return total_reward
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial position - only add small noise to internal joints, NOT the free joint
        self.data.qpos[:] = self.model.qpos0.copy()
        # Free joint is first 7 values (3 pos + 4 quat) - don't perturb these
        # Only add small noise to internal joints (index 7+)
        if self.np_random is not None and self.model.nq > 7:
            noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq - 7)
            self.data.qpos[7:] += noise
        self.data.qvel[:] = 0.0
        
        # Update target if provided
        if options and "target_pos" in options:
            self.target_pos = np.array(options["target_pos"])
            self._update_target_marker()
        
        # Forward the simulation to update visualization
        mujoco.mj_forward(self.model, self.data)
        
        self.steps = 0
        
        obs = self._get_obs()
        info = {"distance_to_target": np.linalg.norm(self.data.qpos[:2] - self.target_pos)}
        
        return obs, info
    
    def _update_target_marker(self):
        """Update the target marker position in the simulation."""
        # Find target marker body
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        if body_id >= 0:
            # Update the body position (this updates the visual marker)
            self.model.body_pos[body_id] = [self.target_pos[0], self.target_pos[1], 0.1]
    
    def render(self, camera_distance=None, camera_azimuth=None, camera_elevation=None, camera_lookat=None):
        """Render the environment with optional custom camera parameters."""
        if self.render_mode is None:
            return None
        
        if self.renderer is None:
            return None
        
        # If custom camera parameters provided, use free camera
        if camera_distance is not None or camera_azimuth is not None or camera_elevation is not None:
            # Use default lookat or provided one
            if camera_lookat is not None:
                self.free_camera.lookat[:] = camera_lookat
            if camera_distance is not None:
                self.free_camera.distance = camera_distance
            if camera_azimuth is not None:
                self.free_camera.azimuth = camera_azimuth
            if camera_elevation is not None:
                self.free_camera.elevation = camera_elevation
            
            # Set camera type to free
            self.free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            
            # Pass free_camera directly to update_scene so it uses our camera
            self.renderer.update_scene(self.data, camera=self.free_camera)
            pixels = self.renderer.render()
        else:
            # Use fixed tracking camera
            self.renderer.update_scene(self.data, camera=self.camera_id if self.camera_id >= 0 else None)
            pixels = self.renderer.render()
        
        if self.render_mode == "rgb_array":
            return pixels
        
        return pixels
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
