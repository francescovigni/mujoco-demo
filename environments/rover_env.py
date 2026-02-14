"""Rover navigation environment using MuJoCo."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from typing import Optional, Tuple, Dict, Any


class RoverNavigationEnv(gym.Env):
    """Custom environment for wheeled rover navigation."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        target_pos: Optional[np.ndarray] = None,
        obstacles: Optional[list] = None,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.target_pos = target_pos if target_pos is not None else np.array([10.0, 5.0])
        
        if obstacles is None:
            self.obstacles = self._generate_obstacles(num_obstacles=8)
        else:
            self.obstacles = obstacles
        
        xml_string = self._build_xml(self.obstacles, self.target_pos)
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        
        self.steps = 0
        self.max_steps = 2000
        
        # Rendering
        if render_mode in ("human", "rgb_array"):
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.free_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self.free_camera)
        else:
            self.renderer = None
            self.free_camera = None
        
        # 4 wheel velocity controls (left_front, right_front, left_rear, right_rear)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,), dtype=np.float32
        )
        
        obs_size = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32
        )
    
    # ------------------------------------------------------------------
    def _generate_obstacles(self, num_obstacles: int = 8) -> list:
        obstacles = []
        np.random.seed(42)
        for i in range(num_obstacles):
            angle = np.random.uniform(0, 2 * np.pi)
            dist  = np.random.uniform(2.5, 8.0)
            x, y  = dist * np.cos(angle), dist * np.sin(angle)
            radius = np.random.uniform(0.3, 0.6)
            height = np.random.uniform(0.5, 1.5)
            color  = [np.random.uniform(0.3, 0.7) for _ in range(3)] + [1.0]
            obstacles.append({
                'position': [x, y], 'radius': radius,
                'height': height, 'color': color, 'id': i
            })
        return obstacles
    
    # ------------------------------------------------------------------
    def _build_xml(self, obstacles, target_pos) -> str:
        xml = """
<mujoco model="rover">
  <compiler angle="degree"/>
  <option timestep="0.002" integrator="implicit" gravity="0 0 -9.81">
    <flag warmstart="enable"/>
  </option>

  <default>
    <geom condim="3" contype="1" conaffinity="1" friction="1.2 0.005 0.0001"/>
    <joint limited="false"/>
    <default class="wheel">
      <joint damping="0.05"/>
    </default>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.05 0.05 0.15"
             width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker"
             rgb1="0.25 0.35 0.45" rgb2="0.15 0.2 0.25" width="512" height="512"/>
    <material name="matplane" texture="texplane" texrepeat="10 10" reflectance="0.15"/>
    <material name="chassis_mat" rgba="0.15 0.25 0.55 1" specular="0.6" shininess="0.4"/>
    <material name="accent_mat" rgba="0.85 0.25 0.15 1" specular="0.5" shininess="0.3"/>
    <material name="wheel_mat" rgba="0.2 0.2 0.2 1" specular="0.2"/>
    <material name="hub_mat" rgba="0.6 0.6 0.65 1" specular="0.8" shininess="0.5"/>
  </asset>

  <worldbody>
    <light diffuse="0.9 0.9 0.9" specular="0.4 0.4 0.4" pos="0 0 8" dir="0 0 -1" directional="true"/>
    <light diffuse="0.3 0.3 0.35" specular="0.1 0.1 0.1" pos="5 -5 6" dir="-0.4 0.4 -1" directional="true"/>

    <geom name="floor" type="plane" size="40 40 0.1" material="matplane"/>

    <!-- ===== Rover chassis ===== -->
    <body name="chassis" pos="0 0 0.12">
      <camera name="track" mode="trackcom" pos="0 -3 1.5" xyaxes="1 0 0 0 0.4 1"/>
      <freejoint name="root"/>

      <!-- Main body -->
      <geom name="chassis_base" type="box" size="0.28 0.18 0.04"
            density="800" material="chassis_mat"/>
      <!-- Top plate -->
      <geom name="chassis_top" type="box" size="0.20 0.14 0.015" pos="0 0 0.055"
            density="200" material="chassis_mat"/>

      <!-- Sensor mast -->
      <geom name="mast" type="cylinder" size="0.015 0.08" pos="0.08 0 0.14"
            material="hub_mat"/>
      <!-- LIDAR dome -->
      <geom name="lidar" type="sphere" size="0.04" pos="0.08 0 0.24"
            material="accent_mat" contype="0" conaffinity="0"/>
      <!-- Antenna -->
      <geom name="antenna" type="capsule" fromto="-0.12 0 0.07 -0.14 0 0.20"
            size="0.008" material="hub_mat" contype="0" conaffinity="0"/>
      <geom name="antenna_tip" type="sphere" size="0.012" pos="-0.14 0 0.20"
            rgba="1 0.2 0.2 1" contype="0" conaffinity="0"/>

      <!-- Battery / electronics box -->
      <geom name="electronics" type="box" size="0.08 0.06 0.025" pos="-0.06 0 0.08"
            rgba="0.12 0.12 0.15 1" contype="0" conaffinity="0"/>

      <!-- Headlights -->
      <geom name="hl_l" type="cylinder" size="0.018 0.005" pos="0.28 0.10 0.0"
            euler="0 90 0" rgba="1 1 0.8 0.9" contype="0" conaffinity="0"/>
      <geom name="hl_r" type="cylinder" size="0.018 0.005" pos="0.28 -0.10 0.0"
            euler="0 90 0" rgba="1 1 0.8 0.9" contype="0" conaffinity="0"/>

      <!-- ===== Wheels ===== -->
      <!-- Front-left -->
      <body name="fl_wheel" pos="0.19 0.22 -0.02">
        <joint name="fl_joint" type="hinge" axis="0 1 0" class="wheel"/>
        <geom name="fl_tire" type="cylinder" size="0.065 0.028" euler="90 0 0"
              material="wheel_mat" friction="1.8 0.005 0.001" density="1200"/>
        <geom name="fl_hub" type="cylinder" size="0.025 0.030" euler="90 0 0"
              material="hub_mat" contype="0" conaffinity="0"/>
      </body>

      <!-- Front-right -->
      <body name="fr_wheel" pos="0.19 -0.22 -0.02">
        <joint name="fr_joint" type="hinge" axis="0 1 0" class="wheel"/>
        <geom name="fr_tire" type="cylinder" size="0.065 0.028" euler="90 0 0"
              material="wheel_mat" friction="1.8 0.005 0.001" density="1200"/>
        <geom name="fr_hub" type="cylinder" size="0.025 0.030" euler="90 0 0"
              material="hub_mat" contype="0" conaffinity="0"/>
      </body>

      <!-- Rear-left -->
      <body name="rl_wheel" pos="-0.19 0.22 -0.02">
        <joint name="rl_joint" type="hinge" axis="0 1 0" class="wheel"/>
        <geom name="rl_tire" type="cylinder" size="0.065 0.028" euler="90 0 0"
              material="wheel_mat" friction="1.8 0.005 0.001" density="1200"/>
        <geom name="rl_hub" type="cylinder" size="0.025 0.030" euler="90 0 0"
              material="hub_mat" contype="0" conaffinity="0"/>
      </body>

      <!-- Rear-right -->
      <body name="rr_wheel" pos="-0.19 -0.22 -0.02">
        <joint name="rr_joint" type="hinge" axis="0 1 0" class="wheel"/>
        <geom name="rr_tire" type="cylinder" size="0.065 0.028" euler="90 0 0"
              material="wheel_mat" friction="1.8 0.005 0.001" density="1200"/>
        <geom name="rr_hub" type="cylinder" size="0.025 0.030" euler="90 0 0"
              material="hub_mat" contype="0" conaffinity="0"/>
      </body>
    </body>
"""
        # Obstacles
        for obs in obstacles:
            x, y = obs['position']
            r, g, b, a = obs['color']
            rad, h = obs['radius'], obs['height']
            oid = obs['id']
            xml += f"""
    <body name="obstacle_{oid}" pos="{x} {y} {h/2}">
      <geom name="obstacle_{oid}" type="cylinder" size="{rad} {h/2}"
            rgba="{r} {g} {b} {a}" contype="1" conaffinity="1"/>
    </body>
"""
        # Target marker
        tx, ty = target_pos[0], target_pos[1]
        xml += f"""
    <body name="target_marker" pos="{tx} {ty} 0.01">
      <geom name="target_base" type="cylinder" size="0.35 0.015"
            rgba="0.15 0.75 0.25 0.7" contype="0" conaffinity="0"/>
      <geom name="target_ring" type="cylinder" size="0.25 0.02"
            rgba="0.2 0.85 0.3 0.8" pos="0 0 0.02" contype="0" conaffinity="0"/>
      <geom name="target_center" type="cylinder" size="0.10 0.025"
            rgba="0.25 1.0 0.35 0.9" pos="0 0 0.04" contype="0" conaffinity="0"/>
      <geom name="target_pole" type="cylinder" size="0.02 0.5"
            rgba="0.2 0.9 0.3 0.6" pos="0 0 0.5" contype="0" conaffinity="0"/>
      <geom name="target_flag" type="box" size="0.12 0.005 0.06"
            rgba="0.2 1.0 0.3 0.9" pos="0.12 0 0.95" contype="0" conaffinity="0"/>
    </body>
"""
        xml += """
  </worldbody>

  <actuator>
    <velocity name="fl_motor" joint="fl_joint" kv="10" ctrlrange="-30 30"/>
    <velocity name="fr_motor" joint="fr_joint" kv="10" ctrlrange="-30 30"/>
    <velocity name="rl_motor" joint="rl_joint" kv="10" ctrlrange="-30 30"/>
    <velocity name="rr_motor" joint="rr_joint" kv="10" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""
        return xml
    
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        pos  = self.data.qpos[:3].copy()      # x, y, z
        quat = self.data.qpos[3:7].copy()     # orientation
        vel  = self.data.qvel[:3].copy()       # linear vel
        angvel = self.data.qvel[3:6].copy()    # angular vel
        wheel_vel = self.data.qvel[6:10].copy()  # wheel speeds
        
        target_dir = self.target_pos - pos[:2]
        target_dist = np.linalg.norm(target_dir)
        target_dir_norm = target_dir / (target_dist + 1e-8)
        
        obs = np.concatenate([
            pos, quat, vel, angvel, wheel_vel,
            target_dir_norm, [target_dist],
        ])
        return obs.astype(np.float32)
    
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Scale action → velocity command  (ctrlrange handles final clamp)
        max_speed = 10.0
        ctrl = action * max_speed
        self.data.ctrl[:] = ctrl
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        
        obs = self._get_obs()
        reward = self._compute_reward()
        
        # Termination: flipped over
        chassis_z = self.data.qpos[2]
        # Check if upside-down using z-component of body up-vector
        quat = self.data.qpos[3:7]
        # up-vector of body: rotate [0,0,1] by quat
        w, x, y, z = quat
        up_z = 1 - 2*(x*x + y*y)
        terminated = up_z < 0.3 or chassis_z < 0.02  # flipped or sunk
        truncated = self.steps >= self.max_steps
        
        rover_pos = self.data.qpos[:2]
        dist_to_target = np.linalg.norm(rover_pos - self.target_pos)
        
        if dist_to_target < 0.5:
            reward += 100.0
            terminated = True
        
        info = {
            "distance_to_target": dist_to_target,
            "chassis_height": float(chassis_z),
            "position": rover_pos.copy(),
            "speed": float(np.linalg.norm(self.data.qvel[:2])),
        }
        
        return obs, reward, terminated, truncated, info
    
    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        pos = self.data.qpos[:2]
        dist = np.linalg.norm(pos - self.target_pos)
        
        reward_dist = -0.1 * dist
        
        # Forward velocity toward target
        to_target = self.target_pos - pos
        to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-8)
        vel_toward = np.dot(self.data.qvel[:2], to_target_norm)
        reward_progress = 0.5 * vel_toward
        
        # Action cost
        action_cost = -0.01 * np.sum(np.square(self.data.ctrl))
        
        return reward_dist + reward_progress + action_cost
    
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0.copy()
        self.data.qvel[:] = 0.0
        
        if options and "target_pos" in options:
            self.target_pos = np.array(options["target_pos"])
            self._update_target_marker()
        
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        obs = self._get_obs()
        info = {"distance_to_target": np.linalg.norm(self.data.qpos[:2] - self.target_pos)}
        return obs, info
    
    def _update_target_marker(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        if body_id >= 0:
            self.model.body_pos[body_id] = [self.target_pos[0], self.target_pos[1], 0.01]
    
    # ------------------------------------------------------------------
    def render(self, camera_distance=None, camera_azimuth=None,
               camera_elevation=None, camera_lookat=None,
               path: np.ndarray = None):
        """Render the scene. If *path* is provided (N×2 array), draw it."""
        if self.renderer is None:
            return None
        
        if camera_distance is not None or camera_azimuth is not None or camera_elevation is not None:
            if camera_lookat is not None:
                self.free_camera.lookat[:] = camera_lookat
            if camera_distance is not None:
                self.free_camera.distance = camera_distance
            if camera_azimuth is not None:
                self.free_camera.azimuth = camera_azimuth
            if camera_elevation is not None:
                self.free_camera.elevation = camera_elevation
            self.free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.renderer.update_scene(self.data, camera=self.free_camera)
        else:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
            self.renderer.update_scene(self.data, camera=cam_id if cam_id >= 0 else None)
        
        # Draw planned path as custom visualisation geoms
        if path is not None and len(path) >= 2:
            self._draw_path(path)
        
        return self.renderer.render()
    
    # ------------------------------------------------------------------
    def _draw_path(self, path: np.ndarray):
        """Inject path markers into the scene before rendering.
        
        Draws small spheres at each waypoint and capsule segments
        connecting consecutive waypoints.
        """
        scene = self.renderer._scene          # mjvScene
        PATH_Z = 0.06                         # height above ground
        SPHERE_RADIUS = 0.04
        LINE_RADIUS = 0.018

        n = len(path)

        for i in range(n):
            if scene.ngeom >= scene.maxgeom:
                break

            # ---- Colour: gradient from cyan (start) → green (end) ----
            t = i / max(n - 1, 1)
            rgba = np.array([0.0, 0.4 + 0.6 * t, 1.0 - 0.7 * t, 0.85], dtype=np.float32)

            # ---- Waypoint sphere ----
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([SPHERE_RADIUS, 0, 0], dtype=np.float64),
                np.array([path[i, 0], path[i, 1], PATH_Z], dtype=np.float64),
                np.eye(3).flatten().astype(np.float64),
                rgba.astype(np.float32),
            )
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1

            # ---- Segment capsule to next waypoint ----
            if i < n - 1:
                if scene.ngeom >= scene.maxgeom:
                    break
                p0 = np.array([path[i, 0], path[i, 1], PATH_Z], dtype=np.float64)
                p1 = np.array([path[i + 1, 0], path[i + 1, 1], PATH_Z], dtype=np.float64)

                g2 = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    g2,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3, dtype=np.float64),   # size filled by connector
                    np.zeros(3, dtype=np.float64),    # pos filled by connector
                    np.eye(3).flatten().astype(np.float64),
                    rgba.astype(np.float32),
                )
                mujoco.mjv_connector(
                    g2,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    LINE_RADIUS,
                    p0, p1,
                )
                g2.category = mujoco.mjtCatBit.mjCAT_DECOR
                scene.ngeom += 1
    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
