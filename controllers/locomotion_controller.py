"Controller for Gymnasium humanoid with position-servo actuators and active balance."

import numpy as np
from typing import Optional


def quat_to_euler(quat):
    "Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]."
    w, x, y, z = quat
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


class SimpleLocomotionController:
    """Position-servo controller with active balance for Gymnasium humanoid.

    17 actuators:
     0: abdomen_y     (torso forward/back bend)
     1: abdomen_z     (torso twist)
     2: abdomen_x     (torso lateral bend)
     3: right_hip_x   (right hip lateral)
     4: right_hip_z   (right hip rotation)
     5: right_hip_y   (right hip forward/back)
     6: right_knee    (right knee)
     7: left_hip_x    (left hip lateral)
     8: left_hip_z    (left hip rotation)
     9: left_hip_y    (left hip forward/back)
    10: left_knee     (left knee)
    11: right_shoulder1
    12: right_shoulder2
    13: right_elbow
    14: left_shoulder1
    15: left_shoulder2
    16: left_elbow
    """

    def __init__(self, action_dim=17):
        self.action_dim = action_dim
        self.phase = 0.0
        self.frequency = 0.8
        self.dt = 0.006

        # Joint indices
        self.ABD_Y, self.ABD_Z, self.ABD_X = 0, 1, 2
        self.R_HIP_X, self.R_HIP_Z, self.R_HIP_Y = 3, 4, 5
        self.R_KNEE = 6
        self.L_HIP_X, self.L_HIP_Z, self.L_HIP_Y = 7, 8, 9
        self.L_KNEE = 10
        self.R_SHOULDER1, self.R_SHOULDER2, self.R_ELBOW = 11, 12, 13
        self.L_SHOULDER1, self.L_SHOULDER2, self.L_ELBOW = 14, 15, 16

        # Standing pose (all zeros = upright with straight joints)
        self.stand_pose = np.zeros(action_dim)

    def get_action(self, target_direction, current_velocity,
                   body_quat=None, body_angvel=None):
        self.phase += 2 * np.pi * self.frequency * self.dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        action = self.stand_pose.copy()

        # Active balance via abdomen, with forward lean for walking
        # NOTE: pitch > 0 means body leaning forward, negative ABD_Y = forward lean
        forward_lean = -0.06  # lean torso forward to initiate walking
        if body_quat is not None:
            roll, pitch, yaw = quat_to_euler(body_quat)
            pr = body_angvel[1] if body_angvel is not None else 0.0
            rr = body_angvel[0] if body_angvel is not None else 0.0
            # Correct negative feedback: pitch>0 (fwd lean) -> positive correction (lean back)
            action[self.ABD_Y] = forward_lean + np.clip(0.8 * pitch + 0.1 * pr, -0.4, 0.4)
            action[self.ABD_X] = np.clip(0.5 * roll + 0.1 * rr, -0.3, 0.3)
        else:
            action[self.ABD_Y] = forward_lean

        # Walking gait
        rp = self.phase
        lp = self.phase + np.pi

        # Hip y: forward/back swing (negative = forward, positive = backward)
        # Asymmetric: more push-off (stance) than swing, with forward bias
        r_swing = max(0, np.sin(rp))
        r_stance = max(0, -np.sin(rp))
        l_swing = max(0, np.sin(lp))
        l_stance = max(0, -np.sin(lp))
        
        hip_fwd_bias = -0.05  # slight forward bias
        action[self.R_HIP_Y] = hip_fwd_bias - 0.15 * r_swing + 0.10 * r_stance
        action[self.L_HIP_Y] = hip_fwd_bias - 0.15 * l_swing + 0.10 * l_stance

        # Knee: bend during swing phase
        action[self.R_KNEE] = -0.2 * r_swing
        action[self.L_KNEE] = -0.2 * l_swing

        # Hip x: no lateral oscillation — let abdomen balance handle it
        # (Gymnasium model has opposite axis signs per side, so simple oscillation causes drift)

        # Turning toward target
        if np.linalg.norm(target_direction) > 0.01:
            turn = np.clip(target_direction[1] * 0.08, -0.1, 0.1)
            action[self.R_HIP_Z] = turn
            action[self.L_HIP_Z] = turn
            action[self.ABD_Z] = turn * 0.5

        # Arms counter-swing
        action[self.R_SHOULDER1] = 0.15 * np.sin(lp)
        action[self.R_SHOULDER2] = -0.1
        action[self.R_ELBOW] = -0.2
        action[self.L_SHOULDER1] = 0.15 * np.sin(rp)
        action[self.L_SHOULDER2] = -0.1
        action[self.L_ELBOW] = -0.2

        return np.clip(action, -1.57, 1.57)

    def reset(self):
        self.phase = 0.0


class NavigationController:
    def __init__(self):
        self.locomotion_controller = SimpleLocomotionController()
        self.target_pos = np.array([0.0, 0.0])

    def set_target(self, target_pos):
        self.target_pos = target_pos

    def get_action(self, current_pos, current_velocity,
                   body_quat=None, body_angvel=None):
        direction = self.target_pos - current_pos
        distance = np.linalg.norm(direction)
        if distance < 0.5:
            action = self.locomotion_controller.stand_pose.copy()
            # Balance even when standing still
            if body_quat is not None:
                roll, pitch, yaw = quat_to_euler(body_quat)
                pr = body_angvel[1] if body_angvel is not None else 0.0
                rr = body_angvel[0] if body_angvel is not None else 0.0
                action[0] = np.clip(1.0 * pitch + 0.15 * pr, -0.5, 0.5)
                action[2] = np.clip(0.5 * roll + 0.1 * rr, -0.3, 0.3)
            return action
        direction = direction / distance
        return self.locomotion_controller.get_action(
            direction, current_velocity, body_quat, body_angvel)

    def reset(self):
        self.locomotion_controller.reset()
