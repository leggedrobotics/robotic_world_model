# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import torch

from mbrl.mbrl.envs import ManagerBasedVisualizeEnv
from .anymal_d_manager_based_mbrl_env import ANYmalDManagerBasedMBRLEnv
from mbrl.mbrl.envs.mdp.events import reset_joints_to_specified, reset_root_velocity_to_specified
from isaaclab.utils.math import quat_apply


class ANYmalDManagerBasedVisualizeEnv(ManagerBasedVisualizeEnv, ANYmalDManagerBasedMBRLEnv):
    
    
    def _reset_imagination_sim(self, parsed_imagination_states):
        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_pos += self.default_joint_pos
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_vel += self.default_joint_vel
        
        root_quat_w = self.scene["robot"].data.root_quat_w[self.env_ids_imagination]
        
        base_lin_vel_w = quat_apply(root_quat_w, base_lin_vel)
        base_ang_vel_w = quat_apply(root_quat_w, base_ang_vel)
                
        velocities = torch.cat([base_lin_vel_w, base_ang_vel_w], dim=1)
        
        reset_joints_to_specified(self, self.env_ids_imagination, joint_pos, joint_vel)
        reset_root_velocity_to_specified(self, self.env_ids_imagination, velocities)
