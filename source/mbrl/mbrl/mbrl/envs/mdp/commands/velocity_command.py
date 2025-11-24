# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg


class UniformVelocityCommand_Visualize(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        
    def _resample_command(self, env_ids: Sequence[int]):
        # intersect the reset envs with only the real envs
        uniques, counts = torch.cat([env_ids, self.env.env_ids_real]).unique(return_counts=True)
        env_ids = uniques[counts > 1]
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids + 1] = self.vel_command_b[env_ids].clone()
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids + 1] = self.heading_target[env_ids].clone()
            # update heading envs
            self.is_heading_env[env_ids + 1] = self.is_heading_env[env_ids].clone()
        # update standing envs
        self.is_standing_env[env_ids + 1] = self.is_standing_env[env_ids].clone()


    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "real_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Model/real"
                self.real_visualizer = VisualizationMarkers(marker_cfg)
                # -- current body pose
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Model/imagination"
                self.imagination_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.real_visualizer.set_visibility(True)
            self.imagination_visualizer.set_visibility(True)
        else:
            if hasattr(self, "real_visualizer"):
                self.real_visualizer.set_visibility(False)
                self.imagination_visualizer.set_visibility(False)
    

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        # update the markers
        # -- real
        marker_position = self.robot.data.root_pos_w.clone()
        marker_position[:, 2] += 1.0
        self.real_visualizer.visualize(marker_position[self.env.env_ids_real])
        # -- imagination
        self.imagination_visualizer.visualize(marker_position[self.env.env_ids_imagination])


class SampleUniformVelocityCommand(UniformVelocityCommand):
    
    def sample_command(self, num_envs: int):
        # sample velocity commands
        r = torch.empty(num_envs, device=self.device)
        vel_command_b = torch.zeros(num_envs, 3, device=self.device)
        # -- linear velocity - x direction
        vel_command_b[:, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        vel_command_b[:, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        vel_command_b[:, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        return vel_command_b

