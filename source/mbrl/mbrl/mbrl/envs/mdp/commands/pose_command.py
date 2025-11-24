# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg


class UniformPoseCommand_Visualize(UniformPoseCommand):
    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        
    def _resample_command(self, env_ids: Sequence[int]):
        # intersect the reset envs with only the real envs
        uniques, counts = torch.cat([env_ids, self.env.env_ids_real]).unique(return_counts=True)
        env_ids = uniques[counts > 1]
        super()._resample_command(env_ids)
        self.pose_command_b[env_ids + 1] = self.pose_command_b[env_ids].clone()

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
