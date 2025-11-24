# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import torch
from tensordict import TensorDict

from mbrl.mbrl.envs import ManagerBasedMBRLEnv


class ANYmalDManagerBasedMBRLEnv(ManagerBasedMBRLEnv):


    def _init_additional_attributes(self):
        self.default_joint_pos = self.scene["robot"].data.default_joint_pos[0]
        self.default_joint_vel = self.scene["robot"].data.default_joint_vel[0]
        self.base_velocity = None


    def _prepare_additional_imagination_attributes(self):
        self.last_air_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self.current_air_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self.last_contact_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self.current_contact_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)

    
    def get_imagination_observation(self, state_history, action_history, observation_noise=True):
        obs_base_lin_vel = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 0:3]
        obs_base_ang_vel = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 3:6]
        obs_projected_gravity = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 6:9]
        obs_joint_pos = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 9:21]
        obs_joint_vel = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 21:33]
        self.obs_last_action = self.imagination_action_normalizer.inverse(action_history[:, -1])
        if observation_noise:
            obs_base_lin_vel += 2 * (torch.rand_like(obs_base_lin_vel) - 0.5) * 0.1
            obs_base_ang_vel += 2 * (torch.rand_like(obs_base_ang_vel) - 0.5) * 0.2
            obs_projected_gravity += 2 * (torch.rand_like(obs_projected_gravity) - 0.5) * 0.05
            obs_joint_pos += 2 * (torch.rand_like(obs_joint_pos) - 0.5) * 0.01
            obs_joint_vel += 2 * (torch.rand_like(obs_joint_vel) - 0.5) * 1.5
        obs = torch.cat([obs_base_lin_vel, obs_base_ang_vel, obs_projected_gravity, self.base_velocity, obs_joint_pos, obs_joint_vel, self.obs_last_action], dim=1)
        obs = TensorDict({"policy": obs}, batch_size=[self.num_imagination_envs], device=self.device)
        self.last_obs = obs
        return obs
    

    def _parse_imagination_states(self, imagination_states_denormalized):
        base_lin_vel = imagination_states_denormalized[:, 0:3]
        base_ang_vel = imagination_states_denormalized[:, 3:6]
        projected_gravity = imagination_states_denormalized[:, 6:9]
        joint_pos = imagination_states_denormalized[:, 9:21]
        joint_vel = imagination_states_denormalized[:, 21:33]
        joint_torque = imagination_states_denormalized[:, 33:45]

        parsed_imagination_states = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
        }
        return parsed_imagination_states

    
    def _parse_extensions(self, extensions):
        if extensions is None:
            return None
        parsed_extensions = {
        }
        return parsed_extensions


    def _parse_contacts(self, contacts):
        thigh_contact = torch.sigmoid(contacts[:, 0:4]).round() if contacts is not None else None
        foot_contact = torch.sigmoid(contacts[:, 4:8]).round() if contacts is not None else None
        
        parsed_contacts = {
            "thigh_contact": thigh_contact,
            "foot_contact": foot_contact,
        }
        return parsed_contacts
    
    
    def _parse_terminations(self, terminations):
        parsed_terminations = torch.sigmoid(terminations).squeeze(-1).round().int() if terminations is not None else None
        return parsed_terminations


    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        
        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        projected_gravity = parsed_imagination_states["projected_gravity"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_torque = parsed_imagination_states["joint_torque"]
        joint_acc = (joint_vel - self.last_obs["policy"][:, 12:24]) / self.step_dt
        thigh_contact = parsed_contacts["thigh_contact"]
        foot_contact = parsed_contacts["foot_contact"]

        lin_vel_error = torch.sum(torch.square(self.base_velocity[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.base_velocity[:, 2] - base_ang_vel[:, 2])
        
        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        lin_vel_z_l2 = torch.square(base_lin_vel[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        dof_torques_l2 = torch.sum(torch.square(joint_torque), dim=1)
        dof_acc_l2 = torch.sum(torch.square(joint_acc), dim=1)
        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)
        if foot_contact is not None:
            first_contact = (self.current_contact_time > 0.0) * (self.current_contact_time < (self.step_dt + 1.0e-8))
            feet_air_time = torch.sum((self.last_air_time - 0.5) * first_contact, dim=1) * (torch.norm(self.base_velocity[:, :2], dim=1) > 0.1)
            
            is_contact = foot_contact.bool()
            is_first_contact = (self.current_air_time > 0) * is_contact
            is_first_detached = (self.current_contact_time > 0) * ~is_contact
            self.last_air_time = torch.where(
                is_first_contact,
                self.current_air_time + self.step_dt,
                self.last_air_time,
            )
            self.current_air_time = torch.where(
                ~is_contact, self.current_air_time + self.step_dt, 0.0
            )
            self.last_contact_time = torch.where(
                is_first_detached,
                self.current_contact_time + self.step_dt,
                self.last_contact_time,
            )
            self.current_contact_time = torch.where(
                is_contact, self.current_contact_time + self.step_dt, 0.0
            )
        else:
            feet_air_time = torch.zeros(self.num_imagination_envs, device=self.device)
        undesired_contacts = torch.sum(thigh_contact, dim=1) if thigh_contact is not None else torch.zeros(self.num_imagination_envs, device=self.device)
        stand_still = torch.sum(torch.abs(joint_pos), dim=1) * (torch.norm(self.base_velocity, dim=1) < 0.05)
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        dof_pos_limits = torch.zeros(self.num_imagination_envs, device=self.device)
        self.imagination_reward_per_step = {
            "track_lin_vel_xy_exp": track_lin_vel_xy_exp,
            "track_ang_vel_z_exp": track_ang_vel_z_exp,
            "lin_vel_z_l2": lin_vel_z_l2,
            "ang_vel_xy_l2": ang_vel_xy_l2,
            "dof_torques_l2": dof_torques_l2,
            "dof_acc_l2": dof_acc_l2,
            "action_rate_l2": action_rate_l2,
            "feet_air_time": feet_air_time,
            "undesired_contacts": undesired_contacts,
            "stand_still": stand_still,
            "flat_orientation_l2": flat_orientation_l2,
            "dof_pos_limits": dof_pos_limits,
        }
        last_obs = torch.cat([base_lin_vel, base_ang_vel, projected_gravity, self.base_velocity, joint_pos, joint_vel, rollout_action], dim=1)
        self.last_obs = TensorDict({"policy": last_obs}, batch_size=[self.num_imagination_envs], device=self.device)
