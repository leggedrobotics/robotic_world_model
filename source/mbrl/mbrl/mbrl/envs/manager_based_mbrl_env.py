# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import torch
from typing import Any

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class ManagerBasedMBRLEnv(ManagerBasedRLEnv):


    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.reward_term_names = self.reward_manager.active_terms
        self.reward_term_names.append("uncertainty")
        self._init_additional_attributes()
        # assigned in runner
        self.num_imagination_envs = None # type: int
        self.num_imagination_steps = None # type: int
        self.imagination_state_normalizer = None # type: Any
        self.imagination_action_normalizer = None # type: Any
        self.system_dynamics = None # type: Any
        self.uncertainty_penalty_weight = None # type: float
        # termination flags
        self.termination_flags = None # type: torch.Tensor | None


    def prepare_imagination(self):
        self.system_dynamics.reset()
        self.system_dynamics_model_ids = torch.randint(0, self.system_dynamics.ensemble_size, (1, self.num_imagination_envs, 1), device=self.device)
        self._reset_imagination_reward_buffer()
        self._prepare_additional_imagination_attributes()
        self.imagination_step_counter = 0
        self.last_obs = None

    
    def sample_imagination_command(self):
        for name, term in self.command_manager._terms.items():
            setattr(self, name, term.sample_command(self.num_imagination_envs))

    
    def _reset_imagination_reward_buffer(self):
        self.imagination_reward_buffer = {
            term: torch.zeros(
                self.num_imagination_envs,
                self.num_imagination_steps,
                device=self.device
                ) for term in self.reward_term_names
            }
        self.imagination_reward_per_step = {
            term: torch.zeros(
                self.num_imagination_envs,
                device=self.device
                ) for term in self.reward_term_names
            }


    def get_imagination_reward_per_step(self):
        per_step_reward_imagination = {term: torch.mean(value) for term, value in self.imagination_reward_buffer.items()}
        return per_step_reward_imagination


    def imagination_step(self, rollout_action, state_history, action_history):
        action_history = torch.cat([action_history[:, 1:].clone(), self.imagination_action_normalizer(rollout_action).unsqueeze(1)], dim=1)
        imagination_states, aleatoric_uncertainty, self.epistemic_uncertainty, extensions, contacts, terminations = self.system_dynamics.forward(state_history, action_history, self.system_dynamics_model_ids)
        imagination_states_denormalized = self.imagination_state_normalizer.inverse(imagination_states)
        parsed_imagination_states = self._parse_imagination_states(imagination_states_denormalized)
        parsed_extensions = self._parse_extensions(extensions)
        parsed_contacts = self._parse_contacts(contacts)
        self.termination_flags = self._parse_terminations(terminations)
        self._compute_imagination_reward_terms(parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts)
        rewards, dones, extras = self._post_imagination_step()
        state_history = torch.cat([state_history[:, 1:].clone(), imagination_states.unsqueeze(1)], dim=1)
        return self.last_obs, rewards, dones, extras, state_history, action_history, self.epistemic_uncertainty
    
    
    def _post_imagination_step(self):
        for term, value in self.imagination_reward_buffer.items():
            if term == "uncertainty":
                value[:, self.imagination_step_counter] = self.uncertainty_penalty_weight * self.epistemic_uncertainty
            else:
                term_cfg = self.reward_manager.get_term_cfg(term)
                term_value = self.imagination_reward_per_step[term]
                value[:, self.imagination_step_counter] = term_cfg.weight * term_value
        
        rewards = torch.sum(
            torch.stack(
                [
                    value[:, self.imagination_step_counter]
                    for value in self.imagination_reward_buffer.values()
                    ]
                ),
            dim=0) * self.step_dt
        
        if self.imagination_step_counter == self.num_imagination_steps - 1:
            dones = torch.ones(self.num_imagination_envs, dtype=torch.int, device=self.device)
            time_outs = torch.ones(self.num_imagination_envs, dtype=torch.int, device=self.device)
        else:
            dones = self.termination_flags if self.termination_flags is not None else torch.zeros(self.num_imagination_envs, dtype=torch.int, device=self.device)
            time_outs = torch.zeros(self.num_imagination_envs, dtype=torch.int, device=self.device)
        
        infos = {"time_outs": time_outs}
        self.imagination_step_counter += 1
        return rewards, dones, infos


    def _init_additional_attributes(self):
        raise NotImplementedError


    def _prepare_additional_imagination_attributes(self):
        raise NotImplementedError


    def get_imagination_observation(self, state_history, action_history, observation_noise=True):
        raise NotImplementedError
    

    def _parse_imagination_states(self, imagination_states_denormalized):
        raise NotImplementedError

    
    def _parse_extensions(self, extensions):
        raise NotImplementedError


    def _parse_contacts(self, contacts):
        raise NotImplementedError


    def _parse_terminations(self, terminations):
        raise NotImplementedError


    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        raise NotImplementedError
