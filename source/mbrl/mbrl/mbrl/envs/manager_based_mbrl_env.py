# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""管理式 MBRL 环境基类。

该类在 Isaac Lab 的 RL 环境上叠加想象 rollout 与动力学模型交互逻辑。
"""

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class ManagerBasedMBRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.reward_term_names = self.reward_manager.active_terms
        self._init_additional_attributes()
        # assigned in runner
        self.num_imagination_envs = None  # type: int
        self.num_imagination_steps = None  # type: int
        self.max_imagination_episode_length = None  # type: int
        self.imagination_command_resample_interval_range = None  # type: list[float] | None
        self.imagination_state_normalizer = None
        self.imagination_action_normalizer = None
        self.system_dynamics = None
        self.uncertainty_penalty_weight = None  # type: float
        # termination flags
        self.termination_flags = None  # type: torch.Tensor | None

    def prepare_imagination(self):
        self.imagination_common_step_counter = 0
        self.system_dynamics_model_ids = torch.randint(
            0,
            self.system_dynamics.ensemble_size,
            (1, self.num_imagination_envs, 1),
            device=self.device,
        )
        self._init_imagination_reward_buffer()
        self._init_intervals()
        self._init_additional_imagination_attributes()
        self._init_imagination_command()
        self.last_obs = TensorDict(
            {
                "policy": torch.zeros(
                    self.num_imagination_envs,
                    self.observation_manager.group_obs_dim["policy"][0],
                )
            },
            batch_size=[self.num_imagination_envs],
            device=self.device,
        )
        self.imagination_extras = {}
        self._reset_imagination_idx(torch.arange(self.num_imagination_envs, device=self.device))

    def _reset_imagination_idx(self, env_ids):
        self.imagination_extras["log"] = dict()
        self.system_dynamics.reset_partial(env_ids)
        self.system_dynamics_model_ids[:, env_ids, :] = torch.randint(
            0,
            self.system_dynamics.ensemble_size,
            (1, len(env_ids), 1),
            device=self.device,
        )
        info = self._reset_imagination_reward_buffer(env_ids)
        self.imagination_extras["log"].update(info)
        self._reset_intervals(env_ids)
        self._reset_additional_imagination_attributes(env_ids)
        self.last_obs["policy"][env_ids] = 0.0

    def _init_imagination_command(self):
        for name, term in self.command_manager._terms.items():
            setattr(self, name, term.sample_command(self.num_imagination_envs))

    def _reset_imagination_command(self, env_ids):
        for name, term in self.command_manager._terms.items():
            getattr(self, name)[env_ids] = term.sample_command(len(env_ids))

    def _init_imagination_reward_buffer(self):
        self.imagination_episode_length_buf = torch.zeros(
            self.num_imagination_envs, device=self.device, dtype=torch.long
        )
        self.imagination_episode_sums = {
            term: torch.zeros(self.num_imagination_envs, device=self.device) for term in self.reward_term_names
        }
        self.imagination_episode_sums["uncertainty"] = torch.zeros(self.num_imagination_envs, device=self.device)
        self.imagination_reward_per_step = {
            term: torch.zeros(self.num_imagination_envs, device=self.device) for term in self.reward_term_names
        }

    def _reset_imagination_reward_buffer(self, env_ids):
        extras = {}
        self.imagination_episode_length_buf[env_ids] = 0
        for term in self.imagination_episode_sums.keys():
            episodic_sum_avg = torch.mean(self.imagination_episode_sums[term][env_ids])
            extras[term] = episodic_sum_avg / (self.max_imagination_episode_length * self.step_dt)
            self.imagination_episode_sums[term][env_ids] = 0.0
        return extras

    def _init_intervals(self):
        if self.imagination_command_resample_interval_range is None:
            return
        else:
            self.imagination_command_resample_intervals = torch.randint(
                self.imagination_command_resample_interval_range[0],
                self.imagination_command_resample_interval_range[1],
                (self.num_imagination_envs,),
                device=self.device,
            )

    def _reset_intervals(self, env_ids):
        if self.imagination_command_resample_interval_range is None:
            return
        else:
            self.imagination_command_resample_intervals[env_ids] = torch.randint(
                self.imagination_command_resample_interval_range[0],
                self.imagination_command_resample_interval_range[1],
                (len(env_ids),),
                device=self.device,
            )

    def imagination_step(self, rollout_action, state_history, action_history):
        rollout_action_normalized = self.imagination_action_normalizer(rollout_action)
        action_history = torch.cat([action_history[:, 1:], rollout_action_normalized.unsqueeze(1)], dim=1)
        (
            imagination_states,
            aleatoric_uncertainty,
            self.epistemic_uncertainty,
            extensions,
            contacts,
            terminations,
        ) = self.system_dynamics.forward(state_history, action_history, self.system_dynamics_model_ids)
        imagination_states_denormalized = self.imagination_state_normalizer.inverse(imagination_states)
        parsed_imagination_states = self._parse_imagination_states(imagination_states_denormalized)
        parsed_extensions = self._parse_extensions(extensions)
        parsed_contacts = self._parse_contacts(contacts)
        self.termination_flags = self._parse_terminations(terminations)
        self._compute_imagination_reward_terms(
            parsed_imagination_states,
            rollout_action,
            parsed_extensions,
            parsed_contacts,
        )
        rewards, dones, extras = self._post_imagination_step()
        command_ids = self._process_command_env_ids()
        self._reset_imagination_command(command_ids)
        state_history = torch.cat([state_history[:, 1:], imagination_states.unsqueeze(1)], dim=1)
        return (
            self.last_obs,
            rewards,
            dones,
            extras,
            state_history,
            action_history,
            self.epistemic_uncertainty,
        )

    def _post_imagination_step(self):
        self.imagination_episode_length_buf += 1
        self.imagination_common_step_counter += 1
        rewards = torch.zeros(self.num_imagination_envs, dtype=torch.float, device=self.device)
        for term in self.imagination_episode_sums.keys():
            if term == "uncertainty":
                rewards += self.uncertainty_penalty_weight * self.epistemic_uncertainty * self.step_dt
                self.imagination_episode_sums[term] += (
                    self.uncertainty_penalty_weight * self.epistemic_uncertainty * self.step_dt
                )
            else:
                term_cfg = self.reward_manager.get_term_cfg(term)
                term_value = self.imagination_reward_per_step[term]
                rewards += term_cfg.weight * term_value * self.step_dt
                self.imagination_episode_sums[term] += term_cfg.weight * term_value * self.step_dt

        terminated = (
            self.termination_flags
            if self.termination_flags is not None
            else torch.zeros(self.num_imagination_envs, dtype=torch.bool, device=self.device)
        )
        time_outs = self.imagination_episode_length_buf >= self.max_imagination_episode_length
        dones = (terminated | time_outs).to(dtype=torch.long)

        reset_env_ids = (terminated | time_outs).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_imagination_idx(reset_env_ids)
        self.imagination_extras["time_outs"] = time_outs
        return rewards, dones, self.imagination_extras

    def _process_command_env_ids(self):
        if self.imagination_command_resample_interval_range is None:
            return torch.empty(0, dtype=torch.int, device=self.device)
        else:
            return (
                (self.imagination_episode_length_buf % self.imagination_command_resample_intervals == 0)
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )

    def _init_additional_attributes(self):
        raise NotImplementedError

    def _init_additional_imagination_attributes(self):
        raise NotImplementedError

    def _reset_additional_imagination_attributes(self, env_ids):
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

    def _compute_imagination_reward_terms(
        self,
        parsed_imagination_states,
        rollout_action,
        parsed_extensions,
        parsed_contacts,
    ):
        raise NotImplementedError
