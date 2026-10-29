"""离线想象环境基类。

定义统一的环境接口、奖励缓存、重置与 rollout 交互流程。
"""

import torch
from tensordict import TensorDict


class BaseEnv:
    def __init__(
        self,
        num_envs: int,
        max_episode_length: int,
        step_dt: float,
        reward_term_weights: dict,
        device: str,
        uncertainty_penalty_weight: float = -0.0,
        observation_noise: bool = False,
        command_resample_interval_range: list | None = [500, 500],
        event_interval_range: list | None = None,
    ):
        self._num_envs = num_envs
        self._max_episode_length = max_episode_length
        self._step_dt = step_dt
        self.device = device

        self.reward_term_weights = reward_term_weights
        self.uncertainty_penalty_weight = uncertainty_penalty_weight
        self.observation_noise = observation_noise
        self.command_resample_interval_range = command_resample_interval_range
        self.event_interval_range = event_interval_range
        self._init_additional_attributes()
        self.dummy_obs = TensorDict(
            {"policy": torch.zeros(num_envs, self.observation_dim)},
            batch_size=[num_envs],
            device=self.device,
        )

        # set in experiment
        self.system_dynamics = None
        self.dataset = None
        self.init_dataset = None
        self.init_data_ratio = None

    def _init_additional_attributes(self):
        pass

    def set_system_dynamics(self, system_dynamics):
        self.system_dynamics = system_dynamics
        self.system_dynamics.eval()

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_init_dataset(self, init_dataset, init_data_ratio=0.0):
        self.init_dataset = init_dataset
        self.init_data_ratio = init_data_ratio

    def prepare_imagination(self):
        self.common_step_counter = 0
        self.system_dynamics.reset()
        self.system_dynamics_model_ids = torch.randint(
            0,
            self.system_dynamics.ensemble_size,
            (1, self._num_envs, 1),
            device=self.device,
        )
        self._init_imagination_reward_buffer()
        self._init_intervals()
        self._init_additional_imagination_attributes()
        state_history, action_history = self._init_imagination_history()
        self._init_imagination_command()
        self.last_obs = TensorDict(
            {"policy": torch.zeros(self._num_envs, self.observation_dim)},
            batch_size=[self._num_envs],
            device=self.device,
        )
        self.extras = {}
        self._reset_idx(torch.arange(self._num_envs, device=self.device))
        return state_history, action_history

    def _reset_idx(self, env_ids):
        self.extras["log"] = dict()
        self.system_dynamics.reset_partial(env_ids)
        self.system_dynamics_model_ids[:, env_ids, :] = torch.randint(
            0,
            self.system_dynamics.ensemble_size,
            (1, len(env_ids), 1),
            device=self.device,
        )
        info = self._reset_imagination_reward_buffer(env_ids)
        self.extras["log"].update(info)
        self._reset_intervals(env_ids)
        self._reset_additional_imagination_attributes(env_ids)
        self.last_obs["policy"][env_ids] = 0.0

    def _init_imagination_history(self):
        num_init_envs = int(self._num_envs * self.init_data_ratio)
        num_random_envs = self._num_envs - num_init_envs
        state_history_init, action_history_init = self.init_dataset.sample_batch(num_init_envs, normalized=True)
        state_history_random, action_history_random = self.dataset.sample_batch(num_random_envs, normalized=True)
        state_history = torch.cat([state_history_init, state_history_random], dim=0)
        action_history = torch.cat([action_history_init, action_history_random], dim=0)
        return state_history, action_history

    def _reset_imagination_history(self, env_ids, state_history, action_history):
        num_reset_envs = len(env_ids)
        num_init_envs = int(num_reset_envs * self.init_data_ratio)
        num_random_envs = num_reset_envs - num_init_envs
        state_history_init, action_history_init = self.init_dataset.sample_batch(num_init_envs, normalized=True)
        state_history_random, action_history_random = self.dataset.sample_batch(num_random_envs, normalized=True)
        state_history[env_ids] = torch.cat([state_history_init, state_history_random], dim=0)[
            :, -state_history.shape[1] :
        ]
        action_history[env_ids] = torch.cat([action_history_init, action_history_random], dim=0)[
            :, -action_history.shape[1] :
        ]
        return state_history, action_history

    def _init_imagination_reward_buffer(self):
        self.episode_length_buf = torch.zeros(self._num_envs, device=self.device, dtype=torch.long)
        self.imagination_episode_sums = {
            term: torch.zeros(self._num_envs, device=self.device) for term in self.reward_term_weights.keys()
        }
        self.imagination_episode_sums["uncertainty"] = torch.zeros(self._num_envs, device=self.device)
        self.imagination_reward_per_step = {
            term: torch.zeros(self._num_envs, device=self.device) for term in self.reward_term_weights.keys()
        }

    def _reset_imagination_reward_buffer(self, env_ids):
        extras = {}
        self.episode_length_buf[env_ids] = 0
        for term in self.imagination_episode_sums.keys():
            episodic_sum_avg = torch.mean(self.imagination_episode_sums[term][env_ids])
            extras[term] = episodic_sum_avg / (self._max_episode_length * self._step_dt)
            self.imagination_episode_sums[term][env_ids] = 0.0
        return extras

    def _init_intervals(self):
        if self.command_resample_interval_range is None:
            return
        else:
            self.command_resample_intervals = torch.randint(
                self.command_resample_interval_range[0],
                self.command_resample_interval_range[1],
                (self._num_envs,),
                device=self.device,
            )
        if self.event_interval_range is None:
            return
        else:
            self.event_intervals = torch.randint(
                self.event_interval_range[0],
                self.event_interval_range[1],
                (self._num_envs,),
                device=self.device,
            )

    def _reset_intervals(self, env_ids):
        if self.command_resample_interval_range is None:
            return
        else:
            self.command_resample_intervals[env_ids] = torch.randint(
                self.command_resample_interval_range[0],
                self.command_resample_interval_range[1],
                (len(env_ids),),
                device=self.device,
            )
        if self.event_interval_range is None:
            return
        else:
            self.event_intervals[env_ids] = torch.randint(
                self.event_interval_range[0],
                self.event_interval_range[1],
                (len(env_ids),),
                device=self.device,
            )

    def imagination_step(self, rollout_action, state_history, action_history):
        _, rollout_action_normalized = self.dataset.normalize(None, rollout_action)
        action_history = torch.cat([action_history[:, 1:], rollout_action_normalized.unsqueeze(1)], dim=1)
        (
            imagination_states,
            aleatoric_uncertainty,
            self.epistemic_uncertainty,
            extensions,
            contacts,
            terminations,
        ) = self.system_dynamics.forward(state_history, action_history, self.system_dynamics_model_ids)
        imagination_states_denormalized, _ = self.dataset.denormalize(imagination_states, None)
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
        event_ids = self._process_event_env_ids()
        imagination_states = self._apply_interval_events(
            imagination_states_denormalized, parsed_imagination_states, event_ids
        )
        state_history = torch.cat([state_history[:, 1:], imagination_states.unsqueeze(1)], dim=1)
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            state_history, action_history = self._reset_imagination_history(
                reset_env_ids, state_history, action_history
            )
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
        self.episode_length_buf += 1
        self.common_step_counter += 1
        rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        for term in self.imagination_episode_sums.keys():
            if term == "uncertainty":
                rewards += self.uncertainty_penalty_weight * self.epistemic_uncertainty * self._step_dt
                self.imagination_episode_sums[term] += (
                    self.uncertainty_penalty_weight * self.epistemic_uncertainty * self._step_dt
                )
            else:
                term_value = self.imagination_reward_per_step[term]
                rewards += self.reward_term_weights[term] * term_value * self._step_dt
                self.imagination_episode_sums[term] += self.reward_term_weights[term] * term_value * self._step_dt

        terminated = (
            self.termination_flags
            if self.termination_flags is not None
            else torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        )
        time_outs = self.episode_length_buf >= self._max_episode_length
        dones = (terminated | time_outs).to(dtype=torch.long)

        reset_env_ids = (terminated | time_outs).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        self.extras["time_outs"] = time_outs
        return rewards, dones, self.extras

    def _process_command_env_ids(self):
        if self.command_resample_interval_range is None:
            return torch.empty(0, dtype=torch.int, device=self.device)
        else:
            return (self.episode_length_buf % self.command_resample_intervals == 0).nonzero(as_tuple=False).squeeze(-1)

    def _process_event_env_ids(self):
        if self.event_interval_range is None:
            return torch.empty(0, dtype=torch.int, device=self.device)
        else:
            return (self.episode_length_buf % self.event_intervals == 0).nonzero(as_tuple=False).squeeze(-1)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def max_episode_length(self):
        return self._max_episode_length

    def _init_additional_imagination_attributes(self):
        raise NotImplementedError

    def _reset_additional_imagination_attributes(self, env_ids):
        raise NotImplementedError

    def _init_imagination_command(self):
        raise NotImplementedError

    def _reset_imagination_command(self, env_ids):
        raise NotImplementedError

    def get_imagination_observation(self, state_history, action_history):
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

    def _apply_interval_events(self, imagination_states_denormalized, parsed_imagination_states, event_ids):
        raise NotImplementedError

    @property
    def state_dim(self):
        raise NotImplementedError

    @property
    def observation_dim(self):
        raise NotImplementedError

    @property
    def action_dim(self):
        raise NotImplementedError
