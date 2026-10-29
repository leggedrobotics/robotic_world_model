"""离线/模型驱动训练入口。

负责数据加载、动力学模型构建、策略训练与日志落盘。
"""

import argparse
import os
import time
import torch
from datetime import datetime
from torch.utils.data import Dataset

import pandas as pd
from configs import BaseConfig, Go2FlatConfig
from envs import BaseEnv, Go2FlatEnv
from policy_training import PolicyTraining
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, SystemDynamicsEnsemble
from rsl_rl.utils import resolve_obs_groups

import wandb


class ModelBasedExperiment:
    def __init__(self, environment, device):
        self.env_cls = self.resolve_environment_cls(environment)
        self.device = device
        self.data_file_idx = 0

    def resolve_environment_cls(self, environment):
        if environment == "go2_flat":
            return Go2FlatEnv
        else:
            raise ValueError(f"Unknown environment: {environment}")

    def _load_data(self, dataset_root, dataset_folder, file_data_size=10000, batch_data_size=50000):
        batch_state_data = []
        batch_action_data = []
        batch_extension_data = []
        batch_contact_data = []
        batch_termination_data = []
        batch_total_num_data = 0
        while True:
            try:
                file = f"state_action_data_{self.data_file_idx}.csv"
                data = pd.read_csv(os.path.join(dataset_root, dataset_folder, file), header=None)
            except FileNotFoundError:
                print(
                    f"[Motion Loader] No data found in {os.path.join(dataset_root, dataset_folder, file)}. Waiting for"
                    " new data."
                )
                time.sleep(1)
                continue
            if len(data) < file_data_size:
                print(
                    f"[Motion Loader] Not enough data in {os.path.join(dataset_root, dataset_folder, file)}. Waiting"
                    " for new data."
                )
                time.sleep(1)
                continue
            state_data = torch.tensor(
                data.iloc[:, : self.state_dim].values,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action_data = torch.tensor(
                data.iloc[:, self.state_dim : self.state_dim + self.action_dim].values,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            extension_data = torch.tensor(
                data.iloc[
                    :,
                    self.state_dim + self.action_dim : self.state_dim + self.action_dim + self.extension_dim,
                ].values,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            contact_data = torch.tensor(
                data.iloc[
                    :,
                    self.state_dim
                    + self.action_dim
                    + self.extension_dim : self.state_dim
                    + self.action_dim
                    + self.extension_dim
                    + self.contact_dim,
                ].values,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            termination_data = torch.tensor(
                data.iloc[
                    :,
                    self.state_dim + self.action_dim + self.extension_dim + self.contact_dim :,
                ].values,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            batch_state_data.append(state_data)
            batch_action_data.append(action_data)
            batch_extension_data.append(extension_data)
            batch_contact_data.append(contact_data)
            batch_termination_data.append(termination_data)
            batch_total_num_data += len(data)
            num_trajs, num_steps, state_dim = state_data.shape
            print(f"[Motion Loader] Loaded {num_trajs} {dataset_folder} trajectories of {num_steps} steps from {file}.")
            print(f"[Motion Loader] Total number of data: {batch_total_num_data} / {batch_data_size}")
            self.data_file_idx += 1
            if batch_total_num_data >= batch_data_size:
                break
        batch_state_data = torch.cat(batch_state_data, dim=1)
        batch_action_data = torch.cat(batch_action_data, dim=1)
        batch_extension_data = torch.cat(batch_extension_data, dim=1)
        batch_contact_data = torch.cat(batch_contact_data, dim=1)
        batch_termination_data = torch.cat(batch_termination_data, dim=1)
        action_dim, extension_dim, contact_dim, termination_dim = (
            action_data.shape[-1],
            extension_data.shape[-1],
            contact_data.shape[-1],
            termination_data.shape[-1],
        )
        print(
            f"[Motion Loader] State dim: {state_dim} | Action dim: {action_dim} | Extension dim: {extension_dim} |"
            f" Contact dim: {contact_dim} | Termination dim: {termination_dim}"
        )
        return (
            batch_state_data,
            batch_action_data,
            batch_extension_data,
            batch_contact_data,
            batch_termination_data,
        )

    def _build_eval_traj_config(
        self,
        eval_state_data,
        eval_action_data,
        eval_extension_data,
        eval_contact_data,
        eval_termination_data,
        num_eval_trajectories,
        num_visualizations,
        len_eval_trajectory,
        state_idx_dict,
    ):
        num_eval_trajs, num_eval_steps, _ = eval_state_data.shape
        ids = torch.randint(0, num_eval_trajs, (num_eval_trajectories,))
        len_eval_trajectory = min(len_eval_trajectory, num_eval_steps)
        start_steps = torch.randint(0, num_eval_steps - len_eval_trajectory + 1, (num_eval_trajectories,))
        start_steps_expanded = start_steps[:, None] + torch.arange(len_eval_trajectory)
        traj_data = [
            eval_state_data[ids[:, None], start_steps_expanded],
            eval_action_data[ids[:, None], start_steps_expanded],
            eval_extension_data[ids[:, None], start_steps_expanded],
            eval_contact_data[ids[:, None], start_steps_expanded],
            eval_termination_data[ids[:, None], start_steps_expanded],
        ]
        self.eval_traj_config = {
            "num_trajs": num_eval_trajectories,
            "num_visualizations": num_visualizations,
            "len_traj": len_eval_trajectory,
            "traj_data": traj_data,
            "state_idx_dict": state_idx_dict,
        }

    def prepare_environment(
        self,
        num_envs,
        max_episode_length,
        step_dt,
        reward_term_weights,
        uncertainty_penalty_weight,
        observation_noise,
        command_resample_interval_range,
        event_interval_range,
    ):
        self.env: BaseEnv = self.env_cls(
            num_envs,
            max_episode_length,
            step_dt,
            reward_term_weights,
            self.device,
            uncertainty_penalty_weight,
            observation_noise,
            command_resample_interval_range,
            event_interval_range,
        )

    def prepare_data(
        self,
        dataset_root,
        dataset_folder,
        file_data_size,
        batch_data_size,
        state_data_mean=None,
        state_data_std=None,
        action_data_mean=None,
        action_data_std=None,
        init_data_ratio=0.0,
        num_eval_trajectories=100,
        num_visualizations=2,
        len_eval_trajectory=400,
        state_idx_dict=None,
    ):
        state_data, action_data, extension_data, contact_data, termination_data = self._load_data(
            dataset_root,
            dataset_folder=dataset_folder,
            file_data_size=file_data_size,
            batch_data_size=batch_data_size,
        )
        self._build_eval_traj_config(
            state_data,
            action_data,
            extension_data,
            contact_data,
            termination_data,
            num_eval_trajectories,
            num_visualizations,
            len_eval_trajectory,
            state_idx_dict,
        )

        class SystemDynamicsDataset(Dataset):
            def __init__(
                self,
                history_horizon,
                forecast_horizon,
                state_data,
                action_data,
                extension_data,
                contact_data,
                termination_data,
                state_data_mean=None,
                state_data_std=None,
                action_data_mean=None,
                action_data_std=None,
            ):
                self.history_horizon = history_horizon
                self.forecast_horizon = forecast_horizon

                self.state_data_mean = (
                    torch.tensor(state_data_mean, device=state_data.device)
                    if state_data_mean is not None
                    else state_data.mean(dim=(0, 1))
                )
                self.state_data_std = (
                    torch.tensor(state_data_std, device=state_data.device)
                    if state_data_std is not None
                    else state_data.std(dim=(0, 1)) + 1e-6
                )
                self.action_data_mean = (
                    torch.tensor(action_data_mean, device=action_data.device)
                    if action_data_mean is not None
                    else action_data.mean(dim=(0, 1))
                )
                self.action_data_std = (
                    torch.tensor(action_data_std, device=action_data.device)
                    if action_data_std is not None
                    else action_data.std(dim=(0, 1)) + 1e-6
                )
                state_data, action_data = self.normalize(state_data, action_data)

                reset_indices = termination_data.flatten().nonzero(as_tuple=False).squeeze(-1)
                if wandb.run is not None:
                    wandb.log(
                        {
                            "Data/num_terminations": len(reset_indices),
                        }
                    )
                valid_indices = []
                for i in range(state_data.shape[1] - history_horizon - forecast_horizon + 1):
                    if not any(
                        reset_indices[
                            (reset_indices >= i) & (reset_indices < i + history_horizon + forecast_horizon - 1)
                        ]
                    ):
                        valid_indices.append(i)
                # (num_groups, history_horizon + forecast_horizon, dim)
                self.state_data = torch.cat(
                    [state_data[:, i : i + history_horizon + forecast_horizon, :] for i in valid_indices],
                    dim=0,
                )
                self.action_data = torch.cat(
                    [action_data[:, i : i + history_horizon + forecast_horizon, :] for i in valid_indices],
                    dim=0,
                )
                self.extension_data = torch.cat(
                    [extension_data[:, i : i + history_horizon + forecast_horizon, :] for i in valid_indices],
                    dim=0,
                )
                self.contact_data = torch.cat(
                    [contact_data[:, i : i + history_horizon + forecast_horizon, :] for i in valid_indices],
                    dim=0,
                )
                self.termination_data = torch.cat(
                    [termination_data[:, i : i + history_horizon + forecast_horizon, :] for i in valid_indices],
                    dim=0,
                )

            def __len__(self):
                return len(self.state_data)

            def __getitem__(self, idx):
                return (
                    self.state_data[idx],
                    self.action_data[idx],
                    self.extension_data[idx],
                    self.contact_data[idx],
                    self.termination_data[idx],
                )

            def normalize(self, state_data=None, action_data=None):
                state_data = (
                    (state_data - self.state_data_mean) / self.state_data_std if state_data is not None else None
                )
                action_data = (
                    (action_data - self.action_data_mean) / self.action_data_std if action_data is not None else None
                )
                return state_data, action_data

            def denormalize(self, state_data, action_data):
                state_data = state_data * self.state_data_std + self.state_data_mean if state_data is not None else None
                action_data = (
                    action_data * self.action_data_std + self.action_data_mean if action_data is not None else None
                )
                return state_data, action_data

            def sample_batch(self, batch_size, normalized=True):
                idx = torch.randint(0, len(self.state_data), (batch_size,))
                if normalized:
                    return (
                        self.state_data[idx, : self.history_horizon],
                        self.action_data[idx, : self.history_horizon],
                    )
                else:
                    return self.denormalize(
                        self.state_data[idx, : self.history_horizon],
                        self.action_data[idx, : self.history_horizon],
                    )

        self.dataset = SystemDynamicsDataset(
            self.history_horizon,
            self.forecast_horizon,
            state_data,
            action_data,
            extension_data,
            contact_data,
            termination_data,
            state_data_mean=state_data_mean,
            state_data_std=state_data_std,
            action_data_mean=action_data_mean,
            action_data_std=action_data_std,
        )
        # init env dataset
        self.env.set_dataset(self.dataset)
        if self.env.init_dataset is None:
            self.env.set_init_dataset(self.dataset, init_data_ratio)

    def prepare_model(
        self,
        history_horizon,
        forecast_horizon,
        extension_dim,
        contact_dim,
        termination_dim,
        ensemble_size,
        architecture_config,
        freeze_auxiliary=False,
        resume_path=None,
    ):
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.system_dynamics = SystemDynamicsEnsemble(
            self.state_dim,
            self.action_dim,
            self.extension_dim,
            self.contact_dim,
            self.termination_dim,
            self.device,
            ensemble_size=ensemble_size,
            history_horizon=self.history_horizon,
            architecture_config=architecture_config,
            freeze_auxiliary=freeze_auxiliary,
        )
        self.model_learning_iteration = 0
        if resume_path is not None:
            print(f"[Prepare Model] Loading model from {resume_path}.")
            loaded_dict = torch.load(resume_path)
            self.system_dynamics.load_state_dict(loaded_dict["system_dynamics_state_dict"])
            self.model_learning_iteration = loaded_dict["iter"]
        # init env system dynamics
        self.env.set_system_dynamics(self.system_dynamics)

    def prepare_policy(
        self,
        observation_dim,
        obs_groups,
        action_dim,
        actor_hidden_dims,
        critic_hidden_dims,
        activation,
        init_noise_std,
        resume_path=None,
    ):
        self.observation_dim = observation_dim
        default_sets = ["critic"]
        obs_groups = resolve_obs_groups(self.env.dummy_obs, obs_groups, default_sets)
        self.actor_critic = ActorCritic(
            obs=self.env.dummy_obs,
            obs_groups=obs_groups,
            num_actions=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        ).to(self.device)
        self.policy_learning_iteration = 0
        if resume_path is not None:
            print(f"[Prepare Policy] Loading policy from {resume_path}.")
            loaded_dict = torch.load(resume_path)
            self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
            self.policy_learning_iteration = loaded_dict["iter"]

    def prepare_algorithm(
        self,
        num_learning_epochs,
        num_mini_batches,
        clip_param,
        gamma,
        lam,
        value_loss_coef,
        entropy_coef,
        learning_rate,
        max_grad_norm,
        use_clipped_value_loss,
        schedule,
        desired_kl,
    ):
        self.alg = PPO(
            self.actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device=self.device,
        )

    def train_policy(self, log_dir, num_steps_per_env, save_interval, max_iterations, export_dir):
        print(f"[Train Policy] Training policy for {max_iterations} iterations.")
        policy_training = PolicyTraining(
            log_dir,
            env=self.env,
            alg=self.alg,
            device=self.device,
            num_steps_per_env=num_steps_per_env,
            save_interval=save_interval,
            max_iterations=max_iterations,
            export_dir=export_dir,
        )
        policy_training.current_learning_iteration = self.policy_learning_iteration
        policy_training.learn()
        self.policy_learning_iteration += policy_training.max_iterations


def run_experiment(config: BaseConfig):
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.run_num}"
    log_dir = os.path.join("logs", config.experiment_name, args_cli.task, run_name)
    wandb.run.name = run_name
    os.makedirs(log_dir, exist_ok=True)
    model_experiment = ModelBasedExperiment(**config.experiment_config.to_dict())
    model_experiment.prepare_environment(**config.environment_config.to_dict())
    model_experiment.prepare_model(**config.model_architecture_config.to_dict())
    model_experiment.prepare_policy(**config.policy_architecture_config.to_dict())
    model_experiment.prepare_algorithm(**config.policy_algorithm_config.to_dict())
    model_experiment.prepare_data(**config.data_config.to_dict())
    model_experiment.train_policy(log_dir, **config.policy_training_config.to_dict())
    print(f"Training completed. Policy saved to {log_dir}.")


def run(config: BaseConfig):
    wandb.init(project=config.experiment_name)
    wandb.config.update(config.to_dict())
    run_experiment(config)


def resolve_task_config(task: str):
    if task == "go2_flat":
        config = Go2FlatConfig()
        return config
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online learning training.")
    parser.add_argument(
        "--task",
        type=str,
        default="go2_flat",
        help="Task to use for the experiment.",
    )
    parser.add_argument(
        "--run_num",
        type=int,
        default=None,
        help="Run number for the experiment on the cluster.",
    )
    args_cli = parser.parse_args()
    config = resolve_task_config(args_cli.task)
    run(config)
