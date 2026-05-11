"""策略训练循环封装。

封装 rollout 收集、PPO 更新、日志统计与检查点保存。
"""

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

from envs.base import BaseEnv
from rsl_rl.algorithms import PPO

import wandb


class PolicyTraining:
    def __init__(
        self,
        log_dir,
        env: BaseEnv,
        alg: PPO,
        device,
        num_steps_per_env=24,
        save_interval=200,
        max_iterations=1000,
        export_dir=None,
    ):
        self.env = env
        self.alg = alg
        self.device = device
        self.alg.init_storage(
            "rl",
            env.num_envs,
            num_steps_per_env,
            env.dummy_obs,
            [env.action_dim],
        )

        self.num_steps_per_env = num_steps_per_env
        self.save_interval = save_interval
        self.max_iterations = max_iterations

        self.log_dir = log_dir
        self.export_dir = log_dir if export_dir is None else export_dir
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def learn(self):
        state_history, action_history = self.env.prepare_imagination()
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + self.max_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            self.log_dict = {}
            epistemic_uncertainty = torch.zeros(self.num_steps_per_env, device=self.device)
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if (
                        self.env.system_dynamics.architecture_config["type"] in ["rnn", "rssm"]
                        and self.env.common_step_counter > 0
                    ):
                        state_history = state_history[:, -1:]
                        action_history = action_history[:, -1:]
                    imagination_obs = self.env.get_imagination_observation(state_history, action_history)
                    imagination_actions = self.alg.act(imagination_obs)
                    (
                        imagination_obs,
                        imagination_rewards,
                        imagination_dones,
                        imagination_extras,
                        state_history,
                        action_history,
                        uncertainty,
                    ) = self.env.imagination_step(imagination_actions, state_history, action_history)
                    self.alg.process_env_step(
                        imagination_obs,
                        imagination_rewards,
                        imagination_dones,
                        imagination_extras,
                    )
                    epistemic_uncertainty[i] = uncertainty.mean(dim=0)

                    if self.log_dir is not None:
                        if "episode" in imagination_extras:
                            ep_infos.append(imagination_extras["episode"])
                        elif "log" in imagination_extras:
                            ep_infos.append(imagination_extras["log"])
                        cur_reward_sum += imagination_rewards
                        cur_episode_length += 1
                        new_ids = (imagination_dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # logs
                num_valid_imagination_envs = self.alg.storage.valid_env_mask.sum()
                epistemic_uncertainty = epistemic_uncertainty.mean(dim=0)

                self.log_dict.update(
                    {
                        "Imagination/epistemic_uncertainty": epistemic_uncertainty,
                        "Imagination/num_valid_imagination_envs": num_valid_imagination_envs,
                    }
                )

                imagination_critic_obs = imagination_obs
                self.alg.compute_returns(imagination_critic_obs)

            # Update policy
            # Note: we keep arguments here since locals() loads them
            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"policy_{it}.pt"))

            ep_infos.clear()
        self.save(os.path.join(self.log_dir, f"policy_{it}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                self.log_dict.update({f"Imagination/{key}": value})
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.log_dict.update({f"Loss/{key}": value})
        self.log_dict.update(
            {
                "Loss/policy_learning_rate": self.alg.learning_rate,
                # -- Policy
                "Policy/mean_noise_std": mean_std.item(),
                # -- Performance
                "Perf/total_fps": fps,
                "Perf/collection time": locs["collection_time"],
                "Perf/learning_time": locs["learn_time"],
            }
        )
        if self.alg.rnd:
            self.log_dict.update({"Loss/rnd": locs["mean_rnd_loss"]})
        if self.alg.symmetry:
            self.log_dict.update({"Loss/symmetry": locs["mean_symmetry_loss"]})

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            self.log_dict.update(
                {
                    "Train/mean_reward": statistics.mean(locs["rewbuffer"]),
                    "Train/mean_episode_length": statistics.mean(locs["lenbuffer"]),
                    "Loss/policy_learning_rate": self.alg.learning_rate,
                    "Policy/mean_noise_std": mean_std.item(),
                    "Perf/total_fps": fps,
                    "Perf/collection time": locs["collection_time"],
                    "Perf/learning_time": locs["learn_time"],
                }
            )

        wandb.log(self.log_dict)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + self.max_iterations - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

    def train_mode(self):
        self.alg.policy.train()

    def eval_mode(self):
        self.alg.policy.eval()
