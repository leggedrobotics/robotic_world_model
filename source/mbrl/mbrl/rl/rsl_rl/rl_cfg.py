# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlSystemDynamicsCfg:
    """Configuration for the system dynamics networks."""
    
    ensemble_size: int = MISSING
    """The ensemble size of the system dynamics network."""

    history_horizon: int = MISSING
    """The prediction horizon of the system dynamics network."""

    architecture_config: dict = MISSING
    """The architecture configuration of the system dynamics network."""
    
    freeze_auxiliary: bool = MISSING
    """Whether to freeze the auxiliary networks."""


@configclass
class RslRlNormalizerCfg:
    """Configuration for the normalizer."""

    mean: list[float] = MISSING
    """The mean of the normalizer."""

    std: list[float] = MISSING
    """The std of the normalizer."""


@configclass
class RslRlMbrlImaginationCfg:
    """Configuration for the imagination."""
    
    num_envs: int = MISSING
    """The number of environments for the imagination."""
    
    num_steps: int = MISSING
    """The number of steps for the imagination."""
    
    command_resample_interval: int = MISSING
    """The resample interval for the command."""
    
    uncertainty_penalty_weight: float = MISSING
    """The weight for the uncertainty penalty."""
    
    state_normalizer: RslRlNormalizerCfg = MISSING
    """The normalizer for the state."""
    
    action_normalizer: RslRlNormalizerCfg = MISSING
    """The normalizer for the action."""


@configclass
class RslRlMbrlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "MBPOPPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    policy_learning_rate: float = MISSING
    """The learning rate for the policy."""

    system_dynamics_learning_rate: float = MISSING
    """The learning rate for the system dynamics."""
    
    system_dynamics_weight_decay: float = MISSING
    """The weight decay for the system dynamics."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    
    system_dynamics_forecast_horizon: int = MISSING
    """The forecast horizon for the system dynamics."""
    
    system_dynamics_loss_weights: dict[str, float] = MISSING
    """The loss weights for the system dynamics."""
    
    system_dynamics_num_mini_batches: int = MISSING
    """The number of mini-batches for the system dynamics."""
    
    system_dynamics_mini_batch_size: int = MISSING
    """The mini-batch size for the system dynamics."""
    
    system_dynamics_replay_buffer_size: int = MISSING
    """The replay buffer size for the system dynamics."""
    
    system_dynamics_num_eval_trajectories: int = MISSING
    """The number of evaluation trajectories for the system dynamics."""
    
    system_dynamics_len_eval_trajectory: int = MISSING
    """The length of the evaluation trajectory for the system dynamics."""
    
    system_dynamics_eval_traj_noise_scale: list[float] = MISSING
    """The noise scale for the evaluation trajectory for the system dynamics."""
