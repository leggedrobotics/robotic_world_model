"""Regression tests for the joint-acceleration reward used by RWM rollouts.

The environment modules normally run inside Isaac Lab, which is not available in
the lightweight test environment.  The tests load the actual classes with small
import stubs and exercise the reward computation on real PyTorch tensors.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_environment_class(relative_path: str, class_name: str, module_name: str):
    """Load an environment module while replacing Isaac-only imports."""

    tensor_dict_module = types.ModuleType("tensordict")
    tensor_dict_module.TensorDict = lambda data, **kwargs: data

    offline_base_module = types.ModuleType("envs.base")
    offline_base_module.BaseEnv = type("BaseEnv", (), {})
    offline_package = types.ModuleType("envs")
    offline_package.__path__ = []

    online_env_module = types.ModuleType("mbrl.mbrl.envs")
    online_env_module.ManagerBasedMBRLEnv = type("ManagerBasedMBRLEnv", (), {})
    online_mbrl_module = types.ModuleType("mbrl.mbrl")
    online_mbrl_module.__path__ = []
    online_package = types.ModuleType("mbrl")
    online_package.__path__ = []

    stubs = {
        "tensordict": tensor_dict_module,
        "envs": offline_package,
        "envs.base": offline_base_module,
        "mbrl": online_package,
        "mbrl.mbrl": online_mbrl_module,
        "mbrl.mbrl.envs": online_env_module,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stubs}
    sys.modules.update(stubs)
    try:
        path = REPO_ROOT / relative_path
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@pytest.mark.parametrize(
    ("relative_path", "class_name", "module_name"),
    [
        (
            "scripts/reinforcement_learning/model_based/envs/anymal_d_flat.py",
            "AnymalDFlatEnv",
            "test_offline_anymal_d_flat",
        ),
        (
            "source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/envs/"
            "anymal_d_manager_based_mbrl_env.py",
            "ANYmalDManagerBasedMBRLEnv",
            "test_online_anymal_d_mbrl",
        ),
    ],
)
def test_joint_acceleration_uses_previous_joint_velocity(relative_path, class_name, module_name):
    """The acceleration penalty is a finite difference of joint velocities."""

    environment_class = _load_environment_class(relative_path, class_name, module_name)
    environment = environment_class.__new__(environment_class)
    num_envs = 2
    step_dt = 0.1

    environment.device = "cpu"
    environment._num_envs = num_envs
    environment.num_envs = num_envs
    environment.num_imagination_envs = num_envs
    environment._step_dt = step_dt
    environment.step_dt = step_dt
    environment.base_velocity = torch.zeros(num_envs, 3)
    environment.obs_last_action = torch.zeros(num_envs, 12)
    for name in ("last_air_time", "current_air_time", "last_contact_time", "current_contact_time"):
        setattr(environment, name, torch.zeros(num_envs, 4))

    # Policy observations are laid out as base velocities (0:9), command
    # velocity (9:12), joint positions (12:24), joint velocities (24:36),
    # and the previous action (36:48).  Make positions deliberately large so
    # accidentally using that slice produces a clearly different penalty.
    previous_joint_velocity = torch.tensor(
        [[1.0] * 12, [-2.0] * 12], dtype=torch.float32
    )
    current_joint_velocity = previous_joint_velocity + torch.tensor(
        [[0.2] * 12, [-0.3] * 12], dtype=torch.float32
    )
    policy_observation = torch.zeros(num_envs, 48)
    policy_observation[:, 12:24] = 100.0
    policy_observation[:, 24:36] = previous_joint_velocity
    environment.last_obs = {"policy": policy_observation}

    parsed_states = {
        "base_lin_vel": torch.zeros(num_envs, 3),
        "base_ang_vel": torch.zeros(num_envs, 3),
        "projected_gravity": torch.zeros(num_envs, 3),
        "joint_pos": torch.zeros(num_envs, 12),
        "joint_vel": current_joint_velocity,
        "joint_torque": torch.zeros(num_envs, 12),
    }
    parsed_contacts = {
        "thigh_contact": torch.zeros(num_envs, 4),
        "foot_contact": torch.zeros(num_envs, 4),
    }

    environment._compute_imagination_reward_terms(
        parsed_states,
        torch.zeros(num_envs, 12),
        None,
        parsed_contacts,
    )

    expected_acceleration = (current_joint_velocity - previous_joint_velocity) / step_dt
    expected_penalty = torch.sum(torch.square(expected_acceleration), dim=1)
    torch.testing.assert_close(environment.imagination_reward_per_step["dof_acc_l2"], expected_penalty)
