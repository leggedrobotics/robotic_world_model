# Robotic World Model Extension for Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository extends [**Isaac Lab**](https://github.com/isaac-sim/IsaacLab) with environments and training pipelines for
- [**Robotic World Model (RWM)**](https://sites.google.com/view/roboticworldmodel/home),
- [**Uncertainty-Aware Robotic World Model (RWM-U)**](https://sites.google.com/view/uncertainty-aware-rwm),

and related model-based reinforcement learning methods.

It enables:
- joint training of policies and neural dynamics models in Isaac Lab (online),
- training of policies with learned neural network dynamics without any simulator (offline),
- evaluation of model-based vs. model-free policies,
- visualization of autoregressive imagination rollouts from learned dynamics,
- visualization of trained policies in Isaac Lab.


<table>
  <tr>
  <td valign="top" width="50%">

![Robotic World Model](rwm.png)

**Paper**: [Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics](https://arxiv.org/abs/2501.10100)  
**Project Page**: [https://sites.google.com/view/roboticworldmodel](https://sites.google.com/view/roboticworldmodel)

  </td>
  <td valign="top" width="50%">

![Uncertainty-Aware Robotic World Model](rwm-u.png)

**Paper**: [Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots](https://arxiv.org/abs/2504.16680)  
**Project Page**: [https://sites.google.com/view/uncertainty-aware-rwm](https://sites.google.com/view/uncertainty-aware-rwm)

  </td>
  </tr>
</table>

**Authors**: [Chenhao Li](https://breadli428.github.io/), [Andreas Krause](https://las.inf.ethz.ch/krausea), [Marco Hutter](https://rsl.ethz.ch/the-lab/people/person-detail.MTIxOTEx.TGlzdC8yNDQxLC0xNDI1MTk1NzM1.html)  
**Affiliation**: [ETH AI Center](https://ai.ethz.ch/), [Learning & Adaptive Systems Group](https://las.inf.ethz.ch/) and [Robotic Systems Lab](https://rsl.ethz.ch/), [ETH Zurich](https://ethz.ch/en.html)


---


## Installation

1. **Install Isaac Lab** (not needed for offline policy training)

Follow the official [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the Conda installation as it simplifies calling Python scripts from the terminal.

2. **Install model-based RSL RL**

Follow the official installation guide of model-based [RSL RL](https://github.com/leggedrobotics/rsl_rl_rwm) for model-based reinforcement learning to replace the `rsl_rl_lib` that comes with Isaac Lab.

3. **Clone this repository** (outside your Isaac Lab directory)

```bash
git clone git@github.com:leggedrobotics/robotic_world_model.git
```

4. **Install the extension** using the Python environment where Isaac Lab is installed

```bash
python -m pip install -e source/mbrl
```

5. **Verify the installation** (not needed for offline policy training)

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Template-Isaac-Velocity-Flat-Anymal-D-Init-v0 --headless
```

---

## World Model Pretraining & Evaluation

Robotic World Model is a model-based reinforcement learning algorithm that learns a dynamics model and a policy concurrently.

### Configure model inputs/outputs

You can configure the model inputs and outputs under `ObservationsCfg_PRETRAIN` in [`AnymalDFlatEnvCfg_PRETRAIN`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py).

Available components:
- `SystemStateCfg`: state input and output head
- `SystemActionCfg`: action input
- `SystemExtensionCfg`: continuous privileged output head (e.g. rewards etc.)
- `SystemContactCfg`: binary privileged output head (e.g. contacts)
- `SystemTerminationCfg`: binary privileged output head (e.g. terminations)

And you can configure the model architecture and training hyperparameters under `RslRlSystemDynamicsCfg` and `RslRlMbrlPpoAlgorithmCfg` in [`AnymalDFlatPPOPretrainRunnerCfg`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py) .

Available options:
- `ensemble_size`: ensemble size for uncertainty estimation
- `history_horizon`: stacked history horizon
- `architecture_config`: architecture configuration
- `system_dynamics_forecast_horizon`: autoregressive prediction steps

### Run dynamics model pretraining:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
  --headless
```

It trains a PPO policy from scratch, while the induced experience during training is used to train the dynamics model.

### Visualize autoregressive predictions

```bash
python scripts/reinforcement_learning/rsl_rl/visualize.py \
  --task Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0 \
  --checkpoint <checkpoint_path> \
  --system_dynamics_load_path <dynamics_model_path>
```

It visualizes the learned dynamics model by rolling out the model autoregressively in imagination, conditioned on the actions from the learned policy.
The `dynamics_model_path` should point to the pretrained dynamics model checkpoint (e.g. `model_<iteration>.pt`) inside the saved run directory.

---

## Model-Based Policy Training & Evaluation

Once a dynamics model is pretrained, you can train a model-based policy purely from **imagined rollouts** generated by the learned dynamics.

There are two options:
- **Option 1: Train policy in imagination *online***, where additional environment interactions are continually collected using the latest policy to update the dynamics model (as implemented with RWM and MBPO-PPO in [Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics](https://arxiv.org/abs/2501.10100)).
- **Option 2: Train policy in imagination *offline*** where no additional environment interactions are collected and the policy has to rely on the static dynamics model (as implemented with RWM-U and MOPO-PPO in [Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots](https://arxiv.org/abs/2504.16680)).

### Option 1: Train policy in imagination *online*

The online data collection relies on interactions with the environment and thus brings up the simulator.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 --headless --checkpoint <checkpoint_path> --system_dynamics_load_path <dynamics_model_path>
```

You can either start the policy from pretrained checkpoints or from scratch by simply omitting the `--checkpoint` argument.

### Option 2: Train policy in imagination *offline*

The offline policy training does not request any new data and thus relies solely on the static dynamics model.
Align the model architecture and specify the model load path under `ModelArchitectureConfig` in [`AnymalDFlatConfig`](scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py).

Additionally, the offline imagination needs to branch off from some initial states. Specify the data path under `DataConfig` in [`AnymalDFlatConfig`](scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py).

```bash
python scripts/reinforcement_learning/model_based/train.py --task anymal_d_flat
```

### Play the learned model-based policy

You can play the learned policies with the original Isaac Lab task registry.

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Anymal-D-Play-v0 --checkpoint <checkpoint_path>
```

---

## Code Structure

We provide a reference pipeline that enables RWM and RWM-U on ANYmal D.

Key files:

**Online**

- Environment configurations + dynamics model setup
  [`flat_env_cfg.py`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py).
- Algorithm configuration + training parameters
  [`rsl_rl_ppo_cfg.py`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py).
- Imagination rollout logic (constructs policy observations & rewards from model outputs)
  [`anymal_d_manager_based_mbrl_env`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/envs/anymal_d_manager_based_mbrl_env.py).
- Visualization environment + rollout reset
  [`anymal_d_manager_based_visualize_env.py`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/envs/anymal_d_manager_based_visualize_env.py).

**Offline**

- Environment configurations + Imagination rollout logic (constructs policy observations & rewards from model outputs)
  [`anymal_d_flat.py`](scripts/reinforcement_learning/model_based/envs/anymal_d_flat.py).
- Algorithm configuration + training parameters
  [`anymal_d_flat_cfg.py`](scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py).
- Pretrained RWM-U checkpoint
  [`pretrain_rnn_ens.pt`](assets/models/pretrain_rnn_ens.pt).
- Initial states for imagination rollout
  [`state_action_data_0.csv`](assets/data/state_action_data_0.csv).


---

## Citation
If you find this repository useful for your research, please consider citing:

```text
@article{li2025robotic,
  title={Robotic world model: A neural network simulator for robust policy optimization in robotics},
  author={Li, Chenhao and Krause, Andreas and Hutter, Marco},
  journal={arXiv preprint arXiv:2501.10100},
  year={2025}
}
@article{li2025uncertainty,
  title={Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots},
  author={Li, Chenhao and Krause, Andreas and Hutter, Marco},
  journal={arXiv preprint arXiv:2504.16680},
  year={2025}
}
```
