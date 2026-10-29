"""Go2 任务下的 RSL-RL/MBPO runner 配置。

定义 init、pretrain、finetune、visualize 四类 runner 的超参数与加载策略。
"""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg

from mbrl.rl.rsl_rl import RslRlMbrlImaginationCfg, RslRlMbrlPpoAlgorithmCfg, RslRlNormalizerCfg, RslRlSystemDynamicsCfg


@configclass
class UnitreeGo2FlatPPOInitRunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.obs_groups = {"policy": ["policy"], "critic": ["policy"]}


@configclass
class UnitreeGo2FlatPPOPretrainRunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    class_name: str = "MBPOOnPolicyRunner"

    system_dynamics = RslRlSystemDynamicsCfg(
        ensemble_size=5,
        history_horizon=32,
        architecture_config={
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
            "state_mean_shape": [128],
            "state_logstd_shape": [128],
            "extension_shape": [128],
            "contact_shape": [128],
            "termination_shape": [128],
        },
        freeze_auxiliary=False,
    )
    imagination = RslRlMbrlImaginationCfg(
        num_envs=0,
        num_steps_per_env=0,
        max_episode_length=0,
        command_resample_interval_range=None,
        uncertainty_penalty_weight=-0.0,
        state_normalizer=RslRlNormalizerCfg(
            mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0] + [0.0] * 36,
            std=[0.5, 0.5, 0.1, 0.3, 0.3, 0.5, 0.02, 0.02, 0.04] + [0.2] * 12 + [2.0] * 12 + [8.0] * 12,
        ),
        action_normalizer=RslRlNormalizerCfg(
            mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )
    algorithm = RslRlMbrlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        policy_learning_rate=1.0e-3,
        system_dynamics_learning_rate=3.0e-4,
        system_dynamics_weight_decay=1.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        system_dynamics_forecast_horizon=8,
        system_dynamics_loss_weights={
            "state": 1.0,
            "sequence": 2.0,
            "bound": 1.0,
            "kl": 0.05,
            "extension": 1.0,
            "contact": 2.0,
            "termination": 2.0,
        },
        system_dynamics_num_mini_batches=32,
        system_dynamics_mini_batch_size=300,
        system_dynamics_replay_buffer_size=1500,
        system_dynamics_num_eval_trajectories=100,
        system_dynamics_len_eval_trajectory=400,
        system_dynamics_eval_traj_noise_scale=[0.1, 0.2, 0.4, 0.5, 0.8],
    )
    run_name = "pretrain"
    load_system_dynamics = False
    system_dynamics_load_path = None
    system_dynamics_warmup_iterations = 0
    system_dynamics_num_visualizations = 4
    system_dynamics_state_idx_dict = {
        r"$v$\n$[m/s]$": [0, 1, 2],
        r"$\omega$\n$[rad/s]$": [3, 4, 5],
        r"$g$\n$[1]$": [6, 7, 8],
        r"$q$\n$[rad]$": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        r"$\dot{q}$\n$[rad/s]$": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        r"$\tau$\n$[Nm]$": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    }
    pca_obs_buf_size = 10000

    def __post_init__(self):
        super().__post_init__()

        self.obs_groups = {"policy": ["policy"], "critic": ["policy"]}
        self.max_iterations = 2000


@configclass
class UnitreeGo2FlatPPOFinetuneRunnerCfg(UnitreeGo2FlatPPOPretrainRunnerCfg):
    resume = True
    load_run = ".*_pretrain"
    load_system_dynamics = True
    system_dynamics_load_path = None
    system_dynamics_warmup_iterations = 500
    run_name = "finetune"

    def __post_init__(self):
        super().__post_init__()
        self.imagination.num_envs = 8192
        self.imagination.num_steps_per_env = 24
        self.imagination.max_episode_length = 256
        self.imagination.command_resample_interval_range = [100, 120]
        self.imagination.uncertainty_penalty_weight = -0.0


@configclass
class UnitreeGo2FlatPPOVisualizeRunnerCfg(UnitreeGo2FlatPPOPretrainRunnerCfg):
    resume = True
    load_run = ".*_pretrain"
    load_system_dynamics = True
    system_dynamics_load_path = None
    run_name = "visualize"
