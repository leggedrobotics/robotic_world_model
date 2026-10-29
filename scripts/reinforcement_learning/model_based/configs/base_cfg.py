"""离线训练基础配置。

定义实验、环境、数据、模型与策略训练的通用参数结构。
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class BaseConfig:
    experiment_name: str = "online"

    @dataclass
    class ExperimentConfig:
        environment: str = "dummy"
        device: str = "cuda"

        def to_dict(self):
            return asdict(self)

    @dataclass
    class EnvironmentConfig:
        num_envs: int = 8192
        max_episode_length: int = 256
        step_dt: float = 0.02
        reward_term_weights: Dict[str, float] = field(default_factory=lambda: {"dummy": 0.0})
        uncertainty_penalty_weight: float = -0.0
        observation_noise: bool = True
        command_resample_interval_range: List[int] | None = None
        event_interval_range: List[int] | None = None

        def to_dict(self):
            return asdict(self)

    @dataclass
    class DataConfig:
        dataset_root: str = "logs/online"
        dataset_folder: str = "train"
        file_data_size: int = 10000
        batch_data_size: int = 50000

        state_idx_dict: Dict[str, List[int]] = field(default_factory=lambda: {"dummy": [0]})
        state_data_mean: List[float] = field(default_factory=lambda: [0.0])
        state_data_std: List[float] = field(default_factory=lambda: [0.0])
        action_data_mean: List[float] = field(default_factory=lambda: [0.0])
        action_data_std: List[float] = field(default_factory=lambda: [0.0])

        init_data_ratio: float = 0.8
        num_eval_trajectories: int = 10
        len_eval_trajectory: int = 400
        num_visualizations: int = 4

        def to_dict(self):
            return asdict(self)

    @dataclass
    class ModelArchitectureConfig:
        history_horizon: int = 1  # the window size of the input state transitions
        forecast_horizon: int = 1  # the autoregressive prediction steps
        extension_dim: int = 0
        contact_dim: int = 0
        termination_dim: int = 0
        ensemble_size: int = 1
        architecture_config: Dict[str, object] = field(
            default_factory=lambda: {
                "type": "mlp",
                "base_shape": [256, 256],
                "state_mean_shape": [128],
                "state_logstd_shape": [128],
                "extension_shape": [128],
                "contact_shape": [128],
                "termination_shape": [128],
            }
        )
        # architecture_config: Dict[str, object] = field(default_factory=lambda: {
        #     "type": "rnn",
        #     "rnn_type": "gru",
        #     "rnn_num_layers": 2,
        #     "rnn_hidden_size": 256,
        #     "state_mean_shape": [128],
        #     "state_logstd_shape": [128],
        #     "extension_shape": [128],
        #     "contact_shape": [128],
        #     "termination_shape": [128],
        # })
        freeze_auxiliary: bool = False
        resume_path: str | None = None

        def to_dict(self):
            return asdict(self)

    @dataclass
    class PolicyArchitectureConfig:
        observation_dim: int = 0
        obs_groups: Dict[str, List[str]] = field(default_factory=lambda: {"policy": ["policy"]})
        action_dim: int = 0
        actor_hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
        critic_hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
        activation: str = "elu"
        init_noise_std: float = 1.0
        resume_path: str | None = None

        def to_dict(self):
            return asdict(self)

    @dataclass
    class PolicyAlgorithmConfig:
        value_loss_coef: float = 1.0
        use_clipped_value_loss: bool = True
        clip_param: float = 0.2
        entropy_coef: float = 0.005
        num_learning_epochs: int = 5
        num_mini_batches: int = 4
        learning_rate: float = 1.0e-3
        schedule: str = "adaptive"
        gamma: float = 0.99
        lam: float = 0.95
        desired_kl: float = 0.01
        max_grad_norm: float = 1.0

        def to_dict(self):
            return asdict(self)

    @dataclass
    class PolicyTrainingConfig:
        num_steps_per_env: int = 24
        save_interval: int = 200
        max_iterations: int = 500
        export_dir: str | None = None

        def to_dict(self):
            return asdict(self)

    def to_dict(self):
        return asdict(self)

    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_architecture_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    policy_architecture_config: PolicyArchitectureConfig = field(default_factory=PolicyArchitectureConfig)
    policy_algorithm_config: PolicyAlgorithmConfig = field(default_factory=PolicyAlgorithmConfig)
    policy_training_config: PolicyTrainingConfig = field(default_factory=PolicyTrainingConfig)
