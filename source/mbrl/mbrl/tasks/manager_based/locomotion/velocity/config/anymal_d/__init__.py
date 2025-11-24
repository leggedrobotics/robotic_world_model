import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Isaac-Velocity-Flat-Anymal-D-Init-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalDFlatEnvCfg_INIT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalDFlatEnvCfg_PRETRAIN",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPOPretrainRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.envs.anymal_d_manager_based_mbrl_env:ANYmalDManagerBasedMBRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalDFlatEnvCfg_FINETUNE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPOFinetuneRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.envs.anymal_d_manager_based_visualize_env:ANYmalDManagerBasedVisualizeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalDFlatEnvCfg_VISUALIZE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPOVisualizeRunnerCfg",
    },
)
