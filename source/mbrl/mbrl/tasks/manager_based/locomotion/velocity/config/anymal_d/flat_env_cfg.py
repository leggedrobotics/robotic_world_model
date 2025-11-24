# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, RewardsCfg

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from mbrl.mbrl.envs.mdp.commands import UniformVelocityCommand_Visualize, SampleUniformVelocityCommand
import mbrl.tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class RewardsCfg_TRAIN(RewardsCfg):
    stand_still = RewTerm(
        func=mdp.joint_pos_stand_still, weight=-1.0, params={"command_name": "base_velocity", "threshold": 0.05}
        )


@configclass
class AnymalDFlatEnvCfg(AnymalDRoughEnvCfg):
    
    rewards: RewardsCfg_TRAIN = RewardsCfg_TRAIN()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class AnymalDFlatEnvCfg_INIT(AnymalDFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # revert rewards
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.feet_air_time.weight = 0.125
        # revert terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 1.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0


@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):

    @configclass
    class SystemStateCfg(ObsGroup):

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=mdp.joint_effort)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class SystemActionCfg(ObsGroup):

        # observation terms (order preserved)
        pred_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class SystemExtensionCfg(ObsGroup):

        pass


    @configclass
    class SystemContactCfg(ObsGroup):

        # observation terms (order preserved)
        thigh_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0})
        foot_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"), "threshold": 1.0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class SystemTerminationCfg(ObsGroup):

        # observation terms (order preserved)
        base_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    # observation groups
    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
    # system_extension: SystemExtensionCfg = SystemExtensionCfg()
    system_contact: SystemContactCfg = SystemContactCfg()
    system_termination: SystemTerminationCfg = SystemTerminationCfg()


@configclass
class AnymalDFlatEnvCfg_PRETRAIN(AnymalDFlatEnvCfg):
    
    # override observation terms
    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()
    

@configclass
class AnymalDFlatEnvCfg_FINETUNE(AnymalDFlatEnvCfg_PRETRAIN):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # override commands
        self.commands.base_velocity.class_type = SampleUniformVelocityCommand


@configclass
class AnymalDFlatEnvCfg_VISUALIZE(AnymalDFlatEnvCfg_PRETRAIN):
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for visualize
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for visualize
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # override commands
        self.commands.base_velocity.class_type = UniformVelocityCommand_Visualize
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)
        # override randomization
        self.events.reset_base.func = mdp.reset_root_state_uniform_visualize
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (1.57, 1.57)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            }
        }
        self.events.reset_robot_joints.func = mdp.reset_joints_by_scale_visualize
