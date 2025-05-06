from typing import Any, Dict, Union

import numpy as np
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.envs.tasks.tabletop import PickCubeEnv
import sapien


@register_env("PickCube-v2", max_episode_steps=50)
class PickCubeEnv2(PickCubeEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(eye=[0, -0.9, 0.6], target=[0, 0, 0.15])
        side_pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0, 0, 0.35])      
        return [CameraConfig("base_camera", base_pose, 84, 84, np.pi/4, 0.01, 100),
                CameraConfig("side_camera", side_pose, 84, 84, 1, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

