# ManiSkill3 Wrapper

from collections import deque
from typing import Any, NamedTuple
import numpy as np
from PIL import Image

import dm_env
from dm_env import specs, TimeStep, StepType
import logging
logging.disable(level=logging.WARN) 


# Add tasks
import envs.tasks_maniskill 
import mani_skill.envs
import gymnasium as gym

from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper




def make(cfg):
    task = cfg.task + "-v2" # , only runs our modified five tasks ("-v2")
    frame_stack = cfg.frame_stack
    seed = cfg.seed
    cameras = cfg.cameras
    obs_mode = getattr(cfg, "ms_mode", "rgb") 
    reward_scale = 1.0
    img_size = getattr(cfg, "img_size", 84)
    sensor_configs = {"width":img_size,"height":img_size}
    human_render_camera_configs = {"render_camera":{"width":512,"height":512}}

    env_kwargs = dict(obs_mode=obs_mode, control_mode="pd_ee_delta_pos", render_mode="all", sim_backend="cpu", reward_mode="normalized_dense", 
                      sensor_configs=sensor_configs, human_render_camera_configs=human_render_camera_configs)
    env = gym.make(task, num_envs=1, **env_kwargs)
    env = CPUGymWrapper(env)
    env = DMWrapper(env, seed, cameras=cameras, frame_stack=frame_stack, reward_scale=reward_scale)
    return env




# Utility, updating camera logic
def rename_cams(new_cams):
    cam_list = []
    for c in new_cams:
        if c == "first": # first
            cam_list.append("hand_camera")
        elif c == "third1": # third A
            cam_list.append("side_camera")
        elif c == "third2": # third B
            cam_list.append("base_camera")
        else:
            cam_list.append(c)
    return cam_list


class DMWrapper(dm_env.Environment):
    """
        Convert environment to a DM environment. Adds success, obs and rendering.
    """
    def __init__(self, env, seed, cameras, frame_stack=1, reward_scale=1.0) -> None:
        self._env = env
        self._reward_scale = reward_scale 
        # Success Metric
        self._is_success = 0
        # Frame Stack
        self._frame_stack = frame_stack
        self._frames = deque([], maxlen=self._frame_stack)
        # Observation and Action Specs, Setting Seed and Cameras
        obs, info = env.reset(seed=seed)
        self._camera_names = rename_cams(cameras)

        image_obs = self.extract_obs(obs)
        state_obs = self.extract_state(obs)
        self._observation_spec = specs.Array(
            shape=image_obs.shape,
            dtype=image_obs.dtype,
            name='observation'
            )
        # State Spec
        self._state_spec = specs.Array(
            shape=state_obs.shape,
            dtype=state_obs.dtype, 
            name='state'
            )
        # Action Spec
        self._action_spec = specs.BoundedArray(
            shape=env.action_space.shape,
            dtype=np.float32,
            minimum=env.action_space.low,
            maximum=env.action_space.high,
            name='action'
            )
        


    # Returns stacked frames 
    def _stack_frames(self, obs):
        if self._frame_stack > 1:
            # Frame Stack Logic
            if len(self._frames) == self._frame_stack:
                self._frames.append(obs)
            else:
                for _ in range(self._frame_stack):
                    self._frames.append(obs)
            assert(len(self._frames)) == self._frame_stack
            obs = np.concatenate(list(self._frames), axis=1) # V, C, H, W
        return obs
    
    def _scale_depth(self, depth):
        # measurement in mm
        clip_beyond = 1500
        depth = np.clip(depth, 0, clip_beyond)
        max_depth = depth.max()
        min_depth = depth.min()
        depth = (depth - min_depth) / (max_depth - min_depth)
        depth = (255 * depth).astype(np.uint8)
        return depth

    # Extracts images and combines them
    def extract_obs(self, obs):
        # Extract Multi View Image
        image_keys = self._camera_names
        multiview_image = []
        for each_key in image_keys:
            d = obs['sensor_data'][each_key]
            view = []
            if 'rgb' in d.keys():
                view.append(d['rgb'])
            if 'depth' in d.keys():
                depth = self._scale_depth(d['depth'])
                view.append(depth)
            view = np.concatenate(view, axis=2) # H, W, C   
            if each_key == "hand_camera":
                view = np.rot90(view)
            multiview_image.append(view) 
        multiview_image = np.stack(multiview_image, axis=0).transpose(0, 3, 1, 2) # V, C, H, W        
        all_images = self._stack_frames(multiview_image)
        return all_images

    # Extracts end effector state from full state
    def extract_state(self, obs): 
        if 'is_grasped' in obs['extra'].keys():
            obs['extra']['is_grasped'] = np.array([int(obs['extra']['is_grasped'])], dtype=np.float32) # Converting is grasped to array

        agent_state = np.concatenate(list(obs['agent'].values()))
        extra_state = np.concatenate(list(obs['extra'].values()))
        full_state = np.concatenate([agent_state, extra_state])
        return full_state
    
    # Changes cameras to the given cameras, currently only supports observations that are already initalized with env creation
    def change_cams(self, new_cams):
        new_cams = rename_cams(new_cams)
        self._camera_names = new_cams
        self._frames.clear() # empty frames


    def step(self, action=None):
        # Action Logic
        if action is None:
            action = np.zeros(self._action_spec.shape, dtype=self._action_spec.dtype)
        else:
            action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum).astype(self._action_spec.dtype)
        # Extracting observations
        obs, reward, term, trun, info = self._env.step(action)
        state = self.extract_state(obs)
        obs = self.extract_obs(obs)
        self._is_success = int(info['success'])
        # Ignore terminated signal since it ends with fewer rewards
        if trun: 
            step_type = StepType.LAST
            discount = 0.0
        else:
            step_type = StepType.MID
            discount = 1.0
        return ExtendedTimeStep(
                        observation=obs,
                        state=state,
                        action=action,
                        reward=reward * self._reward_scale, 
                        discount=discount,
                        step_type=step_type,
                        )

    def reset(self):
        obs, info = self._env.reset()
        self._frames.clear() # empty framestack
        state = self.extract_state(obs)
        obs = self.extract_obs(obs)
        self._is_success = 0
        return ExtendedTimeStep(
                        observation=obs,
                        state=state,
                        action=np.zeros(self._action_spec.shape, dtype=self._action_spec.dtype),
                        reward=0.0,
                        discount=1.0,
                        step_type=StepType.FIRST
                        )

    # Extended view
    def render_multiview(self, img_size=256):
        multiview_image = None
        # Extracting relevant sensors only 
        full_obs = self._env.unwrapped.get_obs()['sensor_data']
        sensor_images = []
        for each_cam in self._camera_names:
            curr_cam = full_obs[each_cam]
            for each_key in curr_cam.keys():
                obs = curr_cam[each_key].cpu().numpy()[0]
                if each_key == "depth":
                    obs = self._scale_depth(obs)
                    obs = np.concatenate([obs,obs,obs], axis=-1) # Repeat depth for visualization
                if each_cam == "hand_camera": #rotate for visualization
                    obs = np.rot90(obs)
                sensor_images.append(obs)
        sensor_images = np.concatenate(sensor_images, axis=1) # H, W, C
        sensor_images = Image.fromarray(sensor_images)
        scale = int(sensor_images.width / sensor_images.height)  # Number of views and sensors
        sensor_images = np.array(sensor_images.resize((scale*img_size, img_size))) # H, W, C
        # Render and add nice view
        nice_image = self.render(img_size=img_size)
        multiview_image = np.concatenate([sensor_images, nice_image], axis=1)
        return multiview_image

    # Singular view
    def render(self, img_size=256, camera_name="agentview"):
        return np.array(Image.fromarray(self._env.unwrapped.render_rgb_array().cpu().numpy()[0]).resize((img_size, img_size))) # H, W, C

    @property
    def is_success(self):
        return int(self._is_success)

    def observation_spec(self):
        return self._observation_spec

    def state_spec(self):
        return self._state_spec

    def action_spec(self):
        return self._action_spec
    
    def close(self):
        self._env.close()
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    

    

class ExtendedTimeStep(NamedTuple):
    observation: Any
    state: Any
    action: Any
    reward: Any
    discount: Any
    step_type: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)
        




