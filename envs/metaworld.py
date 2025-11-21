# Meta-World v2 Wrapper

from collections import deque
from typing import Any, NamedTuple
import numpy as np

import dm_env
from dm_env import specs, TimeStep, StepType
from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE


def make(cfg):
    task = cfg.task
    frame_stack = cfg.frame_stack
    seed = cfg.seed
    cameras = cfg.cameras
    img_size = cfg.img_size
    

    action_repeat = 2
    max_episode_steps = 200
    sparse_rewards = getattr(cfg, 'sparse_rewards', False)

    task += "-v3-goal-observable"

    if task not in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE.keys():
        raise ValueError('Invalid MetaWorld-v3 task:', task)

    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[task](seed=seed)
    
    env._freeze_rand_vec = False
    env = Gym2DMC(env, max_episode_steps = max_episode_steps, sparse_rewards=sparse_rewards) 
    env = ActionWrapper(env, action_repeat=action_repeat, dtype=np.float32, minimum=-1.0, maximum=1.0)
    env = PixelWrapper(env, img_size, cameras=cameras)
    env = FrameStackWrapper(env, frame_stack)   
    env = ExtendedTimeStepWrapper(env)

    env.reset()
    return env


class Gym2DMC(dm_env.Environment):
    """
        Convert a Gym environment to a DMC environment, adds a state and success info, 
        limits episode to max_episode_steps, chooses sparse or dense rewards
    """
    def __init__(self, gym_env, max_episode_steps, sparse_rewards=False) -> None:
        gym_obs_space = gym_env.observation_space
        self._observation_spec = specs.BoundedArray(
            shape=gym_obs_space.shape,
            dtype=gym_obs_space.dtype,
            minimum=gym_obs_space.low,
            maximum=gym_obs_space.high,
            name='observation'
            )
        gym_act_space = gym_env.action_space
        self._action_spec = specs.BoundedArray(
            shape=gym_act_space.shape,
            dtype=gym_act_space.dtype,
            minimum=gym_act_space.low,
            maximum=gym_act_space.high,
            name='action'
            )
        self._env = gym_env
        # Success info
        self._is_success = 0
        # State info
        self._state_obs = None
        self.reset()
        self._state_spec = specs.Array(
            shape=self.state.shape,
            dtype=self.state.dtype,
            name='state'
        )
        # Reward Type
        self._sparse_rewards = sparse_rewards
        # Time Limit
        self._elapsed_steps = None
        self._max_episode_steps = max_episode_steps



    def step(self, action):
        obs, reward, term, trun, info = self._env.step(action)
        if self._sparse_rewards:
            reward = info['success']
        self._state_obs = obs.astype(np.float32)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            step_type = StepType.LAST
            discount = 0.0
            self._is_success = info['success']
        else:
            step_type = StepType.MID
            discount = 1.0
        return TimeStep(step_type=step_type,
                        reward=reward,
                        discount=discount,
                        observation=obs)

    def reset(self):
        obs, info = self._env.reset()
        obs = self._env.step(np.zeros_like(self._env.action_space.sample()))[0]
        self._state_obs = obs.astype(np.float32)
        self._is_success = 0  # Adding success metric
        self._elapsed_steps = 0
        return TimeStep(step_type=StepType.FIRST,
                        reward=0.0,
                        discount=1.0,
                        observation=obs)

    @property
    def state(self):
        state = self._state_obs.astype(np.float32)
        accessible_state = np.concatenate((state[:4], state[18 : 18 + 4])) # Choose only robot-accessible state
        # accessible_state = state # Full state
        return accessible_state

    @property
    def is_success(self):
        return self._is_success

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def observation_spec(self):
        return self._observation_spec

    def state_spec(self):
        return self._state_spec

    def action_spec(self):
        return self._action_spec
    
    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionWrapper(dm_env.Environment):
    """ Action repeat, clip, and cast"""
    def __init__(self, env, action_repeat, dtype, minimum, maximum):
        self._env = env
        self._action_repeat = action_repeat
        self._dtype = dtype
        self._min = minimum
        self._max = maximum
        self._action_spec = self._env.action_spec().replace(dtype=self._dtype)

    def step(self, action):
        action = np.clip(action, self._min, self._max).astype(self._dtype)
        reward = 0.0
        discount = 1.0
        for i in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def state_spec(self):
        return self._env.state_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class PixelWrapper(dm_env.Environment):
    """ Pixel wrapper """
    def __init__(self, env, img_size, cameras=['corner2']):
        super().__init__()
        self._env = env
        self._cameras = cameras
        self._img_size = img_size

        self._observation_spec = specs.BoundedArray(
                        shape=(len(self._cameras), 3, img_size, img_size),
                        dtype=np.uint8, 
                        minimum=0, 
                        maximum=255,
                        name="observation")
        

        # Updating camera logic
        self.change_cams(self._cameras)
        # Third1 Person Corner camera
        self._env.unwrapped.model.camera("corner2").pos = [0.75, 0.075, 0.7]
        # Third2 (Top-dow)n view camera
        self._env.unwrapped.model.camera("corner3").pos = [0.72, 0.1, 1.2] 
        # Front Behind Grippper Camera
        self._env.unwrapped.model.camera("behindGripper").fovy = 90

        # Initialize mujoco renderer viewport
        self._env.unwrapped.mujoco_renderer.render(render_mode="rgb_array")


    def reset(self):
        time_step = self._env.reset()
        return time_step._replace(observation=self._get_pixel_obs())

    def step(self, action):
        time_step = self._env.step(action)
        return time_step._replace(observation=self._get_pixel_obs())

 
    def _get_pixel_obs(self):
        self.update_trackcam_pos() 
        all_cams = []
        for each_cam in self._cameras:
            all_cams.append(
                self.render(width=self._img_size, height=self._img_size, camera_name=each_cam).transpose(2, 0, 1) # C, H, W
            )
        return np.stack(all_cams, axis=0) # V, C, H, W

    # Track cams in Mujoco need to be updated at every step, while fixed cams can be updated only once
    def update_trackcam_pos(self):
        # shift camera pos
        self._env.unwrapped.data.camera("behindGripper").xpos += [0.0, 0.2, 0.00]

        # rotate cameras 180
        yaw = np.radians(180)
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        cam_xmat = np.reshape(self._env.unwrapped.data.camera("corner").xmat, (3,3))
        self._env.unwrapped.data.camera("corner").xmat = np.dot(cam_xmat, rotation_matrix).flatten() 

        cam_xmat = np.reshape(self._env.unwrapped.data.camera("corner2").xmat, (3,3))
        self._env.unwrapped.data.camera("corner2").xmat = np.dot(cam_xmat, rotation_matrix).flatten() 

        cam_xmat = np.reshape(self._env.unwrapped.data.camera("corner3").xmat, (3,3))
        self._env.unwrapped.data.camera("corner3").xmat = np.dot(cam_xmat, rotation_matrix).flatten() 


    # Utility, updating camera logic
    def change_cams(self, new_cams):
        cam_list = []
        for c in new_cams:
            if c == "first":
                cam_list.append("gripperPOV")
            elif c == "third1":
                cam_list.append("corner2")
            elif c == "front":
                cam_list.append("behindGripper")
            elif c == "third2":
                cam_list.append("corner3")
            else:
                cam_list.append(c)
        self._cameras = cam_list

    # Renders with all cameras for visualization
    def render_multiview(self):
        all_cams = []
        for each_cam in self._cameras:
            all_cams.append(
                self.render(camera_name=each_cam)# H, W, C
            )
        return np.concatenate(all_cams, axis=1) # Extended view

    def render(self, width=256, height=256, camera_name='corner2'):
        self._env.unwrapped.mujoco_renderer.viewer.make_context_current()
        self._env.unwrapped.mujoco_renderer.viewer.viewport.width = width
        self._env.unwrapped.mujoco_renderer.viewer.viewport.height = height
        img = self._env.unwrapped.mujoco_renderer.viewer.render(render_mode="rgb_array", camera_id=(self._env.unwrapped.model.camera(camera_name).id))
        flipped_img = img[:, ::-1]
        return flipped_img.copy()

    def observation_spec(self):
        return self._observation_spec

    def state_spec(self):
        return self._env.state_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    


class FrameStackWrapper(dm_env.Environment):
    """ Framestacks pixel and state wrapper """
    def __init__(self, env, frame_stack):
        super().__init__()
        self._env = env
        self._frame_stack = frame_stack


        self._frames = deque([], maxlen=self._frame_stack)
        obs_spec = self._env.observation_spec()
        self._observation_spec = specs.BoundedArray(
                        shape=(obs_spec.shape[0], self._frame_stack*obs_spec.shape[1], *obs_spec.shape[2:]),
                        dtype=obs_spec.dtype, 
                        minimum=obs_spec.minimum, 
                        maximum=obs_spec.maximum,
                        name=obs_spec.name)
        
        self._states = deque([], maxlen=self._frame_stack)
        state_spec = self._env.state_spec()
        self._state_spec = specs.Array(
                        shape=(self._frame_stack * state_spec.shape[0],),
                        dtype=state_spec.dtype, 
                        name=state_spec.name)


    def reset(self):
        time_step = self._env.reset()
        for _ in range(self._frame_stack):
            self._frames.append(time_step.observation)
            self._states.append(self._env.state)
        return time_step._replace(observation=self._stacked_obs()) 

    def step(self, action):
        time_step = self._env.step(action)
        self._frames.append(time_step.observation)
        self._states.append(self._env.state)
        return time_step._replace(observation=self._stacked_obs())


    @property
    def stacked_state(self):
        return self._stacked_state()

    def _stacked_obs(self):
        assert len(self._frames) == self._frame_stack
        return np.concatenate(list(self._frames), axis=1) 

    def _stacked_state(self):
        assert len(self._states) == self._frame_stack
        return np.concatenate(list(self._states), axis=0) 
 
    def observation_spec(self):
        return self._observation_spec

    def state_spec(self):
        return self._state_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    



class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    state: Any

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
        


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0,
                                action=action,
                                state=self._env.stacked_state)

    def observation_spec(self):
        return self._env.observation_spec()

    def state_spec(self):
        return self._env.state_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

