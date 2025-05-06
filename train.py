
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import numpy as np
import torch
from tqdm import tqdm
import shutil


from replay_buffer import ReplayBufferStorage, make_replay_loader
from logger import Logger
import utils

import envs
from dm_env import specs


torch.backends.cudnn.benchmark = True


def make_agent(state_spec, obs_spec, action_spec, cfg):
    cfg.state_shape = state_spec.shape[0]
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent_target, cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create envs
        self.train_env = envs.make(self.cfg)
        self.eval_env = envs.make(self.cfg)
        self.eval_cameras = cfg.eval_cameras

        # create replay buffer
        data_specs = (self.train_env.state_spec(),
                      self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / 'buffer')
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer' , self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        # make sure we have enough memory to train
        utils.calc_memory_gb(data_specs, num_steps=cfg.num_train_steps)

        # create agent
        self.agent = make_agent(self.train_env.state_spec(),
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # create logger
        self.logger = Logger(self.work_dir, self.cfg)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter



    def eval(self):
       # Number of successive episodes to record for video
        num_episodes_to_record = min(self.cfg.num_eval_episodes, self.cfg.num_record_eval_episodes)
        metrics = {}
        for each_cam in self.eval_cameras:
            self.eval_env.change_cams(each_cam) # Changing cam
            step, episode, total_reward, total_success = 0, 0, 0, 0  
            for episode in tqdm(range(self.cfg.num_eval_episodes), leave=False):
                time_step = self.eval_env.reset()
                # Record multiple episode videos
                if (episode == 0) or (episode >= num_episodes_to_record): 
                    self.logger.video_recorder.init(self.eval_env, enabled=(episode==0))
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.state,
                                                time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                    time_step = self.eval_env.step(action)
                    self.logger.video_recorder.record(self.eval_env)
                    total_reward += time_step.reward
                    step += 1
                
                total_success += getattr(self.eval_env, 'is_success', 0)
                # Record multiple episode videos
                if ((episode + 1) == num_episodes_to_record):
                    self.logger.video_recorder.save(file_name=f'{self.cfg.task}_{"+".join(each_cam)}_{self.global_step}', wandb=self.logger._wandb)
            # final increment
            episode += 1
            
            elapsed_time, total_time = self.timer.reset()
            metrics.update({
                'step': self.global_step,
                'episode': self.global_episode,
                'episode_length': step / episode,
                'sps': (step / elapsed_time),
                'total_time': total_time,
                'episode_reward': total_reward / episode,
                'episode_success': total_success / episode,
                '+'.join(each_cam) + '_reward': total_reward / episode,
                '+'.join(each_cam) + '_success': total_success / episode,
            })

        # logging
        self.logger.log(metrics, category="eval")



    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_steps)
        seed_until_step = utils.Until(self.cfg.num_seed_steps)
        eval_every_step = utils.Every(self.cfg.eval_every_steps)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        metrics = {}
        agent_metrics = {}
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # log stats
                elapsed_time, total_time = self.timer.reset()
                metrics.update(agent_metrics)
                metrics.update({
                    'step': self.global_step,
                    'episode': self.global_episode,
                    'episode_length': episode_step, 
                    'sps': (episode_step / elapsed_time),
                    'total_time': total_time,
                    'episode_reward': episode_reward,
                })
                self.logger.log(metrics, category="train")
                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                # save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # evaluate
            if eval_every_step(self.global_step):
                self.eval() 

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.state,
                                        time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # update the agent
            if not seed_until_step(self.global_step):
                agent_metrics.update(self.agent.update(self.replay_iter, self.global_step))

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

        # finished training
        if self.cfg.save_snapshot:
            self.save_snapshot()
        if self.cfg.save_final_video_once:
            self.logger.video_recorder.save_video = True        
        self.eval() 


    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    # Saving model snapshot to wandb, close wandb, delete buffer files, closes envs
    def finish(self):
        snapshot = self.work_dir / 'snapshot.pt'
        self.logger.finish(snapshot)
        try:
            shutil.rmtree("buffer")
        except Exception as e:
            pass
        for each_env in [self.train_env, self.eval_env]:
            try:
                each_env.close()
            except:
                pass

@hydra.main(config_path='.', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    
    # train
    workspace.train()

    # finish
    workspace.finish()


if __name__ == '__main__':
    main()