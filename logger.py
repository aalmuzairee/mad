


import os
import datetime
import re
import numpy as np
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf
import moviepy.editor as mp


CONSOLE_FORMAT = [
    ('step', 'S', 'int'),
    ('episode', 'E', 'int'),
    ('episode_reward', 'R', 'float'),
    ('episode_length', 'L', 'int'),
    ('total_time', 'T', 'time'),
    ('sps', 'SPS', 'float'),
    ('mode', 'M', 'str'), 
    ('episode_success', 'SR', 'float'), # For robotics envs
]


CAT_TO_COLOR = {
    "train": "white",
    "eval": "green",
}

def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def print_run(cfg):
    """Pretty-printing of run information. Call at start of training."""
    prefix, color, attrs = '  ', 'green', ['bold']
    def limstr(s, maxlen=32):
        return str(s[:maxlen]) + '...' if len(str(s)) > maxlen else s
    def pprint(k, v):
        print(prefix + colored(f'{k.capitalize()+":":<16}', color, attrs=attrs), limstr(v))
    kvs = [('task', cfg.task),
           ('algorithm', cfg.agent),
           ('experiment', cfg.exp_name),
           ('seed', cfg.seed),
           ('train steps', f'{int(cfg.num_train_steps):,}'),
           ('observations', 'x'.join([str(s) for s in cfg.obs_shape])),
           ('actions', cfg.action_shape[0]),
           ('use_wandb', cfg.use_wandb)]
    
    w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
    div = '-'*w
    print(div)
    for k,v in kvs:
        pprint(k, v)
    print(div)
    

def cfg_to_group(cfg, return_list=False):
    """Return a wandb-safe group name for logging. Optionally returns group name as list."""
    lst = [cfg.task, cfg.agent, re.sub('[^0-9a-zA-Z]+', '-', cfg.exp_name)]
    return lst if return_list else '-'.join(lst)



class Logger:
    """Primary logging object. Logs either locally or using wandb."""

    def __init__(self, work_dir, cfg):
        self._cfg = cfg
        self._log_dir = make_dir(work_dir / "logs")
        self._save_csv = cfg.save_csv
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._eval = {'eval':[]}
        print_run(cfg)
        self.project = cfg.get("wandb_project", "none")
        self.entity = cfg.get("wandb_entity", "none")
        if not cfg.use_wandb or self.project == "none" or self.entity == "none":
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            self._wandb = None
        else:
            os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
            import wandb
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=str(cfg.seed),
                group=self._group,
                tags=cfg_to_group(cfg, return_list=True) + [f"seed={cfg.seed}"],
                dir=self._log_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
            self._wandb = wandb
        # Video Recorder
        self.video_recorder = VideoRecorder(save_video=self._cfg.save_video, video_dir=self._cfg.video_dir)


    def finish(self, model_path):
        if self._wandb:
            if self._cfg.save_snapshot:
                artifact = self._wandb.Artifact(self._group+'-'+str(self._seed), type='model')
                artifact.add_file(model_path)
                self._wandb.log_artifact(artifact)
            self._wandb.finish()

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "blue")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "blue")} {value:.02f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "blue")} {value}'
        elif ty == "str":
            value = str(value)
            return f'{colored(key+":", "blue")} {value}'
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if (k in d) and d[k] is not None:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))


    def log(self, d, category="train"):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        self.log_wandb(d, category)
        self.log_local(d, category)


    def log_wandb(self, d, category="train", add_step=True):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        if self._wandb:
            step_val = d["step"] if add_step else None
            full_metrics = {}
            for k, v in d.items():
                full_metrics[f'{category}/{k}'] = v
            self._wandb.log(full_metrics, step=step_val)

    def log_local(self, d, category="train"):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        if category == "eval" and self._save_csv:
            remove_keys = ["episode_reward", "episode_success"] # Confusing when there is a reward and success for each camera
            filtered_d = {k: v for k, v in d.items() if k not in remove_keys}
            curr_mode = 'eval'
            self._eval[curr_mode].append(np.array(list(filtered_d.values())))
            pd.DataFrame(np.array(self._eval[curr_mode])).to_csv(
                self._log_dir / f"{curr_mode}_{self._cfg.seed}.csv", header=list(filtered_d.keys()), index=None
            )
        self._print(d, category)



class VideoRecorder:
    """ Saves video locally or to wandb """
    def __init__(self, save_video=False, video_dir=None, render_size=256, fps=30):
        self.save_video = save_video
        self.video_dir = video_dir
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        
        if self.video_dir:
            try: 
                os.makedirs(self.video_dir, exist_ok = True) 
            except OSError as error: 
                self.video_dir = None
                pass

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = enabled
        self.record(env)

    def record(self, env):
        if self.enabled and self.save_video:
            if hasattr(env, 'render_multiview'):
                frame = env.render_multiview()
            elif hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.render()
            self.frames.append(frame)
    
    def save(self, file_name, wandb):
        if self.enabled and self.save_video:
            if self.video_dir:
                path = str(self.video_dir) +  f"/{file_name}.mp4"
                # Using moviepy
                clip = mp.ImageSequenceClip(self.frames, fps=self.fps)
                clip.write_videofile(path, verbose=False, logger=None)
            if wandb:
                frames = np.stack(self.frames).transpose(0, 3, 1, 2)
                wandb.log({file_name: wandb.Video(frames, fps=self.fps, format='mp4')})



