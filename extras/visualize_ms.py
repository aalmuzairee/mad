# Utility script to visualize all enviornments in ManiSkill 

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import time
import numpy as np
import cv2
import os
import numpy as np
from dm_env import StepType

import envs


def make(env_name="PickCube"):
    env_name = env_name
    frame_stack = 1
    seed = 3
    class FakeCfg: pass

    cfg = FakeCfg()
    cfg.task = env_name
    cfg.frame_stack = frame_stack
    cfg.seed = seed
    cfg.img_size = 256
    cfg.reward_scale = 1
    cfg.ms_action = "pos"
    cfg.ms_mode = "rgbd"
    cfg.ms_preprocess_depth = True
    cfg.cameras = ['first','third1','third2']

    env = envs.make(cfg)
    env.reset()

    return env


# Loops over all maniskill envs and displays
def loop():

    # Pulling all ManiSkill3 tasks
    env_names = envs.ms_tasks 
    created_envs = []
    for env_name in env_names:
        print("Instantiating: ", env_name)
        created_envs.append(make(env_name))

    for i in range(len(created_envs)):
        env = created_envs[i]

        step = env.reset()
        count=  0

        # Tune to your liking
        end_count = 80
        reset_count = 10
        render_size = 256

        
        action_shape = env.action_spec().shape
        print("Running: ",env_names[i])
        while True:
            step = env.step((np.random.random(action_shape)*2)-1.0)
            rgb = step.observation.astype(np.float32)
            
            # Multiview  
            rgb = rgb.astype(np.uint8).transpose(0,2,3,1) # V, H, W, C            
            num_views, h, w, c = rgb.shape
            if c > 3: # Depth
                depth = rgb[:, :, :, 3]
                depth = np.stack([depth,depth,depth], axis=-1)
                rgb = rgb[:, :, :, :3] # Remove depth
                rgb = np.concatenate([rgb, depth], axis=0) # Combine over views
                num_views = num_views * 2
                c = c - 1
            splitter = np.split(rgb, num_views, axis=0)
            splitter = [v[0] for v in splitter]
            rgb = np.concatenate(splitter, axis=1) # Concatenate over width
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb = cv2.resize(rgb, dsize=(num_views * render_size, render_size))

            print("Step: ", count,"/", end_count, step.step_type, end="\r")
            cv2.imshow("Curr", rgb)
            cv2.waitKey(30)
            if (step.step_type == StepType.LAST) or (count % reset_count) == 0:
                env.reset()
            count += 1
            # End loop
            if count > end_count:
                env.close()
                cv2.destroyAllWindows()
                break
        print("Finished.                      ")




if __name__ == '__main__':
    loop()
