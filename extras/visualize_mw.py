# Utility script to visualize all enviornments in MetaWorld 

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import time
import numpy as np
import cv2
import numpy as np
from dm_env import StepType

import envs 




def make(env_name="door-open"):
    env_name = env_name
    frame_stack = 1
    action_repeat = 1
    seed = 3
    class FakeCfg: pass

    cfg = FakeCfg()
    cfg.task = env_name
    cfg.frame_stack = frame_stack
    cfg.action_repeat = action_repeat
    cfg.seed = seed
    cfg.img_size = 256
    cfg.increase_rand = False
    cfg.sparse_rewards = False
    cfg.cameras = ['first','third1', 'third2']

    env = envs.make(cfg)
    return env



# Loops over all metaworld envs and displays
def loop():

    env_names = envs.mw_tasks
    created_envs = []
    for env_name in env_names:
        print("Instantiating: ", env_name)
        created_envs.append(make(env_name))

    for i in range(len(created_envs)):
        env = created_envs[i]
        env.reset()
        count=  0
        
        # Tune to your liking
        end_count = 80
        reset_count = 10
        render_size = 256


        action_shape = env.action_spec().shape
        print("Running: ",env_names[i])
        while True:
            step = env.step(np.random.random(action_shape))
            rgb = step.observation.astype(np.float32)

            # Multi view visualizze
            rgb = rgb.astype(np.uint8).transpose(0,2,3,1) # V, H, W, C
            num_views, h, w, c = rgb.shape
            splitter = np.split(rgb, num_views, axis=0)
            splitter = [v[0] for v in splitter]
            rgb = np.concatenate(splitter, axis=1) # Concatenate over width
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb = cv2.resize(rgb, dsize=(num_views * render_size, render_size))
            print("Step: ", count,"/", end_count, step.step_type, end="\r")

            cv2.imshow("Curr", rgb)
            cv2.waitKey(30)
            if count % reset_count == 0:
                env.reset()

            if step.step_type == StepType.LAST:
                env.reset()
            count += 1
            # End loop
            if count > end_count:
                cv2.destroyAllWindows()
                break
        print("Finished.                      ")



if __name__ == '__main__':
    loop()
