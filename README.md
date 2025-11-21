

<h1> Merging and Disentangling Views in Visual Reinforcement Learning for Robotic Manipulation</span></h1>

Pytorch implementation of

[Merging and Disentangling Views in Visual Reinforcement Learning for Robotic Manipulation](https://arxiv.org/abs/2505.04619) by

[Abdulaziz Almuzairee](https://aalmuzairee.github.io), [Rohan Patil](https://rohanpatil.me/), [Dwait Bhatt](https://dwaitbhatt.com/), [Henrik I. Christensen](https://hichristensen.com) (UC San Diego)</br>

</br><img width="100%" src="https://github.com/aalmuzairee/mad/blob/gh-pages/static/videos/repo_header.gif"></br>

[[Website]](https://aalmuzairee.github.io/mad) [[Paper]](https://arxiv.org/abs/2505.04619) 

-----

## 🎥 TLDR;

We offer a method called MAD. Using MAD, a reinforcement learning agent can easily merge multiple camera views to gain higher sample efficiency while still being able to function with any singular camera view alone. MAD achieves that by 
1) encoding camera views individually,
2) merging the individual camera view features through summation to create a multi camera view representation,
3) framing all the singular view representations as augmentations to the multi camera view representation during the learning.

-----

## Citation

If you find our work useful, please consider citing our paper:

```
@misc{almuzairee2025merging,
      title={Merging and Disentangling Views in Visual Reinforcement Learning for Robotic Manipulation}, 
      author={Abdulaziz Almuzairee and Rohan Patil and Dwait Bhatt and Henrik I. Christensen},
      year={2025},
      eprint={2505.04619},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.04619}, 
}
```

-----

## ❗ Updates

- **[Nov/20/2025]** Added a seperate branch for Meta-world-v3 support and eased installation process.

-----

## Getting Started

### ⚙️ Installation

System requirements:

- GPU: Minimum 11 GB RAM, supports CUDA 11.0 or later 
- CPU RAM:  95 GB  (Meta-World tasks), 135GB (ManiSkill3 tasks) -- replay buffer implementation is not optimized
- CPU Cores: 4 cores
- Average Runtime:  4hrs (Meta-World tasks), 6hrs (ManiSkill3 tasks)
- Recommended Base Docker Image: `nvidia/cudagl:11.3.0-base-ubuntu18.04`
- Conda

If you're ready, clone this repo:
```sh
git clone https://github.com/aalmuzairee/mad.git
cd mad
```

Then to run Meta-World you need to install mujoco210. We provide a utility script that installs it:

```sh
apt update
apt-get install gcc libosmesa6-dev wget git -y
. ./extras/install_mw.sh
```

After installing mujoco, you can then install the packages by running:

```sh
cd mad
conda env create -f environment.yaml
conda activate mad
```
Finally you need to install gym=0.21 for Meta-World and setup the GPU links:
```sh
pip install pip==24.0
pip install gym==0.21

# some setups need the following symlink command
ln -s /usr/local/cuda /usr/local/nvidia 
```

-----

## 📜 Example usage

We provide examples on how to train below.

```sh
# Train MAD (Ours) with all three cameras and evaluate on all three singular and combined
python train.py agent=mad task=basketball

# Train MVD (Baseline) with two cameras, and evaluating on singular and combined cameras
python train.py agent=baselines.mvd task=hammer cameras=[first,third1] eval_cameras=[[first],[third1],[first,third1]]

# Train VIB (Baseline) with two cameras, and evaluating only on combined cameras
python train.py agent=baselines.vib task=soccer cameras=[first,third1] eval_cameras=[[first,third1]]
```
 where the log outputs will be: 

 ```sh
 eval    S: 0            E: 0            R: 3.60         L: 100          T: 0:00:07      SPS: 246.49     SR: 0.00                                                                                                            
 train   S: 100          E: 1            R: 6.42         L: 100          T: 0:00:08      SPS: 115.77
 train   S: 200          E: 2            R: 3.17         L: 100          T: 0:00:09      SPS: 115.18 
```

with each letter corresponding to:

 ```sh
 eval    S: Steps        E: Episode      R: Episode Reward     L: Episode Length    T: Time Elapsed      SPS: Steps Per Second    SR: Success Rate   
```


For logging, we recommend configuring [Weights and Biases](https://wandb.ai) (`wandb`) in `config.yaml` to track training progress.
We further provide limited logging in local csv files.


-----

## 📖 Config options

Please refer to `config.yaml` for a full list of options.

#### Algorithms

There are four algorithms that you can choose from:

- `mad` : [MAD (Almuzairee et al., 2025)](https://github.com/aalmuzairee/mad)
- `baselines.mvd` : [MVD (Dunion et al., 2024)](https://github.com/uoe-agents/MVD)
- `baselines.vib` : [VIB (Hsu et al., 2022)](https://github.com/moojink/cube-grasping)
- `baselines.drq` : [DrQ (Kostrikov et al., 2020)](https://github.com/denisyarats/drq)

by setting the `agent` variable in the `config.yaml` file or a commandline argument like `agent=mad`.

#### Tasks

We test on **20** Visual RL tasks from Meta-World and ManiSkill, you can find them listed in [`envs/__init__.py`](https://github.com/aalmuzairee/mad/blob/master/envs/__init__.py)

#### Cameras

As stated in the paper, we use three cameras: 

- First Person: `first`
- Third Person A: `third1`
- Third Person B: `third2`

which can be set for training in cameras:

`cameras=[first,third1,third2]`

and for evaluation you can use any combination, but it must be a list of lists. One example when evaluating on all singular and combined cameras would be:

`eval_cameras=[[first],[third1],[third2],[first,third1,third2]]`

which can all be set in `config.yaml` file or as a command line argument to `train.py`.

-----

## ✨ Visualization

We further offer on-screen rendering visualization scripts that help you visualize the environments, they can be found in extras/ and can be run using:
```sh
 # visualize metaworld
python extras/visualize_mw.py

# visualize maniskill
python extras/visualize_ms.py
```

## ❓ FAQ

#### Q: Can I train with lower CPU RAM?

**A:** If you set `frame_stack=1` in the config, you will be able to train with 35 GB (Meta-World tasks) or 45GB (ManiSkill3 tasks). We haven't verified the results, but we would expect a ~5-10% drop in final performance.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
