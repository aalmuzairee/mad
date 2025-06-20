cd ~; mkdir .mujoco; cd .mujoco; 
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz; 
tar -xzf mujoco210-linux-x86_64.tar.gz; rm mujoco210-linux-x86_64.tar.gz; 
wget https://www.roboti.us/file/mjkey.txt; 
cd ~; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin; export MUJOCO_GL egl;  

