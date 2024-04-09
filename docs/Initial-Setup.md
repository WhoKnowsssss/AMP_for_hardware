# Run AMP and Diffusion on hardware #

```bash
conda create -p ./.conda-env/ python==3.8
```

```bash
conda activate ./.conda-env/
pip install -r requirements.txt
```

```bash
cd /rscratch/tk/Documents/isaacgym/python/
pip install -e .
```

```bash
cd ./rsl_rl/
pip install -e .
cd ../
pip install -e .
```

```bash
pip install setuptools==59.5.0
```


## Diffusion


1. Install diffusion policy

```https://github.com/Ruofeng-Wang/diffusion_policy```


```bash
pip install -e .
```

2. Install dependencies
You don't have to install all the dependencies. I installed these: 

```bash
pip install einop
pip install diffusers
pip install zarr
pip install omegaconf
pip install dill
```

### Isaac Gym

3. Modify ```legged_gym/scripts/play_diff.py``` to load model and normalizer correctly. 

4. Run command

```bash
python legged_gym/scripts/play_diff.py --task=a1_amp --sim_device=cpu --rl_device=cpu
```


## Qiayuan Build

```bash
cd ~/qiayuan_ws/
catkin build
# catkin build legged_controllers legged_unitree_description -DPYTHON_EXECUTABLE=/usr/bin/python3
# catkin build legged_rl_controllers legged_unitree_description -DPYTHON_EXECUTABLE=/usr/bin/python3
# catkin build legged_gazebo -DPYTHON_EXECUTABLE=/usr/bin/python3
# catkin build legged_unitree_hw -DPYTHON_EXECUTABLE=/usr/bin/python3
```
