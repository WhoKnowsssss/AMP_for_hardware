# Run AMP and Diffusion on hardware #

## Follow the original readme to install this package

## Hardware Setup
1. ROS Comm: 

Set Mini PC as master node, policy PC as compute node

Master PC: 
```
export ROS_MASTER_URI=http://master_ip:11311
export ROS_IP=master_ip
```

Compute PC: 
```
export ROS_MASTER_URI=http://master_ip:11311
export ROS_IP=compute_ip
```

2. Run ROS on master PC
```
roscore
```

3. Run Hardware/Gazebo

Hardware: 
```
roslaunch legged_unitree_hw legged_unitree_hw.launch
```

Gazebo: 
```
roslaunch legged_unitree_description empty_world.launch
```

4. Run lowlevel controller

```
roslaunch legged_rl_controllers load_rl_controller.launch
```

## AMP on hardware
1. Install Robostack

```https://robostack.github.io/GettingStarted.html```

replace 
```
conda config --env --add channels robostack-staging
```

with 
```
conda config --env --add channels robostack
```

2. Run command

```
python legged_gym/scripts/play_real.py --task=real_amp --sim_device=cpu --rl_device=cpu
```

3. Start Topic_Controller in rqt, then use joystick to turn on (A) / off (B) the policy

## Diffusion on hardware


2. Install diffusion policy

```https://github.com/Ruofeng-Wang/diffusion_policy```

You don't have to install all the dependencies. I installed these: 
```
pip install einop
pip install diffusers
pip install zarr
```

3. Modify ```legged_gym/scripts/play_diff.py``` to load model and normalizer correctly. 

Note: focus on model first. Normalizer shouldn't be hard


4. Run command
```
python legged_gym/scripts/play_diff.py --task=real_amp --sim_device=cpu --rl_device=cpu
```


# Robot Dog Init

L2 + A

L2 + B

L1 + L2 + Start

