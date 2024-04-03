# Run AMP and Diffusion on hardware #

## System Setup

There are two ways of setting up the experiment system. 

First way is to use a computer to connect to Go1, and another computer to run the diffusion model.

Second way is to use the Mini PC to both connect to Go1 and run the diffusion model.

We will introduce the second method.

Make sure that the Mini PC is under `Go1` Wired connection setting:

IP: `192.168.123.29`

Mask: `255.255.255.0`


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


Connect Mini PC (Master PC) to dog via ethernet cable

Make sure the master is `192.168.123.11`. If not, right click the system ethernet connection and select the other profile.


2. Run ROS on master PC

Terminator launch four windows

### Window #1:

```bash
source ~/qiayuan_ws/devel/setup.bash
roscore
```

### Window #2 (Run Hardware/Gazebo):

Hardware: 

```bash
source ~/qiayuan_ws/devel/setup.bash
roslaunch legged_unitree_hw legged_unitree_hw.launch
```

The dog motor should be enabled.

Gazebo: 
```bash
source ~/qiayuan_ws/devel/setup.bash
roslaunch legged_unitree_description empty_world.launch
```

### Window #3 (Run lowlevel controller)

```bash
source ~/qiayuan_ws/devel/setup.bash
roslaunch legged_rl_controllers load_legged_rl_controller.launch
```

### Window #4

```bash
rqt
```

Under "Controller manager", click namespace, select the available namespace

Under controller, "target controller", right click, select "Start"

Dog should stand up by itself.



## AMP on hardware
1. Install Robostack

https://robostack.github.io/GettingStarted.html

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

3. Start Topic_Controller in rqt, then use joystick to turn on (A) / off (B) the policy. 

Note: You can also send a Float64MultiArray message to topic go1_lowlevel/status_flag ([0] would turn off and [1] would turn on the policy). The easiest way is to install rqt_publisher, start rqt and find it in plugins, then publish the on/off message. 

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

### Gazebo/Hardware

3. Modify ```legged_gym/scripts/play_diff_real.py``` to load model and normalizer correctly. 

4. Run command

```bash
python legged_gym/scripts/play_diff_real.py --task=real_amp --sim_device=cpu --rl_device=cpu
```

5. Start Topic_Controller in rqt, then use joystick to turn on (A) / off (B) the policy. 

Note: You can also send a Float64MultiArray message to topic go1_lowlevel/status_flag ([0] would turn off and [1] would turn on the policy). The easiest way is to install rqt_publisher, start rqt and find it in plugins, then publish the on/off message. 


# Robot Dog Init

L2 + A

L2 + B

L1 + L2 + Start

