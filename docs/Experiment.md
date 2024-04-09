# Experiment Setup

## System Setup

There are two ways of setting up the experiment system. 

First way is to use a computer to connect to Go1, and another computer to run the diffusion model.

Second way is to use the Mini PC to both connect to Go1 and run the diffusion model.

We will introduce the second method.

Make sure that the Mini PC is under `Go1` Wired connection setting:

IP: `192.168.123.29`

Mask: `255.255.255.0`


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
conda activate amp_hw
source ./env.sh
roscore
```

### Window #2 (Run Hardware/Gazebo):

Go1: 

```bash
conda activate amp_hw
source ./env.sh
roslaunch legged_unitree_hw legged_unitree_hw.launch
```

The dog motor should be enabled.

Gazebo: 

```bash
conda activate amp_hw
source ./env.sh
roslaunch legged_unitree_description empty_world.launch
```

### Window #3 (Run lowlevel controller)

```bash
conda activate amp_hw
source ./env.sh
roslaunch legged_rl_controllers load_legged_rl_controller.launch
```

### Window #4

```bash
conda activate amp_hw
source ./env.sh
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

# Robot Dog Init

L2 + A

L2 + B

L1 + L2 + Start

