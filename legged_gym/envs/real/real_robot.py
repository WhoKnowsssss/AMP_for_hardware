# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import numpy as np
from scipy.spatial.transform import Rotation

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
# from torch.tensor import Tensor
try:
    import rospy 
    from std_msgs.msg import Float32MultiArray, Float32, String
    from sensor_msgs.msg import Joy
except:
    print("rospy not found, please install ros to do real experiments.")

class RealMiniCheetah(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        
        # force set to 1.
        self.cfg.env.num_envs=1
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.last_time = time.time()
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self._recv_motor_angles=np.zeros(12)
        self._recv_motor_vels=np.zeros(12)
        self._tweaked_actions=np.zeros(12)
        self.motor_direction_correction=np.array([1,1,1,1,1,1,1,1,1,1,1,1])

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.time_per_render = self.cfg.sim.dt * self.cfg.control.decimation
        # self._ref_motion = refmotion()
        is_local=False   # hard code
        # self._udp=cheetah_udp_interface(is_local=is_local)
        rospy.init_node('cheetah_udp_interface', anonymous=True)
        env_ids =torch.ones([self.num_envs]).bool().nonzero(as_tuple=False).flatten()
        # first receive observation before reset ref motion.
        self.raw_observation=np.zeros(37)
        rospy.Subscriber("/go1_lowlevel/robot_state", Float32MultiArray, self._cheetah_obs_callback)
        rospy.Subscriber("/go1_lowlevel/status_flag", Float32MultiArray, self._status_callback)
        rospy.Subscriber("/go1_lowlevel/robot_command", Float32MultiArray, self._command_callback)
        rospy.Subscriber("/joy", Joy, self._joystick_callback)
        self.publisher = rospy.Publisher('/go1_lowlevel/actions', Float32MultiArray, queue_size=1)
        self._state_flag = 0

        #self._recv_commands[0:10] = np.array([0.3, 0.0, 0.0, 0.0, np.pi, np.pi, 0, 0.6, 0.12, 0.35])
        self._recv_commands = np.array([0.6, 0, 0.])

    def _cheetah_obs_callback(self, data):
        self.raw_observation[:]=np.array(data.data)
        # print(self.raw_observation[7])
        
    def _status_callback(self, data):
        self._state_flag = np.array(data.data)[0]
    
    def _command_callback(self, data):
        self._recv_commands = np.array(data.data)

    def _joystick_callback(self, data):
        if data.buttons[1]==1:
            self._state_flag = 1
            print("pressed A!")
        elif data.buttons[2]==1:
            self._state_flag = 0
        
        commanded_velocity_x = data.axes[3] if data.axes[3] > 0 else 0
        commanded_velocity_y = data.axes[0]
        commanded_velocity_z = data.axes[2]

        self._recv_commands[0] = commanded_velocity_x
        self._recv_commands[1] = commanded_velocity_y
        self._recv_commands[2] = commanded_velocity_z

        # elif data.buttons[3]==1:
        #     self._recv_commands = np.array([0.1, 0, 0.])
        # elif data.buttons[0]==1:
        #     self._recv_commands = np.array([1., 0, 0.])


    def _clip_max_change(self):
        MAX_CHANGE=0.3
        self.actions[0]=torch.clip(self.actions[0],(self.dof_pos-self.default_dof_pos-MAX_CHANGE)/self.cfg.control.action_scale,
                                                            (self.dof_pos-self.default_dof_pos+MAX_CHANGE)/self.cfg.control.action_scale)

    # prevent calling reset_idx from parent class.
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
    def step(self,actions):
        # self.filter(actions)  TODO disable filter?
        # clip
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        if int(self._state_flag) !=  1:
            self.actions*=0

        # step physics and render each frame
        # self._clip_max_change()
        self._process_action_to_real(self.actions)

        self.publisher.publish(Float32MultiArray(data=self.final_actions[0]))

        self.decode_observation()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, None, None

    def _rpy_to_quat(self, rpy):
        rot = Rotation.from_euler('xyz', rpy, degrees=False)
        rot_quat = rot.as_quat()
        return rot_quat

    def decode_observation(self):
        # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
        # print(self.raw_observation)

        self._recv_orientation = self.raw_observation[3:7]

        self._recv_rpyrate = np.array(self.raw_observation[10:13])
        
        # print("real_xyzw",self._recv_orientation)
        # print("rpy_rate", self._recv_rpyrate)

        self._tweak_motor_order_direction()

        # turn numpy to tensors:
        self.base_quat = to_torch(self._recv_orientation,device=self.device).view(1,-1)
        self.base_ang_vel = to_torch(self._recv_rpyrate,device=self.device).view(1,-1)
        self.dof_pos = to_torch(self._recv_motor_angles,device=self.device).view(1,-1)
        self.dof_vel = to_torch(self._recv_motor_vels,device=self.device).view(1,-1)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.commands = to_torch(self._recv_commands,device=self.device).view(1,-1)
        # self.commands = to_torch(self._recv_commands,device=self.device).view(1,-1)
        # self._receive_gamepad_commands_once()
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_ang_vel)
        # print(self.commands[:,0:10])
        self.obs_buf_without_history = torch.cat((  
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # print(self.obs_buf_without_history)
        # fill in self.obs_buf
        self.obs_buf = self.obs_buf_without_history


    def _tweak_motor_order_direction(self):
        # isaac order:  FL FR HL HR     abduction +hip+ knee
        # cheetah software order: FR FL HR HL adduction + hip + knee
        dof_state = self.raw_observation[13:37].reshape(12,-1)
        raw_angles = dof_state[:,0]
        raw_vels = dof_state[:,1]
        self._recv_motor_angles[6:9] = raw_angles[3:6]
        self._recv_motor_angles[3:6] = raw_angles[6:9]

        self._recv_motor_angles[0:3] = raw_angles[0:3]
        self._recv_motor_angles[9:12] = raw_angles[9:12]

        self._recv_motor_vels[6:9] = raw_vels[3:6]
        self._recv_motor_vels[3:6] = raw_vels[6:9]

        self._recv_motor_vels[0:3] = raw_vels[0:3]
        self._recv_motor_vels[9:12] = raw_vels[9:12]


        # change directions
        self._recv_motor_angles = self.motor_direction_correction * self._recv_motor_angles
        self._recv_motor_vels = self.motor_direction_correction * self._recv_motor_vels
    
    def _process_action_to_real(self,raw_actions):
        # raw_actions=self.actions.clone() *motor_direction
        raw_actions = raw_actions.cpu().numpy()
        self._tweaked_actions[6:9] = raw_actions[0,3:6]
        self._tweaked_actions[3:6] = raw_actions[0,6:9]

        self._tweaked_actions[0:3] = raw_actions[0,0:3]
        self._tweaked_actions[9:12] = raw_actions[0,9:12]
        

        actions_scaled = self._tweaked_actions * self.cfg.control.action_scale
        final_actions = actions_scaled + self.default_dof_pos.cpu().numpy()
        self.final_actions = final_actions * self.motor_direction_correction
        #self.final_actions[:, [0, 3, 6, 9]] = 0.