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

import os
import time
import threading

import numpy as np
from scipy.spatial.transform import Rotation
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from cc.udp import UDPTx, UDPRx

from legged_gym.envs.base.legged_robot import LeggedRobot
# from torch.tensor import Tensor

import time
import threading
from ctypes import *

import numpy as np

HOST_IP     = b"192.168.44.101"
HOST_PORT   = 9000
ROBOT_IP = "192.168.44.1"
ROBOT_PORT = 8000

N_OBSERVATIONS  = 45
N_ACTIONS = 12

communication_freq = 100

# print current directory
udp = cdll.LoadLibrary("/home/tk/Desktop/playground/AMP_for_hardware/libudp.so")


class UDPRx(Structure):
    _fields_ = [
        ("sockfd", c_int),
        ("thread_id", c_ulonglong),
        ("robot_addr", c_byte * 16),
        ("host_addr", c_byte * 16),
        ("obs", c_float * N_OBSERVATIONS),
    ]

class RealCyberDog2(LeggedRobot):
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
        env_ids =torch.ones([self.num_envs]).bool().nonzero(as_tuple=False).flatten()
        # first receive observation before reset ref motion.
        self.raw_observation=np.zeros((N_OBSERVATIONS, ))
        

        # self.rx_udp = UDPRx((HOST_IP, HOST_PORT))
        self.rx_udp = UDPRx()
        error = udp.initialize(byref(self.rx_udp), HOST_IP, HOST_PORT, ROBOT_IP, ROBOT_PORT)
        if error:
            print("Error initializing UDPRx")
            exit()
        self.tx_udp = UDPTx((ROBOT_IP, ROBOT_PORT))


        self.rx_buffer = None
        self.tx_buffer = np.zeros((N_ACTIONS, ), dtype=np.float32)

        self.is_running = threading.Event()

        self._state_flag = 0

        #self._recv_commands[0:10] = np.array([0.3, 0.0, 0.0, 0.0, np.pi, np.pi, 0, 0.6, 0.12, 0.35])
        self._recv_commands = np.array([0.5, 0.0, 0.0, ])

    def _cheetah_obs_callback(self, data):
        # self.raw_observation[:]=np.array(data.data)
        pass
        
    def _status_callback(self, data):
        self._state_flag = np.array(data.data)[0]

    def _status_callback_2(self, data):
        if data.buttons[1]==1:
            self._state_flag = 1
            print("pressed A!")
        elif data.buttons[2]==1:
            self._state_flag = 0


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
        
    def step(self, actions):
        # # self.filter(actions)  TODO disable filter?
        # # clip
        # clip_actions = self.cfg.normalization.clip_actions
        # self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # # if int(self._state_flag) != 1:
        # #     self.actions *= 0

        # # step physics and render each frame
        # # self._clip_max_change()

        # # self.actions[:, :] = 0
        # # self.actions[:, 2] = 0.6 * np.sin(2 * time.time())
        # # self.actions[:, 5] = 0.6 * np.sin(2 * time.time())
        # # self.actions[:, 8] = 0.6 * np.sin(2 * time.time())
        # # self.actions[:, 11] = 0.6 * np.sin(2 * time.time())

        # self.txHandler(self.actions[0].detach().cpu().numpy())

        # self.decode_observation()

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, None, None

    def _compute_real_observations(self, actions):
        used_obs = np.array(self.rx_udp.obs).reshape(1, -1)
        used_obs[:, 6:9] = self._recv_commands[:3] * self.commands_scale.cpu().numpy()

        # used_obs = np.concatenate([used_obs[:, :3], used_obs[:, 6:-12], actions[None, :]], axis=-1) # HACK for current AMP
        used_obs = np.concatenate([used_obs[:, :3], used_obs[:, 6:]], axis=-1) # HACK for current AMP
        self.obs_buf = torch.from_numpy(used_obs).float().to(self.device)   

        return self.obs_buf

    def _rpy_to_quat(self, rpy):
        rot = Rotation.from_euler('xyz', rpy, degrees=False)
        rot_quat = rot.as_quat()
        return rot_quat

    def decode_observation(self):
        # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
        # print(self.raw_observation)
        used_obs = self.raw_observation.copy().reshape(1, -1)
        used_obs[:, 6:9] = self._recv_commands * self.commands_scale.cpu().numpy()

        used_obs = np.concatenate([used_obs[:, :3], used_obs[:, 6:-12], self.actions], axis=-1) # HACK for current AMP


        self.obs_buf = torch.from_numpy(used_obs).float().to(self.device)

        # print(self.obs_buf)


    def rxHandler(self):
        # self.rx_udp._sock.settimeout(timeout)
        while not self.is_running.is_set():
            n_elements = 45
            dtype = np.float32

            rx_buffer = self.rx_udp.recvNumpy(
                bufsize=n_elements*np.dtype(dtype).itemsize, 
                dtype=dtype, 
                timeout=0.01)
            if rx_buffer is not None:
                self.raw_observation[:] = rx_buffer

    def txHandler(self, actions: np.ndarray):
        actions = actions.astype(np.float32)

        self.tx_udp.send(actions)
        # print("TX message: %s" % actions)

