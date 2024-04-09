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

import sys
import os
import time
import threading
import time
import queue
import signal

import isaacgym
import numpy as np
import torch
from matplotlib import pyplot as plt

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger


def sigint_handler(signum, frame):
	print("  Ctrl + C Exit!")
	sys.exit(0)
# signal.signal(signal.SIGINT, sigint_handler) 


class logger_config:
    EXPORT_POLICY=False
    RECORD_FRAMES=False
    robot_index=0
    joint_index=1

def add_input(input_queue, stop_event):
    while not stop_event.is_set():
        input_queue.put(sys.stdin.read(1))


_convert_obs_dict_to_tensor = lambda obs, device: torch.tensor(np.concatenate([
        obs["ProjectedGravity"], obs["FakeCommand"], obs["MotorAngle"],
        obs["MotorVelocity"], obs["LastAction"]]), device=device).float()


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.num_envs = 1
    # env_cfg.env.get_commands_from_joystick = False
    env_cfg.terrain.num_rows = 1    # 5
    env_cfg.terrain.num_cols = 1    # 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.randomize_gains = False
    # env_cfg.domain_rand.randomize_base_mass = False
    
    num_log_steps = 100
    log_counter = 0
    obs_log = []
    timestr = time.strftime("%m%d-%H%M%S")
    logdir = f'experiement_log_{timestr}'
    # os.mkdir(logdir)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    input_queue = queue.Queue()
    stop_input = threading.Event()
    input_thread = threading.Thread(target=add_input, args=(input_queue,stop_input))
    input_thread.daemon = True
    input_thread.start()

    obss = np.zeros((6660, 344))
    filter_obss = np.zeros((6660, 12))
    idx = 0
    global_idx = 0
    start_idx = np.inf
    stand_override = False

    print("running")

    s = time.perf_counter()
    s2 = time.perf_counter()

    def infer_action_callback():
        nonlocal env, policy, obs, obss, idx, s, start_idx, global_idx, stand_override

        global_idx += 1

        if not stand_override:
            with torch.no_grad():
                actions = policy(obs.detach())
                obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        s = time.perf_counter()    

    def call_every(seconds, callback, stop_event):
        t1 = time.perf_counter()
        t2 = time.perf_counter()
        
        while not stop_event.wait(seconds - (t1-t2)):
            t2 = time.perf_counter()
            callback()
            # print("freq", 1/(time.perf_counter()-t1))
            t1 = time.perf_counter()
        

    def start_call_every_thread(seconds, callback):
        stop_event = threading.Event()
        thread = threading.Thread(target=call_every, args=(seconds, callback, stop_event), daemon=True)
        thread.start()
        return stop_event
    
    action_stop_event = start_call_every_thread(0.02, infer_action_callback)
    save_flag = LOG_EXP

    # stop_event.set()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Ctrl+C pressed")
        
        env.is_running.set()

        print("Exiting...")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    LOG_EXP = False
    args = get_args()
    play(args)
    
