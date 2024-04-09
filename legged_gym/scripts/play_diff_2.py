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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from legged_gym.envs.diffusion.bc_lowdim_policy_nsteps import DiffusionTransformerLowdimPolicy
from legged_gym.envs.diffusion.diffusion_env_wrapper import DiffusionEnvWrapper
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer

try:
    from trt_model import TRTModel
except:
    print("Install TRT for real experiments")
def play(args):
    ckpt_name = 'converted'
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False

    train_cfg.runner.amp_num_preload_transitions = 1


    use_trt_acceleration = False


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy 
    # TODO: change to TRT model
    if use_trt_acceleration:
        model = TRTModel("./checkpoints/{}_model.plan".format(ckpt_name))
    else:
        # converted_model.pt already contains the trained weights
        model = torch.load("./checkpoints/{}_model.pt".format(ckpt_name))
        model.eval()
    # model = None
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=20,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small", # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
        clip_sample=True, # required when predict_epsilon=False
        prediction_type="epsilon" # or sample

    )
    normalizer = LinearNormalizer()
    config_dict, normalizer_ckpt = torch.load("./checkpoints/{}_config_dict.pt".format(ckpt_name))
    normalizer._load_from_state_dict(normalizer_ckpt, 'normalizer.', None, None, None, None, None)

    horizon = config_dict['horizon']
    obs_dim = 45
    action_dim = env.num_actions
    n_action_steps = 3
    num_inference_steps=config_dict['num_inference_steps']
    n_obs_steps = config_dict['n_obs_steps']
    obs_as_cond=True
    pred_action_steps_only=False
    policy = DiffusionTransformerLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        normalizer = normalizer,
        horizon=horizon, 
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        n_action_steps=n_action_steps, 
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        obs_as_cond=obs_as_cond,
        pred_action_steps_only=pred_action_steps_only,
    )

    env = DiffusionEnvWrapper(env=env, policy=policy, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

    for i in range(1000000):
        env.step()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
