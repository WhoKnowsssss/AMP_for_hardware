from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import threading

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time
import sys
import threading
import queue

from matplotlib import pyplot as plt

from legged_gym.envs.diffusion.diffusion_policy import DiffusionTransformerLowdimPolicy
from legged_gym.envs.diffusion.diffusion_env_wrapper import DiffusionEnvWrapper
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer

from trt_model import TRTModel

class logger_config:
    EXPORT_POLICY=False
    RECORD_FRAMES=False
    robot_index=0
    joint_index=1

def add_input(input_queue, stop_event):
    while not stop_event.is_set():
        input_queue.put(sys.stdin.read(1))

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False


    use_trt_acceleration = True

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
    if use_trt_acceleration:
        model = TRTModel("./checkpoints/model.plan")
    else:
        # converted_model.pt already contains the trained weights
        model = torch.load("./checkpoints/converted_model.pt")
        model.eval()
    
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
    original_ckpt = torch.load("./checkpoints/latest.ckpt")
    original_ckpt = {k: v for k, v in original_ckpt['state_dicts']['model'].items() if "normalizer" in k}
    normalizer._load_from_state_dict(original_ckpt, 'normalizer.', None, None, None, None, None)
    horizon = 16
    obs_dim = env.num_obs
    action_dim = env.num_actions
    n_action_steps = 6
    n_obs_steps = 8
    num_inference_steps=20
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


    idx = 0
    global_idx = 0
    start_idx = np.inf
    stand_override = False

    s = time.perf_counter()

    def infer_action_callback():
        nonlocal env, policy, obs, idx, s, start_idx, global_idx, stand_override

        global_idx += 1
    
        env.step()
        # print('freq: ', 1/(time.perf_counter()-s))
        s = time.perf_counter()
    

    def call_every(seconds, callback, stop_event):
        t1 = time.perf_counter()
        t2 = time.perf_counter()
        while not stop_event.wait(seconds - (t1-t2)):
            t2 = time.perf_counter()
            callback()
            # print("freq", callback, 1/(time.perf_counter()-t1))
            t1 = time.perf_counter()
            # print('wait, ', seconds - (t1-t2))

    def start_call_every_thread(seconds, callback):
        stop_event = threading.Event()
        thread = threading.Thread(target=call_every, args=(seconds, callback, stop_event), daemon=True)
        thread.start()
        return stop_event

    # receive_stop_event = start_call_every_thread(1/1000, receive_obs_callback)

    # TODO: set the frequency of diffusion policy here. 
    # The frequency should be 30 / n_action_steps
    action_stop_event = start_call_every_thread(1/5, infer_action_callback) 
    save_flag = LOG_EXP

    # stop_event.set()
    while True:
        # if idx > 600:
        #     env.a = 0.05
        # # if idx > 600:
        # #     env.a = 0.1
        # else:
        #     env.a = 0.0
        time.sleep(100)
        # env.raw_observation=env._udp.receive_observation()
        # time_now = int(time.time() * 1000)
        # print("time now = ",time_now,"   time udp = ",data)
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    LOG_EXP = False
    args = get_args()
    play(args)
    
