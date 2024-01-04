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

import signal
import sys

def sigint_handler(signum, frame):
	print("  Ctrl + C Exit!")
	sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler) 

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
    # train_cfg.runner.resume = True
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

    s = time.perf_counter()


    # with open('../test_results_plotting/real_obs_0503_5.pkl', 'rb') as f:
    #     import pickle
    #     gz_obss, _ = pickle.load(f)
    #     gz_obss = torch.tensor(gz_obss)
        
    # asynchronous step
    # def receive_obs_callback():
    #     # update obs. calculate reward
    #     nonlocal env, idx, s
    #     # env.raw_observation=env._udp.receive_observation()
    #     # s = time.perf_counter()


    def infer_action_callback():
        nonlocal env, policy, obs, obss, idx, s, start_idx, global_idx, stand_override

        global_idx += 1

        # if start_idx < global_idx:
        #     env._state_flag = 1
        # if env._state_flag == 1 and LOG_EXP:
        #     idx += 1
            # obss[idx,:] = obs[0].detach().cpu().numpy()
            # filter_obss[idx,:] = env.final_actions[0]
            # obs[0] = gz_obss[idx]
        # obs, _, rews, dones, infos = env.get_observations()

        if not stand_override:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
        print('freq: ', 1/(time.perf_counter()-s))
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
    action_stop_event = start_call_every_thread(1/30, infer_action_callback)
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
    
