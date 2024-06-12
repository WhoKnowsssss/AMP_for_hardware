import sys
import time
import threading
import queue

import numpy as np
import isaacgym
import torch
from matplotlib import pyplot as plt

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, loadModel, Gamepad, Hand
from legged_gym.envs.diffusion.diffusion_policy import DiffusionTransformerLowdimPolicy
from legged_gym.envs.diffusion.diffusion_env_wrapper_new import DiffusionEnvWrapper, N_ACTION_STEPS
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer



ACTION_UPDATE_FREQUENCY = 50
DIFFUSE_UPDATE_FREQUENCY = 100




class logger_config:
    EXPORT_POLICY=False
    RECORD_FRAMES=False
    robot_index=0
    joint_index=1

def add_input(input_queue, stop_event):
    while not stop_event.is_set():
        input_queue.put(sys.stdin.read(1))

def play(args):
    # gamepad = Gamepad(0)
    # gamepad.start()


    checkpoint: str = args.checkpoint
    normalizer_ckpt_name = checkpoint.replace('.plan', '_config_dict.pt') #"./checkpoints/converted_config_dict.pt"

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    timestr = time.strftime("%m%d-%H%M%S")
    logdir = f'experiement_log_{timestr}'
    # os.mkdir(logdir)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # load policy
    model = loadModel(args.checkpoint)
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small", # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
        clip_sample=True, # required when predict_epsilon=False
        prediction_type="epsilon" # or sample

    )


    # noise_scheduler = DDIMScheduler(
    #     num_train_timesteps= 10,
    #     beta_start= 0.0001,
    #     beta_end= 0.02,
    #     # beta_schedule is important
    #     # this is the best we found
    #     beta_schedule= 'squaredcos_cap_v2',
    #     clip_sample= True,
    #     set_alpha_to_one= True,
    #     steps_offset= 0,
    #     prediction_type= 'epsilon' # or sample

    # )
    normalizer = LinearNormalizer()
    config_dict, normalizer_ckpt = torch.load(normalizer_ckpt_name)
    normalizer._load_from_state_dict(normalizer_ckpt, 'normalizer.', None, None, None, None, None)
    horizon = config_dict['horizon']
    obs_dim = 45
    action_dim = env.num_actions
    n_action_steps = N_ACTION_STEPS
    n_obs_steps = config_dict['n_obs_steps']
    num_inference_steps=10
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


    # TODO: set the frequency of diffusion policy here. 
    # The frequency should be 30 / n_action_steps
    # diff_event = IntervalTimer(1. / DIFFUSE_UPDATE_FREQUENCY, infer_diffusion_callback) 


    # diff_event.start()
    idx = 0
    try:
        while True:
            # if idx % 10 == 0:
                # velx = np.abs(gamepad.getY(Hand.left)) * 0.7 + 0.3
                # env.set_command(velx)

            # print("outer: ", env.c_wrapper.stepDiffusionFlag)
            if env.c_wrapper.stepDiffusionFlag == 1:
                env.c_wrapper.stepDiffusionFlag = 0
                # s2 = time.perf_counter()
                env.step_diffusion_new()
                # print("diff time:", (time.perf_counter() - s2))
            # time.sleep(0.001)
            idx += 1
    except KeyboardInterrupt:
        exit()
        # gamepad.stop()
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    LOG_EXP = False
    args = get_args()
    play(args)
    
