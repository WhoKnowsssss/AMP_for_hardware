import time

import numpy as np
import torch

from legged_gym.envs.real.real_cyberdog2 import RealCyberDog2, UDPRx

from ctypes import *

udp = cdll.LoadLibrary("/home/tk/Desktop/playground/AMP_for_hardware/libudp.so")



N_OBSERVATIONS  = 45
N_ACTIONS = 12

N_ACTION_STEPS = 1
N_OBS_STEPS = 8

class DiffusionWrapper(Structure):
    _fields_ = [
        ("thread_id", c_ulonglong),
        ("action_queue", c_float * N_ACTION_STEPS * N_ACTIONS),
        ("observation_history", c_float * (N_OBS_STEPS+1) * N_OBSERVATIONS),
        ("stepDiffusionFlag", c_uint8),
        ("new_action_queue", POINTER(c_float)),
        ("rx_udp", POINTER(UDPRx))
    ]

class DiffusionEnvWrapper:
    def __init__(self, env, policy, n_obs_steps=8, n_action_steps=8):

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.policy = policy
        history = self.n_obs_steps
        device = env.device
        self.state_history = torch.zeros((env.num_envs, history+1, 45), dtype=torch.float32, device=device)
        self.state_history_numpy = np.zeros((history+1, 45), dtype=np.float32)
        self.action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=torch.float32, device=device)
        actions = np.zeros((n_action_steps, env.num_actions), dtype=np.float32)
        self.env: RealCyberDog2 = env
        obs = env.get_observations()

        self.state_history[:,:,:] = env.get_diffusion_observation().to(device)

        self.diffusion_action_queues = torch.zeros((env.num_envs, n_action_steps, env.num_actions), dtype=torch.float32, device=device)
        self.diffusion_action_queues_new = torch.zeros((env.num_envs, n_action_steps, env.num_actions), dtype=torch.float32, device=device)

      
        self.idx = 0        
        self.c_wrapper = DiffusionWrapper()
        actions_ptr = actions.ctypes.data_as(POINTER(c_float))
        udp.init_diffusion_wrapper(byref(self.c_wrapper), byref(self.env.rx_udp), actions_ptr)
        
    # def step(self):
    #     history = self.n_obs_steps
    #     obs_dict = {'obs': self.state_history[:,:-1]}
    #     action_dict = self.policy.predict_action(obs_dict)
    #     pred_action = action_dict['action_pred']
       
    #     action = pred_action[:,history:history+self.n_action_steps,:]

    #     start_t = time.perf_counter()

    #     # step env
    #     for i in range(self.n_action_steps):
    #         action_step = action[:,i,:]
    #         action_step = self.env.getFilteredAction(action_step)
    #         obs, _, rews, dones, infos  = self.env.step(action_step.detach())
        
    #         self.state_history = torch.roll(self.state_history, shifts=-1, dims=1)
    #         self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
    #         self.state_history[:,-1,:] = self.env.get_diffusion_observation().to(self.env.device)
    #         self.action_history[:,-1,:] = action_step
    #         # single_obs_dict = {'obs': self.state_history[:,-1,:].to('cuda:0')}

    #         elapsed_t = time.perf_counter() - start_t
    #         freq = 1/elapsed_t
    #         print("step freq: ", freq)

    def step_diffusion_new(self):

        history = self.n_obs_steps
        memmove(self.state_history_numpy.ctypes.data, byref(self.c_wrapper.observation_history), self.state_history_numpy.nbytes)

        self.state_history_numpy[:,6:9] = self.env._recv_commands * self.env.commands_scale.cpu().numpy()
        obs_dict = {'obs': torch.from_numpy(self.state_history_numpy).unsqueeze(0).to(self.env.device)[:,1:]}
        # obs_dict = {'obs': self.state_history[:,1:]}
        # print("new actions start")
        # print("obs_dict: ", obs_dict['obs'].shape) 
        # print(torch.all(obs_dict['obs'] == 0, dim=-1))
        print("obs_dict: ", obs_dict['obs'][:,:,0])
        action_dict = self.policy.predict_action(obs_dict)
        pred_action = action_dict['action_pred']
       
        actions = pred_action[:,history:history+self.n_action_steps,:]
        # Print pred_actions
        # print(actions)

        # for i in range(self.n_action_steps):
        #     actions[:,i,:] = self.env.getFilteredAction(actions[:,i,:])

        # set the new address in C
        actions: np.array = actions.detach().cpu().numpy()
        actions_ptr = actions.ctypes.data_as(POINTER(c_float))
        udp.set_new_action_queue(byref(self.c_wrapper), actions_ptr)