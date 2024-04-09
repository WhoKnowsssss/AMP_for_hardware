import torch         
import time


class DiffusionEnvWrapper:
    def __init__(self, env, policy, n_obs_steps=8, n_action_steps=8):

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.policy = policy
        history = self.n_obs_steps
        device = 'cuda:0'
        self.state_history = torch.zeros((env.num_envs, history+1, 45), dtype=torch.float32, device=device)
        self.action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=torch.float32, device=device)
        self.env = env
        obs = env.get_observations()

        self.state_history[:,:,:] = env.get_diffusion_observation().to(device)

        self.diffusion_action_queues = torch.zeros((env.num_envs, n_action_steps, env.num_actions), dtype=torch.float32, device=device)
        self.diffusion_action_queues_new = torch.zeros((env.num_envs, n_action_steps, env.num_actions), dtype=torch.float32, device=device)

      
        self.idx = 0

        self.step_diffusion_flag = True

    
    def step(self):
        history = self.n_obs_steps
        obs_dict = {'obs': self.state_history[:,:-1]}
        action_dict = self.policy.predict_action(obs_dict)
        pred_action = action_dict['action_pred']
       
        action = pred_action[:,history:history+self.n_action_steps,:]

        start_t = time.perf_counter()

        # step env
        for i in range(self.n_action_steps):
            action_step = action[:,i,:]
            obs, _, rews, dones, infos,  = self.env.step(action_step.detach())
        
            self.state_history = torch.roll(self.state_history, shifts=-1, dims=1)
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.state_history[:,-1,:] = self.env.get_diffusion_observation().to(self.env.device)
            self.action_history[:,-1,:] = action_step
            # single_obs_dict = {'obs': self.state_history[:,-1,:].to('cuda:0')}

            elapsed_t = time.perf_counter() - start_t
            freq = 1/elapsed_t
            print("step freq: ", freq)


    def step_action(self):
        self.idx = self.idx % self.n_action_steps
        self.idx = 0
        if self.idx == 0:
            self.diffusion_action_queues[:] = self.diffusion_action_queues_new[:]
            # print("load new actions")
        
        action_step = self.diffusion_action_queues[:, self.idx, :]
        # print("current idx: ", self.idx, " took action")

        obs, _, rews, dones, infos, _, _ = self.env.step(action_step.detach())
        # print("current idx: ", self.idx, " step")

    
        self.state_history = torch.roll(self.state_history, shifts=-1, dims=1)
        self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
        self.state_history[:,-1,:] = self.env.get_diffusion_observation().to(self.device)
        self.action_history[:,-1,:] = action_step

        if self.idx == self.n_action_steps - 2:
            self.step_diffusion_flag = True

        self.idx += 1

    def step_diffusion(self):

        history = self.n_obs_steps
        obs_dict = {'obs': self.state_history[:,1:]}
        # print("new actions start")
        action_dict = self.policy.predict_action(obs_dict)
        pred_action = action_dict['action_pred']
       
        self.diffusion_action_queues_new[:] = pred_action[:,history:history+self.n_action_steps,:]
        # print("new actions done")
