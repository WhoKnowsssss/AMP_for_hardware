import torch         
import time


class DiffusionEnvWrapper:
    def __init__(self, env, policy, n_obs_steps=8, n_action_steps=8):

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.policy = policy
        history = self.n_obs_steps
        device = 'cuda:0'
        self.state_history = torch.zeros((env.num_envs, history+1, env.num_obs), dtype=torch.float32, device=device)
        self.action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=torch.float32, device=device)
        self.env = env
        obs = env.get_observations()

        self.state_history[:,-1,:] = obs

        self.diffusion_action_queues = torch.zeros((env.num_envs, n_action_steps, env.num_actions), dtype=torch.float32, device=device)
        self.diffusion_action_queues_new = torch.zeros((env.num_envs, n_action_steps, env.num_actions), dtype=torch.float32, device=device)

      
        self.idx = 0

        self.step_diffusion_flag = False

    
    def step(self):
        history = self.n_obs_steps
        obs_dict = {'obs': self.state_history[:,:-1]}
        action_dict = self.policy.predict_action(obs_dict)
        pred_action = action_dict['action_pred']
       
        action = pred_action[:,history:history+self.n_action_steps,:]
        if action.shape[1] == 0:
            action = pred_action[:,-1:,:]
        # step env
        for i in range(self.n_action_steps):
            action_step = action[:,i,:]
            obs, _, rews, dones, infos, _, _ = self.env.step(action_step.detach())
        
            self.state_history = torch.roll(self.state_history, shifts=-1, dims=1)
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.state_history[:,-1,:] = obs
            self.action_history[:,-1,:] = action_step
            # single_obs_dict = {'obs': self.state_history[:,-1,:].to('cuda:0')}

            # if (i < self.n_action_steps - 1):
            #     time.sleep(1/30)
            time.sleep(1/30)

    def step_action(self):
        history = self.n_obs_steps
        obs_dict = {'obs': self.state_history[:,1:].cuda()}
        # print("new actions start")
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        pred_action = action_dict['action_pred'].cpu()

        action_step = pred_action[:,-1]
        # print("current idx: ", self.idx, " took action")

        obs, _, rews, dones, infos, _, _ = self.env.step(action_step.detach())
        # print("current idx: ", self.idx, " step")

        self.state_history = torch.roll(self.state_history, shifts=-1, dims=1)
        self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
        self.state_history[:,-1,:] = obs
        self.action_history[:,-1,:] = action_step