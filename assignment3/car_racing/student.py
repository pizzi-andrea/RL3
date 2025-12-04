import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        
        self.N = 10  # envs
        self.M = 128 # trajectory lengths
        self.K = 5   # num actions
        self.I = 10  # train PPO

        self.epsilon = 0.1
        self.gamma = 0.99
        self.gae_lambda = 0.95

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            
        ) # TODO

        self.normalizer = nn.BatchNorm1d(num_features=1)
        self.flatten = nn.Flatten()
        self.V = nn.Linear(in_features=256, out_features=1, bias=True, device=device) #critic
        self.P = nn.Linear(in_features=256, out_features=5, bias=True, device=device) # actor
        
       
        self.device = device
        self.envs = [gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human') for _ in range(self.N)]
        
        self.o_next = np.array([env.reset()[0] for env in self.envs], dtype=np.float32)
        self.d_next = np.zeros(shape=self.N, dtype=bool)
        
        self.params = list(self.V.parameters()) + list(self.P.parameters()) + list(self.encoder.parameters())
        self.optim = torch.optim.AdamW(self.params, lr=0.001)


    def forward(self, x): 

        fm = self.encoder(x)
        embs = self.flatten(fm)

        v = self.V(embs)
        policy_logits = self.P(embs)
        
        return policy_logits, v

        #return x
    
    def act(self, state):
        state = torch.from_numpy(state).to(self.device).permute(0, 3, 1, 2)
        state = state.float() / 255.0
        policy_logits, v = self.forward(state)
        dist = torch.distributions.Categorical(probs=policy_logits)
        action = dist.sample()

        action_log = dist.log_prob(action)
        return action, action_log, v

    def train(self):
        # Step 1: Rollout

        # D # 
        obs_buf = torch.zeros((self.M, self.N, 96, 96, 3), dtype=torch.uint8)
        rew_buf = torch.zeros((self.M, self.N))
        v_buf = torch.zeros((self.M, self.N))
        logp_buf = torch.zeros((self.M, self.N))
        act_buf = torch.zeros((self.M, self.N))
        done_buf = torch.zeros((self.M, self.N))
        # # #
        with torch.no_grad():
            for t in range(self.M):
                o_t_next = torch.from_numpy(self.o_next).to(self.device)

                # actor 
                a_t,log_t,v_t = self.act(o_t_next)

                obs_buf[t] = self.o_next
                obs_buf[t] = self.o_next
                v_buf[t] = v_t.flatten() 
                logp_buf[t] = log_t.flatten()
                act_buf[t] = a_t
                done_buf[t] = self.d_next

                # parallel step execution
                for i in range(self.N):
                    # rollout phase
                    o_next_single, r_t, terminated, truncated, _ = self.envs[i].step(a_t[i])
                    
                    # Aggiorna il buffer per il prossimo stato t+1
                    self.o_next[i] = o_next_single
                    self.d_next[i] = terminated or truncated
                    rew_buf[t, i] = r_t
                
                
        
        # step 2: Learning Phase

        return 

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    def TD(self, gamma):
        pass 

    def GAE(r, v, d, v_next, d_next, lamb):
        pass 

    def loss(y , v):

        pass
    
