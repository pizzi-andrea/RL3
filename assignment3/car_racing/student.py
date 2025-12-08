import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        
        # https://youtu.be/MEt6rrxH8W4
        def layer_init_(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer
        super(Policy, self).__init__()
        super(Policy, self).train(True)
        self.device = device
        
        # hyper-parameters ##########################
        self.N = 8   # envs
        self.M = 512  # trajectory lengths
        self.K = 5    # num actions
        self.I = 500     # train PPO
        
        self.img_size =(96,96)
        self.epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.grad_norm = 0.5
        self.clip_v = 0.2
        self.c2 = 0.02 # entropy coeff 1 - entropy
        self.c1 = 0.5 # value coeff 2  - value
        #############################################

        # CNN backbone ######################################################################
        self.res_blk1 = nn.Sequential(
            layer_init_(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, bias=True)), 
            nn.ReLU(),
        )

        self.res_blk2 = nn.Sequential(
            layer_init_(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=True)),
            nn.ReLU(),
        )

        self.res_blk3 = nn.Sequential(
            layer_init_(nn.Conv2d (in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=True) ),
            nn.ReLU(),
        )
        #####################################################################################
            
        self.flatten = nn.Sequential(
            nn.Flatten(),
            layer_init_(nn.Linear(4096, 1024)),
            nn.ReLU(),
        )

        # Value function Estimator ########################
        self.V = nn.Sequential(
            layer_init_(nn.Linear(in_features=1024, out_features=512)),
            nn.Tanh(),
            layer_init_(nn.Linear(in_features=512, out_features=128)),
            nn.Tanh(),
            layer_init_(nn.Linear(in_features=128, out_features=1), std=1.)

        )
        #####################################################

        # Policy Estimator ##################################
        self.P = nn.Sequential(
            layer_init_(nn.Linear(in_features=1024, out_features=1024)),
            nn.Tanh(),
            layer_init_(nn.Linear(in_features=1024, out_features=512)),
            nn.Tanh(),
            layer_init_(nn.Linear(in_features=512, out_features=self.K), std=0.01)
        )
        ######################################################
        
        self.envs = gym.make_vec(
            'CarRacing-v2',
            num_envs=self.N,
            vectorization_mode="async",
            
            # Wrappers pipline
            wrappers=[ # add in test
                #lambda env: gym.wrappers.GrayScaleObservation(env, keep_dim=True), # require CV
                #lambda env: gym.wrappers.ResizeObservation(env, self.img_size),    # not use
                #lambda env: gym.wrappers.FrameStack(env, num_stack=4)
            ],
            
            # envs costructor parameters
            continuous=self.continuous,
            render_mode='rgb_array'
        )
        

        # ste N parallel envs

        self.o_next, _ = self.envs.reset()
       
        self.o_next = self.to_tensor(self.o_next)
        
        self.d_next = torch.from_numpy(np.zeros(shape=self.N, dtype=bool))
        self.optim = torch.optim.Adam(self.parameters(), lr=3e-4, eps=1e-5) # see p
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=1.0, end_factor=0, total_iters=self.I) # lr scheduler


    def forward(self, x): 
        x = self.res_blk1(x) 
        x = self.res_blk2(x)
        x = self.res_blk3(x)  

      
        embs = self.flatten(x)
        v = self.V(embs)
        policy_logits = self.P(embs)
        
        return policy_logits, v
        #return x
    
    def to_tensor(self, x):
        
        if isinstance(x, np.ndarray):    
            
            if len(x.shape) == 4:
                x = torch.from_numpy(x).to(self.device).float().permute(0,3,1,2)
            elif len(x.shape) == 3:
                x = torch.from_numpy(x).to(self.device).float().permute(2,0,1)
        elif isinstance(x, torch.Tensor):
            if len(x.shape) == 4:
                x = x.to(self.device).float().permute(0,3,1,2)
            elif len(x.shape) == 3:
                x = x.to(self.device).float().permute(2,0,1)
        elif isinstance(x, list) or isinstance(x, tuple):
            
            x = np.array(x)
            if len(x.shape) == 4:
                x = torch.from_numpy(x).to(self.device).float().permute(0,3,1,2)
            elif len(x.shape) == 3:
                x = torch.from_numpy(x).to(self.device).float().permute(2,0,1)
        else:
            raise TypeError("invalid type")


        x = x/255.0
        
        return x

    def act(self, state): # in batch
        state = state[np.newaxis,:] # add batch dimension
        state = self.to_tensor(state)
        a, _, _ = self._act(state)
        return a.item()
        
    def _act(self, state):
        
        policy_logits, v = self.forward(state)
        if not self.training:
            return torch.argmax(policy_logits)
        
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()
        action_log = dist.log_prob(action)
        return action, action_log, v

    def train(self):
    
        for PPO_epoch in range(1, self.I +1):

            print(f"[Start PPO iteration: {PPO_epoch}]")
            
            # buffer D ########################################
            obs_buf = torch.zeros((self.M, self.N, 3, self.img_size[0], self.img_size[1]))
            reward_buf = torch.zeros((self.M, self.N))
            values_buf = torch.zeros((self.M, self.N))
            logp_buf = torch.zeros((self.M, self.N))
            action_buf = torch.zeros((self.M, self.N))
            done_buf = torch.zeros((self.M, self.N))
            ##########################################
            
            
            #print(f"--Start rollout phase envs={self.N}, trajectory={self.M}--")
            # Step 1: Rollout 
            with torch.no_grad():
                # horizont M
                for t in range(self.M):
                    
                    #cache
                    envs_next_o = self.o_next # (num_envs x (3, 96, 96))
                    envs_next_d = self.d_next # (num_envs x 1)

                    # prepare batchs

                    # actor 
                    curent_action, current_action_logp, current_value = self._act(envs_next_o)
                    
                    # remove batch 
                    current_value = torch.squeeze(current_value, dim=1)
                    
                    #rollout phase
                    
                    # parallel execution (True parallel)
                    o, r, terminated, truncated,  infos = self.envs.step(curent_action.numpy())
                    
                    
                    # store all data in buffers
                    dones = terminated | truncated
                    np_dones = np.array(envs_next_d, dtype=np.float32)
                    
                    # save next states

                    #cache
                    o = self.to_tensor(o)
                    self.o_next = o
                    self.d_next = torch.from_numpy(np_dones).float()
                    
                    obs_buf[t]  = envs_next_o
                    reward_buf[t] = torch.from_numpy(r).float()
                    action_buf[t] = curent_action.long()
                    logp_buf[t] = current_action_logp
                    values_buf[t] = current_value
                    
                    # recover last observatio when env is reset
                    for i in range(len(dones)):
                        if dones[i]:
                            if "final_observation" in infos:
                                obs_buf[t][i] = self.to_tensor(infos["final_observation"][i])
                                self.o_next[i] =  o[i]
                                print(f"Reset env {i}, term_normal={dones[i]}")
                    
                    # next value for TD
                
               
                _, _, last_value = self._act(self.o_next)
                last_value = last_value.squeeze() # Assicuriamoci che sia (N,)

                
                # compute Advantages and target return foreach envs and step
           
                advantages = torch.zeros_like(reward_buf)
                last_gae_lam = 0
                for t in reversed(range(self.M)):
                    if t == self.M - 1:
                        next_non_terminal = 1.0 - self.d_next # Se l'ultimo step è done, next_value è ignorato
                        next_val = last_value
                    else:
                        next_non_terminal = 1.0 - done_buf[t+1]
                        next_val = values_buf[t+1]
                    
                    delta = reward_buf[t] + self.gamma * next_val * next_non_terminal - values_buf[t]
                    last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                    advantages[t] = last_gae_lam
                
                


                target_return = advantages + values_buf

                
            
                    
                #print(a_flat.shape)
                #print(logp_flat.shape)
                #print(A.shape)
                #print(R.shape)
                #print(v_buf.shape)

                # flat batch 
                
                obs_buf = obs_buf.flatten(0,1)
                done_buf = done_buf.flatten(start_dim=0)
                values_buf = values_buf.flatten(start_dim=0)
                action_buf = action_buf.flatten().long()
                logp_buf = logp_buf.flatten(start_dim=0)
                reward_buf = reward_buf.flatten(start_dim=0)
                advantages = advantages.flatten(start_dim=0)
                target_return = target_return.flatten(start_dim=0) 
                print(f"[GAE={advantages.mean()}]\n MR=[{reward_buf.mean()}]")
            
            # step 2: Learning Phase
            dataset = torch.utils.data.TensorDataset(obs_buf, action_buf, logp_buf, advantages, target_return, values_buf)
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
            EPOCHS = 5
            
            # sub-step 2.1 Train for n epochs
            for epoch in range(1, EPOCHS +1):
 
                #print(f"[train epoch: {epoch}]")
                loss_epoch = 0
                c = 0
                for obs, action, logp, advantage, target_return, value in loader:
                    
                    obs = obs.to(self.device)
                    action = action.to(self.device)
                    logp = logp.to(self.device)
                    advantage = advantage.to(self.device)
                    target_return = target_return.to(self.device)
                    
                    #advantage = (advantage - torch.mean(advantage))/(torch.std(advantage) + 1e-8)
                    # ratio computation
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    policy_logits, current_values = self.forward(obs)
                    current_values = current_values.flatten()
                    dist = torch.distributions.Categorical(logits=policy_logits)
                    
                    current_policy_logp = dist.log_prob(action)
                    ratio = torch.exp(current_policy_logp - logp)
                    #print(ratio)
                    
                    
                    # policy loss
                    policy_ratio = advantage * ratio
                    policy_clip = advantage * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    policy_loss = -torch.min(policy_ratio, policy_clip).mean()
                    
                    # Value loss
                    critic_loss_clip = value + torch.clamp(current_values - value, -self.clip_v, self.clip_v) 
                    critic_loss = F.mse_loss(target_return, critic_loss_clip)

                    # entropy term
                    entropy_loss  = -dist.entropy().mean()
                    
                    loss = policy_loss + self.c1 * critic_loss + self.c2 * entropy_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # TODO

                    self.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
                    self.optim.step()
                    
                    
                    c += 1
                    loss_epoch += loss.item()
                    #print(f"mean entropy batch {-entropy_loss.item()}")
                #print (f"[epoch {epoch} loss: {loss_epoch/c}]")
            self.scheduler.step()
        return 

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
        
    
    
