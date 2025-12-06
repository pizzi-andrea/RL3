import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        
        self.N = 4   # envs
        self.M = 512  # trajectory lengths
        self.K = 5    # num actions
        self.I = 2     # train PPO
        

        self.epsilon = 0.1
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.grad_norm = 0.5
        self.epsilon_v = 0.5


        self.c1 = 0.01 # entropy coeff 1 - entropy
        self.c2 = 0.5 # entropy coeff 2  - value
        # Resnet 8
        self.res_blk1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, bias=True), 
            nn.BatchNorm2d(num_features=(32)),
            nn.ReLU(),
        )

        self.res_blk2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=True),
            nn.BatchNorm2d(num_features=(64)),
            nn.ReLU(),
        )

        self.res_blk3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(num_features=(64)),
            nn.ReLU(),
        )
            
        

        


        self.flatten = nn.Flatten()


        self.V = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.Tanh(),
            nn.Linear(in_features=4096, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=1)

        )

        self.P = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.Tanh(),
            nn.Linear(in_features=4096, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.K)
        )

        self.device = device
        self.envs = [gym.make('CarRacing-v2', continuous=self.continuous, render_mode='rgb_array') for _ in range(self.N)]
        
        self.o_next = np.array([env.reset()[0] for env in self.envs], dtype=np.float32)
        self.d_next = torch.from_numpy(np.zeros(shape=self.N, dtype=bool))
        
        self.params = self.parameters()
        self.optim = torch.optim.Adam(self.params, lr=3e-4, eps=1e-8) # see p


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
        
        if not isinstance(x, torch.Tensor):
            if len(x.shape) == 4:
                state = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2)
            elif len(x.shape) == 3:
                state = torch.from_numpy(x).to(self.device).permute(3, 1, 2)
        else:
            if len(x.shape) == 4:
                state = x.to(self.device).permute(0, 3, 1, 2)
            elif len(x.shape) == 3:
                state = x.to(self.device).permute(3, 1, 2)

        state = state.float() / 255.0
        return state

    def act(self, state): # in batch
        state = state[np.newaxis,:]
        state = self.to_tensor(state)
        a, _, _ = self._act(state)
        return a.item()
        
       
    
    def _act(self, state):
        
        policy_logits, v = self.forward(state)
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()

        action_log = dist.log_prob(action)
        return action, action_log, v

    def train(self):
    
        def GAE(last_values, dones):
            v_arr = v_buf.clone().cpu().numpy()
            r_arr = r_buf.clone().cpu().numpy()
            d_arr = d_buf.clone().cpu().numpy()

            next_values = last_values.clone().cpu().numpy()
            next_non_terminal = 1.0 - dones.clone().cpu().numpy()

            T = len(v_arr)
            adv_buff = np.zeros_like(v_arr)
            last_gae_lam = np.zeros_like(next_values)

            for i in reversed(range(T)):
                if i == T - 1:
                    next_vals = next_values
                    next_non_term = next_non_terminal
                else:
                    next_vals = v_arr[i + 1]
                    next_non_term = 1.0 - d_arr[i + 1]

                delta = r_arr[i] + self.gamma * next_vals * next_non_term - v_arr[i]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_term * last_gae_lam
                adv_buff[i] = last_gae_lam

            return adv_buff.flatten(), v_arr.flatten()

        # D # 
        o_buf = torch.zeros((self.M, self.N, 96, 96, 3))
        r_buf = torch.zeros((self.M, self.N))
        v_buf = torch.zeros((self.M, self.N))
        logp_buf = torch.zeros((self.M, self.N))
        a_buf = torch.zeros((self.M, self.N))
        d_buf = torch.zeros((self.M, self.N))
        # # #
        for PPO_epoch in range(1, self.I +1):

            print(f"[Start PPO iteration: {PPO_epoch}]")
            o_buf[:] = 0.
            r_buf[:] = 0.
            v_buf[:] = 0.
            logp_buf[:] = 0.
            a_buf[:] = 0.
            d_buf[:] = 0.
            
            
            

            # Step 1: Rollout 
            with torch.no_grad():
                # horizont M
                for t in range(self.M):
                    batch_o_t = self.o_next # computed in previus step
                    batch_b_t = self.d_next # 


                    # actor 
                    batch_a_t, batch_log_t, batch_v_t = self._act(self.to_tensor(batch_o_t))

                
                    # store and remove batch
                    v_buf[t, :] = batch_v_t.squeeze()
                    a_buf[t] = batch_a_t.flatten()
                    logp_buf[t, :] = batch_log_t.squeeze()
                    

                    # parallel step execution (for each env)
                    for i in range(self.N):
                        # rollout phase

                        o_next_ti, r_ti, terminated_i, truncated_i, _ = self.envs[i].step(batch_a_t[i].item())
                        
                        # Update buffer with data get by rollout phase
                        o_buf[t,i] = torch.from_numpy(batch_o_t[i])
                        d_buf[t,i] = batch_b_t[i]
                        r_buf[t,i] = torch.tensor(r_ti)

                        # update env i state
                        self.d_next[i] = (truncated_i or terminated_i)
                        
                        # reset env if ended
                        if terminated_i or truncated_i:
                            self.o_next[i] = self.envs[i].reset()[0]
                        else:
                            self.o_next[i] = o_next_ti
                
               
                _, _, last_values = self._act(self.to_tensor(self.o_next))
                last_values = last_values.squeeze() # Assicuriamoci che sia (N,)

            # step 2: Learning Phase


            A, R = GAE(last_values, self.d_next)
            A = torch.from_numpy(A)
            R = torch.from_numpy(R)
            #r_flat = r_buf.flatten()
            v_buf = v_buf.flatten()
            logp_flat = logp_buf.flatten()
            a_flat = a_buf.flatten()
            #d_flat = d_buf.flatten()
            o_flat = o_buf.reshape(self.M * self.N, *o_buf.shape[2:])
            
            
    
            #print(a_flat.shape)
            #print(logp_flat.shape)
            #print(A.shape)
            #print(R.shape)
            #print(v_buf.shape)
            dataset = torch.utils.data.TensorDataset(
                o_flat,
                a_flat,
                logp_flat,
                A,
                R,
                v_buf,
                
            )

            BS = 64
            loader  = torch.utils.data.DataLoader(dataset, batch_size=BS, shuffle=True)
            EPOCHS = 4
            
            # sub-step 2.1 Train for n epochs
            for epoch in range(1, EPOCHS +1):

                
                print(f"[train epoch: {epoch}]")
                loss_epoch = 0
                c = 0
                for o, a, logpa, adv, td, v in loader:
                    logpa = logpa.detach()
                    adv = adv.detach()
                    td = td.detach()
                    v = v.detach()

                    logpa = logpa.to(self.device)
                    adv = adv.to(self.device)
                    td = td.to(self.device)
                    a = a.to(self.device)
                    
                    action_logits, values = self.forward(self.to_tensor(o))
                    dist = torch.distributions.Categorical(logits=action_logits)

                    action_log = dist.log_prob(a)

                    
                    ratio = torch.exp(action_log - logpa)
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    
                    # Policy (critic) loss
                    policy_ratio = adv * ratio
                    policy_clip = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    policy_loss = -torch.min(policy_ratio, policy_clip).mean()

                    # Value loss
                    value_predict = v + torch.clamp(values.squeeze() - v, -self.epsilon_v, self.epsilon_v)
                    value_loss = F.mse_loss(td, value_predict)

                    # Entropy
                    entropy  = -dist.entropy().mean()

                    # final loss 
                    loss = policy_loss + self.c1*entropy +self.c2 * value_loss

                    # surogate clip loss
                    self.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
                    self.optim.step()

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # TODO

                    c += 1
                    loss_epoch += loss.item()
                print (f"[epoch {epoch} loss: {loss_epoch/c}]")
        return 

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
        
    
    
