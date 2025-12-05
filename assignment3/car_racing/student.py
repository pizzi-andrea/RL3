import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        
        self.N = 3   # envs
        self.M = 256  # trajectory lengths
        self.K = 5    # num actions
        self.I = 10     # train PPO
        

        self.epsilon = 0.1
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.grad_norm = 0.5


        self.c1 = 0.5 # entropy coeff 1
        self.c2 = 0.01 # entropy coeff 2
        # Resnet 8
        self.res_blk1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, bias=True), 
            nn.BatchNorm2d(num_features=(64)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.res_blk2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=(64)),
            nn.ReLU(),
        )

        self.res_blk3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=(64)),
            nn.ReLU(),
        )
            
        self.res_blk4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias=True), 
            nn.BatchNorm2d(num_features=(128)),
            nn.ReLU(),
        )

        self.res_blk5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=(128)),
            nn.ReLU(),
        )

        self.res_blk6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=2, bias=True),
            nn.BatchNorm2d(num_features=(128)),
            nn.ReLU(),
        )

        self.res_blk7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=2, bias=True), 
            nn.BatchNorm2d(num_features=(256)),
            nn.ReLU(),
        )

        self.res_blk8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, bias=True),
            nn.BatchNorm2d(num_features=(256)),
            nn.ReLU(),
        )

        


        self.flatten = nn.Flatten()


        self.V = nn.Sequential(
            nn.Linear(in_features=2304, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=1)

        )

        self.P = nn.Sequential(
            nn.Linear(in_features=2304, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.K)
        )


        
        
       
        self.device = device
        self.envs = [gym.make('CarRacing-v2', continuous=self.continuous, render_mode='rgb_array') for _ in range(self.N)]
        
        self.o_next = np.array([env.reset()[0] for env in self.envs], dtype=np.float32)
        self.d_next = torch.from_numpy(np.zeros(shape=self.N, dtype=bool))
        
        self.params = self.parameters()
        self.optim = torch.optim.Adam(self.params, lr=0.001, eps=1e-8) # see p


    def forward(self, x): 
        
        x = self.res_blk1(x) 
        x = self.res_blk2(x)
        x = self.res_blk3(x)  
        
        residual_2 = x 
        
        x = self.res_blk4(x)
        x = self.res_blk5(x)

        residual_2 = self.res_blk6(residual_2)
        
        x = x + residual_2

        x = self.res_blk7(x)
        x = self.res_blk8(x)
        



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
    
        def GAE():
            o_next = self.to_tensor(self.o_next)
            _, _, v_next = self._act(o_next) # bootstrap
            lamb = self.gae_lambda
            gamma = self.gamma
            M, N = r_buf.shape
            
            # Rimuovi la dimensione superflua (1) da v_next
            v_next = v_next.squeeze(1)  # Ora forma (N,)

            # Aggiungi una dimensione fittizia all'inizio per poter concatenare
            v_next = v_next.unsqueeze(0) # Ora forma (1, N)

            # Concatenazione: Rimuoviamo l'ultima riga di v_buf e aggiungiamo v_next_expanded come l'ultima riga
            # Usiamo -1 per il taglio, assumendo che i passi siano la dim=0
            v_next_buf = torch.cat((v_buf[:-1, :], v_next), dim=0)
            
            # Convert boolean 'done' flags to float, compute logical NOT (1 - done).
            # This ensures the future value term is zeroed out if the episode terminated.
            not_done = torch.logical_not(d_buf).float()
            
            # --- 2. Calculate TD Residuals (Deltas) Vettorized (M, N) ---
            # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            deltas = r_buf + gamma * v_next_buf * not_done - v_buf

            # --- 3. Recursive GAE Calculation (Loop over M, Reverse over N) ---
            advantages = torch.zeros_like(r_buf, dtype=torch.float32)
            last_gae = torch.zeros(M)
            
            term = gamma * lamb

            # Iterate backwards over time steps (N)
            for t in reversed(range(N)):
                # GAE(t) = delta(t) + gamma * lambda * GAE(t+1) * (1 - done(t))
                # The element-wise multiplication efficiently handles all M environments simultaneously
                last_gae = deltas[:, t] + term * not_done[:, t] * last_gae
                advantages[:, t] = last_gae
                
            # --- 4. Calculate Returns for the Critic ---
            # Return_t = Advantage_t + V(s_t)
            returns = advantages + v_buf
            
            # --- 5. Flattening and Return ---
            # Flatten from (M, N) to (M*N) for policy gradient updates
            return advantages.flatten(), returns.flatten()
        
        for PPO_epoch in range(1, self.I +1):

            print(f"[Start PPO iteration: {PPO_epoch}]")
            # D # 
            o_buf = torch.zeros((self.M, self.N, 96, 96, 3))
            r_buf = torch.zeros((self.M, self.N))
            v_buf = torch.zeros((self.M, self.N))
            logp_buf = torch.zeros((self.M, self.N))
            a_buf = torch.zeros((self.M, self.N))
            d_buf = torch.zeros((self.M, self.N))
            # # #

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
                        self.o_next[i] = o_next_ti

            # step 2: Learning Phase
            A, R = GAE()
            
            A = A.flatten()
            R = R.flatten()
            #r_flat = r_buf.flatten()
            v_buf = v_buf.flatten()
            logp_flat = logp_buf.flatten()
            #a_flat = a_buf.flatten()
            #d_flat = d_buf.flatten()
            o_flat = o_buf.reshape(self.M * self.N, *o_buf.shape[2:])
            
            
            

            EPOCHS = 3
            BS = 32
            
            #print(a_flat.shape)
            #print(logp_flat.shape)
            #print(A.shape)
            #print(R.shape)
            #print(v_buf.shape)
            dataset = torch.utils.data.TensorDataset(
                o_flat,
                #a_flat,
                logp_flat,
                A,
                R,
                v_buf,
                
            )

            loader  = torch.utils.data.DataLoader(dataset, batch_size=BS, shuffle=True)
            # sub-step 2.1 Train for n epochs

            
            for epoch in range(1, EPOCHS +1):

                
                print(f"[train epoch: {epoch}]")
                loss_epoch = 0
                c = 0
                for o, logpa, adv, td, v in loader:
                    logpa = logpa.detach()
                    adv = adv.detach()
                    td = td.detach()
                    v = v.detach() 
                    _, act_log, v_n = self._act(self.to_tensor(o))

                    
                    ratio = torch.exp(act_log - logpa)
                    
                    actor_unclip_loss = ratio*adv 
                    actor_clip_loss = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)*adv
                    actor_loss = torch.min(actor_unclip_loss, actor_clip_loss).mean()

                    
                    
                    
                    critic_loss_unclipped = torch.pow(v_n - td, exponent=2)
                    V_clipped = torch.clip(v_n, v - self.epsilon, v + self.epsilon)
                    critic_loss_clipped = torch.pow(V_clipped - td, exponent=2)
                    critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

                    entropy = torch.distributions.Categorical(act_log).entropy().mean()

                    loss = -actor_loss + self.c1*critic_loss -self.c2*entropy
                    self.optim.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_norm)
                    self.optim.step()

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
    
    

    

        

    def loss(self, y , v):

        pass

    def A(self, td_errors):
        td_errors = 0
        
    
    
