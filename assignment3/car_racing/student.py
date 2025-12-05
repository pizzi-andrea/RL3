import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        
        self.N = 10   # envs
        self.M = 128  # trajectory lengths
        self.K = 5    # num actions
        self.I = 10     # train PPO
        

        self.epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.grad_norm = 0.5


        self.c1 = 0.5 # entropy coeff 1
        self.c2 = 0.01 # entropy coeff 2

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

        self.normalizer = nn.BatchNorm1d(num_features=self.N*self.M)
        self.flatten = nn.Flatten()
        self.V = nn.Linear(in_features=256, out_features=1, bias=True, device=device) #critic
        self.P = nn.Linear(in_features=256, out_features=5, bias=True, device=device) # actor
        
       
        self.device = device
        self.envs = [gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human') for _ in range(self.N)]
        
        self.o_next = torch.from_numpy(np.array([env.reset()[0] for env in self.envs], dtype=np.float32))
        self.d_next = torch.from_numpy(np.zeros(shape=self.N, dtype=bool))
        
        self.params = list(self.V.parameters()) + list(self.P.parameters()) + list(self.encoder.parameters())
        self.optim = torch.optim.AdamW(self.params, lr=0.001)


    def forward(self, x): 

        fm = self.encoder(x)
        embs = self.flatten(fm)

        v = self.V(embs)
        policy_logits = self.P(embs)
        
        return policy_logits, v

        #return x
    
    def act(self, state): # in batch 
        state = torch.from_numpy(state).to(self.device).permute(0, 3, 1, 2)
        state = state.float() / 255.0
        policy_logits, v = self.forward(state)
        dist = torch.distributions.Categorical(probs=policy_logits)
        action = dist.sample()

        action_log = dist.log_prob(action)
        return action, action_log, v

    def train(self):
    
        def GAE():
            _, _, v_next = self.act(self.o_next) # bootstrap
            lamb = self.gae_lambda
            gamma = self.gamma
            M, N = r_buf.shape
            
            # --- 1. Prepare V(s_{t+1}) for TD Residual Calculation ---
            # V(s_{t+1}) is V(s_t) shifted, with V(s_T) (v_next) at the end.
            v_next_buf = torch.cat((v_buf[:, 1:], v_next.unsqueeze(1)), dim=1) 
            
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
            o_buf = torch.zeros((self.M, self.N, 96, 96, 3), dtype=torch.uint8)
            r_buf = torch.zeros((self.M, self.N))
            v_buf = torch.zeros((self.M, self.N))
            logp_buf = torch.zeros((self.M, self.N))
            a_buf = torch.zeros((self.M, self.N), dtype=torch.int)
            d_buf = torch.zeros((self.M, self.N), dtype=torch.bool)
            # # #

            # Step 1: Rollout 
            with torch.no_grad():
                # horizont M
                for t in range(self.M):
                    batch_o_t = self.o_next # computed in previus step
                    batch_b_t = self.d_next # 


                    # actor 
                    batch_a_t, batch_log_t, batch_v_t = self.act(batch_o_t)

                
                    # store and remove batch

                    v_buf[t, :] = batch_v_t.squeeze()
                    a_buf[t] = batch_a_t.flatten()
                    logp_buf[t, :] = batch_log_t.squeeze()
                    

                    # parallel step execution (for each env)
                    for i in range(self.N):
                        # rollout phase

                        o_next_ti, r_ti, terminated_i, truncated_i, _ = self.envs[i].step(batch_a_t[i])
                        
                        # Update buffer with data get by rollout phase
                        o_buf[t,i] = batch_o_t[i]
                        d_buf[t,i] = batch_b_t[i]
                        r_buf[t,i] = r_ti

                        # update env i state
                        self.d_next[i] = (truncated_i or terminated_i)
                        self.o_next[i] = o_next_ti

                # step 2: Learning Phase
                A, R = GAE()
                
                A = A.flatten()
                R = R.flatten()
                #r_flat = r_buf.flatten()
                #v_flat = v_buf.flatten()
                logp_flat = logp_buf.flatten()
                a_flat = a_buf.flatten()
                #d_flat = d_buf.flatten()
                o_flat = o_buf.reshape(self.M * self.N, *o_buf.shape[2:])
                
                

                EPOCHS = 8
                BS = 16
                batch = batch.squeeze()
                dataset = torch.utils.data.TensorDataset(
                    o_flat,
                    a_flat,
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
                    for o, a, logp, adv, td, v in loader:

                        a = self.normalizer(a)
                        ratio = torch.exp(logp - logp_flat)
                        
                        actor_unclip_loss = ratio*a 
                        actor_clip_loss = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)*a
                        actor_loss = torch.min(actor_unclip_loss, actor_clip_loss)

                        _, act_log, v_n = self.act(o)
                        
                        
                        critic_loss_unclipped = torch.pow(v_n - R, exponent=2)
                        V_clipped = torch.clip(v_n, v - self.epsilon, v + self.epsilon)
                        critic_loss_clipped = torch.pow(V_clipped - R, exponent=2)
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
        
    
    
