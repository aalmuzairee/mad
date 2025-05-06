# Reimplementation of (https://arxiv.org/pdf/2404.14064) from (https://github.com/uoe-agents/MVD)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from algos.baselines.info_nce import InfoNCE


class Encoder(nn.Module):
    def __init__(self, state_shape, obs_shape):
        super().__init__()

        assert len(obs_shape) == 4 # [View, Channel, Height, Width]
        self.num_views = obs_shape[0]
        self.repr_dim = 50 * 2 # 2 for number of encoders

        self.aug = utils.RandomShiftsAug(pad=4)
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(obs_shape[1], 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(49 * 64, 50), nn.LayerNorm(50), nn.Tanh()
        )
        self.private_encoder = nn.Sequential(
            nn.Conv2d(obs_shape[1], 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(49 * 64, 50), nn.LayerNorm(50), nn.Tanh()
        )


        self.apply(utils.weight_init)

    def forward(self, state, obs, mvd_loss=False):
        b, num_views, c, h, w = obs.shape
        obs = obs.view(b*num_views, c, h, w)
        obs = self.aug(obs)
        obs = obs / 255.0 - 0.5
        h_shared = self.shared_encoder(obs.clone())
        h_private = self.private_encoder(obs.clone())
        if mvd_loss:
            z = torch.cat([h_shared, h_private], dim=-1) 
            z = z.view(b, num_views, -1) 
        # Selection Logic from MVD: https://github.com/uoe-agents/MVD/blob/main/algorithms/models.py#L288
        # ---------------------------------------------
        h_shared = h_shared.view(b, num_views, -1)
        h_private = h_private.view(b, num_views, -1)
        idx_shared = torch.randperm(num_views)[0]
        idx_private = torch.randperm(num_views)[0]
        h_shared = h_shared[:, idx_shared]
        h_private = h_private[:, idx_private]
        # ---------------------------------------------
        x = torch.cat((h_shared, h_private), dim=-1)
        if mvd_loss:
            x = (x, z)
        return x
    

class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, state_shape, repr_dim, action_shape, feature_dim, hidden_dim, log_std_bounds):
        super().__init__()
        
        self.policy = nn.Sequential(nn.Linear(repr_dim + state_shape, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 2*action_shape[0]))
        self.log_std_bounds = log_std_bounds
  

        self.apply(utils.weight_init)

    def forward(self, state, obs):
        h = torch.cat([state, obs], dim=1)
        mu, log_std = self.policy(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, state_shape, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0] + state_shape, hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0] + state_shape, hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, state, obs, action):
        h_action = torch.cat([obs, action, state], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class AGENT:
    def __init__(self, cfg):

        self.cfg = cfg
        self.device = cfg.device
        self.critic_target_tau = cfg.critic_target_tau
        self.update_every_steps = cfg.update_every_steps
        self.num_expl_steps = cfg.num_expl_steps
        
        # models
        self.encoder = Encoder(cfg.state_shape, cfg.obs_shape).to(cfg.device)
        print(self.encoder)

        self.actor = Actor(cfg.state_shape, self.encoder.repr_dim, cfg.action_shape, 
                           cfg.feature_dim, cfg.hidden_dim, cfg.log_std_bounds).to(cfg.device)
        self.critic = Critic(cfg.state_shape, self.encoder.repr_dim, cfg.action_shape, 
                             cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.critic_target = Critic(cfg.state_shape, self.encoder.repr_dim, cfg.action_shape, 
                                    cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(cfg.device)
        self.log_alpha.requires_grad = True
        
        # set target entropy to -|A|
        self.target_entropy = -cfg.action_shape[0]

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.lr) 

        # MVD optimizer
        self.mvd_opt =  torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state, obs, step, eval_mode):
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        obs = self.encoder(state, obs)
        dist = self.actor(state, obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)

        return action.cpu().detach().numpy()[0]


    def update_critic(self, state, obs, action, reward, discount, next_state, next_obs):
        metrics = dict()
        with torch.no_grad():
            dist = self.actor(next_state, next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_state, next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)

        # get current Q estimates
        Q1, Q2 = self.critic(state, obs, action)
        critic_loss = (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)) 

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics


    def update_actor(self, state, obs):
        metrics = dict()

        dist = self.actor(state, obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(state, obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        # optimize the actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # optimize temperature
        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_entropy'] = -log_prob.mean().item()
        metrics['actor_target_entropy'] = self.target_entropy
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha_value'] = self.alpha

        return metrics
    

    def update_mvd(self, state, obs, next_state, next_obs):
        metrics = dict()

        _, z = self.encoder(state, obs, mvd_loss=True)
        with torch.no_grad():
            _, next_z = self.encoder(next_state, next_obs, mvd_loss=True)
        mvd_loss = self.get_mvd_loss(z, next_z)
        self.mvd_opt.zero_grad(set_to_none=True)
        mvd_loss.backward()
        self.mvd_opt.step()
        metrics['mvd_loss'] = mvd_loss.item()

        return metrics


    def get_mvd_loss(self, z, next_z):
        z_shared, z_private = z.chunk(2, dim=-1)
        next_z_shared, next_z_private = next_z.chunk(2, dim=-1)
        num_cameras = z.shape[1]

        shared_loss = InfoNCE(negative_mode="mixed")
        private_loss = InfoNCE(negative_mode="paired")

        pos_idx = np.random.randint(num_cameras)
        anchor_idxs = [i for i in range(num_cameras) if i != pos_idx]
        multi_view_loss = 0

        for anchor_idx in anchor_idxs:
            shared_query = z_shared[:, anchor_idx]
            private_query = z_private[:, anchor_idx]
    
            # No target encoders in our implementation, we detach the keys below 
            # (https://github.com/uoe-agents/MVD/blob/main/algorithms/multi_view_disentanglement.py#L55)
            shared_positive_key = z_shared[:, pos_idx].detach()
            private_positive_key = next_z_private[:, anchor_idx].detach()

            shared_negative_key = z_private.detach()
            private_negative_key = z_private[:, [pos_idx]].detach()

            multi_view_loss += shared_loss(shared_query, shared_positive_key, shared_negative_key)
            multi_view_loss += private_loss(private_query, private_positive_key, private_negative_key)

        multi_view_loss /= len(anchor_idxs)
        return multi_view_loss



    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        state, obs, action, reward, discount, next_state, next_obs = utils.to_torch(
            batch, self.device)

        # encode
        encoded_obs = self.encoder(state, obs, mvd_loss=False)
        with torch.no_grad():
            encoded_next_obs = self.encoder(next_state, next_obs, mvd_loss=False)

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(state, encoded_obs, action, reward, discount, next_state, encoded_next_obs))

        # update actor
        metrics.update(self.update_actor(state, encoded_obs.detach()))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        # update mvd
        metrics.update(self.update_mvd(state, obs, next_state, next_obs))


        return metrics