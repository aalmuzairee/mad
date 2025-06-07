import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Encoder(nn.Module):
    def __init__(self, state_shape, obs_shape):
        super().__init__()

        assert len(obs_shape) == 4 # [View, Channel, Height, Width]
        self.num_views = obs_shape[0]
        self.num_channels = obs_shape[1]
        self.repr_dim = 49 * 64 

        self.aug = utils.RandomShiftsAug(pad=4)
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[1], 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        self.apply(utils.weight_init)

    def forward(self, state, obs, single_view_aug=False):
        b, num_views, c, h, w = obs.shape
        obs = obs.view(b*num_views, c, h, w)
        obs = self.aug(obs)
        obs = obs / 255.0 - 0.5
        x = self.conv(obs)
        x = x.view(b, num_views, -1)
        x_multi_view = x.sum(dim=1)
        # MAD
        if single_view_aug == True:
            x_single_view = x.transpose(0, 1).reshape(num_views * b, -1)
            x_multi_view = torch.cat([x_multi_view, x_single_view], dim=0)
        return x_multi_view
    

class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, state_shape, repr_dim, action_shape, feature_dim, hidden_dim, log_std_bounds):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(feature_dim + state_shape, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 2*action_shape[0]))
        self.log_std_bounds = log_std_bounds
  

        self.apply(utils.weight_init)

    def forward(self, state, obs):
        h_image = self.trunk(obs)
        h = torch.cat([state, h_image], dim=1)
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

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0] + state_shape, hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0] + state_shape, hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, state, obs, action):
        h_image = self.trunk(obs)
        h_action = torch.cat([h_image, action, state], dim=-1)
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

        self.train()
        self.critic_target.train()

        # MAD params
        self.num_views = cfg.obs_shape[0]
        self.mad_alpha = cfg.mad_alpha
        self.mad_beta = (1 - self.mad_alpha) 
        self.num_repetitions = self.num_views + 1 # repeat once for each singular view, and an extra one for all views combined


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

        # MAD Loss
        target_Q = torch.cat([target_Q, target_Q], dim=0)
        action = torch.cat([action] * self.num_repetitions, dim=0)
        state = torch.cat([state] * self.num_repetitions, dim=0)

        # get current Q estimates
        Q1, Q2 = self.critic(state, obs, action)

        # MAD Loss - Critic
        chunked_Q1 = Q1.chunk(self.num_repetitions, dim=0)
        chunked_Q2 = Q2.chunk(self.num_repetitions, dim=0)
        combined_Qs = [torch.cat([c1, c2], dim=0) for c1, c2 in zip(chunked_Q1, chunked_Q2)]
        Q_reg = combined_Qs[0] # (batch_size, 2*Q)
        Q_augs = torch.cat(combined_Qs[1:], dim=0) # (num_views*batch_size, 2*Q)
        critic_reg_loss = (F.mse_loss(Q_reg, target_Q) * 2) * self.mad_alpha
        critic_aug_loss = (F.mse_loss(Q_augs, target_Q.repeat(self.num_views, 1)) * 2) * self.mad_beta
        critic_loss = critic_reg_loss + critic_aug_loss

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_reg_loss'] = critic_reg_loss.item()
        metrics['critic_aug_loss'] = critic_aug_loss.item()
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

        # MAD Loss
        chunked_obs = obs.chunk(self.num_repetitions, dim=0)
        obs_combined = chunked_obs[0]
        critic_obs = torch.cat([obs_combined] * self.num_repetitions, dim=0)
        state = torch.cat([state] * self.num_repetitions, dim=0)

        dist = self.actor(state, obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(state, critic_obs, action)
        Q = torch.min(Q1, Q2)

        # MAD Loss - Actor
        chunked_log_prob = log_prob.chunk(self.num_repetitions, dim=0)
        chunked_Q = Q.chunk(self.num_repetitions, dim=0)
        reg_log_prob = chunked_log_prob[0]
        aug_log_prob = torch.cat(chunked_log_prob[1:], dim=0)
        reg_Q = chunked_Q[0]
        aug_Q = torch.cat(chunked_Q[1:], dim=0)

        actor_loss_reg = (self.alpha.detach() * reg_log_prob - reg_Q).mean()
        actor_loss_aug = (self.alpha.detach() * aug_log_prob - aug_Q).mean()
        actor_loss = (actor_loss_reg * self.mad_alpha) + (actor_loss_aug * self.mad_beta)

        # optimize the actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # MAD Loss - Alpha
        alpha_loss_reg = (self.alpha * (-reg_log_prob - self.target_entropy).detach()).mean()
        alpha_loss_aug = (self.alpha * (-aug_log_prob - self.target_entropy).detach()).mean()
        alpha_loss = (alpha_loss_reg * self.mad_alpha) + (alpha_loss_aug * self.mad_beta)

        # optimize temperature
        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        metrics['actor_loss_reg'] = actor_loss_reg.item()
        metrics['actor_loss_aug'] = actor_loss_aug.item()
        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_entropy'] = -log_prob.mean().item()
        metrics['actor_target_entropy'] = self.target_entropy
        metrics['alpha_loss_reg'] = alpha_loss_reg.item()
        metrics['alpha_loss_aug'] = alpha_loss_aug.item()
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha_value'] = self.alpha

        return metrics


    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        state, obs, action, reward, discount, next_state, next_obs = utils.to_torch(
            batch, self.device)

        # encode
        obs = self.encoder(state, obs, single_view_aug=True)
        with torch.no_grad():
            next_obs = self.encoder(next_state, next_obs)

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(state, obs, action, reward, discount, next_state, next_obs))

        # update actor
        metrics.update(self.update_actor(state.detach(), obs.detach()))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics