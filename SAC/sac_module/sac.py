import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from SAC.sac_module.actor import DiagGaussianActor
from SAC.sac_module.critic import DoubleQCritic
# from tensorboardX import SummaryWriter

from SAC.sac_module import utils

import abc


class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def update(self, replay_buffer, logger, step):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class SACAgent(Agent):
    """SAC algorithm."""

    def __init__(self,
                 obs_dim,
                 action_dim,
                 action_range,
                 device=device,
                 # critic_cfg,
                 # actor_cfg,
                 discount=0.99,
                 init_temperature=0.1,
                 alpha_lr=5e-4,
                 alpha_betas=(0.9, 0.999),
                 actor_lr=5e-4,
                 actor_betas=(0.9, 0.999),
                 actor_update_frequency=1,
                 critic_lr=5e-4,
                 critic_betas=(0.9, 0.99),
                 critic_tau=0.005,
                 critic_target_update_frequency=2,
                 batch_size=512,
                 learnable_temperature=True):
        super().__init__()

        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        # self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        # self.critic_target = hydra.utils.instantiate(critic_cfg).to(
        #     self.device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        #
        # self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = DoubleQCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=1024, hidden_depth=2).to(
            self.device)
        self.critic_target = DoubleQCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=1024, hidden_depth=2).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = DiagGaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=1024, hidden_depth=2,
                                       log_std_bounds=(-5, 2)).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, writer,
                      step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        writer.add_scalar('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(writer, step)

    def update_actor_and_alpha(self, obs, writer, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        writer.add_scalar('train_actor/loss', actor_loss, step)
        writer.add_scalar('train_actor/target_entropy',
                          self.target_entropy, step)
        writer.add_scalar('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(writer, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            # logger.log('train_alpha/loss', alpha_loss, step)
            # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, writer, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        writer.add_scalar('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                           writer, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, writer, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
        # Save model parameters

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

        # Load model parameters

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(
                actor_path, map_location=torch.device('cpu')))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(
                critic_path, map_location=torch.device('cpu')))
