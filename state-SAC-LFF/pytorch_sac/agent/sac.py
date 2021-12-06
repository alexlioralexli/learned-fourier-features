import numpy as np
import torch
import torch.nn.functional as F
import abc
import pytorch_sac.utils as utils
from pytorch_sac.agent.critic import DoubleQCritic
from pytorch_sac.agent.actor import DiagGaussianActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train_mode(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def train(self, replay_buffer, batch_size=256):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""


class SACAgent(Agent):
    """SAC algorithm."""

    def __init__(self,
                 state_dim,
                 action_dim,
                 action_range,
                 network_class,
                 network_kwargs,
                 discount=0.99,
                 init_temperature=0.1,
                 alpha_lr=1e-4,
                 alpha_betas=(0.9, 0.999),
                 actor_lr=1e-4,
                 actor_betas=(0.9, 0.999),
                 actor_update_frequency=1,
                 critic_lr=1e-4,
                 critic_betas=(0.9, 0.999),
                 tau=0.005,
                 critic_target_update_frequency=2,
                 batch_size=1024,
                 learnable_temperature=True):
        super().__init__()
        print('Pytorch SAC time!')
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        q1 = network_class(state_dim + action_dim, 1, add_tanh=False, **network_kwargs)
        q2 = network_class(state_dim + action_dim, 1, add_tanh=False, **network_kwargs)
        # q1 = utils.mlp(state_dim + action_dim, 1024, 1, 2)
        # q2 = utils.mlp(state_dim + action_dim, 1024, 1, 2)
        q1_target = network_class(state_dim + action_dim, 1, add_tanh=False, **network_kwargs)
        q2_target = network_class(state_dim + action_dim, 1, add_tanh=False, **network_kwargs)
        self.critic = DoubleQCritic(q1, q2).to(self.device)
        self.critic_target = DoubleQCritic(q1_target, q2_target).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # policy_net = utils.mlp(state_dim, 1024, 2 * action_dim, 2)
        policy_net = network_class(state_dim, action_dim * 2, add_tanh=False, **network_kwargs)
        self.actor = DiagGaussianActor(policy_net, log_std_bounds=[-5, 2]).to(self.device)
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

        self.train_mode()
        self.critic_target.train()
        self.step = 0

    def train_mode(self, training=True):
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

    def select_action(self, obs, evaluate=True):
        return self.act(obs, sample=not evaluate)

    def update_critic(self, obs, action, reward, next_obs, not_done):
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

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def train(self, replay_buffer, batch_size=256):
        obs, action, next_obs, reward, not_done_no_max = replay_buffer.sample(batch_size)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max)

        if self.step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs)

        if self.step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

        self.step += 1

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_target = copy.deepcopy(self.actor)