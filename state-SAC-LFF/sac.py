"""
Taken from D2RL codebase here: https://github.com/pairlab/d2rl/blob/main/sac/sac.py
"""

import copy
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac_utils import soft_update, GaussianPolicy
from TD3 import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 network_class,
                 network_kwargs,
                 max_action,
                 lr=3e-4,
                 discount=0.99,
                 tau=0.005,
                 alpha=0.1,
                 policy_freq=2,
                 automatic_entropy_tuning=True,
                 dmc=False):

        self.gamma = discount
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = policy_freq
        self.automatic_entropy_tuning = automatic_entropy_tuning

        policy_net = network_class(state_dim, action_dim * 2, add_tanh=False, **network_kwargs)
        self.policy = GaussianPolicy(policy_net, max_action).to(
            device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr, betas=(0.9, 0.999))

        q1 = network_class(state_dim + action_dim, 1, **network_kwargs)
        q2 = network_class(state_dim + action_dim, 1, **network_kwargs)
        self.critic = Critic(q1, q2).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr, betas=(0.9, 0.999))

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = - action_dim
            # self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()  # something weird here
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            if dmc:
                self.alpha_optim = Adam([self.log_alpha], lr=1e-4, betas=(0.9, 0.999))
            else:
                self.alpha_optim = Adam([self.log_alpha], lr=1e-4, betas=(0.9,0.999))

        self.total_it = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self, memory, batch_size=256):
        self.total_it += 1
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if self.total_it % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.policy.state_dict(), filename + "_actor")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_target = copy.deepcopy(self.critic)

        self.policy.load_state_dict(torch.load(filename + "_actor"))
