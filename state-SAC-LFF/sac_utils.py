"""
Taken from D2RL codebase here: https://github.com/pairlab/d2rl/blob/main/sac/sac.py
"""
import math
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class GaussianPolicy(nn.Module):
    """
    Note that this differs from the D2RL approach, which doesn't concatenate x before the final layer.
    We do here, and I think that excluding it would be weird.
    """
    def __init__(self, policy_net, max_action):
        super(GaussianPolicy, self).__init__()
        self.policy_net = policy_net
        self.max_action = max_action

    def forward(self, state):
        x = self.policy_net(state)
        mean, log_std = torch.chunk(x, 2, dim=1)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.max_action
        return action, log_prob, mean