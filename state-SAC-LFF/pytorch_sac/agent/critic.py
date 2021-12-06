import torch
from torch import nn
import pytorch_sac.utils as utils


class DoubleQCritic(nn.Module):
    """Critic network, employs double Q-learning."""
    def __init__(self, q1, q2):
        super().__init__()
        self.Q1 = q1
        self.Q2 = q2
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2

