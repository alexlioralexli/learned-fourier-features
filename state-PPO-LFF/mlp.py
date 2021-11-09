import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_hidden=1, hidden_dim=256, first_dim=0, nonlinearity='tanh',
                 add_tanh=False, add_relu=False):
        super().__init__()

        if nonlinearity == 'relu':
            raise NotImplementedError
            nonlin_cls = nn.ReLU
        elif nonlinearity == 'tanh':
            nonlin_cls = nn.Tanh
        else:
            raise RuntimeError
        first_dim = max(hidden_dim, first_dim)
        layers = [nn.Linear(input_size, first_dim), nonlin_cls()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(first_dim, hidden_dim))
            first_dim = hidden_dim
            layers.append(nonlin_cls())
        layers.append(nn.Linear(first_dim, output_size))
        if add_tanh:
            layers.append(nn.Tanh())
        if add_relu:
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: tensor of shape [batch_size, input_size]
        :return: logits: tensor of shape [batch_size, n_classes]
        """
        x = x.view(len(x), -1)  # flatten
        return self.mlp.forward(x)


class FourierMLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_hidden=1,
                 hidden_dim=256,
                 nonlinearity='tanh',
                 sigma=1.0,
                 fourier_dim=256,
                 train_B=False,
                 concatenate_fourier=False,
                 add_tanh=False,
                 add_relu=False):
        super().__init__()

        # create B
        b_shape = (input_size, fourier_dim // 2)
        self.sigma = sigma
        self.B = nn.Parameter(torch.normal(torch.zeros(*b_shape), torch.full(b_shape, sigma)))
        self.B.requires_grad = train_B
        if nonlinearity == 'relu':
            nonlin_cls = nn.ReLU
        elif nonlinearity == 'tanh':
            nonlin_cls = nn.Tanh
        else:
            raise RuntimeError

        self.concatenate_fourier = concatenate_fourier
        if self.concatenate_fourier:
            mlp_input_dim = fourier_dim + input_size
        else:
            mlp_input_dim = fourier_dim

        # create rest of the network
        layers = [nn.Linear(mlp_input_dim, hidden_dim), nonlin_cls()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nonlin_cls())
        layers.append(nn.Linear(hidden_dim, output_size))
        if add_tanh:
            layers.append(nn.Tanh())
        if add_relu:
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: tensor of shape [batch_size, input_size]
        :return: logits: tensor of shape [batch_size, n_classes]
        """
        x = x.view(len(x), -1)  # flatten
        # create fourier features
        proj = (2 * np.pi) * torch.matmul(x, self.B)
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.concatenate_fourier:
            ff = torch.cat([x, ff], dim=-1)
        return self.mlp.forward(ff)
