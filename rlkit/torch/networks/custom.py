"""
Random networks
"""
from rlkit.torch.core import PyTorchModule
import torch
from torch import nn
import torch.nn.functional as F
from rlkit.torch import pytorch_util as ptu

class LinearMlp(PyTorchModule):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        self.to(ptu.device)
        

    def forward(self, x):
        return self.fc(x)


class AutoEncoderMlp(PyTorchModule):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.e1 = nn.Linear(state_dim + action_dim, 256)
        self.e2 = nn.Linear(256, 256)

        self.r1 = nn.Linear(256, 1, bias=False)

        self.a1 = nn.Linear(256, 256)
        self.a2 = nn.Linear(256, action_dim)

        self.d1 = nn.Linear(256, 256)
        self.d2 = nn.Linear(256, state_dim)
        self.to(ptu.device)

    def forward(self, obs, action):
        x = F.relu(self.e1(torch.cat([obs, action], axis=1)))
        x = F.relu(self.e2(x))
        # x now is the latent representation

        # reward prediction
        reward_prediction = self.r1(x)

        # action reconstruction
        action_rec = F.relu(self.a1(x))
        action_rec = self.a2(action_rec)

        # next state prediction
        next_state_prediction = F.relu(self.d1(x))
        next_state_prediction = self.d2(next_state_prediction)

        return next_state_prediction, action_rec, reward_prediction

    def latent(self, obs, action):
        x = F.relu(self.e1(torch.cat([obs, action], axis=1)))
        x = F.relu(self.e2(x))
        return x