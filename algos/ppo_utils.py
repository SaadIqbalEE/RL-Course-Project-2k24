import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
import  torch

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1_a = torch.nn.Linear(state_space, hidden_size)
        self.fc2_a = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_a_mean = torch.nn.Linear(hidden_size, action_space)
        self.fc3_a_logstd = torch.nn.Linear(hidden_size, action_space)
        
        self.fc1_c = torch.nn.Linear(state_space, hidden_size)
        self.fc2_c = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_c = torch.nn.Linear(hidden_size, 1)
        
        self.log_std = torch.nn.Parameter(torch.ones(action_space))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_a = self.fc1_a(x)
        x_a = F.relu(x_a)
        x_a = self.fc2_a(x_a)
        x_a = F.relu(x_a)
        
        mean = self.fc3_a_mean(x_a)  # Mean vector μ
        
        std = torch.exp( torch.clamp(self.log_std, -2.0, 1.0))  # Ensure positivity of std
        
        # Multivariate Gaussian distribution with diagonal covariance
        action_dist = Independent(Normal(mean, std), 1)
        
        x_c = self.fc1_c(x)
        x_c = F.relu(x_c)
        x_c = self.fc2_c(x_c)
        x_c = F.relu(x_c)
        value = self.fc3_c(x_c)  # Value estimate

        return action_dist, value
    
    def set_logstd_ratio(self, ratio):
        """
        Adjusts the policy's log_std parameter based on the provided ratio.

        Args:
            ratio (float): Ratio used to scale the randomness. Should decrease over time (0 to 1).
        """
        with torch.no_grad():
            self.log_std.data = self.log_std.data * ratio