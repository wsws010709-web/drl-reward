import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class NetworkBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class MLP(NetworkBase):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=128,
                 layer_num=2,
                 activation_function=torch.relu,
                 last_activation=None):
        super().__init__()
        self.activation = activation_function
        self.last_activation = last_activation

        dims = [input_dim] + [hidden_dim] * max(0, layer_num - 1) + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.network_init()

    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x


class Critic(MLP):
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 hidden_dim=128,
                 layer_num=2,
                 activation_function=torch.relu,
                 last_activation=None):
        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         hidden_dim=hidden_dim,
                         layer_num=layer_num,
                         activation_function=activation_function,
                         last_activation=last_activation)

    def forward(self, *x):
        if len(x) == 1:
            x = x[0]
        else:
            x = torch.cat(x, dim=-1)
        return super().forward(x)


class StateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim, activation_function=torch.relu):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, encode_dim)
        self.activation_function = activation_function
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = self.activation_function(self.fc1(state))
        x = self.activation_function(self.fc2(x))
        x = self.activation_function(self.fc3(x))
        return x


class ActionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim, activation_function=torch.relu):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, encode_dim)
        self.activation_function = activation_function
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, action):
        x = self.activation_function(self.fc1(action))
        x = self.activation_function(self.fc2(x))
        x = self.activation_function(self.fc3(x))
        return x


class ForwardModel(nn.Module):
    def __init__(self, encode_dim, output_dim=1, hidden_dim=128, activation_function=torch.relu, last_activation=None):
        super().__init__()
        self.fc1 = nn.Linear(encode_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation_function = activation_function
        self.last_activation = last_activation
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state_embedding, action_embedding):
        x = torch.cat([state_embedding, action_embedding], dim=-1)
        x = self.activation_function(self.fc1(x))
        x = self.fc2(x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x


class Reward(nn.Module):
    """
    Learned dense reward R_omega(s, a).

    In this task, state should be a graph-level state embedding and action should
    be the embedding of the selected edge produced by the policy encoder.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=128,
                 encode_dim=128,
                 output_dim=1,
                 activation_function=torch.relu,
                 last_activation=None):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim, hidden_dim, encode_dim, activation_function)
        self.action_encoder = ActionEncoder(action_dim, hidden_dim, encode_dim, activation_function)
        self.forward_model = ForwardModel(encode_dim, output_dim, hidden_dim, activation_function, last_activation)

    def forward(self, state, action):
        state_embedding = self.state_encoder(state)
        action_embedding = self.action_encoder(action)
        reward = self.forward_model(state_embedding, action_embedding)
        return reward
