import torch
from torch import nn
from torch.distributions import Normal

from pole import ACTION_DIM, OBS_DIM


class CartPoleActorCritic(nn.Module):
    """PPO actor and critic for the pole's observation (pole.OBS_DIM) and action (pole.ACTION_DIM)."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )
        self.critic = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.5))

    def distribution(self, observation: torch.Tensor) -> Normal:
        return Normal(self.actor(observation), self.log_std.exp())

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    def act(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        raw_action = distribution.sample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        return action, raw_action, log_prob, self.value(observation)

    def evaluate(
        self, observation: torch.Tensor, raw_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, self.value(observation)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observation))


CartPolePolicy = CartPoleActorCritic