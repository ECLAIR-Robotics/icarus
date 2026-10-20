import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from policy import CartPoleActorCritic
from warp_env import MarsWarpEnv


@dataclass(frozen=True)
class PPOConfig:
    nworld: int = 512
    rollout_steps: int = 128
    iterations: int = 500
    update_epochs: int = 4
    minibatch_size: int = 4096
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 0


@dataclass
class Rollout:
    observations: torch.Tensor
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    mean_reward: float
    done_rate: float


def collect_rollout(
    env: MarsWarpEnv,
    policy: CartPoleActorCritic,
    observation: torch.Tensor,
    config: PPOConfig,
) -> tuple[Rollout, torch.Tensor]:
    device = observation.device
    observations = []
    raw_actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []

    for _ in range(config.rollout_steps):
        with torch.no_grad():
            action, raw_action, log_prob, value = policy.act(observation)
        next_observation, reward, done = env.pole_step(action.cpu().numpy())
        observations.append(observation)
        raw_actions.append(raw_action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.as_tensor(reward, device=device))
        dones.append(torch.as_tensor(done, device=device))
        observation = torch.as_tensor(
            next_observation, dtype=torch.float32, device=device
        )

    with torch.no_grad():
        next_value = policy.value(observation)

    reward_tensor = torch.stack(rewards)
    done_tensor = torch.stack(dones)
    value_tensor = torch.stack(values)
    advantages = torch.empty_like(reward_tensor)
    gae = torch.zeros(config.nworld, device=device)

    for step in reversed(range(config.rollout_steps)):
        not_done = (~done_tensor[step]).float()
        delta = (
            reward_tensor[step]
            + config.gamma * next_value * not_done
            - value_tensor[step]
        )
        gae = delta + config.gamma * config.gae_lambda * not_done * gae
        advantages[step] = gae
        next_value = value_tensor[step]

    rollout = Rollout(
        observations=torch.stack(observations).flatten(0, 1),
        raw_actions=torch.stack(raw_actions).flatten(0, 1),
        old_log_probs=torch.stack(log_probs).flatten(),
        old_values=value_tensor.flatten(),
        returns=(advantages + value_tensor).flatten(),
        advantages=advantages.flatten(),
        mean_reward=reward_tensor.mean().item(),
        done_rate=done_tensor.float().mean().item(),
    )
    return rollout, observation


def update_policy(
    policy: CartPoleActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: PPOConfig,
) -> tuple[float, float]:
    advantages = (rollout.advantages - rollout.advantages.mean()) / (
        rollout.advantages.std() + 1e-8
    )
    sample_count = rollout.observations.shape[0]
    policy_losses = []
    value_losses = []

    for _ in range(config.update_epochs):
        for indices in torch.randperm(
            sample_count, device=rollout.observations.device
        ).split(config.minibatch_size):
            log_prob, entropy, value = policy.evaluate(
                rollout.observations[indices], rollout.raw_actions[indices]
            )
            ratio = (log_prob - rollout.old_log_probs[indices]).exp()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
            policy_loss = -torch.minimum(
                unclipped, clipped * advantages[indices]
            ).mean()

            value_loss = nn.functional.mse_loss(value, rollout.returns[indices])
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def train(config: PPOConfig, checkpoint: Path) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MarsWarpEnv(nworld=config.nworld, pole=True, seed=config.seed)
    policy = CartPoleActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    observation = torch.as_tensor(env.pole_observation(), dtype=torch.float32, device=device)

    for iteration in range(1, config.iterations + 1):
        rollout, observation = collect_rollout(env, policy, observation, config)
        policy_loss, value_loss = update_policy(policy, optimizer, rollout, config)
        print(
            f"{iteration:04d} reward={rollout.mean_reward:.3f} "
            f"falls={rollout.done_rate:.4f} "
            f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f}"
        )
        if iteration % 25 == 0:
            save_checkpoint(checkpoint, policy, optimizer, iteration, config)

    save_checkpoint(checkpoint, policy, optimizer, config.iterations, config)


def save_checkpoint(
    path: Path,
    policy: CartPoleActorCritic,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: PPOConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "config": config,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MARS cartpole policy with PPO."
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--nworld", type=int, default=512)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/cartpole_ppo.pt")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        PPOConfig(iterations=args.iterations, nworld=args.nworld),
        checkpoint=args.checkpoint,
    )
