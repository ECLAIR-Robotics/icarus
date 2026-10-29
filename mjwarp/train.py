"""PPO (rsl_rl) on MarsNavEnv. One process per GPU; rsl_rl averages gradients across them.

    torchrun --standalone --nproc_per_node=2 train.py --num-envs 4096   # both GPUs
    python train.py --num-envs 4096                                      # one GPU

One-time setup (from mjwarp/; --system-site-packages reuses TACC's python/3.12 module):

    python3 -m venv --system-site-packages .venv && source .venv/bin/activate
    pip install "mujoco>=3.8" mujoco-warp warp-lang rsl-rl-lib tensorboard

Run via `mjwarp/train.sh`, which activates the venv. Logs and checkpoints go to
logs/nav/<timestamp>/ (TensorBoard: `tensorboard --logdir logs/nav`).
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from nav_env import MarsNavEnv

TRAIN_CFG = {
    "num_steps_per_env": 32,
    "save_interval": 100,
    "obs_groups": {"actor": ["policy"], "critic": ["critic"]},  # critic: clean state + hidden params
    "logger": "tensorboard",
    "algorithm": {
        "class_name": "PPO",
        "learning_rate": 1e-3,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.005,
        "max_grad_norm": 1.0,
    },
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.8},
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096, help="worlds per GPU")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs" / "nav")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint (model_*.pt) to continue from")
    args = parser.parse_args()

    # Set by torchrun; a plain `python train.py` is a single-GPU run.
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))
    device = f"cuda:{local_rank}"
    torch.manual_seed(args.seed + rank)  # different worlds per GPU; rsl_rl syncs the weights

    env = MarsNavEnv(args.num_envs, device=device)
    log_dir = args.log_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runner = OnPolicyRunner(env, TRAIN_CFG, log_dir=str(log_dir), device=device)
    if args.resume is not None:
        runner.load(str(args.resume), map_location=device)
    runner.learn(args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
