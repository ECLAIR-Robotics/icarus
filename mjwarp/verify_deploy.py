"""Checks the robot-side runner (skills/rl_nav.py + rl_nav_utils.py) against the trainer before
it goes on the robot, using whatever is installed in skills/rl_nav_policy/.

1. Equivalence: the numpy policy (and the onnxruntime one, if onnxruntime is installed) vs the
   torch policy, bin_scan vs the env's lidar binning, and build_observation vs the env's actor
   observation, on live env states.
2. Closed loop: runs the real RlNavigate.execute against MarsNavEnv (all noise on) through a
   stand-in for innate-os's skill API, one episode per seed.

    python verify_deploy.py --checkpoint logs/nav/<run>/model_<N>.pt --episodes 8
"""

import argparse
import copy
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import nav_env
from nav_env import MarsNavEnv
from play import SKILL_POLICY_DIR, latest_checkpoint
from train import TRAIN_CFG

ICARUS_DIR = Path(__file__).resolve().parents[1]


def check_equivalence(checkpoint: Path, device: str) -> None:
    from skills.rl_nav_utils import NavPolicy, NavSpec, OnnxNavPolicy, bin_scan, build_observation, goal_in_base

    spec, numpy_policy = NavSpec.load(SKILL_POLICY_DIR / "policy_io.json"), NavPolicy(SKILL_POLICY_DIR / "policy.npz")
    env = MarsNavEnv(64, device)
    runner = OnPolicyRunner(env, copy.deepcopy(TRAIN_CFG), log_dir=None, device=device)
    runner.load(str(checkpoint), map_location=device)
    torch_policy = runner.get_inference_policy(device)
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(20):
            obs, *_ = env.step(torch_policy(obs))
        torch_actions = torch_policy(obs).cpu().numpy()
    actor = obs["policy"].cpu().numpy()
    numpy_actions = np.stack([numpy_policy(o) for o in actor])

    raw = env._lidar.cpu().numpy()
    binned = np.stack([bin_scan(spec, r, -math.pi, 2 * math.pi / raw.shape[1], 0.0, 100.0) for r in raw])
    est, goal, vel = env.est_pose.cpu().numpy(), env.goal.cpu().numpy(), env.meas_vel.cpu().numpy()
    history, scan = env.action_history.cpu().numpy(), env.scan_obs.cpu().numpy()
    rebuilt = np.stack(
        [
            build_observation(spec, scan[i], goal_in_base(tuple(goal[i]), tuple(est[i])), vel[i, 0], vel[i, 1], history[i])
            for i in range(env.num_envs)
        ]
    )
    diffs = {
        "numpy vs torch policy": _action_diff(numpy_actions, torch_actions),
        **_onnx_diff(OnnxNavPolicy, actor, torch_actions),
        "bin_scan vs env binning": np.abs(binned - env._bin(env._scan_ranges()).cpu().numpy()).max(),
        "build_observation vs env actor obs": np.abs(rebuilt - actor).max(),
    }
    for name, diff in diffs.items():
        print(f"{name}: max diff {diff:.2e}")
    if max(diffs.values()) > 1e-4:
        raise SystemExit("MISMATCH: the robot would not see what the policy trained on")
    del env, runner


def _onnx_diff(policy_class: type, actor: np.ndarray, torch_actions: np.ndarray) -> dict[str, float]:
    try:
        onnx_policy = policy_class(SKILL_POLICY_DIR / "policy.onnx")
    except ImportError:
        print("onnxruntime not installed: skipping the policy.onnx check")
        return {}
    return {"onnxruntime vs torch policy": _action_diff(np.stack([onnx_policy(o) for o in actor]), torch_actions)}


def _action_diff(actions: np.ndarray, reference: np.ndarray) -> float:
    """Compared after the clamp to [-1, 1] that every runner applies: the raw outputs can be in the
    hundreds, where float32 rounding alone differs by ~1e-3 between implementations."""
    return float(np.abs(np.clip(actions, -1.0, 1.0) - np.clip(reference, -1.0, 1.0)).max())


def _fake_innate() -> types.ModuleType:
    """Just the parts of innate-os's `innate` module that rl_nav.py uses."""
    innate = types.ModuleType("innate")

    class SkillOutput:
        def __init__(self, message: str, data: object = None) -> None:
            self.message, self.data = message, data

    class Skill:
        on_sleep = None

        def sleep(self, seconds: float) -> None:
            type(self).on_sleep(seconds)

        def wait_for(self, read, timeout: float):  # noqa: ANN001, ANN202
            return read()

        def fail(self, message: str) -> None:
            raise RuntimeError(message)

        def feedback(self, message: str) -> None:
            print("  feedback:", message)

    innate.Skill, innate.SkillOutput, innate.SkillReturn = Skill, SkillOutput, object
    innate.Lidar = innate.Mobility = innate.Odometry = object
    innate.resource = property
    return innate


def run_episode(seed: int, device: str) -> None:
    """One RlNavigate run in a fresh random world. The skill sees the env's noisy odometry and
    noisy raw lidar beams; each self.sleep() advances the sim one control step."""
    from skills import rl_nav

    torch.manual_seed(seed)
    env = MarsNavEnv(1, device)
    env.max_episode_length = 10**9
    goal_radius, nav_env.GOAL_RADIUS = nav_env.GOAL_RADIUS, -1.0  # the skill decides when it arrived
    command, clock = [0.0, 0.0], [0.0]

    class Mobility:
        def send_cmd_vel(self, linear_x: float = 0.0, angular_z: float = 0.0, duration: float | None = None) -> None:
            command[:] = [linear_x, angular_z]

        def stop(self) -> None:
            command[:] = [0.0, 0.0]

    def advance(_seconds: float) -> None:
        action = torch.tensor([[command[0] / nav_env.MAX_FORWARD_SPEED, command[1] / nav_env.MAX_TURN_RATE]], device=device)
        _, _, done, _ = env.step(action)
        clock[0] += nav_env.CONTROL_DT
        if done.any():
            raise RuntimeError("collision")

    def odom() -> types.SimpleNamespace:
        (x, y, theta), (v, w) = env.est_pose[0].tolist(), env.meas_vel[0].tolist()
        return types.SimpleNamespace(x=x, y=y, theta=theta, linear_velocity=v, angular_velocity=w, stamp=clock[0])

    def lidar() -> types.SimpleNamespace:
        beams = env._noisy_scan()[0].tolist()
        return types.SimpleNamespace(
            ranges=beams, angle_min=-math.pi, angle_increment=2 * math.pi / len(beams), range_min=0.15, range_max=12.0, stamp=clock[0]
        )

    rl_nav.Skill.on_sleep = staticmethod(advance)
    skill_class = type("SimRlNavigate", (rl_nav.RlNavigate,), {"odom": property(lambda s: odom()), "lidar": property(lambda s: lidar())})
    skill = skill_class()
    skill.mobility = Mobility()

    (x, y, theta), (gx, gy) = env._pose()[0].tolist(), env.goal[0].tolist()
    scale = min(1.0, (rl_nav.MAX_GOAL_DISTANCE - 0.1) / math.hypot(gx - x, gy - y))
    gx, gy = x + (gx - x) * scale, y + (gy - y) * scale
    forward = math.cos(theta) * (gx - x) + math.sin(theta) * (gy - y)
    left = -math.sin(theta) * (gx - x) + math.cos(theta) * (gy - y)
    obstacles = int((env.shape_xy[0, nav_env.N_WALLS :, 0] < nav_env.PARK_X - 1).sum())
    try:
        out = skill.execute(forward, left)
        x, y, _ = env._pose()[0].tolist()
        print(f"seed {seed}: goal ({forward:+.2f}, {left:+.2f}) m, {obstacles} obstacles -> {out.message}; "
              f"{clock[0]:.1f} s, true error {math.hypot(gx - x, gy - y):.2f} m, safety stops {out.data.safety_stops}")
    except RuntimeError as e:
        print(f"seed {seed}: goal ({forward:+.2f}, {left:+.2f}) m, {obstacles} obstacles -> FAILED: {e} at {clock[0]:.1f} s")
    finally:
        nav_env.GOAL_RADIUS = goal_radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None, help="must be the one installed; default: newest")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(ICARUS_DIR))
    sys.modules["innate"] = _fake_innate()
    check_equivalence(args.checkpoint or latest_checkpoint(Path(__file__).resolve().parent / "logs" / "nav"), args.device)
    for seed in range(args.episodes):
        run_episode(seed, args.device)


if __name__ == "__main__":
    main()
