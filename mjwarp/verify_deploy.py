"""Checks the robot-side runner (skills/rl_nav_utils.py, used by icarus/rl_nav_node.py) against the trainer before
it goes on the robot, using whatever is installed in skills/rl_nav_policy/.

1. Equivalence: the numpy policy (and the onnxruntime one, if onnxruntime is installed) vs the
   torch policy, bin_scan and bin_camera vs the env's lidar and camera binning, and
   build_observation vs the env's actor observation, on live env states.
2. Closed loop: runs NavController (what icarus/rl_nav_node.py runs on the robot) against
   MarsNavEnv with all noise on, one episode per seed.

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
MAX_GOAL_DISTANCE = 5.0  # rl_nav_node's default max_goal_distance


def check_equivalence(checkpoint: Path, device: str) -> None:
    from skills.rl_nav_utils import NavPolicy, NavSpec, OnnxNavPolicy, bin_camera, bin_scan, build_observation, goal_in_base

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
    camera = env.camera_obs.cpu().numpy() if spec.camera is not None else [None] * env.num_envs
    rebuilt = np.stack(
        [
            build_observation(spec, scan[i], goal_in_base(tuple(goal[i]), tuple(est[i])), vel[i, 0], vel[i, 1], history[i], camera[i])
            for i in range(env.num_envs)
        ]
    )
    diffs = {
        "numpy vs torch policy": _action_diff(numpy_actions, torch_actions),
        **_onnx_diff(OnnxNavPolicy, actor, torch_actions),
        "bin_scan vs env binning": np.abs(binned - env._bin(env._scan_ranges()).cpu().numpy()).max(),
        "build_observation vs env actor obs": np.abs(rebuilt - actor).max(),
    }
    if spec.camera is not None:
        for noisy in (False, True):
            points, depth = (t.cpu().numpy() for t in env._camera_points(noisy))
            env_bins = _torch_bins(env, points, depth)
            robot_bins = np.stack([bin_camera(spec.camera, points[i], depth[i]) for i in range(env.num_envs)])
            diffs[f"bin_camera vs env camera binning ({'noisy' if noisy else 'clean'})"] = np.abs(robot_bins - env_bins).max()
    for name, diff in diffs.items():
        print(f"{name}: max diff {diff:.2e}")
    if max(diffs.values()) > 1e-4:
        raise SystemExit("MISMATCH: the robot would not see what the policy trained on")
    del env, runner


def _torch_bins(env: MarsNavEnv, points: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """env._camera_bins' binning applied to these points (env._camera_points would draw fresh noise)."""
    original = env._camera_points
    env._camera_points = lambda noisy, dist=None: (torch.from_numpy(points).to(env.device), torch.from_numpy(depth).to(env.device))
    try:
        return env._camera_bins(True).cpu().numpy()
    finally:
        env._camera_points = original


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


def run_episode(seed: int, device: str) -> None:
    """One closed-loop run in a fresh random world through NavController, as rl_nav_node drives
    it on the robot: it sees the env's noisy odometry, noisy raw lidar beams and (camera policies)
    the env's noisy camera bins, and each tick advances the sim one control step."""
    from skills.rl_nav_utils import NavController, NavSpec, goal_in_base, load_policy

    spec = NavSpec.load(SKILL_POLICY_DIR / "policy_io.json")
    controller = NavController(spec, load_policy(SKILL_POLICY_DIR))
    torch.manual_seed(seed)
    env = MarsNavEnv(1, device)
    env.max_episode_length = 10**9
    goal_radius, nav_env.GOAL_RADIUS = nav_env.GOAL_RADIUS, -1.0  # the controller's caller decides when it arrived
    (x, y, theta), (gx, gy) = env._pose()[0].tolist(), env.goal[0].tolist()
    scale = min(1.0, (MAX_GOAL_DISTANCE - 0.1) / math.hypot(gx - x, gy - y))
    goal = (x + (gx - x) * scale, y + (gy - y) * scale)  # the true goal; the robot tracks it in odometry
    distance = math.dist(goal, (x, y))
    est = env.est_pose[0].tolist()
    goal_odom = (est[0] + goal[0] - x, est[1] + goal[1] - y)
    shapes = env.shape_xy[0, :, 0] < nav_env.PARK_X - 1
    tall, low = int(shapes[nav_env.N_WALLS : nav_env.N_WALLS + nav_env.N_TALL].sum()), int(shapes[nav_env.N_WALLS + nav_env.N_TALL :].sum())
    clock, camera, camera_stamp, reverse_steps = 0.0, None, 0.0, 0
    controller.reset(0.0)
    try:
        for _ in range(int(max(20.0, 10.0 * distance) * nav_env.CONTROL_HZ)):
            est = env.est_pose[0].tolist()
            goal_body = goal_in_base(goal_odom, tuple(est))
            if math.hypot(*goal_body) < spec.goal_radius:
                break
            beams = env._noisy_scan()[0].cpu().numpy()
            scan = types.SimpleNamespace(
                ranges=beams, angle_min=-math.pi, angle_increment=2 * math.pi / len(beams), range_min=0.15, range_max=12.0, stamp=clock
            )
            if spec.camera is not None:
                bins = env.camera_obs[0].cpu().numpy()
                if camera is None or not np.array_equal(bins, camera):
                    camera, camera_stamp = bins, clock
            cmd = controller.step(goal_body, tuple(est), clock, scan, clock, camera, camera_stamp)
            reverse_steps += cmd.linear < -0.02
            action = torch.tensor([[cmd.linear / nav_env.MAX_FORWARD_SPEED, cmd.angular / nav_env.MAX_TURN_RATE]], device=device)
            _, _, done, _ = env.step(action)
            clock += nav_env.CONTROL_DT
            if done.any():
                raise RuntimeError("collision")
        x, y, _ = env._pose()[0].tolist()
        outcome = "Reached the goal" if math.hypot(*goal_in_base(goal_odom, tuple(env.est_pose[0].tolist()))) < spec.goal_radius else "Timed out"
        print(
            f"seed {seed}: goal {distance:.1f} m, {tall} tall + {low} low obstacles -> {outcome}; "
            f"{clock:.1f} s, true error {math.dist(goal, (x, y)):.2f} m, safety stops {controller.safety_stops}, reversing {reverse_steps / max(1, clock * nav_env.CONTROL_HZ):.0%}"
        )
    except RuntimeError as e:
        print(f"seed {seed}: goal {distance:.1f} m, {tall} tall + {low} low obstacles -> FAILED: {e} at {clock:.1f} s")
    finally:
        nav_env.GOAL_RADIUS = goal_radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None, help="must be the one installed; default: newest")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(ICARUS_DIR))
    check_equivalence(args.checkpoint or latest_checkpoint(Path(__file__).resolve().parent / "logs" / "nav"), args.device)
    for seed in range(args.episodes):
        run_episode(seed, args.device)


if __name__ == "__main__":
    main()
