"""Pure numpy pieces of the RL nav runner: the exported policy, the observation it expects, and
the per-tick controller (policy + safety) shared by the RlNavigate skill and icarus/rl_nav_node.py.
No ROS and no torch, so they run on the Jetson as-is and can be checked against the trainer
(icarus/mjwarp/nav_env.py, whose deploy_spec() is the policy_io.json read here)."""

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class NavSpec:
    control_hz: float
    max_forward_speed: float
    max_turn_rate: float
    goal_radius: float
    goal_obs_scale: float
    action_history: int
    lidar_bins: int
    lidar_min_range: float
    lidar_max_range: float

    @classmethod
    def load(cls, path: Path) -> "NavSpec":
        raw = json.loads(path.read_text())
        lidar = raw["lidar"]
        return cls(
            control_hz=raw["control_hz"],
            max_forward_speed=raw["max_forward_speed"],
            max_turn_rate=raw["max_turn_rate"],
            goal_radius=raw["goal_radius"],
            goal_obs_scale=raw["goal_obs_scale"],
            action_history=raw["action_history"],
            lidar_bins=lidar["bins"],
            lidar_min_range=lidar["min_range"],
            lidar_max_range=lidar["max_range"],
        )

    def bin_centers(self) -> np.ndarray:
        """Angle (rad, base frame, CCW, 0 = forward) at the middle of each lidar bin."""
        width = 2 * math.pi / self.lidar_bins
        return -math.pi + (np.arange(self.lidar_bins) + 0.5) * width

    def observation_size(self) -> int:
        return self.lidar_bins + 5 + 2 + 2 * self.action_history


class Policy(Protocol):
    backend: str
    obs_size: int

    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


class NavPolicy:
    """The actor MLP exported by play.py: observation normalizer, ELU hidden layers, linear
    output (the deterministic action)."""

    backend = "numpy"

    def __init__(self, path: Path) -> None:
        weights = np.load(path)
        self._mean = weights["obs_mean"]
        self._scale = 1.0 / (weights["obs_std"] + weights["obs_eps"])
        n_layers = int(weights["n_layers"])
        self._layers = [(weights[f"w{i}"], weights[f"b{i}"]) for i in range(n_layers)]
        self.obs_size = self._mean.shape[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs.astype(np.float32) - self._mean) * self._scale
        for w, b in self._layers[:-1]:
            x = w @ x + b
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        w, b = self._layers[-1]
        return w @ x + b


class OnnxNavPolicy:
    """The same actor as policy.onnx, on onnxruntime's CPU provider with one thread and no
    spinning: at batch size 1 a ~110k-weight MLP is faster on one CPU core than on the GPU,
    and it leaves the other cores and the Jetson's shared RAM to the rest of the stack."""

    backend = "onnxruntime"

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        self.obs_size = int(inp.shape[-1])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._session.run(None, {self._input: obs.astype(np.float32).reshape(1, -1)})[0][0]


def load_policy(policy_dir: Path, backend: str = "auto") -> Policy:
    """`backend`: "onnxruntime", "numpy", or "auto" (onnxruntime when it is installed and
    policy.onnx exists, numpy otherwise)."""
    if backend not in ("auto", "onnxruntime", "numpy"):
        raise ValueError(f"unknown policy backend {backend!r}")
    onnx_path = policy_dir / "policy.onnx"
    if backend == "onnxruntime" or (backend == "auto" and onnx_path.exists()):
        try:
            return OnnxNavPolicy(onnx_path)
        except ImportError:
            if backend == "onnxruntime":
                raise
    return NavPolicy(policy_dir / "policy.npz")


@lru_cache(maxsize=8)
def _bin_layout(n_beams: int, angle_min: float, angle_increment: float, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Beam order grouping the beams by bin, the bins that get at least one beam, and where each
    of those bins starts in that order. A scan's geometry is fixed, so this runs once."""
    angles = angle_min + np.arange(n_beams) * angle_increment
    wrapped = np.mod(angles + math.pi, 2 * math.pi)
    index = np.minimum((wrapped / (2 * math.pi / bins)).astype(np.int64), bins - 1)
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    present = np.unique(sorted_index)
    return order, present, np.searchsorted(sorted_index, present)


def bin_scan(
    spec: NavSpec, ranges: np.ndarray, angle_min: float, angle_increment: float, range_min: float, range_max: float
) -> np.ndarray:
    """LaserScan -> the policy's lidar observation (lidar_bins values in 0-1, 1 = nothing within max range)."""
    r = np.asarray(ranges, dtype=np.float32)
    valid = np.isfinite(r) & (r >= max(range_min, spec.lidar_min_range)) & (r <= range_max)
    r = np.where(valid, np.minimum(r, spec.lidar_max_range), spec.lidar_max_range)
    order, present, starts = _bin_layout(r.size, float(angle_min), float(angle_increment), spec.lidar_bins)
    out = np.full(spec.lidar_bins, spec.lidar_max_range, dtype=np.float32)
    out[present] = np.minimum.reduceat(r[order], starts)
    return out / spec.lidar_max_range


def goal_in_base(goal_xy: tuple[float, float], pose: tuple[float, float, float]) -> tuple[float, float]:
    x, y, theta = pose
    dx, dy = goal_xy[0] - x, goal_xy[1] - y
    return math.cos(theta) * dx + math.sin(theta) * dy, -math.sin(theta) * dx + math.cos(theta) * dy


def relative_to_odom(pose: tuple[float, float, float], forward: float, left: float) -> tuple[float, float]:
    x, y, theta = pose
    return x + math.cos(theta) * forward - math.sin(theta) * left, y + math.sin(theta) * forward + math.cos(theta) * left


def build_observation(
    spec: NavSpec,
    lidar_bins: np.ndarray,
    goal_body: tuple[float, float],
    linear_velocity: float,
    angular_velocity: float,
    action_history: np.ndarray,
) -> np.ndarray:
    """Same layout as MarsNavEnv's actor observation; `action_history` is (action_history, 2), newest first."""
    gx, gy = goal_body
    heading = math.atan2(gy, gx)
    goal = [gx / spec.goal_obs_scale, gy / spec.goal_obs_scale, math.hypot(gx, gy) / spec.goal_obs_scale]
    velocity = [linear_velocity / spec.max_forward_speed, angular_velocity / spec.max_turn_rate]
    return np.concatenate(
        (lidar_bins, goal, [math.cos(heading), math.sin(heading)], velocity, action_history.ravel())
    ).astype(np.float32)


class OdomVelocity:
    """Body velocity the way the trainer measures it (MarsNavEnv._update_odometry): the odometry
    displacement along the previous heading and the yaw change, over the time between the two
    poses. Odometry.twist can't be used instead: the sim driver leaves it at zero."""

    def __init__(self) -> None:
        self._last: tuple[tuple[float, float, float], float] | None = None
        self.value = (0.0, 0.0)

    def update(self, pose: tuple[float, float, float], stamp: float) -> tuple[float, float]:
        if self._last is not None:
            (x0, y0, theta0), stamp0 = self._last
            dt = stamp - stamp0
            if dt <= 0.0:
                return self.value
            forward = math.cos(theta0) * (pose[0] - x0) + math.sin(theta0) * (pose[1] - y0)
            dyaw = math.atan2(math.sin(pose[2] - theta0), math.cos(pose[2] - theta0))
            self.value = (forward / dt, dyaw / dt)
        self._last = (pose, stamp)
        return self.value


class Scan(Protocol):
    """innate's Lidar, or anything else with these LaserScan fields (stamp in seconds)."""

    ranges: tuple | np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    stamp: float


@dataclass
class NavCommand:
    linear: float
    """m/s for cmd_vel.linear.x (0 while stopped by a safety check)."""
    angular: float
    action: np.ndarray
    """The policy's clamped action, before the safety checks."""
    bins: np.ndarray
    """What the policy saw of the lidar, 0-1."""
    blocked: bool
    """Forward motion zeroed: a return within safety_stop_range in the direction of travel."""
    stale: bool
    """Both speeds zeroed: the scan or odometry stopped updating."""


@dataclass
class NavController:
    """One control tick: lidar bins + odometry -> policy -> clamped command -> safety checks.

    The caller owns the goal and its frame and hands in the goal in the base frame each tick;
    call reset() at the start of every run (the trainer starts episodes with zero history).
    """

    spec: NavSpec
    policy: Policy
    safety_stop_range: float = 0.25
    """m from the lidar; no driving toward a return this close, whatever the policy says."""
    safety_sector: float = math.radians(60)
    stale_scan_s: float = 0.6
    stale_odom_s: float = 0.3
    safety_stops: int = 0
    _history: np.ndarray = field(init=False, repr=False)
    _velocity: OdomVelocity = field(init=False, repr=False)
    _bins: np.ndarray = field(init=False, repr=False)
    _scan_stamp: float | None = field(init=False, repr=False)
    _changed_at: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.policy.obs_size != self.spec.observation_size():
            raise ValueError("the policy and policy_io.json are from different exports")
        self.reset()

    def reset(self, now: float = 0.0) -> None:
        self._history = np.zeros((self.spec.action_history, 2), dtype=np.float32)
        self._velocity = OdomVelocity()
        self._bins = np.ones(self.spec.lidar_bins, dtype=np.float32)
        self._scan_stamp = None
        self._changed_at = {"scan": (None, now), "odom": (None, now)}
        self.safety_stops = 0

    def step(self, goal_body: tuple[float, float], pose: tuple[float, float, float], odom_stamp: float, scan: Scan, now: float) -> NavCommand:
        """`pose` is the odometry pose (x, y, yaw) stamped `odom_stamp` (s); `now` is a local
        monotonic time (s), used only to notice feeds that stopped updating."""
        if scan.stamp != self._scan_stamp:
            self._scan_stamp = scan.stamp
            self._bins = bin_scan(self.spec, scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max)
        linear_velocity, angular_velocity = self._velocity.update(pose, odom_stamp)
        obs = build_observation(self.spec, self._bins, goal_body, linear_velocity, angular_velocity, self._history)
        action = np.clip(self.policy(obs), -1.0, 1.0).astype(np.float32)
        self._history = np.roll(self._history, 1, axis=0)
        self._history[0] = action
        linear, angular = float(action[0]) * self.spec.max_forward_speed, float(action[1]) * self.spec.max_turn_rate

        scan_fresh = self._fresh("scan", scan.stamp, self.stale_scan_s, now)
        odom_fresh = self._fresh("odom", odom_stamp, self.stale_odom_s, now)
        stale = not (scan_fresh and odom_fresh)
        blocked = False
        if stale:
            linear, angular = 0.0, 0.0
        elif self._blocked(linear):
            linear, blocked = 0.0, True
            self.safety_stops += 1
        return NavCommand(linear, angular, action, self._bins, blocked, stale)

    def _fresh(self, name: str, stamp: float, limit: float, now: float) -> bool:
        """Tracks when each feed's stamp last changed by the local clock, so it works whatever
        clock the stamps use."""
        last, changed_at = self._changed_at[name]
        if stamp != last:
            self._changed_at[name] = (stamp, now)
            return True
        return now - changed_at <= limit

    def _blocked(self, linear: float) -> bool:
        """A return within safety_stop_range in the direction of travel."""
        if linear == 0.0:
            return False
        centers = self.spec.bin_centers()
        ahead = np.abs(centers) <= self.safety_sector if linear > 0 else np.abs(centers) >= math.pi - self.safety_sector
        return bool((self._bins[ahead] * self.spec.lidar_max_range < self.safety_stop_range).any())
