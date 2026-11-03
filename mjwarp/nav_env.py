"""Goal reaching with obstacle avoidance for the MARS base, as an rsl_rl VecEnv, with the
noise, latency and variety the real robot will have.

Each world is a walled room of random size (walls sometimes missing) with a random set of
cylinders and rotated boxes. The robot drives to a goal given in its odometry frame.

What the policy does not get to see cleanly (each parameter is drawn per episode):
- Commands reach the motors after a Gaussian latency (plus per-step jitter), then go through
  a randomized motor model: gain error, turn bias, deadband, first-order lag, acceleration
  limit, and per-step noise. Physics runs in 10 ms chunks so the latency has that resolution.
- The lidar refreshes at a random 5-10 Hz (the robot's /scan is throttled to 6 Hz), with
  range noise, dropouts, spurious short returns and a mount-yaw offset.
- The goal and velocity come from simulated wheel odometry: scale error, random-walk drift,
  blind to sideways slip, and noisy velocity estimates.
- Random pushes perturb the base.
- The head's stereo camera sees what the lidar can't: low obstacles (6-14 cm, under the lidar's
  17 cm plane) are scattered in many rooms. A fan of rays from the camera, intersected analytically
  with the floor and the (box and cylinder) obstacles, stands in for the stereo point cloud
  (/mars/main_camera/points), with its 0.25-2 m depth clamp, holes, range-squared noise, a slow
  frame rate and a pitch error. The camera only looks forward, and reversing is
  penalized.

The critic gets the clean state plus these hidden parameters (asymmetric actor-critic).
Rewards and terminations use the true state.

Everything stays on the GPU. Warp and torch share one CUDA stream, so the torch views of
mjwarp's arrays (qpos, ctrl, sensordata, ...) can be written and read without syncs.

Anything the robot-side code must reproduce is in `deploy_spec()`, which play.py exports as
policy_io.json.
"""

import math

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from robot_env import build_model
from warp_env import MarsWarpEnv

# Robot interface (mirrored on the robot via deploy_spec()).
CONTROL_HZ = 10.0
CONTROL_DT = 1.0 / CONTROL_HZ
# innate-os mars_control/config/motion_control.yaml
MAX_FORWARD_SPEED = 0.4  # m/s
MAX_TURN_RATE = 1.0  # rad/s
LIDAR_BINS = 40
# Closer returns are the robot's own body: in sim, beams ~10 deg left hit it at 0.15-0.19 m.
LIDAR_MIN_RANGE = 0.2
LIDAR_MAX_RANGE = 4.0
GOAL_OBS_SCALE = 5.0
ACTION_HISTORY = 3
GOAL_RADIUS = 0.3
EPISODE_SECONDS = 45.0
# Stereo camera (mars_cam stereo_depth_estimator: depth clamped to 0.25-2 m, <= 8 Hz). Its points,
# in base_link, between CAMERA_MIN/MAX_HEIGHT above the floor are obstacles; they are min-pooled by
# bearing into CAMERA_BINS sectors across CAMERA_HFOV, as horizontal range from base_link / max.
CAMERA_BINS = 16
CAMERA_HFOV = math.radians(112.0)  # the lens sees ~116 deg; the outer columns are unreliable
CAMERA_MIN_DEPTH = 0.25
CAMERA_MAX_DEPTH = 2.0
# The floor-rejection height grows with range: a 1 deg head pitch error lifts the floor by ~1.7 cm/m.
CAMERA_MIN_HEIGHT = 0.04
CAMERA_MIN_HEIGHT_PER_M = 0.02
CAMERA_MAX_HEIGHT = 0.40  # the robot (head included) is ~0.36 m tall; higher things are overhead
CAMERA_MAX_RANGE = 2.0

# World.
ROOM_HALF_RANGE = (2.0, 4.0)  # per axis, so rooms are 4-8 m rectangles
WALL_PRESENT_PROB = 0.85
WALL_HALF_LENGTH = 4.6
WALL_HALF_THICKNESS = 0.05
OBSTACLE_HEIGHT = 0.5
CYLINDER_RADII = (0.08, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
BOX_HALF_SIZES = ((0.05, 0.3), (0.05, 0.6), (0.15, 0.15), (0.25, 0.25), (0.2, 0.4), (0.3, 0.6), (0.4, 0.4), (0.1, 1.0))
OBSTACLE_DENSITY_RANGE = (0.1, 0.7)  # per-episode probability that each obstacle is in the room
# Below the lidar's plane: only the camera sees these (shoes, books, toys, cable trays).
LOW_BOX_SIZES = ((0.10, 0.06, 0.06), (0.15, 0.10, 0.08), (0.20, 0.05, 0.10), (0.08, 0.08, 0.12), (0.25, 0.15, 0.07), (0.12, 0.30, 0.14))  # half x, half y, height
LOW_CYLINDER_SIZES = ((0.06, 0.10), (0.10, 0.07), (0.15, 0.12))  # radius, height
LOW_OBSTACLE_EPISODE_PROB = 0.7  # rooms that have any
LOW_OBSTACLE_DENSITY_RANGE = (0.2, 0.7)
# Unused walls and obstacles are parked out here, beyond the lidar's max range.
PARK_X = 30.0
OVERVIEW_CAMERA = "overview"

# The chassis is centred 5.9 cm behind base_link; this circle around that point covers it and the wheels.
ROBOT_CENTER_OFFSET = -0.0587
ROBOT_RADIUS = 0.2
MIN_GOAL_DISTANCE = 1.0
SPAWN_MARGIN = 0.5
OBSTACLE_SPAWN_CLEARANCE = 0.25

# Command path: policy -> Zenoh -> bringup -> I2C motor controller.
CMD_LATENCY_MEAN = 0.07  # s
CMD_LATENCY_STD = 0.025  # across episodes
CMD_LATENCY_JITTER_STD = 0.01  # within an episode, per step
CMD_LATENCY_MAX = 0.25
MOTOR_GAIN_RANGE = (0.85, 1.15)
TURN_BIAS_STD = 0.03  # rad/s
DEADBAND_MAX = (0.03, 0.1)  # m/s, rad/s
MOTOR_TAU_RANGE = (0.05, 0.25)  # s
MOTOR_ACCEL_RANGE = ((0.6, 2.0), (1.5, 5.0))  # m/s^2, rad/s^2
CMD_NOISE_STD = (0.02, 0.05)
PUSH_PROB = 0.02  # per step
PUSH_STD = (0.15, 0.5)  # m/s, rad/s

# Lidar (RPLidar; innate-os throttles /scan to 6 Hz).
SCAN_PERIOD_RANGE = (0.1, 0.2)
LIDAR_NOISE_STD = 0.01
LIDAR_NOISE_REL = 0.01
LIDAR_DROPOUT_RANGE = (0.0, 0.1)
LIDAR_SPURIOUS_PROB = 0.002
LIDAR_YAW_OFFSET_STD_DEG = 2.0

# Stereo camera noise.
CAMERA_PERIOD_RANGE = (0.125, 0.25)  # 4-8 Hz
CAMERA_DROPOUT_RANGE = (0.0, 0.4)  # holes: textureless surfaces, speckle and edge filters
CAMERA_NOISE_K_RANGE = (0.005, 0.02)  # depth std = k * depth^2 (m)
CAMERA_SPURIOUS_PROB = 0.0005  # the real pipeline's speckle filter drops most isolated matches
CAMERA_PITCH_STD_DEG = 1.0  # head calibration / tilt error
# The ray fan (sim only; the robot uses every point). Denser near the horizon, where the floor
# is far and a low obstacle spans few degrees.
CAMERA_RAY_AZIMUTHS_DEG = tuple(float(a) for a in range(-56, 57, 4))
CAMERA_RAY_ELEVATIONS_DEG = (-40, -35, -30, -26, -22, -19, -16, -14, -12, -10, -8, -6, -4, -2, 1, 5)

# Wheel odometry.
ODOM_SCALE_RANGE = ((0.95, 1.05), (0.95, 1.05))  # distance, yaw
ODOM_STEP_NOISE = (0.002, 0.003)  # m, rad per control step (random walk)
ODOM_VEL_NOISE = (0.02, 0.05)  # m/s, rad/s

# mjwarp's per-world contact/constraint buffers. The default njmax (64) overflows once the
# robot touches an obstacle; an overflowing world drops constraints and can go NaN.
NCONMAX = 64
NJMAX = 512

PHYSICS_DT = 0.002
CHUNK_SUBSTEPS = 5
CHUNK_DT = CHUNK_SUBSTEPS * PHYSICS_DT
CHUNKS_PER_STEP = round(CONTROL_DT / CHUNK_DT)
CMD_HISTORY = math.ceil(CMD_LATENCY_MAX / CONTROL_DT) + 1
LIDAR_BEAMS = 360

REWARD_PROGRESS = 10.0
REWARD_GOAL = 10.0
REWARD_COLLISION = -10.0
REWARD_TIME = -0.01
REWARD_PROXIMITY = -0.1
PROXIMITY_MARGIN = 0.2
REWARD_ACTION_RATE = -0.01
# Per step, times the reverse speed as a fraction of the max: the camera only looks forward.
REWARD_REVERSE = -0.25

N_WALLS = 4
WALL_HALF_SIZES = ((WALL_HALF_LENGTH, WALL_HALF_THICKNESS),) * 2 + ((WALL_HALF_THICKNESS, WALL_HALF_LENGTH),) * 2
N_TALL = len(CYLINDER_RADII) + len(BOX_HALF_SIZES)
N_LOW = len(LOW_BOX_SIZES) + len(LOW_CYLINDER_SIZES)
# Obstacles as rounded rectangles: half extents plus a corner radius (a cylinder is all radius).
SHAPE_HALF = (
    WALL_HALF_SIZES
    + ((0.0, 0.0),) * len(CYLINDER_RADII)
    + BOX_HALF_SIZES
    + tuple((hx, hy) for hx, hy, _ in LOW_BOX_SIZES)
    + ((0.0, 0.0),) * len(LOW_CYLINDER_SIZES)
)
SHAPE_ROUND = (
    (0.0,) * N_WALLS
    + CYLINDER_RADII
    + (0.0,) * len(BOX_HALF_SIZES)
    + (0.0,) * len(LOW_BOX_SIZES)
    + tuple(r for r, _ in LOW_CYLINDER_SIZES)
)
N_SHAPES = len(SHAPE_HALF)
SHAPE_HEIGHT = (
    (1.0,) * N_WALLS
    + (OBSTACLE_HEIGHT,) * N_TALL
    + tuple(z for _, _, z in LOW_BOX_SIZES)
    + tuple(z for _, z in LOW_CYLINDER_SIZES)
)


def _camera_ray_angles() -> tuple[list[float], list[float]]:
    """(azimuth, elevation) of every ray in radians, row-major over elevations."""
    az, el = [], []
    for e in CAMERA_RAY_ELEVATIONS_DEG:
        for a in CAMERA_RAY_AZIMUTHS_DEG:
            az.append(math.radians(a))
            el.append(math.radians(e))
    return az, el



def deploy_spec() -> dict:
    """Everything the robot-side runner needs to rebuild the actor's observation and apply its action."""
    return {
        "control_hz": CONTROL_HZ,
        "max_forward_speed": MAX_FORWARD_SPEED,
        "max_turn_rate": MAX_TURN_RATE,
        "goal_radius": GOAL_RADIUS,
        "goal_obs_scale": GOAL_OBS_SCALE,
        "action_history": ACTION_HISTORY,
        "lidar": {
            "bins": LIDAR_BINS,
            "min_range": LIDAR_MIN_RANGE,
            "max_range": LIDAR_MAX_RANGE,
            "frame": "base_laser (same axes as base_link)",
            "binning": "beam angle a (rad, CCW, 0 = forward) wrapped to [-pi, pi) goes to bin "
            "floor((a + pi) / (2 pi / bins)); each bin is the min valid range; returns that are "
            "non-finite, outside [range_min, range_max] of the scan or below min_range count as "
            "max_range; then clip to max_range and divide by it",
        },
        "camera": {
            "bins": CAMERA_BINS,
            "hfov": CAMERA_HFOV,
            "min_depth": CAMERA_MIN_DEPTH,
            "max_depth": CAMERA_MAX_DEPTH,
            "min_height": CAMERA_MIN_HEIGHT,
            "min_height_per_m": CAMERA_MIN_HEIGHT_PER_M,
            "max_height": CAMERA_MAX_HEIGHT,
            "max_range": CAMERA_MAX_RANGE,
            "topic": "/mars/main_camera/points",
            "binning": "points (camera_optical_frame) with optical-axis depth z in [min_depth, max_depth] "
            "are moved to base_link; with bearing b = atan2(y, x) and range r = hypot(x, y) (both from "
            "base_link's origin), those with height z in [min_height + min_height_per_m * r, max_height] "
            "are obstacles; points "
            "with |b| < hfov/2 go to bin floor((b + hfov/2) / (hfov / bins)), each bin is the min r; "
            "empty bins are max_range; then clip to max_range and divide by it",
        },
        "observation": [
            ["lidar", LIDAR_BINS, "binned scan, 0-1"],
            ["goal_body_xy", 2, "goal in base frame (x fwd, y left) from odometry, m / goal_obs_scale"],
            ["goal_distance", 1, "m / goal_obs_scale"],
            ["goal_heading_cos_sin", 2, "cos, sin of atan2(goal_y, goal_x)"],
            ["forward_velocity", 1, "odom linear.x / max_forward_speed"],
            ["yaw_rate", 1, "odom angular.z / max_turn_rate"],
            ["action_history", 2 * ACTION_HISTORY, "last clamped actions, newest first, (fwd, turn) pairs"],
            ["camera", CAMERA_BINS, "binned stereo obstacle points, 0-1, bin 0 = rightmost"],
        ],
        "action": [
            ["forward", "clamp to [-1, 1], times max_forward_speed -> cmd_vel.linear.x"],
            ["turn", "clamp to [-1, 1], times max_turn_rate -> cmd_vel.angular.z"],
        ],
    }


def _scene_xml() -> str:
    def mocap_body(name: str, geom: str, z: float) -> str:
        return f'<body name="{name}" mocap="true" pos="{PARK_X} 0 {z}">{geom}</body>'

    walls = [
        mocap_body(f"wall_{i}", f'<geom type="box" size="{hx} {hy} 0.5" rgba="0.7 0.7 0.75 1"/>', 0.5)
        for i, (hx, hy) in enumerate(WALL_HALF_SIZES)
    ]
    h = OBSTACLE_HEIGHT / 2
    cylinders = [
        mocap_body(f"cylinder_{i}", f'<geom type="cylinder" size="{r} {h}" rgba="0.6 0.4 0.2 1"/>', h)
        for i, r in enumerate(CYLINDER_RADII)
    ]
    boxes = [
        mocap_body(f"box_{i}", f'<geom type="box" size="{hx} {hy} {h}" rgba="0.35 0.45 0.65 1"/>', h)
        for i, (hx, hy) in enumerate(BOX_HALF_SIZES)
    ]
    low_boxes = [
        mocap_body(f"low_box_{i}", f'<geom type="box" size="{hx} {hy} {z / 2}" rgba="0.75 0.3 0.55 1"/>', z / 2)
        for i, (hx, hy, z) in enumerate(LOW_BOX_SIZES)
    ]
    low_cylinders = [
        mocap_body(f"low_cylinder_{i}", f'<geom type="cylinder" size="{r} {z / 2}" rgba="0.75 0.3 0.55 1"/>', z / 2)
        for i, (r, z) in enumerate(LOW_CYLINDER_SIZES)
    ]
    bodies = "\n        ".join(walls + cylinders + boxes + low_boxes + low_cylinders)
    return f"""
<mujoco model="mars_nav_arena">
    <option timestep="{PHYSICS_DT}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>
    <default>
        <geom rgba="0.7 0.7 0.75 1"/>
    </default>
    <worldbody>
        <light type="directional" pos="0 0 3" dir="0.3 0.5 -1" diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35"/>
        <geom name="ground" type="plane" size="60 60 0.1" friction="0.9 0.01 0.001" rgba="0.35 0.35 0.35 1"/>
        {bodies}
        <!-- Visual only: no collision, and too low for the lidar's horizontal beams to hit. -->
        <body name="goal" mocap="true" pos="{PARK_X} -5 0">
            <geom type="cylinder" size="{GOAL_RADIUS} 0.002" pos="0 0 0.002" contype="0" conaffinity="0" rgba="0.1 0.9 0.2 1"/>
        </body>
        <!-- Top-down view of the largest room, for play.py's videos. -->
        <camera name="{OVERVIEW_CAMERA}" pos="0 0 10.5" fovy="45" resolution="480 480"/>
    </worldbody>
</mujoco>
"""


def _camera_self_occlusion(m: mujoco.MjModel, azimuths: list[float], elevations: list[float]) -> tuple[list[float], list[bool]]:
    """The head camera's position in base_link at the home pose, and which rays the robot's own
    body (chassis, arm) blocks within CAMERA_MIN_DEPTH; those never return a valid point, on the
    robot either. Rays start a few cm out: the optical centre sits inside the head housing."""
    data = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, data, m.key("home").id)
    mujoco.mj_forward(m, data)
    base = m.body("robot_base_link").id
    rot, origin = data.xmat[base].reshape(3, 3), data.xpos[base]
    camera = data.xpos[m.body("robot_camera_optical_frame").id]
    robot_bodies = {i for i in range(m.nbody) if m.body(i).name.startswith("robot_")}
    start_offset = 0.05
    geom = np.zeros(1, dtype=np.int32)
    blocked = []
    for a, e in zip(azimuths, elevations):
        direction = rot @ np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        dist = mujoco.mj_ray(m, data, camera + start_offset * direction, direction, None, 1, m.body("robot_camera_optical_frame").id, geom)
        blocked.append(bool(dist >= 0 and geom[0] >= 0 and m.geom_bodyid[geom[0]] in robot_bodies and dist + start_offset < 1.0))
    return (rot.T @ (camera - origin)).tolist(), blocked


def _shape_names() -> list[str]:
    return (
        [f"wall_{i}" for i in range(N_WALLS)]
        + [f"cylinder_{i}" for i in range(len(CYLINDER_RADII))]
        + [f"box_{i}" for i in range(len(BOX_HALF_SIZES))]
        + [f"low_box_{i}" for i in range(len(LOW_BOX_SIZES))]
        + [f"low_cylinder_{i}" for i in range(len(LOW_CYLINDER_SIZES))]
    )


class MarsNavEnv(VecEnv):
    def __init__(self, num_envs: int, device: str = "cuda:0") -> None:
        self.num_envs = num_envs
        self.num_actions = 2
        self.device = device
        self.max_episode_length = int(EPISODE_SECONDS * CONTROL_HZ)
        self.cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple, str))}

        # One stream for both libraries: warp's kernels and graphs then order correctly with
        # torch's reads and writes of the same memory. Graph capture needs a non-default stream.
        wp.set_device(device)
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device)
        torch.cuda.set_stream(stream)
        wp.set_stream(wp.stream_from_torch(stream))

        mj_model = build_model(_scene_xml())
        self.sim = MarsWarpEnv(num_envs, control_hz=CONTROL_HZ, mj_model=mj_model, nconmax=NCONMAX, njmax=NJMAX)
        self._init_constants(mj_model)
        self._init_views()
        self._chunk_graph = self._capture_chunk()
        self._forward_graph = self._capture_forward()
        self._reset_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._reset_mask_wp = wp.from_torch(self._reset_mask)
        self._init_state()
        self._reset_worlds()
        self._obs = self._compute_obs()

    def _init_constants(self, m: mujoco.MjModel) -> None:
        dev = self.device
        self._base_x = m.joint("robot_base_x").qposadr[0]
        self._base_y = m.joint("robot_base_y").qposadr[0]
        self._base_yaw = m.joint("robot_base_yaw").qposadr[0]
        self._base_dofs = [m.joint(f"robot_base_{axis}").dofadr[0] for axis in ("x", "y", "yaw")]
        self._forward_id = m.actuator("base_forward").id
        self._yaw_id = m.actuator("base_yaw").id
        self._shape_mocap = [m.body_mocapid[m.body(name).id] for name in _shape_names()]
        self._goal_mocap = m.body_mocapid[m.body("goal").id]
        home = m.key("home")
        self._home_qpos = torch.tensor(home.qpos, dtype=torch.float32, device=dev)
        self._home_ctrl = torch.tensor(home.ctrl, dtype=torch.float32, device=dev)
        self._shape_half = torch.tensor(SHAPE_HALF, device=dev)
        self._shape_round = torch.tensor(SHAPE_ROUND, device=dev)
        self._park_xy = torch.tensor([[PARK_X + 2.5 * i, 0.0] for i in range(N_SHAPES)], device=dev)
        self._max_cmd = torch.tensor([MAX_FORWARD_SPEED, MAX_TURN_RATE], device=dev)
        self._beam_index = torch.arange(LIDAR_BEAMS, device=dev)
        az, el = _camera_ray_angles()
        self._ray_az = torch.tensor(az, device=dev)
        self._ray_el = torch.tensor(el, device=dev)
        # The head holds its home pose, where the camera looks level: ray angles are base-frame angles.
        self._ray_dir = torch.stack((self._ray_el.cos() * self._ray_az.cos(), self._ray_el.cos() * self._ray_az.sin(), self._ray_el.sin()), dim=1)
        # Optical-axis depth per metre along each ray, for the depth clamp.
        self._ray_depth_per_m = self._ray_el.cos() * self._ray_az.cos()
        camera_pos, blocked = _camera_self_occlusion(m, az, el)
        self._camera_pos = torch.tensor(camera_pos, dtype=torch.float32, device=dev)
        self._ray_blocked = torch.tensor(blocked, device=dev)
        self._shape_height = torch.tensor(SHAPE_HEIGHT, device=dev)
        self._shape_is_cylinder = self._shape_half.abs().sum(dim=1) == 0

    def _init_views(self) -> None:
        data = self.sim.data
        self._qpos = wp.to_torch(data.qpos)
        self._qvel = wp.to_torch(data.qvel)
        self._ctrl = wp.to_torch(data.ctrl)
        self._mocap_pos = wp.to_torch(data.mocap_pos)
        self._mocap_quat = wp.to_torch(data.mocap_quat)
        self._lidar = wp.to_torch(data.sensordata)[:, self.sim.lidar_slice]

    def _init_state(self) -> None:
        n, dev = self.num_envs, self.device
        zeros = lambda *shape: torch.zeros(n, *shape, device=dev)  # noqa: E731
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.goal = zeros(2)
        self.shape_xy = zeros(N_SHAPES, 2)
        self.shape_yaw = zeros(N_SHAPES)
        self.prev_goal_dist = zeros()
        self.action_history = zeros(ACTION_HISTORY, 2)
        self.cmd_history = zeros(CMD_HISTORY, 2)
        self.motor_vel = zeros(2)
        self.prev_pose = zeros(3)
        self.est_pose = zeros(3)
        self.meas_vel = zeros(2)
        self.scan_age = zeros()
        self.scan_obs = torch.ones(n, LIDAR_BINS, device=dev)
        self.camera_age = zeros()
        self.camera_obs = torch.ones(n, CAMERA_BINS, device=dev)
        # Per-episode hidden parameters.
        self.latency = zeros()
        self.motor_gain = zeros(2)
        self.turn_bias = zeros()
        self.deadband = zeros(2)
        self.motor_tau = zeros(2)
        self.motor_accel = zeros(2)
        self.scan_period = zeros()
        self.lidar_dropout = zeros()
        self.lidar_shift = torch.zeros(n, dtype=torch.long, device=dev)
        self.odom_scale = zeros(2)
        self.camera_period = zeros()
        self.camera_dropout = zeros()
        self.camera_noise_k = zeros()
        self.camera_pitch = zeros()

    def _capture_chunk(self) -> wp.Graph:
        """CHUNK_SUBSTEPS physics steps with sensors off; _forward_graph computes them once per control step."""
        model, data = self.sim.model, self.sim.data
        flags = model.opt.disableflags
        with wp.ScopedCapture() as capture:
            model.opt.disableflags = flags | mjw.DisableBit.SENSOR
            for _ in range(CHUNK_SUBSTEPS):
                mjw.step(model, data)
            model.opt.disableflags = flags
        return capture.graph

    def _capture_forward(self) -> wp.Graph:
        with wp.ScopedCapture() as capture:
            mjw.forward(self.sim.model, self.sim.data)
        return capture.graph

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = actions.clamp(-1.0, 1.0)
        self.action_history = torch.cat((actions[:, None], self.action_history[:, :-1]), dim=1)
        self.cmd_history = torch.cat(((actions * self._max_cmd)[:, None], self.cmd_history[:, :-1]), dim=1)
        self._push()
        self._run_physics()
        self.episode_length_buf += 1

        # A world whose physics blew up is cut off as a timeout (no penalty) and reset, rather
        # than letting its NaNs reach the policy update.
        diverged = ~torch.isfinite(self._qpos).all(dim=1) | ~torch.isfinite(self._qvel).all(dim=1)
        self._qpos.masked_fill_(diverged[:, None], 0.0)
        self._qvel.masked_fill_(diverged[:, None], 0.0)
        self._update_odometry()

        goal_dist = torch.linalg.norm(self.goal - self._pose()[:, :2], dim=1)
        clearance = self._clearance(self._robot_center())
        pose = self._pose()
        forward_vel = pose[:, 2].cos() * self._qvel[:, self._base_dofs[0]] + pose[:, 2].sin() * self._qvel[:, self._base_dofs[1]]
        reversing = (-forward_vel / MAX_FORWARD_SPEED).clamp(0.0, 1.0)
        reached = goal_dist < GOAL_RADIUS
        collided = clearance < 0.0
        terminated = (reached | collided) & ~diverged
        timed_out = ((self.episode_length_buf >= self.max_episode_length) | diverged) & ~terminated
        rewards = torch.where(
            diverged,
            0.0,
            REWARD_PROGRESS * (self.prev_goal_dist - goal_dist)
            + REWARD_GOAL * reached
            + REWARD_COLLISION * collided
            + REWARD_TIME
            + REWARD_PROXIMITY * ((PROXIMITY_MARGIN - clearance).clamp(min=0.0) / PROXIMITY_MARGIN)
            + REWARD_ACTION_RATE * (self.action_history[:, 0] - self.action_history[:, 1]).square().sum(dim=1)
            + REWARD_REVERSE * reversing,
        )
        self.prev_goal_dist = goal_dist

        dones = terminated | timed_out
        odom_error = torch.linalg.norm(self.est_pose[:, :2] - self._pose()[:, :2], dim=1)
        extras = {
            "time_outs": timed_out,
            "log": {
                "Episode/success": reached[dones].float(),
                "Episode/collision": collided[dones].float(),
                "Episode/timeout": timed_out[dones].float(),
                "Episode/diverged": diverged[dones].float(),
                "Episode/odom_error_m": odom_error[dones],
                "Policy/reversing": (forward_vel < -0.05).float().mean(),
                "Policy/forward_speed": forward_vel.clamp(min=0.0).mean(),
            },
        }

        self._reset_mask.copy_(dones)
        self._reset_worlds()
        self._obs = self._compute_obs()
        return self._obs, rewards, dones, extras

    def _push(self) -> None:
        pushed = torch.rand(self.num_envs, device=self.device) < PUSH_PROB
        push = torch.randn(self.num_envs, 3, device=self.device) * torch.tensor(
            [PUSH_STD[0], PUSH_STD[0], PUSH_STD[1]], device=self.device
        )
        self._qvel[:, self._base_dofs] += push * pushed[:, None]

    def _run_physics(self) -> None:
        """One control period: the command each chunk sees is the one issued `latency` ago,
        passed through the motor model."""
        n = self.num_envs
        latency = (self.latency + torch.randn(n, device=self.device) * CMD_LATENCY_JITTER_STD).clamp(0.0, CMD_LATENCY_MAX)
        cmd_noise = torch.randn(n, 2, device=self.device) * torch.tensor(CMD_NOISE_STD, device=self.device)
        alpha = 1.0 - torch.exp(-CHUNK_DT / self.motor_tau)
        max_delta = self.motor_accel * CHUNK_DT
        # reset_data zeroes ctrl in reset worlds, so the arm/head targets are rewritten every step.
        self._ctrl.copy_(self._home_ctrl.expand_as(self._ctrl))
        for chunk in range(CHUNKS_PER_STEP):
            steps_back = torch.ceil((latency - chunk * CHUNK_DT) / CONTROL_DT).clamp(0, CMD_HISTORY - 1).long()
            cmd = self.cmd_history.gather(1, steps_back[:, None, None].expand(-1, 1, 2)).squeeze(1)
            target = cmd * self.motor_gain + cmd_noise
            target[:, 1] += self.turn_bias
            target = torch.where(target.abs() < self.deadband, 0.0, target)
            self.motor_vel += ((target - self.motor_vel) * alpha).clamp(-max_delta, max_delta)
            self._ctrl[:, self._forward_id] = self.motor_vel[:, 0]
            self._ctrl[:, self._yaw_id] = self.motor_vel[:, 1]
            wp.capture_launch(self._chunk_graph)

    def _update_odometry(self) -> None:
        """Integrates wheel odometry from the true motion: scale error, random-walk noise, and no
        sideways component (wheel encoders can't see slip)."""
        pose = self._pose()
        delta = pose[:, :2] - self.prev_pose[:, :2]
        forward = self.prev_pose[:, 2].cos() * delta[:, 0] + self.prev_pose[:, 2].sin() * delta[:, 1]
        dyaw = _wrap(pose[:, 2] - self.prev_pose[:, 2])
        noise = torch.randn(self.num_envs, 2, device=self.device) * torch.tensor(ODOM_STEP_NOISE, device=self.device)
        measured = torch.stack((forward, dyaw), dim=1) * self.odom_scale + noise
        heading = self.est_pose[:, 2] + measured[:, 1] / 2
        self.est_pose[:, 0] += measured[:, 0] * heading.cos()
        self.est_pose[:, 1] += measured[:, 0] * heading.sin()
        self.est_pose[:, 2] = _wrap(self.est_pose[:, 2] + measured[:, 1])
        vel_noise = torch.randn(self.num_envs, 2, device=self.device) * torch.tensor(ODOM_VEL_NOISE, device=self.device)
        self.meas_vel = measured / CONTROL_DT + vel_noise
        self.prev_pose = pose

    def _pose(self) -> torch.Tensor:
        return self._qpos[:, [self._base_x, self._base_y, self._base_yaw]]

    def _robot_center(self) -> torch.Tensor:
        pose = self._pose()
        return pose[:, :2] + ROBOT_CENTER_OFFSET * torch.stack((pose[:, 2].cos(), pose[:, 2].sin()), dim=1)

    def _shape_distance(
        self, points: torch.Tensor, shape_xy: torch.Tensor | None = None, shape_yaw: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Signed distance (n, N_SHAPES) from one point per world to every wall/obstacle outline
        (the current layout unless one is given)."""
        shape_xy = self.shape_xy if shape_xy is None else shape_xy
        shape_yaw = self.shape_yaw if shape_yaw is None else shape_yaw
        d = points[:, None, :] - shape_xy
        cos, sin = shape_yaw.cos(), shape_yaw.sin()
        local = torch.stack((cos * d[..., 0] + sin * d[..., 1], -sin * d[..., 0] + cos * d[..., 1]), dim=-1)
        q = local.abs() - self._shape_half
        return torch.linalg.norm(q.clamp(min=0.0), dim=-1) + q.max(dim=-1).values.clamp(max=0.0) - self._shape_round

    def _clearance(self, center: torch.Tensor) -> torch.Tensor:
        """Gap (m) between the robot's footprint circle and the nearest wall or obstacle; < 0 is a collision."""
        return self._shape_distance(center).min(dim=1).values - ROBOT_RADIUS

    def _reset_worlds(self) -> None:
        """Re-randomizes the worlds in `_reset_mask`. Samples for every world and keeps the
        masked ones, so there is no host sync on which worlds are done."""
        mask = self._reset_mask
        mjw.reset_data(self.sim.model, self.sim.data, self._reset_mask_wp)
        keep = lambda new, old: torch.where(mask.view(-1, *([1] * (old.dim() - 1))), new, old)  # noqa: E731

        room_half, robot_pose, goal, shapes_xy, shapes_yaw = self._sample_layout()
        self.shape_xy = keep(shapes_xy, self.shape_xy)
        self.shape_yaw = keep(shapes_yaw, self.shape_yaw)
        self.goal = keep(goal, self.goal)
        qpos = self._home_qpos.expand(self.num_envs, -1).clone()
        qpos[:, self._base_x] = robot_pose[:, 0]
        qpos[:, self._base_y] = robot_pose[:, 1]
        qpos[:, self._base_yaw] = robot_pose[:, 2]
        self._qpos.copy_(keep(qpos, self._qpos))
        self._mocap_pos[:, self._shape_mocap, :2] = self.shape_xy
        half_yaw = self.shape_yaw / 2
        self._mocap_quat[:, self._shape_mocap] = torch.stack(
            (half_yaw.cos(), torch.zeros_like(half_yaw), torch.zeros_like(half_yaw), half_yaw.sin()), dim=-1
        )
        self._mocap_pos[:, self._goal_mocap, :2] = self.goal

        self._sample_hidden_params(keep)
        pose = self._pose()
        self.prev_pose = keep(pose, self.prev_pose)
        self.est_pose = keep(pose, self.est_pose)
        for name in ("action_history", "cmd_history", "motor_vel", "meas_vel"):
            setattr(self, name, keep(torch.zeros_like(getattr(self, name)), getattr(self, name)))
        self.episode_length_buf.masked_fill_(mask, 0)
        self.prev_goal_dist = keep(torch.linalg.norm(self.goal - pose[:, :2], dim=1), self.prev_goal_dist)
        wp.capture_launch(self._forward_graph)

    def _sample_layout(self) -> tuple[torch.Tensor, ...]:
        n, dev = self.num_envs, self.device
        rand = lambda *shape: torch.rand(n, *shape, device=dev)  # noqa: E731
        lo, hi = ROOM_HALF_RANGE
        room_half = lo + (hi - lo) * rand(2)
        spawn_half = room_half - SPAWN_MARGIN
        robot_xy = (rand(2) * 2 - 1) * spawn_half
        robot_pose = torch.cat((robot_xy, ((rand(1) * 2 - 1) * math.pi)), dim=1)
        goal = (rand(2) * 2 - 1) * spawn_half
        for _ in range(10):
            too_close = torch.linalg.norm(goal - robot_xy, dim=1) < MIN_GOAL_DISTANCE
            goal = torch.where(too_close[:, None], (rand(2) * 2 - 1) * spawn_half, goal)

        wall_offset = room_half + WALL_HALF_THICKNESS
        zero = torch.zeros(n, device=dev)
        walls = torch.stack(
            (
                torch.stack((zero, wall_offset[:, 1]), dim=1),
                torch.stack((zero, -wall_offset[:, 1]), dim=1),
                torch.stack((wall_offset[:, 0], zero), dim=1),
                torch.stack((-wall_offset[:, 0], zero), dim=1),
            ),
            dim=1,
        )
        n_obstacles = N_SHAPES - N_WALLS
        obstacles = (rand(n_obstacles, 2) * 2 - 1) * room_half[:, None, :]
        shapes_xy = torch.cat((walls, obstacles), dim=1)
        shapes_yaw = torch.cat((torch.zeros(n, N_WALLS, device=dev), rand(n_obstacles) * math.pi), dim=1)

        lo, hi = OBSTACLE_DENSITY_RANGE
        density = lo + (hi - lo) * rand(1)
        lo, hi = LOW_OBSTACLE_DENSITY_RANGE
        low_density = (lo + (hi - lo) * rand(1)) * (rand(1) < LOW_OBSTACLE_EPISODE_PROB)
        active = torch.cat((rand(N_WALLS) < WALL_PRESENT_PROB, rand(N_TALL) < density, rand(N_LOW) < low_density), dim=1)
        # Obstacles that would overlap the spawn or the goal are left parked rather than resampled.
        spawn_clear = ROBOT_RADIUS + abs(ROBOT_CENTER_OFFSET) + OBSTACLE_SPAWN_CLEARANCE
        is_obstacle = torch.arange(N_SHAPES, device=dev) >= N_WALLS
        active &= ~is_obstacle | (self._shape_distance(robot_xy, shapes_xy, shapes_yaw) > spawn_clear)
        active &= ~is_obstacle | (self._shape_distance(goal, shapes_xy, shapes_yaw) > GOAL_RADIUS + OBSTACLE_SPAWN_CLEARANCE)
        shapes_xy = torch.where(active[..., None], shapes_xy, self._park_xy)
        return room_half, robot_pose, goal, shapes_xy, shapes_yaw

    def _sample_hidden_params(self, keep) -> None:  # noqa: ANN001 -- the masked-select closure from _reset_worlds
        n, dev = self.num_envs, self.device

        def uniform(lo_hi: tuple, *shape: int) -> torch.Tensor:
            lo, hi = torch.tensor(lo_hi, device=dev).unbind(-1) if isinstance(lo_hi[0], tuple) else lo_hi
            return lo + (hi - lo) * torch.rand(n, *shape, device=dev)

        latency = (CMD_LATENCY_MEAN + CMD_LATENCY_STD * torch.randn(n, device=dev)).clamp(0.0, CMD_LATENCY_MAX)
        self.latency = keep(latency, self.latency)
        self.motor_gain = keep(uniform(MOTOR_GAIN_RANGE, 2), self.motor_gain)
        self.turn_bias = keep(TURN_BIAS_STD * torch.randn(n, device=dev), self.turn_bias)
        self.deadband = keep(torch.rand(n, 2, device=dev) * torch.tensor(DEADBAND_MAX, device=dev), self.deadband)
        self.motor_tau = keep(uniform(MOTOR_TAU_RANGE, 2), self.motor_tau)
        self.motor_accel = keep(uniform(MOTOR_ACCEL_RANGE, 2), self.motor_accel)
        period = uniform(SCAN_PERIOD_RANGE)
        self.scan_period = keep(period, self.scan_period)
        # _compute_obs adds CONTROL_DT next: this forces a fresh scan now and leaves the episode
        # at a random phase of the lidar's cycle.
        self.scan_age = keep(period * (1 + torch.rand(n, device=dev)) - CONTROL_DT, self.scan_age)
        self.lidar_dropout = keep(uniform(LIDAR_DROPOUT_RANGE), self.lidar_dropout)
        shift = torch.round(torch.randn(n, device=dev) * LIDAR_YAW_OFFSET_STD_DEG * LIDAR_BEAMS / 360).long()
        self.lidar_shift = keep(shift, self.lidar_shift)
        self.odom_scale = keep(uniform(ODOM_SCALE_RANGE, 2), self.odom_scale)
        period = uniform(CAMERA_PERIOD_RANGE)
        self.camera_period = keep(period, self.camera_period)
        self.camera_age = keep(period * (1 + torch.rand(n, device=dev)) - CONTROL_DT, self.camera_age)
        self.camera_dropout = keep(uniform(CAMERA_DROPOUT_RANGE), self.camera_dropout)
        self.camera_noise_k = keep(uniform(CAMERA_NOISE_K_RANGE), self.camera_noise_k)
        self.camera_pitch = keep(torch.randn(n, device=dev) * math.radians(CAMERA_PITCH_STD_DEG), self.camera_pitch)

    def _scan_ranges(self) -> torch.Tensor:
        """Raw beams with misses and self-hits as max range: what the lidar sees with no noise."""
        r = self._lidar
        return torch.where((r < LIDAR_MIN_RANGE), LIDAR_MAX_RANGE, r).clamp(max=LIDAR_MAX_RANGE)

    def _noisy_scan(self) -> torch.Tensor:
        n, dev = self.num_envs, self.device
        index = (self._beam_index + self.lidar_shift[:, None]) % LIDAR_BEAMS
        r = self._lidar.gather(1, index)
        hit = r > 0
        r = r + torch.randn_like(r) * (LIDAR_NOISE_STD + LIDAR_NOISE_REL * r.clamp(min=0.0))
        spurious = torch.rand(n, LIDAR_BEAMS, device=dev) < LIDAR_SPURIOUS_PROB
        r = torch.where(spurious, LIDAR_MIN_RANGE + torch.rand_like(r) * (LIDAR_MAX_RANGE - LIDAR_MIN_RANGE), r)
        dropped = torch.rand(n, LIDAR_BEAMS, device=dev) < self.lidar_dropout[:, None]
        invalid = (~hit & ~spurious) | dropped | (r < LIDAR_MIN_RANGE)
        return torch.where(invalid, LIDAR_MAX_RANGE, r).clamp(max=LIDAR_MAX_RANGE)

    def _bin(self, ranges: torch.Tensor) -> torch.Tensor:
        """Min-pools beams into LIDAR_BINS sectors; must match the robot-side binning in deploy_spec()."""
        return ranges.view(self.num_envs, LIDAR_BINS, -1).min(dim=2).values / LIDAR_MAX_RANGE

    def _camera_ray_distances(self) -> torch.Tensor:
        """(n, rays) distance from the camera to the first floor or obstacle hit; inf for a miss or a
        ray the robot's own body blocks. Exact ray-box / ray-cylinder intersection in each shape's frame."""
        pose = self._pose()
        cos, sin = pose[:, 2].cos()[:, None], pose[:, 2].sin()[:, None]
        rel = self.shape_xy - pose[:, None, :2]  # (n, S, 2), world axes
        rel = torch.stack((cos * rel[..., 0] + sin * rel[..., 1], -sin * rel[..., 0] + cos * rel[..., 1]), dim=-1)  # base axes
        psi = self.shape_yaw - pose[:, 2:3]  # shape yaw in the base frame
        c, s = psi.cos()[..., None], psi.sin()[..., None]  # (n, S, 1)
        # Ray origin and horizontal direction in each shape's frame.
        o = self._camera_pos[None, None, :2] - rel  # (n, S, 2)
        ox, oy = (c[..., 0] * o[..., 0] + s[..., 0] * o[..., 1])[..., None], (-s[..., 0] * o[..., 0] + c[..., 0] * o[..., 1])[..., None]
        dx = c * self._ray_dir[:, 0] + s * self._ray_dir[:, 1]  # (n, S, R)
        dy = -s * self._ray_dir[:, 0] + c * self._ray_dir[:, 1]
        dz = self._ray_dir[:, 2]
        # Vertical slab [0, height].
        oz = self._camera_pos[2]
        tz0, tz1 = (0.0 - oz) / dz, (self._shape_height[:, None] - oz) / dz
        t_in, t_out = torch.minimum(tz0, tz1), torch.maximum(tz0, tz1)  # (S, R), broadcast over n
        # Boxes: x and y slabs.
        half = self._shape_half[None, :, :, None]  # (1, S, 2, 1)
        tx0, tx1 = (-half[:, :, 0] - ox) / dx, (half[:, :, 0] - ox) / dx
        ty0, ty1 = (-half[:, :, 1] - oy) / dy, (half[:, :, 1] - oy) / dy
        box_in = torch.maximum(torch.minimum(tx0, tx1), torch.minimum(ty0, ty1))
        box_out = torch.minimum(torch.maximum(tx0, tx1), torch.maximum(ty0, ty1))
        # Cylinders: |o + t d| = radius in the horizontal plane.
        a = dx.square() + dy.square()
        b = 2 * (ox * dx + oy * dy)
        k = ox.square() + oy.square() - self._shape_round[None, :, None].square()
        root = (b.square() - 4 * a * k).clamp(min=0.0).sqrt()
        miss = b.square() - 4 * a * k < 0
        cyl_in = torch.where(miss, torch.inf, (-b - root) / (2 * a))
        cyl_out = torch.where(miss, -torch.inf, (-b + root) / (2 * a))
        cylinder = self._shape_is_cylinder[None, :, None]
        enter = torch.maximum(torch.where(cylinder, cyl_in, box_in).nan_to_num(-torch.inf), t_in)
        leave = torch.minimum(torch.where(cylinder, cyl_out, box_out).nan_to_num(torch.inf), t_out)
        hit = (leave >= enter) & (leave > 0)
        shape_dist = torch.where(hit, enter.clamp(min=0.0), torch.inf).amin(dim=1)  # (n, R)
        floor = torch.where(dz < 0, -oz / dz, torch.inf)
        dist = torch.minimum(shape_dist, floor.expand_as(shape_dist))
        return torch.where(self._ray_blocked, torch.inf, dist)

    def _camera_points(self, noisy: bool, dist: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """The ray fan as stereo points: (n, rays, 3) in base_link and (n, rays) optical-axis depth;
        misses are inf. `noisy` adds the stereo depth noise, holes, spurious matches and the pitch
        error (the points are placed with the nominal geometry while the head is off by camera_pitch).
        `dist` reuses this step's _camera_ray_distances()."""
        n = self.num_envs
        dist = self._camera_ray_distances() if dist is None else dist
        elevation = self._ray_el.expand(n, -1)
        if noisy:
            depth = dist * self._ray_depth_per_m
            dist = dist + torch.randn_like(dist) * self.camera_noise_k[:, None] * depth.clamp(max=10.0).square() / self._ray_depth_per_m
            spurious = torch.rand_like(dist) < CAMERA_SPURIOUS_PROB
            fake = (CAMERA_MIN_DEPTH + torch.rand_like(dist) * (CAMERA_MAX_DEPTH - CAMERA_MIN_DEPTH)) / self._ray_depth_per_m
            dist = torch.where(spurious, fake, dist)
            dist = torch.where(torch.rand_like(dist) < self.camera_dropout[:, None], torch.inf, dist)
            elevation = elevation - self.camera_pitch[:, None]
        horizontal = dist * elevation.cos()
        points = torch.stack(
            (
                self._camera_pos[0] + horizontal * self._ray_az.cos(),
                self._camera_pos[1] + horizontal * self._ray_az.sin(),
                self._camera_pos[2] + dist * elevation.sin(),
            ),
            dim=-1,
        )
        return points, dist * self._ray_depth_per_m

    def _camera_bins(self, noisy: bool, dist: torch.Tensor | None = None) -> torch.Tensor:
        """Stereo points -> CAMERA_BINS obstacle ranges (0-1), as deploy_spec() describes; must match
        the robot-side skills/rl_nav_utils.bin_camera."""
        points, depth = self._camera_points(noisy, dist)
        x, y, z = points.unbind(-1)
        r = torch.hypot(x, y).nan_to_num(CAMERA_MAX_RANGE)
        bearing = torch.atan2(y, x)
        obstacle = (
            (depth >= CAMERA_MIN_DEPTH)
            & (depth <= CAMERA_MAX_DEPTH)
            & (z >= CAMERA_MIN_HEIGHT + CAMERA_MIN_HEIGHT_PER_M * r)
            & (z <= CAMERA_MAX_HEIGHT)
            & (bearing.abs() < CAMERA_HFOV / 2)
        )
        index = ((bearing.nan_to_num(0.0) + CAMERA_HFOV / 2) / (CAMERA_HFOV / CAMERA_BINS)).long().clamp(0, CAMERA_BINS - 1)
        ranges = torch.where(obstacle, r, CAMERA_MAX_RANGE).clamp(max=CAMERA_MAX_RANGE)
        out = torch.full((self.num_envs, CAMERA_BINS), CAMERA_MAX_RANGE, device=self.device)
        return out.scatter_reduce(1, index, ranges, reduce="amin") / CAMERA_MAX_RANGE

    def _goal_features(self, pose: torch.Tensor) -> torch.Tensor:
        cos, sin = pose[:, 2].cos(), pose[:, 2].sin()
        delta = self.goal - pose[:, :2]
        body = torch.stack((cos * delta[:, 0] + sin * delta[:, 1], -sin * delta[:, 0] + cos * delta[:, 1]), dim=1)
        heading = torch.atan2(body[:, 1], body[:, 0])
        distance = torch.linalg.norm(delta, dim=1, keepdim=True)
        return torch.cat((body / GOAL_OBS_SCALE, distance / GOAL_OBS_SCALE, heading.cos()[:, None], heading.sin()[:, None]), 1)

    def _compute_obs(self) -> TensorDict:
        self.scan_age += CONTROL_DT
        fresh = self.scan_age >= self.scan_period
        self.scan_age = torch.where(fresh, self.scan_age - self.scan_period, self.scan_age)
        self.scan_obs = torch.where(fresh[:, None], self._bin(self._noisy_scan()), self.scan_obs)
        self.camera_age += CONTROL_DT
        fresh = self.camera_age >= self.camera_period
        self.camera_age = torch.where(fresh, self.camera_age - self.camera_period, self.camera_age)
        ray_dist = self._camera_ray_distances()
        self.camera_obs = torch.where(fresh[:, None], self._camera_bins(True, ray_dist), self.camera_obs)
        history = self.action_history.flatten(1)
        actor = torch.cat(
            (self.scan_obs, self._goal_features(self.est_pose), self.meas_vel / self._max_cmd, history, self.camera_obs), dim=1
        )

        pose = self._pose()
        vel = self._qvel[:, self._base_dofs]
        true_vel = torch.stack((pose[:, 2].cos() * vel[:, 0] + pose[:, 2].sin() * vel[:, 1], vel[:, 2]), dim=1)
        odom_error = self.est_pose - pose
        odom_error[:, 2] = _wrap(odom_error[:, 2])
        hidden = torch.cat(
            (
                self.latency[:, None] / CMD_LATENCY_MAX,
                self.motor_gain,
                self.turn_bias[:, None] / MAX_TURN_RATE,
                self.motor_tau / MOTOR_TAU_RANGE[1],
                self.motor_accel / torch.tensor([r[1] for r in MOTOR_ACCEL_RANGE], device=self.device),
                self.motor_vel / self._max_cmd,
                self.scan_age[:, None] / SCAN_PERIOD_RANGE[1],
                odom_error * 10.0,
                self.camera_age[:, None] / CAMERA_PERIOD_RANGE[1],
                self.camera_dropout[:, None],
                self.camera_pitch[:, None] / math.radians(CAMERA_PITCH_STD_DEG),
            ),
            dim=1,
        )
        critic = torch.cat(
            (self._bin(self._scan_ranges()), self._goal_features(pose), true_vel / self._max_cmd, history, self._camera_bins(False, ray_dist), hidden),
            dim=1,
        )
        return TensorDict({"policy": actor, "critic": critic}, batch_size=[self.num_envs], device=self.device)


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
