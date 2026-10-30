"""UNSAFE: runs the legacy nav policy in mjwarp/nav/exported/ on the robot, with no safety checks.

That export is from the first, noise-free training run: 49 observations (one past action), actions
scaled to 0.6 m/s and 2.5 rad/s, and an ONNX file with its weights in policy.onnx.data. It learned
to reach goals mostly in reverse: it turns until the goal is behind it, then backs up. It doesn't
fit skills/rl_nav_utils.py, so this node builds its observation itself, from the layout in its
policy_io.json. It was never checked with mjwarp/verify_deploy.py.

Unlike rl_nav_node there is no hard stop near obstacles, no stale-feed stop, no goal distance limit
and no timeout. It still uses the same topics, so `rlnav goal`, `rlnav cancel` and `rlnav status`
work with it. Don't run it at the same time as rl_nav_node or the RlNavigate skill.

Run it with `ros2 run icarus rl_nav_unsafe_node` (not ros2 launch: it waits for Enter on stdin).
Ctrl-C (or SIGTERM) publishes a zero Twist before exiting; the cmd_vel mux also stops the base
within 0.5 s of the node going silent.
"""

import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

GOAL_TOPIC = "/icarus/rl_nav/goal"
CANCEL_TOPIC = "/icarus/rl_nav/cancel"
STATUS_TOPIC = "/icarus/rl_nav/status"

# The legacy export's contract (mjwarp/nav/exported/policy_io.json). It predates
# nav_env.deploy_spec(), so its scalings are only in its notes; they are written out here.
CONTROL_HZ = 10.0
LIDAR_BINS = 40
LIDAR_MAX_RANGE = 4.0
GOAL_OBS_SCALE = 5.0
MAX_FORWARD_SPEED = 0.6
MAX_TURN_RATE = 2.5
GOAL_RADIUS = 0.3
OBS_SIZE = LIDAR_BINS + 2 + 1 + 2 + 1 + 1 + 2
# In training these bins (0-18 deg left of forward) always saw the robot's own body at ~0.16 m
# (normalizer mean 0.04, std 0.01). Any other value is ~100 std out of distribution and saturates
# the policy, so they are pinned to what it saw in training. It is blind to obstacles there.
PINNED_BINS = (20, 21)
PINNED_VALUE = 0.04

WARNING = f"""
\033[1;31m================================ WARNING ================================
 rl_nav_unsafe_node drives the robot with the LEGACY policy and NO SAFETY CHECKS:
   - no hard stop near obstacles, no stale-sensor stop, no goal limit, no timeout
   - commands up to {MAX_FORWARD_SPEED} m/s (the checked policy uses 0.4); turn rate capped by the
     max_angular_speed parameter (default 0.5 rad/s, the policy was trained at up to {MAX_TURN_RATE})
   - this policy was never verified against the robot-side code
   - it mostly DRIVES BACKWARDS: it turns the goal behind it, then reverses at full speed
   - it is blind to obstacles 0-18 deg left of straight ahead (bins pinned to the self-hit)
 Keep the E-stop in hand. Ctrl-C stops the robot.
=========================================================================\033[0m
"""

Pose2 = tuple[float, float, float]


def _yaw(q) -> float:  # noqa: ANN001 -- geometry_msgs Quaternion
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def bin_scan(msg: LaserScan) -> np.ndarray:
    """Beams CCW from -pi, min-pooled into LIDAR_BINS bins; misses count as max range; 0-1."""
    r = np.asarray(msg.ranges, dtype=np.float32)
    valid = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
    r = np.where(valid, np.minimum(r, LIDAR_MAX_RANGE), LIDAR_MAX_RANGE)
    angles = msg.angle_min + np.arange(r.size) * msg.angle_increment
    index = np.minimum((np.mod(angles + math.pi, 2 * math.pi) / (2 * math.pi / LIDAR_BINS)).astype(np.int64), LIDAR_BINS - 1)
    out = np.full(LIDAR_BINS, LIDAR_MAX_RANGE, dtype=np.float32)
    np.minimum.at(out, index, r)
    return out / LIDAR_MAX_RANGE


class RlNavUnsafeNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_nav_unsafe")
        p = self.declare_parameter
        default_dir = Path(get_package_share_directory("icarus")) / "rl_nav" / "legacy_policy"
        policy_dir = Path(p("policy_dir", str(default_dir)).value)
        # Multiplies the commands only; the velocity observation keeps the training scalings.
        self._speed_scale = float(p("speed_scale", 1.0).value)
        # Caps the published turn rate only; the policy's last-action input stays as it chose.
        self._max_angular_speed = float(p("max_angular_speed", 0.5).value)
        cmd_vel_topic = p("cmd_vel_topic", "/cmd_vel_skills").value

        self._session = onnxruntime.InferenceSession(str(policy_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
        model_input = self._session.get_inputs()[0]
        if model_input.shape[-1] != OBS_SIZE:
            raise RuntimeError(f"{policy_dir / 'policy.onnx'} takes {model_input.shape[-1]} observations, this node builds {OBS_SIZE}")
        self._input_name = model_input.name

        self._bins: np.ndarray | None = None
        self._scan_received = 0.0
        self._odom: Odometry | None = None
        self._odom_received = 0.0
        self._goal: tuple[float, float] | None = None
        self._last_action = np.zeros(2, dtype=np.float32)
        self._cmd = (0.0, 0.0)
        self._state, self._message = "idle", "waiting for a goal"
        self._step_us = 0

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self._status_pub = self.create_publisher(String, STATUS_TOPIC, 10)
        self.create_subscription(LaserScan, p("scan_topic", "/scan").value, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, p("odom_topic", "/odom").value, self._on_odom, 10)
        self.create_subscription(PoseStamped, GOAL_TOPIC, self._on_goal, 10)
        self.create_subscription(Bool, CANCEL_TOPIC, self._on_cancel, 10)
        self.create_timer(1.0 / CONTROL_HZ, self._tick)
        self.get_logger().warn(
            f"UNSAFE legacy policy from {policy_dir}, driving {cmd_vel_topic} "
            f"(speed_scale {self._speed_scale}, turn rate capped at {self._max_angular_speed} rad/s)"
        )

    def _on_scan(self, msg: LaserScan) -> None:
        self._bins = bin_scan(msg)
        self._scan_received = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg
        self._odom_received = time.monotonic()

    def _pose(self) -> Pose2:
        pose = self._odom.pose.pose
        return pose.position.x, pose.position.y, _yaw(pose.orientation)

    def _on_goal(self, msg: PoseStamped) -> None:
        if self._odom is None:
            self._set_state("error", "no odometry yet")
            return
        odom_frame = self._odom.header.frame_id
        x, y = msg.pose.position.x, msg.pose.position.y
        if msg.header.frame_id not in ("", odom_frame):
            # Expressed in odom once, when it arrives: a map goal won't follow later AMCL corrections.
            try:
                t = self._tf.lookup_transform(odom_frame, msg.header.frame_id, Time()).transform
            except TransformException as e:
                self._set_state("error", f"no transform {msg.header.frame_id} -> {odom_frame}: {e}")
                return
            theta = _yaw(t.rotation)
            x, y = (
                t.translation.x + math.cos(theta) * x - math.sin(theta) * y,
                t.translation.y + math.sin(theta) * x + math.cos(theta) * y,
            )
        self._goal = (x, y)
        self._last_action[:] = 0.0
        self._set_state("driving", f"goal ({x:.2f}, {y:.2f}) in {odom_frame}")

    def _on_cancel(self, msg: Bool) -> None:
        if msg.data and self._goal is not None:
            self._end_run("cancelled", "cancelled")

    def _set_state(self, state: str, message: str) -> None:
        self._state, self._message = state, message
        self.get_logger().info(f"{state}: {message}")

    def _end_run(self, state: str, message: str) -> None:
        self._goal = None
        self.stop_robot()
        self._set_state(state, message)

    def stop_robot(self) -> None:
        self._cmd = (0.0, 0.0)
        self._cmd_pub.publish(Twist())

    def _tick(self) -> None:
        remaining = None
        if self._goal is not None and self._bins is not None and self._odom is not None:
            x, y, theta = self._pose()
            dx, dy = self._goal[0] - x, self._goal[1] - y
            gx, gy = math.cos(theta) * dx + math.sin(theta) * dy, -math.sin(theta) * dx + math.cos(theta) * dy
            remaining = math.hypot(gx, gy)
            if remaining < GOAL_RADIUS:
                self._end_run("reached", f"Reached the goal ({remaining:.2f} m off)")
            else:
                self._drive(gx, gy, remaining)
        self._publish_status(remaining)

    def _drive(self, gx: float, gy: float, distance: float) -> None:
        twist = self._odom.twist.twist
        bins = self._bins.copy()
        bins[list(PINNED_BINS)] = PINNED_VALUE
        obs = np.concatenate(
            [
                bins,
                [gx / GOAL_OBS_SCALE, gy / GOAL_OBS_SCALE, distance / GOAL_OBS_SCALE, gx / distance, gy / distance],
                [twist.linear.x / MAX_FORWARD_SPEED, twist.angular.z / MAX_TURN_RATE],
                self._last_action,
            ]
        ).astype(np.float32)
        start = time.perf_counter()
        raw = self._session.run(None, {self._input_name: obs[None]})[0][0]
        self._step_us = int((time.perf_counter() - start) * 1e6)
        self._last_action = np.clip(raw, -1.0, 1.0).astype(np.float32)
        msg = Twist()
        msg.linear.x = float(self._last_action[0]) * MAX_FORWARD_SPEED * self._speed_scale
        turn = float(self._last_action[1]) * MAX_TURN_RATE * self._speed_scale
        msg.angular.z = max(-self._max_angular_speed, min(self._max_angular_speed, turn))
        self._cmd = (msg.linear.x, msg.angular.z)
        self._cmd_pub.publish(msg)

    def _publish_status(self, remaining: float | None) -> None:
        now = time.monotonic()
        status = {
            "state": self._state,
            "message": self._message,
            "frame": self._odom.header.frame_id if self._odom else "odom",
            "robot": [round(v, 3) for v in self._pose()] if self._odom else None,
            "goal": [round(v, 3) for v in self._goal] if self._goal else None,
            "remaining": round(remaining, 2) if remaining is not None else None,
            "time_left": None,
            "cmd": [round(v, 2) for v in self._cmd],
            "blocked": None,
            "stale": None,
            "safety_stops": None,
            "bins": self._bins.round(3).tolist() if self._bins is not None else [],
            "lidar_max_range": LIDAR_MAX_RANGE,
            "dry_run": False,
            "backend": "onnxruntime (legacy, UNSAFE)",
            "step_us": self._step_us,
            "scan_age": round(now - self._scan_received, 2) if self._bins is not None else None,
            "odom_age": round(now - self._odom_received, 2) if self._odom else None,
        }
        self._status_pub.publish(String(data=json.dumps(status)))


def _interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> None:
    print(WARNING, file=sys.stderr, flush=True)
    try:
        input("Press Enter to start the node (Ctrl-C to abort): ")
    except (EOFError, KeyboardInterrupt):
        print("\naborted", file=sys.stderr)
        return
    # rclpy's own SIGINT handler shuts the context down before we could publish the final stop.
    # SIGINT is set explicitly too: a process started in the background inherits it ignored.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = None
    try:
        node = RlNavUnsafeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            for _ in range(3):
                node.stop_robot()
                time.sleep(0.05)
            node.get_logger().warn("stopped: sent zero velocity")
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
