"""Runs the MJWarp-trained RL nav policy on the robot as a ROS 2 node, and serves its web UI.

Inputs:  /scan (sensor_msgs/LaserScan), /odom (nav_msgs/Odometry), TF (goal frame -> odom,
         base_link -> laser), and goals on /icarus/rl_nav/goal (geometry_msgs/PoseStamped, in
         `map` or `odom`; orientation is ignored, the policy only drives to a point).
Output:  geometry_msgs/Twist on /cmd_vel_skills, the skills input of innate-os's cmd_vel mux,
         so teleop still overrides it. Nothing is published while there is no goal.
Control: std_msgs/Bool true on /icarus/rl_nav/cancel stops the run (std_msgs/Empty can't be
         published through rws); the `dry_run` parameter runs the policy without driving.
Status:  std_msgs/String JSON on /icarus/rl_nav/status: state, poses in the display frame
         (map when map->odom exists, else odom), the scan, what the policy sees, its command.
UI:      http://<robot>:<http_port>/ (static page that talks to rosbridge, port 9090).

The policy, observation and safety checks are skills/rl_nav_utils.py, shared with the RlNavigate
skill and checked against the trainer by mjwarp/verify_deploy.py. The goal is held in its own
frame and re-expressed in odom every tick, so a map goal follows AMCL's corrections of odometry
drift; the policy itself still sees the goal relative to odometry, as in training.
"""

import importlib.util
import json
import math
import sys
import threading
import time
import types
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

GOAL_TOPIC = "/icarus/rl_nav/goal"
CANCEL_TOPIC = "/icarus/rl_nav/cancel"
STATUS_TOPIC = "/icarus/rl_nav/status"
# Wait this long for the first scan and odometry of a run, or for a lost goal-frame transform.
FEED_TIMEOUT_S = 3.0
TF_TIMEOUT_S = 1.0
STATUS_SCAN_POINTS = 720

Pose2 = tuple[float, float, float]


def _load_utils(path: Path) -> types.ModuleType:
    """skills/rl_nav_utils.py is a file of the skill pack, not of a Python package."""
    spec = importlib.util.spec_from_file_location("rl_nav_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rl_nav_utils"] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw(q) -> float:  # noqa: ANN001 -- geometry_msgs Quaternion
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _compose(a: Pose2, b: Pose2) -> Pose2:
    """Pose b, given in frame a, expressed in a's parent frame."""
    x, y, theta = a
    return (x + math.cos(theta) * b[0] - math.sin(theta) * b[1], y + math.sin(theta) * b[0] + math.cos(theta) * b[1], _wrap(theta + b[2]))


def _stamp(msg_stamp) -> float:  # noqa: ANN001 -- builtin_interfaces Time
    return msg_stamp.sec + msg_stamp.nanosec * 1e-9


@dataclass
class _Scan:
    """The LaserScan fields NavController reads, with the stamp in seconds."""

    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    stamp: float
    frame_id: str
    received: float


@dataclass
class _Odom:
    pose: Pose2
    stamp: float
    frame_id: str
    received: float


@dataclass
class _Run:
    frame: str
    goal: tuple[float, float]
    dry_run: bool
    started: float
    deadline: float | None = None
    tf_ok_at: float = 0.0


class _UiHandler(SimpleHTTPRequestHandler):
    """The static UI, plus /config.json telling it where rosbridge and the topics are."""

    config: dict = {}

    def do_GET(self) -> None:  # noqa: N802 -- http.server's name
        if self.path.split("?")[0] != "/config.json":
            super().do_GET()
            return
        body = json.dumps(self.config).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- http.server's name
        pass


class RlNavNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_nav")
        share = Path(get_package_share_directory("icarus")) / "rl_nav"
        p = self.declare_parameter
        skills_dir = Path(p("skills_dir", str(share)).value)
        policy_dir = Path(p("policy_dir", str(skills_dir / "rl_nav_policy")).value)
        backend = p("backend", "auto").value
        self._cmd_vel_topic = p("cmd_vel_topic", "/cmd_vel_skills").value
        scan_topic = p("scan_topic", "/scan").value
        odom_topic = p("odom_topic", "/odom").value
        self._map_frame = p("map_frame", "map").value
        self._base_frame = p("base_frame", "base_link").value
        self._max_goal_distance = p("max_goal_distance", 5.0).value
        self._time_limit_per_meter = p("time_limit_per_meter", 10.0).value
        self._min_time_limit = p("min_time_limit", 20.0).value
        p("dry_run", False)
        status_hz = p("status_hz", 5.0).value
        http_port = p("http_port", 8080).value
        rosbridge_port = p("rosbridge_port", 9090).value

        self._u = _load_utils(skills_dir / "rl_nav_utils.py")
        self._spec = self._u.NavSpec.load(policy_dir / "policy_io.json")
        self._controller = self._u.NavController(
            self._spec,
            self._u.load_policy(policy_dir, backend),
            safety_stop_range=p("safety_stop_range", 0.25).value,
            safety_sector=math.radians(p("safety_sector_deg", 60.0).value),
            stale_scan_s=p("stale_scan_s", 0.6).value,
            stale_odom_s=p("stale_odom_s", 0.3).value,
        )

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._scan: _Scan | None = None
        self._odom: _Odom | None = None
        self._run: _Run | None = None
        self._result: dict | None = None
        self._last_cmd = None
        self._remaining: float | None = None
        self._step_us = 0.0
        self._laser_in_base: dict[str, Pose2] = {}

        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._status_pub = self.create_publisher(String, STATUS_TOPIC, 10)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(PoseStamped, GOAL_TOPIC, self._on_goal, 10)
        self.create_subscription(Bool, CANCEL_TOPIC, self._on_cancel, 10)
        self.create_timer(1.0 / self._spec.control_hz, self._tick)
        self.create_timer(1.0 / status_hz, self._publish_status)

        _UiHandler.config = {
            "rosbridge_port": rosbridge_port,
            "node": self.get_fully_qualified_name(),
            "goal_topic": GOAL_TOPIC,
            "cancel_topic": CANCEL_TOPIC,
            "status_topic": STATUS_TOPIC,
            "map_topic": "/map",
        }
        self._http = ThreadingHTTPServer(("0.0.0.0", http_port), partial(_UiHandler, directory=str(share / "web")))
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        self.get_logger().info(
            f"RL nav ready: {self._controller.policy.backend} policy from {policy_dir}, driving {self._cmd_vel_topic}; "
            f"UI on http://<robot>:{http_port}/"
        )

    # --- inputs -----------------------------------------------------------------------------

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = _Scan(
            np.asarray(msg.ranges, dtype=np.float32), msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max,
            _stamp(msg.header.stamp), msg.header.frame_id, time.monotonic(),
        )  # fmt: skip

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self._odom = _Odom(
            (pose.position.x, pose.position.y, _yaw(pose.orientation)), _stamp(msg.header.stamp), msg.header.frame_id or "odom", time.monotonic()
        )

    def _on_goal(self, msg: PoseStamped) -> None:
        now = time.monotonic()
        frame = msg.header.frame_id or self._display_frame()
        dry_run = self.get_parameter("dry_run").value
        self._run = _Run(frame, (msg.pose.position.x, msg.pose.position.y), dry_run, now, tf_ok_at=now)
        self._result = None
        self._remaining = None
        self._controller.reset(now)
        self.get_logger().info(f"Goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) in {frame}{' (dry run)' if dry_run else ''}")

    def _on_cancel(self, msg: Bool) -> None:
        if msg.data and self._run is not None:
            self._finish("cancelled", "Stopped from the UI")

    # --- control loop -----------------------------------------------------------------------

    def _tick(self) -> None:
        run = self._run
        if run is None:
            return
        now = time.monotonic()
        scan, odom = self._scan, self._odom
        if scan is None or odom is None:
            if now - run.started > FEED_TIMEOUT_S:
                self._finish("error", "No lidar or odometry")
            return

        goal = self._goal_in(odom.frame_id, run)
        if goal is None:
            if now - run.tf_ok_at > TF_TIMEOUT_S:
                self._finish("error", f"No transform from {run.frame} to {odom.frame_id}")
            else:
                self._send(0.0, 0.0)
            return
        run.tf_ok_at = now

        goal_body = self._u.goal_in_base(goal[:2], odom.pose)
        remaining = math.hypot(*goal_body)
        self._remaining = remaining
        if run.deadline is None:
            if remaining > self._max_goal_distance:
                self._finish("error", f"Goal is {remaining:.1f} m away; the policy handles up to {self._max_goal_distance:.1f} m")
                return
            run.deadline = now + max(self._min_time_limit, self._time_limit_per_meter * remaining)
        if remaining < self._spec.goal_radius:
            self._finish("reached", f"Reached the goal ({remaining:.2f} m off)")
            return
        if now > run.deadline:
            self._finish("timeout", f"Timed out {remaining:.2f} m from the goal")
            return

        start = time.perf_counter()
        cmd = self._controller.step(goal_body, odom.pose, odom.stamp, scan, now)
        self._step_us = 0.9 * self._step_us + 0.1 * (time.perf_counter() - start) * 1e6 if self._step_us else (time.perf_counter() - start) * 1e6
        self._last_cmd = cmd
        self._send(cmd.linear, cmd.angular)

    def _send(self, linear: float, angular: float) -> None:
        if self._run is not None and self._run.dry_run:
            return
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self._cmd_pub.publish(twist)

    def _finish(self, state: str, message: str) -> None:
        run, self._run = self._run, None
        if run is not None and not run.dry_run:
            self._cmd_pub.publish(Twist())  # then silence: the mux hands the base back to nav
        self._result = {"state": state, "message": message, "safety_stops": self._controller.safety_stops}
        self._last_cmd = None
        self.get_logger().info(message)
        self._publish_status()

    # --- frames -----------------------------------------------------------------------------

    def _transform(self, target: str, source: str, stamp: float | None = None) -> Pose2 | None:
        """The latest transform, or the one at `stamp` (s) if TF has it."""
        if target == source:
            return (0.0, 0.0, 0.0)
        for at in ([Time(nanoseconds=int(stamp * 1e9))] if stamp else []) + [Time()]:
            try:
                t = self._tf.lookup_transform(target, source, at).transform
            except TransformException:
                continue
            return (t.translation.x, t.translation.y, _yaw(t.rotation))
        return None

    def _goal_in(self, frame: str, run: _Run) -> Pose2 | None:
        t = self._transform(frame, run.frame)
        return None if t is None else _compose(t, (*run.goal, 0.0))

    def _display_frame(self) -> str:
        odom_frame = self._odom.frame_id if self._odom else "odom"
        return self._map_frame if self._transform(self._map_frame, odom_frame) is not None else odom_frame

    def _laser_pose(self, frame_id: str) -> Pose2:
        """base_link -> laser is static (URDF), so it is looked up once; identity until TF has it."""
        if frame_id not in self._laser_in_base:
            t = self._transform(self._base_frame, frame_id)
            if t is None:
                return (0.0, 0.0, 0.0)
            self._laser_in_base[frame_id] = t
        return self._laser_in_base[frame_id]

    # --- status -----------------------------------------------------------------------------

    def _publish_status(self) -> None:
        now = time.monotonic()
        run, scan, odom = self._run, self._scan, self._odom
        frame = self._display_frame()
        status: dict = {
            "state": ("dry_run" if run.dry_run else "driving") if run else (self._result or {}).get("state", "idle"),
            "message": (self._result or {}).get("message", ""),
            "frame": frame,
            "dry_run": self.get_parameter("dry_run").value,
            "backend": self._controller.policy.backend,
            "step_us": round(self._step_us, 1),
            "cmd_vel_topic": self._cmd_vel_topic,
            "goal_radius": self._spec.goal_radius,
            "max_goal_distance": self._max_goal_distance,
            "lidar_max_range": self._spec.lidar_max_range,
            "safety_stops": self._controller.safety_stops if run else (self._result or {}).get("safety_stops", 0),
            "scan_age": round(now - scan.received, 2) if scan else None,
            "odom_age": round(now - odom.received, 2) if odom else None,
        }
        robot = None
        if odom is not None:
            t = self._transform(frame, odom.frame_id)
            robot = _compose(t, odom.pose) if t is not None else None
        status["robot"] = [round(v, 3) for v in robot] if robot else None
        if run is not None:
            goal = self._goal_in(frame, run)
            status["goal"] = [round(goal[0], 3), round(goal[1], 3)] if goal else None
            status["remaining"] = round(self._remaining, 2) if self._remaining is not None else None
            status["time_left"] = round(run.deadline - now, 1) if run.deadline else None
        cmd = self._last_cmd
        if cmd is not None:
            status.update(
                cmd=[round(cmd.linear, 3), round(cmd.angular, 3)], action=[round(float(a), 3) for a in cmd.action], blocked=cmd.blocked, stale=cmd.stale
            )
        if scan is not None:
            bins = self._u.bin_scan(self._spec, scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max)
            status["bins"] = [round(float(b), 3) for b in bins]
            if robot is not None:
                status["laser"], status["scan"] = self._scan_points(scan, frame, robot)
        self._status_pub.publish(String(data=json.dumps(status, separators=(",", ":"))))

    def _scan_points(self, scan: _Scan, frame: str, robot: Pose2) -> tuple[list[float], list[float]]:
        """The laser pose and the valid returns, flattened [x0, y0, x1, ...], in the display frame,
        where the laser was when it took the scan: at 1 rad/s a 6 Hz scan placed at the current
        pose would be skewed by up to 10 degrees."""
        laser = self._transform(frame, scan.frame_id, scan.stamp) or _compose(robot, self._laser_pose(scan.frame_id))
        r = scan.ranges
        angles = scan.angle_min + np.arange(r.size) * scan.angle_increment + laser[2]
        valid = np.isfinite(r) & (r >= max(scan.range_min, self._spec.lidar_min_range)) & (r <= scan.range_max)
        idx = np.flatnonzero(valid)[:: max(1, int(valid.sum()) // STATUS_SCAN_POINTS + 1)]
        xy = np.stack((laser[0] + r[idx] * np.cos(angles[idx]), laser[1] + r[idx] * np.sin(angles[idx])), axis=1)
        return [round(v, 3) for v in laser], np.round(xy, 2).ravel().tolist()

    def stop(self) -> None:
        if self._run is not None and not self._run.dry_run:
            self._cmd_pub.publish(Twist())
        self._http.shutdown()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RlNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
