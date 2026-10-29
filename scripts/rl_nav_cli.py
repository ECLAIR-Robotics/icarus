#!/usr/bin/env python3
"""Command-line checks for rl_nav_node on the robot (see RL_NAV_ROBOT_CHECKLIST.md).

    python3 scripts/rl_nav_cli.py status [--watch]       one line from /icarus/rl_nav/status
    python3 scripts/rl_nav_cli.py goal FWD LEFT          goal FWD m ahead, LEFT m left of the robot now (sent in odom)
    python3 scripts/rl_nav_cli.py cancel                 stop the run
    python3 scripts/rl_nav_cli.py scan                   /scan geometry, rate and nearest raw return (no node needed)
    python3 scripts/rl_nav_cli.py pose                   odometry pose
    python3 scripts/rl_nav_cli.py drive LIN ANG SECS     MOVES THE BASE: /cmd_vel_skills at 10 Hz, capped, then zero

Angles are in the robot frame: 0 deg ahead, +90 left, -90 right.
"""

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

STATUS_TOPIC = "/icarus/rl_nav/status"
GOAL_TOPIC = "/icarus/rl_nav/goal"
CANCEL_TOPIC = "/icarus/rl_nav/cancel"
CMD_TOPIC = "/cmd_vel_skills"
# drive is for sign checks only: well under the policy's 0.4 m/s and 1 rad/s.
DRIVE_MAX_LINEAR = 0.2
DRIVE_MAX_ANGULAR = 0.5
DRIVE_MAX_SECONDS = 2.0


def _yaw(q) -> float:  # noqa: ANN001 -- geometry_msgs Quaternion
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wait(node: Node, msg_type: type, topic: str, timeout: float = 5.0, qos=10, count: int = 1) -> list:  # noqa: ANN001
    got: list = []
    sub = node.create_subscription(msg_type, topic, got.append, qos)
    end = time.monotonic() + timeout
    while len(got) < count and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if not got:
        raise SystemExit(f"no message on {topic} within {timeout:.0f} s")
    return got


def _pose(node: Node) -> tuple[float, float, float, str]:
    msg = _wait(node, Odometry, "/odom")[0]
    p = msg.pose.pose
    return p.position.x, p.position.y, _yaw(p.orientation), msg.header.frame_id or "odom"


def _status_line(data: str) -> str:
    d = json.loads(data)
    bins = d.get("bins") or []
    nearest = "-"
    if bins:
        i = min(range(len(bins)), key=bins.__getitem__)
        angle = math.degrees(-math.pi + (i + 0.5) * 2 * math.pi / len(bins))
        nearest = f"{bins[i] * d['lidar_max_range']:.2f} m at {angle:+.0f} deg"
    fields = {
        "state": d["state"],
        "msg": json.dumps(d["message"]),
        "frame": d["frame"],
        "robot": d.get("robot"),
        "goal": d.get("goal"),
        "remaining": d.get("remaining"),
        "time_left": d.get("time_left"),
        "cmd": d.get("cmd"),
        "blocked": d.get("blocked"),
        "stale": d.get("stale"),
        "safety_stops": d.get("safety_stops"),
        "nearest": nearest,
        "dry_run": d["dry_run"],
        "backend": d["backend"],
        "step_us": d["step_us"],
        "scan_age": d["scan_age"],
        "odom_age": d["odom_age"],
    }
    return " ".join(f"{k}={v}" for k, v in fields.items())


def status(node: Node, watch: bool) -> None:
    if not watch:
        print(_status_line(_wait(node, String, STATUS_TOPIC).pop().data))
        return
    node.create_subscription(String, STATUS_TOPIC, lambda m: print(_status_line(m.data), flush=True), 10)
    rclpy.spin(node)


def goal(node: Node, forward: float, left: float) -> None:
    x, y, theta, frame = _pose(node)
    pub = node.create_publisher(PoseStamped, GOAL_TOPIC, 10)
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = x + math.cos(theta) * forward - math.sin(theta) * left
    msg.pose.position.y = y + math.sin(theta) * forward + math.cos(theta) * left
    msg.pose.orientation.w = 1.0
    _publish_once(node, pub, msg)
    print(
        f"goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) in {frame}: {forward:+.2f} m ahead, {left:+.2f} m left"
    )


def cancel(node: Node) -> None:
    _publish_once(node, node.create_publisher(Bool, CANCEL_TOPIC, 10), Bool(data=True))
    print("cancel sent")


def _publish_once(node: Node, pub, msg) -> None:  # noqa: ANN001
    """Waits for the subscriber to be discovered, so the one message isn't dropped."""
    end = time.monotonic() + 3.0
    while pub.get_subscription_count() == 0 and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        raise SystemExit(f"nobody subscribes to {pub.topic_name}: is rl_nav_node running?")
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.2)


def scan(node: Node) -> None:
    msgs = _wait(node, LaserScan, "/scan", timeout=6.0, qos=qos_profile_sensor_data, count=6)
    m = msgs[-1]
    stamps = [s.header.stamp.sec + s.header.stamp.nanosec * 1e-9 for s in msgs]
    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if stamps[-1] > stamps[0] else float("nan")
    print(
        f"frame={m.header.frame_id} beams={len(m.ranges)} rate={rate:.1f} Hz "
        f"angle_min={math.degrees(m.angle_min):+.1f} angle_max={math.degrees(m.angle_max):+.1f} "
        f"increment={math.degrees(m.angle_increment):.3f} deg range=[{m.range_min:.2f}, {m.range_max:.2f}] m"
    )
    valid = [(r, i) for i, r in enumerate(m.ranges) if math.isfinite(r) and m.range_min <= r <= m.range_max]
    for cutoff, label in ((0.0, "nearest return"), (0.2, "nearest return beyond 0.2 m (what the policy can see)")):
        beyond = [(r, i) for r, i in valid if r >= cutoff]
        if not beyond:
            print(f"{label}: none")
            continue
        r, i = min(beyond)
        angle = math.degrees(
            math.atan2(math.sin(m.angle_min + i * m.angle_increment), math.cos(m.angle_min + i * m.angle_increment))
        )
        print(f"{label}: {r:.2f} m at {angle:+.0f} deg")


def pose(node: Node) -> None:
    x, y, theta, frame = _pose(node)
    print(f"x={x:.3f} y={y:.3f} yaw={math.degrees(theta):+.1f} deg ({frame})")


def drive(node: Node, linear: float, angular: float, seconds: float) -> None:
    if abs(linear) > DRIVE_MAX_LINEAR or abs(angular) > DRIVE_MAX_ANGULAR or not 0 < seconds <= DRIVE_MAX_SECONDS:
        raise SystemExit(
            f"drive is capped at |lin| <= {DRIVE_MAX_LINEAR}, |ang| <= {DRIVE_MAX_ANGULAR}, 0 < secs <= {DRIVE_MAX_SECONDS}"
        )
    x0, y0, t0, _ = _pose(node)
    pub = node.create_publisher(Twist, CMD_TOPIC, 10)
    twist = Twist()
    twist.linear.x, twist.angular.z = float(linear), float(angular)
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            pub.publish(twist)
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        pub.publish(Twist())
    time.sleep(0.8)
    x1, y1, t1, _ = _pose(node)
    forward = math.cos(t0) * (x1 - x0) + math.sin(t0) * (y1 - y0)
    dyaw = math.degrees(math.atan2(math.sin(t1 - t0), math.cos(t1 - t0)))
    print(
        f"commanded lin={linear:+.2f} m/s ang={angular:+.2f} rad/s for {seconds:.1f} s -> odom moved {forward:+.3f} m forward, yaw {dyaw:+.1f} deg (+ = left/CCW)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").add_argument("--watch", action="store_true")
    g = sub.add_parser("goal")
    g.add_argument("forward", type=float)
    g.add_argument("left", type=float)
    sub.add_parser("cancel")
    sub.add_parser("scan")
    sub.add_parser("pose")
    d = sub.add_parser("drive")
    for name in ("linear", "angular", "seconds"):
        d.add_argument(name, type=float)
    args = parser.parse_args()

    rclpy.init()
    node = Node("rl_nav_cli")
    try:
        if args.cmd == "status":
            status(node, args.watch)
        elif args.cmd == "goal":
            goal(node, args.forward, args.left)
        elif args.cmd == "cancel":
            cancel(node)
        elif args.cmd == "scan":
            scan(node)
        elif args.cmd == "pose":
            pose(node)
        else:
            drive(node, args.linear, args.angular, args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
