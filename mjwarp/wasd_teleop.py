"""Hold-to-move WASD teleop, no X11 required: publishes /cmd_vel while a drive key is
actively being held, zero the moment it's released.

Raw terminal input has no real key-up event -- this approximates one using the OS's own
keyboard-repeat stream: each byte that arrives refreshes that key's "last seen" time, and
a key counts as released once RELEASE_TIMEOUT has passed without another repeat. This only
works as well as your terminal's key-repeat rate is faster than RELEASE_TIMEOUT; an unusually
slow repeat rate could read as an early release mid-hold.
"""

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

SPEED = 0.3
TURN = 0.8
RELEASE_TIMEOUT = 0.2  # seconds; longer than a terminal's OS key-repeat interval

VX_KEYS = {"w": 1.0, "x": -1.0}
WZ_KEYS = {"a": 1.0, "d": -1.0}


class WasdTeleop(Node):
    def __init__(self) -> None:
        super().__init__("wasd_teleop")
        self._pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._last_seen = dict.fromkeys(VX_KEYS, 0.0) | dict.fromkeys(WZ_KEYS, 0.0)
        self.create_timer(0.05, self._on_timer)

    def on_key(self, key: str) -> None:
        if key in self._last_seen:
            self._last_seen[key] = self._now()

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_timer(self) -> None:
        now = self._now()
        held = {k for k, t in self._last_seen.items() if now - t < RELEASE_TIMEOUT}
        vx = sum(v for k, v in VX_KEYS.items() if k in held) * SPEED
        wz = sum(v for k, v in WZ_KEYS.items() if k in held) * TURN
        msg = Twist()
        msg.linear.x = vx
        msg.angular.z = wz
        self._pub.publish(msg)


def _read_key(timeout: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else None


def main() -> None:
    rclpy.init()
    node = WasdTeleop()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print("Hold w/a/s/d to move/turn the base; release to stop. q: quit.")
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = _read_key(0.02)
            if key == "q":
                break
            if key:
                node.on_key(key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
