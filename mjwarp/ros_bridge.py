"""ROS 2 interface for one simulated MARS, with the same topics, frames and rates as
Innate's sim driver (mars_sim_driver/node.py), so Nav2, SLAM and skills run unchanged.
robot_state_publisher supplies the URDF's static frames; see sim.launch.py."""

import math

import mujoco.viewer
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
from std_msgs.msg import Float64MultiArray, Header, Int32
from tf2_ros import TransformBroadcaster

import _driver_pkg  # noqa: F401 -- puts mars_sim_driver on sys.path
from mars_sim_driver.constants import CAMERA_CX, CAMERA_CY, CAMERA_FX, CAMERA_FY, CAMERA_HEIGHT, CAMERA_WIDTH
from robot_env import ARM_JOINTS, DRIVEN_JOINTS, HEAD_CAMERA, LIDAR_RAYS, WRIST_CAMERA
from warp_env import MarsWarpEnv

ODOM_HZ = 30.0
JOINT_STATE_HZ = 30.0
SCAN_HZ = 6.0
HEAD_CAMERA_HZ = 10.0
WRIST_CAMERA_HZ = 6.0
DEPTH_HZ = 8.0

CMD_VEL_TIMEOUT_S = 0.5
LIDAR_RANGE_MIN, LIDAR_RANGE_MAX = 0.15, 12.0
# The real stereo depth pipeline only reports this band; outside it pixels are 0 (invalid).
DEPTH_MIN_M, DEPTH_MAX_M = 0.25, 2.0

HEAD_FRAME = "camera_optical_frame"
WRIST_FRAME = "arm_camera_link"


def vector3(values: np.ndarray) -> Vector3:
    x, y, z = values
    return Vector3(x=x, y=y, z=z)


def image_msg(pixels: np.ndarray, encoding: str, header: Header) -> Image:
    # Built by hand: cv_bridge needs OpenCV and numpy<2, neither of which this environment has.
    height, width = pixels.shape[:2]
    return Image(
        header=header, height=height, width=width, encoding=encoding, step=pixels.strides[0], data=pixels.tobytes()
    )


def head_camera_info(header: Header) -> CameraInfo:
    return CameraInfo(
        header=header,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        distortion_model="plumb_bob",
        d=[0.0] * 5,
        k=[CAMERA_FX, 0.0, CAMERA_CX, 0.0, CAMERA_FY, CAMERA_CY, 0.0, 0.0, 1.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[CAMERA_FX, 0.0, CAMERA_CX, 0.0, 0.0, CAMERA_FY, CAMERA_CY, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


class MarsSimBridge(Node):
    def __init__(self) -> None:
        super().__init__("mars_warp_sim")
        self.env = MarsWarpEnv(cameras=True)
        self.state = self.env.snapshot()
        self._last_cmd_vel = self.get_clock().now()
        # snapshot() refills the same MjData every step, so the viewer always shows the latest state.
        self._viewer = (
            mujoco.viewer.launch_passive(self.env.mj_model, self.state)
            if self.declare_parameter("viewer", False).value
            else None
        )

        self._tf = TransformBroadcaster(self)
        self._odom_pub = self.create_publisher(Odometry, "/odom", 1)
        self._joint_states_pub = self.create_publisher(JointState, "/joint_states", 1)
        self._arm_state_pub = self.create_publisher(JointState, "/mars/arm/state", 1)
        self._scan_pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self._head_pub = self.create_publisher(Image, "/mars/main_camera/left/image_raw", qos_profile_sensor_data)
        self._head_info_pub = self.create_publisher(CameraInfo, "/mars/main_camera/left/camera_info", 10)
        self._depth_pub = self.create_publisher(
            Image, "/mars/main_camera/depth/image_rect_raw", qos_profile_sensor_data
        )
        self._wrist_pub = self.create_publisher(Image, "/mars/arm/image_raw", qos_profile_sensor_data)

        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 1)
        self.create_subscription(Float64MultiArray, "/mars/arm/commands", self._on_arm_command, qos_profile_sensor_data)
        self.create_subscription(Int32, "/mars/head/set_position", self._on_head_position, 10)

        self.create_timer(self.env.control_period, self._step)
        self.create_timer(1.0 / ODOM_HZ, self._publish_odom)
        self.create_timer(1.0 / JOINT_STATE_HZ, self._publish_joint_states)
        self.create_timer(1.0 / SCAN_HZ, self._publish_scan)
        self.create_timer(1.0 / HEAD_CAMERA_HZ, self._publish_head_camera)
        self.create_timer(1.0 / WRIST_CAMERA_HZ, self._publish_wrist_camera)
        self.create_timer(1.0 / DEPTH_HZ, self._publish_depth)

    def destroy_node(self) -> None:
        if self._viewer is not None:
            self._viewer.close()  # its render thread would otherwise keep the process alive
        super().destroy_node()

    def _header(self, frame_id: str) -> Header:
        return Header(stamp=self.get_clock().now().to_msg(), frame_id=frame_id)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.env.drive(msg.linear.x, msg.angular.z)
        self._last_cmd_vel = self.get_clock().now()

    def _on_arm_command(self, msg: Float64MultiArray) -> None:
        self.env.set_arm(np.asarray(msg.data[: len(ARM_JOINTS)]))

    def _on_head_position(self, msg: Int32) -> None:
        self.env.set_head(math.radians(msg.data))

    def _step(self) -> None:
        if (self.get_clock().now() - self._last_cmd_vel).nanoseconds > CMD_VEL_TIMEOUT_S * 1e9:
            self.env.drive(0.0, 0.0)  # a stale cmd_vel stops the base, like the real base watchdog
        self.env.step()
        self.state = self.env.snapshot()
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def _publish_odom(self) -> None:
        x, y, z = self.state.sensor("odom_pos").data
        qw, qx, qy, qz = self.state.sensor("odom_quat").data
        rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        header = self._header("odom")

        odom = Odometry(header=header, child_frame_id="base_link")
        odom.pose.pose = Pose(position=Point(x=x, y=y, z=z), orientation=rotation)
        odom.twist.twist = Twist(
            linear=vector3(self.state.sensor("odom_linear_vel").data),
            angular=vector3(self.state.sensor("odom_angular_vel").data),
        )
        self._odom_pub.publish(odom)
        transform = Transform(translation=Vector3(x=x, y=y, z=z), rotation=rotation)
        self._tf.sendTransform(TransformStamped(header=header, child_frame_id="base_link", transform=transform))

    def _publish_joint_states(self) -> None:
        positions = [float(self.state.sensor(f"{name}_pos").data[0]) for name in DRIVEN_JOINTS]
        velocities = [float(self.state.sensor(f"{name}_vel").data[0]) for name in DRIVEN_JOINTS]
        header = self._header("")
        self._joint_states_pub.publish(
            JointState(header=header, name=DRIVEN_JOINTS, position=positions, velocity=velocities)
        )
        arm = slice(len(ARM_JOINTS))  # DRIVEN_JOINTS starts with the arm joints
        self._arm_state_pub.publish(JointState(header=header, name=ARM_JOINTS, position=positions[arm]))

    def _publish_scan(self) -> None:
        ranges = self.state.sensordata[self.env.lidar_slice]
        increment = 2 * math.pi / LIDAR_RAYS
        self._scan_pub.publish(
            LaserScan(
                header=self._header("base_laser"),
                angle_min=-math.pi,
                angle_max=math.pi - increment,
                angle_increment=increment,
                scan_time=1.0 / SCAN_HZ,
                range_min=LIDAR_RANGE_MIN,
                range_max=LIDAR_RANGE_MAX,
                # A rangefinder reports -1 for no hit; LaserScan's convention is +inf (REP 117).
                ranges=np.where((ranges < 0) | (ranges > LIDAR_RANGE_MAX), np.inf, ranges).tolist(),
            )
        )

    def _publish_head_camera(self) -> None:
        header = self._header(HEAD_FRAME)
        self._head_info_pub.publish(head_camera_info(header))
        if self._head_pub.get_subscription_count() == 0:
            return
        self.env.render()
        self._head_pub.publish(image_msg(self.env.rgb(HEAD_CAMERA)[0], "rgb8", header))

    def _publish_wrist_camera(self) -> None:
        if self._wrist_pub.get_subscription_count() == 0:
            return
        self.env.render()
        self._wrist_pub.publish(image_msg(self.env.rgb(WRIST_CAMERA)[0], "rgb8", self._header(WRIST_FRAME)))

    def _publish_depth(self) -> None:
        if self._depth_pub.get_subscription_count() == 0:
            return
        self.env.render()
        depth = self.env.depth(HEAD_CAMERA)[0]
        in_band = (depth >= DEPTH_MIN_M) & (depth <= DEPTH_MAX_M)  # NaN compares False
        millimetres = np.where(in_band, depth * 1000.0, 0.0).astype(np.uint16)
        self._depth_pub.publish(image_msg(millimetres, "16UC1", self._header(HEAD_FRAME)))


def main() -> None:
    rclpy.init()
    bridge = MarsSimBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
