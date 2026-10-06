import math
import mujoco
import rclpy
from geometry_msgs.msg import Twist, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

import _driver_pkg
from mars_sim_driver.world import KP_FORWARD, KP_LATERAL, KP_YAW
from warp_env import MarsWarpEnv

ARM_JOINTS = [f"joint{i+1}" for i in range(6)] + ["joint_head"]

class MjWarpBridge(Node):
    def __init__(self):
        super().__init__("mjwarp_bridge")
        self.mwe = MarsWarpEnv(1)
        model = self.mwe.mj_model

        self._joint_qpos_adr = {j: model.joint(f"robot_{j}").qposadr[0] for j in ARM_JOINTS}
        self._base_x_adr = model.joint("robot_base_x").qposadr[0]
        self._base_y_adr = model.joint("robot_base_y").qposadr[0]
        self._base_yaw_adr = model.joint("robot_base_yaw").qposadr[0]
        self._base_x_dof = model.joint("robot_base_x").dofadr[0]
        self._base_y_dof = model.joint("robot_base_y").dofadr[0]
        self._base_yaw_dof = model.joint("robot_base_yaw").dofadr[0]
        self._base_x_act = model.actuator("robot_base_x_act").id
        self._base_y_act = model.actuator("robot_base_y_act").id
        self._base_yaw_act = model.actuator("robot_base_yaw_act").id

        self._target_vx = 0.0
        self._target_wz = 0.0
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 1)

        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_timer(model.opt.timestep, self._on_timer)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._target_vx = msg.linear.x
        self._target_wz = msg.angular.z

    def _apply_drive_force(self) -> None:
        """Body-frame forward/lateral/yaw velocity PD, same gains and math as
        sim/sandbox/common.py's apply_drive_force -- but writing to the base's motor
        actuators instead of xfrc_applied, since the planar base has no free joint here."""
        qpos = self.mwe.data.qpos.numpy()[0]
        qvel = self.mwe.data.qvel.numpy()[0]

        yaw = qpos[self._base_yaw_adr]
        cos, sin = math.cos(yaw), math.sin(yaw)
        vx_world, vy_world = qvel[self._base_x_dof], qvel[self._base_y_dof]
        v_forward = vx_world * cos + vy_world * sin
        v_lateral = -vx_world * sin + vy_world * cos
        wz_actual = qvel[self._base_yaw_dof]

        force_forward = KP_FORWARD * (self._target_vx - v_forward)
        force_lateral = -KP_LATERAL * v_lateral
        torque_yaw = KP_YAW * (self._target_wz - wz_actual)

        ctrl = self.mwe.data.ctrl.numpy()
        ctrl[0, self._base_x_act] = force_forward * cos - force_lateral * sin
        ctrl[0, self._base_y_act] = force_forward * sin + force_lateral * cos
        ctrl[0, self._base_yaw_act] = torque_yaw
        self.mwe.data.ctrl.assign(ctrl)

    def _on_timer(self):
        self._apply_drive_force()
        self.mwe.step()
        qpos = self.mwe.data.qpos.numpy()[0]

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ARM_JOINTS
        js.position = [float(qpos[self._joint_qpos_adr[j]]) for j in ARM_JOINTS]
        self.joint_pub.publish(js)

        tf = TransformStamped()
        tf.header.stamp = js.header.stamp
        tf.header.frame_id = "world"
        tf.child_frame_id = "base_link"  # raw URDF root -- NOT "robot_base_link"
        tf.transform.translation.x = float(qpos[self._base_x_adr])
        tf.transform.translation.y = float(qpos[self._base_y_adr])
        yaw = float(qpos[self._base_yaw_adr])
        tf.transform.rotation.z = math.sin(yaw / 2)
        tf.transform.rotation.w = math.cos(yaw / 2)
        self.tf_broadcaster.sendTransform(tf)

def main():
    rclpy.init()
    node = MjWarpBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()