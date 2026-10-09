from launch import LaunchDescription
from launch_ros.actions import Node

URDF_PATH = "/root/icarus/innate-os/ros2_ws/src/mars_bot/mars_sim/urdf/mars.urdf"

def generate_launch_description():
    with open(URDF_PATH) as f:
        robot_description = f.read()
    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],

        ),
    ])