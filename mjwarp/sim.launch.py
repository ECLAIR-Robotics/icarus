"""The MuJoCo Warp sim behind Innate's ROS interface, plus robot_state_publisher for the
URDF's static frames. Run with ROS sourced:
`ros2 launch mjwarp/sim.launch.py [viewer:=true] [rviz:=true]`."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
MARS_SIM = HERE.parent / "innate-os" / "ros2_ws" / "src" / "mars_bot" / "mars_sim"


def robot_description() -> str:
    # mars_sim isn't built outside Innate's workspace, so RViz can't resolve package:// mesh URIs.
    return (MARS_SIM / "urdf" / "mars.urdf").read_text().replace("package://mars_sim/", f"file://{MARS_SIM}/")


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("viewer", default_value="false", description="Show the sim in MuJoCo's viewer"),
            DeclareLaunchArgument("rviz", default_value="false", description="Open RViz with mars.rviz"),
            # Strips rosidl's per-byte asserts, which make publishing raw images slow (as in Innate's launch).
            SetEnvironmentVariable("PYTHONOPTIMIZE", "1"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description()}],
            ),
            ExecuteProcess(
                cmd=[
                    "uv",
                    "run",
                    "python",
                    "ros_bridge.py",
                    "--ros-args",
                    "-p",
                    ["viewer:=", LaunchConfiguration("viewer")],
                ],
                cwd=str(HERE),
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(HERE / "mars.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
