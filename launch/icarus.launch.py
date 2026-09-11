from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='icarus',
            executable='icarus_node',
            name='icarus_node',
            output='screen',
        ),
    ])
