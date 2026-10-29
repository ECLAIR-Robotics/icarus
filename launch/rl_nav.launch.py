from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = {
        'backend': 'auto',  # onnxruntime if installed, else numpy
        'cmd_vel_topic': '/cmd_vel_skills',
        'http_port': '8080',
        'dry_run': 'false',
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=value) for name, value in args.items()),
        Node(
            package='icarus',
            executable='rl_nav_node',
            name='rl_nav',
            output='screen',
            parameters=[{
                'backend': LaunchConfiguration('backend'),
                'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                'http_port': LaunchConfiguration('http_port'),
                'dry_run': LaunchConfiguration('dry_run'),
            }],
        ),
    ])
