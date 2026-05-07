from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    with open('/workspace/steve_new_gripper.urdf', 'r') as f:
        urdf = f.read()

    return LaunchDescription([
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            parameters=[{'robot_description': urdf}],
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': urdf}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        )
    ])