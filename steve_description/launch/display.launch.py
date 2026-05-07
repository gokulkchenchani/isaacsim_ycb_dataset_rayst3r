
""" 
Launch file to display the robot model in RViz2

Author:
W.M. Nipun Dhananjaya
29-11-2024
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory path
    pkg_dir = get_package_share_directory('steve_description')
    # Get URDF file path
    urdf_file = os.path.join(pkg_dir, 'urdf', 'steve_main.urdf')

    # Read URDF file content
    with open(urdf_file, 'r') as file:
        robot_description = file.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
        ),
    
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
            parameters=[{'use_gui': 'false'}]
        ),
    
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(pkg_dir, 'rviz', 'rviz.rviz')]
        )
    ])