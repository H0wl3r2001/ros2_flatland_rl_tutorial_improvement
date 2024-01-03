from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindExecutable
import launch.conditions as conditions
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

import yaml

def generate_launch_description():
    pkg_share = FindPackageShare("serp_rl")

    world_path = LaunchConfiguration("world_path")
    update_rate = LaunchConfiguration("update_rate")
    step_size = LaunchConfiguration("step_size")
    show_viz = LaunchConfiguration("show_viz")
    viz_pub_rate = LaunchConfiguration("viz_pub_rate")
    use_sim_time = LaunchConfiguration("use_sim_time")

    #world = '2'

    ld = LaunchDescription(
        [
            # Flatland parameters.
            # You can either change these values directly here or override them in the launch command default values. Example:
            #   ros2 launch serp_rl serp_rl.launch.py update_rate:="20.0"
            DeclareLaunchArgument(name="update_rate", default_value="1000.0"),
            DeclareLaunchArgument(name="step_size", default_value="0.01"),
            DeclareLaunchArgument(name="show_viz", default_value="true"),
            DeclareLaunchArgument(name="viz_pub_rate", default_value="30.0"),
            DeclareLaunchArgument(name="use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                name="world_path",
                default_value=PathJoinSubstitution([pkg_share, "world/world.yaml"]),
            ),
            
            SetEnvironmentVariable(name="ROSCONSOLE_FORMAT", value="[${severity} ${time} ${logger}]: ${message}"),

            # runs the code in the file serp_rl/__init__.py that is used to control the robot
            Node(
                name="object",
                package="object",
                executable="object",
                output="screen",
            ),
        ]
    )
    return ld


if __name__ == "__main__":
    generate_launch_description()
