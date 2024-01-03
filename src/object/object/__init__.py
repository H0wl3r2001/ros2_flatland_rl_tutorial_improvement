#!/usr/bin/env python3
import rclpy
from rclpy.publisher import Publisher
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import LaserScan
from flatland_msgs.srv import MoveModel
from flatland_msgs.msg import Collisions
from rcl_interfaces.msg import Parameter

from gym import Env
from gym.spaces import Discrete, Box

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import numpy as np
import time
import threading

import yaml
import math

class MovableObjectEnv(Node, Env):
    def __init__(self) -> None:
        super().__init__("MovableObjectEnv")

        # **** Create publishers ****
        self.pub:Publisher = self.create_publisher(Twist, "/cmd_vel", 1)
        # ***************************

    def run_rl_alg(self):
        print('lol')


def main(args = None):
    rclpy.init(args = args)
    
    obj = MovableObjectEnv()

    thread = threading.Thread(target=obj.run_rl_alg)
    thread.start()

    rclpy.spin(obj)


if __name__ == "__main__":
    main()