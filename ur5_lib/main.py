#!/usr/bin/env python 

'''
Main function for robot manipulation and data collection
Shubham Kanitkar (shubhamkanitkar32@gmail.com) June, 2021
Hung-Jui Huang (hungjuih@andrew.cmu.edu) Sept, 2021
'''

import rospy
import tf

from robot import Robot
from gripper import Gripper
import utils

TABLE_CENTER = [-0.1, 0.5, 0.25]
PARENT_DIR = '/media/okemo/joehuang/example'

def main():
  rospy.init_node('main', anonymous=True)
  robot = Robot()
  gripper = Gripper()
  try:
    gripper.homing()
    robot.move_to_location(
        TABLE_CENTER[0], TABLE_CENTER[1], TABLE_CENTER[2],
        ask_confirmation=False)
    robot.move_offset_distance(0, 0, -0.05, False)
    gripper.graspWithForce()
    gripper.homing()
    robot.joint_angle_homing(ask_confirmation=False)
  except (tf.LookupException, tf.ConnectivityException,
      tf.ExtrapolationException):
    pass

if __name__ == '__main__':
  main()
