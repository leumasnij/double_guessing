#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
Hung-Jui Huang (hungjuih@andrew.cmu.edu) Sept, 2021
'''

import rospy
from std_srvs.srv import Empty
from wsg_50_common.srv import Move, Conf

def homing():
  """ Moves fingers to home position (maximum opening). """
  rospy.wait_for_service('/wsg_50_driver/homing')
  try:
    service = rospy.ServiceProxy('/wsg_50_driver/homing', Empty)
    service()
  except rospy.ServiceException as e:
    print("Service call failed: %s"%e)

def set_force(force):
  """ Set the grasping force. """
  rospy.wait_for_service('/wsg_50_driver/set_force')
  try:
    service = rospy.ServiceProxy('/wsg_50_driver/set_force', Conf)
    service(force)
  except rospy.ServiceException as e:
    print ("Service Call Failed: %s"%e)

def move(width, speed = 50):
  """
  Moves fingers to an absolute position at a specific velocity,
  do not try to grasp with it. It will throw error.
  """
  rospy.wait_for_service('/wsg_50_driver/move')
  try:
    service = rospy.ServiceProxy('/wsg_50_driver/move', Move)
    service(width, speed)
  except rospy.ServiceException as e:
    print("Service call failed: %s"%e)

def grasp(width, speed = 50):
  """ Grasps an object of a specific width at a specific velocity. """
  rospy.wait_for_service('/wsg_50_driver/grasp')
  try:
    service = rospy.ServiceProxy('/wsg_50_driver/grasp', Move)
    service(width, speed)
  except rospy.ServiceException as e:
    print ("Service Call Failed: %s"%e)
