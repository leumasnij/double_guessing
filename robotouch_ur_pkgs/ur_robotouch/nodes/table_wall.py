#!/usr/bin/env python 

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import rospy
import math
import tf
import geometry_msgs.msg
import numpy as np

if __name__ == '__main__':
  rospy.init_node('collision', anonymous=True)

  scene = moveit_commander.PlanningSceneInterface()
  robot = moveit_commander.RobotCommander()

  rospy.sleep(2)

  p = geometry_msgs.msg.PoseStamped()
  p.header.frame_id = robot.get_planning_frame()
  p.pose.position.x = 0.
  p.pose.position.y = +0.37
  p.pose.position.z = 0
  # print(p.pose.orientation.x)
  scene.add_box("table", p, (3, 3, 0.05))

  r1 = geometry_msgs.msg.PoseStamped()
  r1.header.frame_id = robot.get_planning_frame()
  r1.pose.position.x = -0.4
  r1.pose.position.y = 0.
  r1.pose.position.z = 0.
  scene.add_box("right_wall", r1, (0.05, 3, 3))

  r2 = geometry_msgs.msg.PoseStamped()
  r2.header.frame_id = robot.get_planning_frame()
  r2.pose.position.x = 0.2
  r2.pose.position.y = 0.
  r2.pose.position.z = 0.
  scene.add_box("left_wall", r2, (0.05, 3, 3))

  q = geometry_msgs.msg.PoseStamped()
  q.header.frame_id = robot.get_planning_frame()
  q.pose.position.x = 0
  q.pose.position.y = -0.25
  q.pose.position.z = 0
  q.pose.orientation.x =  0.7068252
  q.pose.orientation.y =  0
  q.pose.orientation.z =  0
  q.pose.orientation.w =  0.7073883
  # print(q.pose.orientation.x)
  scene.add_box("wall", q, (3, 3, 0.05))

