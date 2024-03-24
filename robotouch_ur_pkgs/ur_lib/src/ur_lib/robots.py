#!/usr/bin/env python 

'''
Version:
    0.3.0
Authors:
    Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
    Hung-Jui Huang (hungjuih@andrew.cmu.edu) Sept, 2021
    Achu Wilson (achuw@andrew.cmu.edu) Nov 2021
'''

import sys
import utils
import time

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg 

class UR5E():
  """ The robot object. """
  def __init__(self, acc_scale_factor = 0.3, vel_scale_factor = 0.3):
    """
    The robot class to control the robot arm.

    :param acc_scale_factor: float; scaling factor that reduce the
      max joint acceleration.
    :param vel_scale_factor: float; scaling factor that reduce the
      max joint velocity.
    """
    moveit_commander.roscpp_initialize(sys.argv)
    self.robot = moveit_commander.RobotCommander()
    self.move_group = moveit_commander.MoveGroupCommander("manipulator")
    self.display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    self.display_trajectory_publisher = rospy.Publisher(
        '/move_group/display_planned_path',
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20)
    self.move_group.set_planner_id('RRTstar')
    self.move_group.set_max_acceleration_scaling_factor(acc_scale_factor)
    self.move_group.set_max_velocity_scaling_factor(vel_scale_factor)
    self.move_group.set_planning_time(1.5)

  def _ask_confirmation(self):
    """ Ask user to confirm the execution. """
    print('Movement Access?? [y/n]: ')
    key = raw_input()
    if key == 'y':
      return True
    elif key == 'n':
      print('Motion Declined!\n')
      return False
    else:
      print('Invalid Input. Please select [y/n]\n')
      return self._ask_confirmation()

  def _plan_trajectory(self, ask_confirmation=True):
    """ Plan and execute the trajectory. """
    self.plan = self.move_group.plan()
    self.display_trajectory.trajectory_start = self.robot.get_current_state()
    self.display_trajectory.trajectory.append(self.plan)
    self.display_trajectory_publisher.publish(self.display_trajectory)
    print('Publishing motion trajectory in RVIZ!')
    is_confirmed = True
    if ask_confirmation:
      is_confirmed = self._ask_confirmation()
    if is_confirmed:
      # Execute planned trajectory plan
      self.move_group.execute(self.plan)
      self.move_group.stop()
      # Clear pose target before starting new planning
      self.move_group.clear_pose_targets()

  def set_scale_factor(self, acc_scale_factor = 0.1, vel_scale_factor = 0.2):
    """
    The robot class to control the robot arm.

    :param acc_scale_factor: float; scaling factor that reduce the
      max joint acceleration.
    :param vel_scale_factor: float; scaling factor that reduce the
      max joint velocity.
    """
    self.move_group.set_max_acceleration_scaling_factor(acc_scale_factor)
    self.move_group.set_max_velocity_scaling_factor(vel_scale_factor)

  def move_joint_space_goal(self, angle_base, angle_shoulder, angle_elbow,
                            angle_wrist1, angle_wrist2, angle_wrist3,
                            ask_confirmation=True):
    """
    Move the robot arm to specified arm configuration.
    :param angle_base: float; the base joint angle in radians.
    :param angle_shoulder: float; the shoulder joint angle in radians.
    :param angle_elbow: float; the elbow joint angle in radians.
    :param angle_wrist1: float; the wrist1 joint angle in radians.
    :param angle_wrist2: float; the wrist2 joint angle in radians.
    :param angle_wrist3: float; the wrist3 joint angle in radians.
    :param ask_confirmation: bool; ask user to confirm before execute.
    """
    joint_group_values = self.move_group.get_current_joint_values()
    joint_group_values[0] = angle_base
    joint_group_values[1] = angle_shoulder
    joint_group_values[2] = angle_elbow
    joint_group_values[3] = angle_wrist1
    joint_group_values[4] = angle_wrist2
    joint_group_values[5] = angle_wrist3
    self.move_group.set_joint_value_target(joint_group_values)
    # Plan and execute the trajectory
    self._plan_trajectory(ask_confirmation)

  def move_offset_distance(
      self, offset_x, offset_y, offset_z, ask_confirmation=True):
    """
    Move an offset distance.

    :param offset_x: float; the offset x distance in meters.
    :param offset_y: float; the offset y distance in meters.
    :param offset_z: float; the offset z distance in meters.
    :param ask_confirmation: bool; ask user to confirm before execute.
    """
    current_pose = self.move_group.get_current_pose()
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position = current_pose.pose.position
    pose_goal.orientation = current_pose.pose.orientation
    pose_goal.position.x += offset_x
    pose_goal.position.y += offset_y
    pose_goal.position.z += offset_z
    self.move_group.set_pose_target(pose_goal)
    # Plan and execute the trajectory
    self._plan_trajectory(ask_confirmation)

  def move_to_location(self, loc_x, loc_y, loc_z, ask_confirmation=True):
    """
    Move to specified location.

    :param loc_x: float; the absolution x location in meters.
    :param loc_y: float; the absolution y location in meters.
    :param loc_z: float; the absolution z location in meters.
    :param ask_confirmation: bool; ask user to confirm before execute.
    """
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation = self.move_group.get_current_pose().pose.orientation
    pose_goal.position.x = loc_x
    pose_goal.position.y = loc_y
    pose_goal.position.z = loc_z
    self.move_group.set_pose_target(pose_goal)
    # Plan and execute the trajectory
    self._plan_trajectory(ask_confirmation)

  def joint_angle_homing(self, ask_confirmation=True):
    """ Go to the pre-grasp position. """
    print("Joint-Space Based Pre-grasp Position")
    self.move_joint_space_goal(angle_base = 1.5708,
                               angle_shoulder = -1.9199,
                               angle_elbow =  1.9199,
                               angle_wrist1 = -1.5708,
                               angle_wrist2 = -1.5708,
                               angle_wrist3 = -1.5708,
                               ask_confirmation = ask_confirmation)

  def joint_angle_horizontal_homing(self, ask_confirmation=True):
    """ Go to the horizontal pre-grasp position. """
    print("Joint-Space Horizontal Pre-grasp Position")
    self.move_joint_space_goal(angle_base = 1.5708,
                               angle_shoulder = -1.9199,
                               angle_elbow =  1.9199,
                               angle_wrist1 = -1.5708,
                               angle_wrist2 = 0,
                               angle_wrist3 = -1.5708,
                               ask_confirmation = ask_confirmation)

  def manual_control(self, gripper, ask_confirmation=True):
    """ Manual control of the robot arm + gripper system. """
    is_confirmed = True
    if ask_confirmation:
      is_confirmed = self._ask_confirmation()
    if is_confirmed:
      while True:
        direction = utils.robot_movement_direction_user_input()
        if direction == 'g':
          gripper.graspWithForce()
        elif direction == 'r':
          gripper.homing()
        elif direction == 'q':
          break
        # Need to move robot arm
        else:
          offset = float(raw_input('Enter Movement Distance: '))
          if direction == 'w':
            self.move_offset_distance(0, 0, offset, ask_confirmation=False)
          elif direction == 's':
            self.move_offset_distance(0, 0, -offset, ask_confirmation=False)
          elif direction == 'a':
            self.move_offset_distance(offset, 0, 0, ask_confirmation=False)
          elif direction == 'd':
            self.move_offset_distance(-offset, 0, 0, ask_confirmation=False)
          elif direction == 'z':
            self.move_offset_distance(0, offset, 0, ask_confirmation=False)
          elif direction == 'x':
            self.move_offset_distance(0, -offset, 0, ask_confirmation=False)
import rospy

def test():
    print("Hello World")
