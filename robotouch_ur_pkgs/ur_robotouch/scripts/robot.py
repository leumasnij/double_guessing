#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
'''

import sys
import utils

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg 

#from record import start_data_collection, stop_data_collection

## Global
moveit_commander.roscpp_initialize(sys.argv)
robot = moveit_commander.RobotCommander()
group_name = "manipulator"
move_group = moveit_commander.MoveGroupCommander(group_name)
display_trajectory = moveit_msgs.msg.DisplayTrajectory()
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
												moveit_msgs.msg.DisplayTrajectory,
												queue_size=20)
vel_scale_factor = 0.6
acc_scale_factor = 0.1

class Robot():
	def __init__(self):
		self.robot = robot
		self.move_group = move_group
		self.display_trajectory = display_trajectory
		self.display_trajectory_publisher = display_trajectory_publisher

		self.move_group.set_planner_id('RRTstar')
		self.move_group.set_max_acceleration_scaling_factor(acc_scale_factor)
		self.move_group.set_max_velocity_scaling_factor(vel_scale_factor)

	def _user_input(self,override=False):
		if override==True:
			self.move_group.execute(self.plan)											# Execute planned trajectory plan
			self.move_group.stop()														# Stop robot movement
			self.move_group.clear_pose_targets()										# Clear pose target before starting new planning
			print('Motion Completed!')
			print('\n')
		else:
			print('Movement Access?? [y/n]: ')
			key = raw_input()																# User input
			if key == 'y' or override==True:
				self.move_group.execute(self.plan)											# Execute planned trajectory plan
				self.move_group.stop()														# Stop robot movement
				self.move_group.clear_pose_targets()										# Clear pose target before starting new planning
				print('Motion Completed!')
				print('\n')
			elif key == 'n':
				print('Motion Declined!')
				print('\n')
			else:
				print('Invalid Input. Please select [y/n]')
				print('\n')
				self._user_input()

	def _plan_trajectory(self,override=False):
		self.plan = self.move_group.plan()												# Plan the motion trajectory
		self.display_trajectory.trajectory_start = self.robot.get_current_state()		# Get robot current state
		self.display_trajectory.trajectory.append(self.plan)							# Schedule planned trajectory plan
		self.display_trajectory_publisher.publish(self.display_trajectory)				# Publish planned trajectory plan
		print('Publishing motion trajectory in RVIZ!')
		#check for movement authorization by user or override
		self._user_input(override)

	def move_joint_space_goal(self, angle_base, angle_shoulder, angle_elbow, \
							angle_wrist1, angle_wrist2, angle_wrist3,override=False):

		joint_group_values = self.move_group.get_current_joint_values()					# Get current joint angle values
		joint_group_values[0] = angle_base												# Set new base joint value
		joint_group_values[1] = angle_shoulder											# Set new shoulder joint value
		joint_group_values[2] = angle_elbow												# Set new elbow joint value
		joint_group_values[3] = angle_wrist1											# Set new wrist1 joint value
		joint_group_values[4] = angle_wrist2											# Set new wrist2 joint value
		joint_group_values[5] = angle_wrist3											# Set new wrist3 joint value
		self.move_group.set_joint_value_target(joint_group_values)						# Set target joint values

		self._plan_trajectory(override)

	def move_offset_distance(self, offset_x, offset_y, offset_z,override=False):

		pose_values = self.move_group.get_current_pose()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation = pose_values.pose.orientation
		pose_goal.position.x += offset_x
		pose_goal.position.y += offset_y
		pose_goal.position.z += offset_z
		self.move_group.set_pose_target(pose_goal)

		self._plan_trajectory(override)

	def move_to_location(self, loc_x, loc_y, loc_z, override=False, vel = vel_scale_factor, acc=acc_scale_factor):

		pose_values = self.move_group.get_current_pose()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation = pose_values.pose.orientation
		pose_goal.position.x = loc_x
		pose_goal.position.y = loc_y
		pose_goal.position.z = loc_z

		
		self.move_group.set_pose_target(pose_goal)

		self._plan_trajectory(override)

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
		 
	# Helper functions

	# Joint Angle Based Homing 
	def joint_angle_homing(self,override=False):
			print("Joint-Space Based Pre-grasp Position")
			self.move_joint_space_goal(angle_base = -1.57, angle_shoulder = -1.57, angle_elbow = -1.91, \
										angle_wrist1 = -1.22, angle_wrist2 = 1.57, angle_wrist3 = 1.57, override = override)

	# Joint Angle Horizontal Homing 
	def joint_angle_horizontal_homing(self,override=False):
			print("Joint-Space Horizontal Pre-grasp Position")
			self.move_joint_space_goal(angle_base = -1.57, angle_shoulder = -1.57, angle_elbow = -1.91, \
										angle_wrist1 = -1.22, angle_wrist2 = 0, angle_wrist3 = 1.57,override = override)


	def manual_control(self, gripper):

		while True:
			direction = utils.robot_movement_direction_user_input()
			offset = float(raw_input('Enter Movement Distance: '))

			if direction == 'w':
				self.move_offset_distance(offset_x=0, offset_y=0, offset_z=offset)

			if direction == 's':
				self.move_offset_distance(offset_x=0, offset_y=0, offset_z=-offset)

			if direction == 'a':
				self.move_offset_distance(offset_x=offset, offset_y=0, offset_z=0)

			if direction == 'd':
				self.move_offset_distance(offset_x=-offset, offset_y=0, offset_z=0)

			if direction == 'z':
				self.move_offset_distance(offset_x=0, offset_y=offset, offset_z=0)

			if direction == 'x':
				self.move_offset_distance(offset_x=0, offset_y=-offset, offset_z=0)

			if direction == 'g':
				gripper.graspWithForce()

			if direction == 'r':
				gripper.homing()

			if (utils.reached_destination()):
				break

	def _horizontal_scan(self, pose_goal):
		itr = 5
		for i in range(itr):
			pose_goal.position.x += 0.02
			self.move_group.go(pose_goal, wait=True)
			self.move_group.stop()
			if (utils.reached_destination()):
				return True
			pose_goal.position.x -= 0.02
			self.move_group.go(pose_goal, wait=True)
			self.move_group.stop()

			if (i != (itr-1)):
				pose_goal.position.y -= 0.01
				self.move_group.go(pose_goal, wait=True)
				self.move_group.stop()

		pose_goal.position.y += 0.01*itr
		pose_goal.position.z -= 0.01
		self.move_group.go(pose_goal, wait=True)
		self.move_group.stop()
		return False


	def move_and_poke(self):
		print('Robot Working in the Move and Mode')

		pose_values = self.move_group.get_current_pose()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation = pose_values.pose.orientation
		for i in range(5):
			if (self._horizontal_scan(pose_goal)):
				return True
		return False


	def swiping(self, trial_dir, stages, insert):
		print('Robot working in the Swiping Mode')
		itr = 5
		choices = []

		pose_values = self.move_group.get_current_pose()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation = pose_values.pose.orientation
		for v in range(itr):
			timestamp = utils.get_current_time_stamp()
			stages.append(['start_'+str(v+1), timestamp])

			pose_goal.position.x += 0.01 * int(insert)
			self.move_group.go(pose_goal, wait=True)
			self.move_group.stop()


			pose_goal.position.z -= 0.1
			self.move_group.go(pose_goal, wait=True)
			self.move_group.stop()

			timestamp = utils.get_current_time_stamp()
			stages.append(['end_'+str(v+1), timestamp])

			choice = utils.label_data(trial_dir, v+1)
			choices.append(choice)

			# if (v != (itr-1)):
			pose_goal.position.x -= 0.01 * int(insert)
			self.move_group.go(pose_goal, wait=True)
			self.move_group.stop()

			pose_goal.position.z += 0.1
			pose_goal.position.y -= 0.01
			self.move_group.go(pose_goal, wait=True)
			self.move_group.stop()

		return stages, choices

'''
# Under Work
	def iid_swiping(self, trial_dir, stages, insert):
		print('Robot working in One Swipe Mode')
		itr = 5
		choices = []

		pose_values = self.move_group.get_current_pose()
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position = pose_values.pose.position
		pose_goal.orientation = pose_values.pose.orientation

		for v in range(itr):
			azure, tactile, wsg50, usbcam = start_data_collection(trial_dir)

			timestamp = utils.get_current_time_stamp()
			stages.append(['start_'+str(v+1), timestamp])

			pose_goal.position.x += 0.01 * int(insert)
			self.move_group.go(pose_goal, wait=True)
			self.move_group.stop()


			pose_goal.position.z -= 0.1
			self.move_group.go(pose_goal, wait=True)
			self.move_group.stop()

			timestamp = utils.get_current_time_stamp()
			stages.append(['end_'+str(v+1), timestamp])

			stop_data_collection(azure, tactile, wsg50, usbcam)

			choice = utils.label_data(trial_dir, v+1)
			choices.append(choice)

			pose_goal.position.x -= 0.01 * int(insert)
			self.move_group.go(pose_goal, wait=True)
			self.move_group.stop()

'''
# 	def exploration_sliding(self):
# 		pass