
import cv2
import os
import argparse
import numpy as np
import rospy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from ur5_lib import gripper
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from wsg_50_common.msg import Status, Cmd
from scipy.interpolate import interp1d
from scipy import interpolate
from image_decomp import image_decomp_main
from read_npz import flow_rate_to_time_matrix
"""
Using haptics to predict flow and thickness.
Then given the stroke we want to draw, draw a line.
"""


class DishGarneshing(object):
  def __init__(self):

    # Grid poses
    self.adr = '/home/rocky/samueljin/pancake_bot/active_shake_ws/src/pancake_drawing/file4.jpg'
    self.flow_rate_adr = '/home/rocky/samueljin/pancake_bot/batter_wide/1/weights.npz'
    data = np.load('/home/rocky/samueljin/pancake_bot/active_shake_ws/src/pancake_low.npz')
    self.interp = RegularGridInterpolator((data["xs"], data["ys"]), data["joint_poses"])
    self.xmin = np.min(data["xs"])
    self.xmax = np.max(data["xs"])
    self.ymin = np.min(data["ys"])
    self.ymax = np.max(data["ys"])
    # Key poses
    # self.dripping_pose = np.array([2.1213, -1.5316, 1.3533, -0.5717, -0.2125, -0.0182])
    self.dripping_pose = data["joint_poses"][0][0]
    self.non_drip_pos = np.array([0., 0., 0., 0., 0., np.pi/2+0.05])
    self.non_drip_pos2 = np.array([0., 0., 0., 0., 0., np.pi])
    # States
    self.start_time = None
    self.is_drawing = False
    self.joint_state = None
    self.total_volume = 0.0
    self.sauce_count = 0

    # Subscribers
    rospy.Subscriber('/joint_states', JointState, self.cb_joint_states)
    # Publishers
    self.pos_controller = rospy.Publisher('/scaled_pos_joint_traj_controller/command',
                                          JointTrajectory, queue_size=20)
    self.finger_traj_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=1)
    self.finger_speed_pub = rospy.Publisher('/wsg_50_driver/goal_speed', Float32, queue_size=1)
    
  def get_joint_angle(self, s):
    x = max(min(self.s2x(s), self.xmax), self.xmin)
    y = max(min(self.s2y(s), self.ymax), self.ymin)
    return self.interp([np.array([x, y])])[0]


  def cb_joint_states(self, msg):
    self.joint_state = np.array(msg.position)


  def follow_trajectory(self, joint_space_goals, times):
    """ follow joint trajectory. """
    pos_message = JointTrajectory()
    pos_message.joint_names = ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint",
                               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    for joint_space_goal, time in zip(joint_space_goals, times):
      pos_message_point = JointTrajectoryPoint()
      pos_message_point.positions = joint_space_goal
      pos_message_point.time_from_start = rospy.Duration(time)
      pos_message.points.append(pos_message_point)
    self.pos_controller.publish(pos_message)

  def move_to_joint(self, joint_space_goal, time = 2.0):
    """ Move to joint state. """
    pos_message = JointTrajectory()
    pos_message.joint_names = ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint",
                               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    pos_message_point = JointTrajectoryPoint()
    pos_message_point.positions = joint_space_goal
    pos_message_point.time_from_start = rospy.Duration(time)
    pos_message.points.append(pos_message_point)
    self.pos_controller.publish(pos_message)


  def reset_squeeze(self, cur_pos):
    # Reset Squeeze

    self.finger_speed_pub.publish(Float32(60.0))
    self.move_to_joint(cur_pos + self.non_drip_pos, time = 2)
    rospy.sleep(2.)


  def star_squeeze(self):
    rospy.sleep(2.)
    gripper.move(75, 50)
    self.move_to_joint(self.interp([0,0])[0] + self.non_drip_pos, time = 2)
    rospy.sleep(2)
    velocity = flow_rate_to_time_matrix(self.flow_rate_adr)
    flag = 1
    for j in range(7):
        i = np.array([[40*j+5, a] for a in range(251)])
        
        ys = -(i[:,0]/250.)*0.20 + 0.10
        xs = (i[:,1]/250.)*0.20 - 0.10

        joint_angles = []
        times = []
        time = []
        joint_angle = []
        velocity = (j+2)*0.025
        for j in range(len(xs)):
            if(velocity*j<20):
              time.append(velocity*j+0.05)
              joint_angle.append(self.interp([np.array([xs[j], ys[j]])])[0])
            else:
              break

        times.append(time)
        joint_angles.append(joint_angle)

        while flag:
          command = raw_input("Is bottle placed? [y]")
          if command == 'y':
              flag = False
              break
          else:
              print("Command '%s' unrecognized"%command)
              continue
        self.follow_trajectory([joint_angles[0][0] + self.non_drip_pos], [2])
        rospy.sleep(2.) 
        for k in range(len(times)):
            #print(times[k])
            self.follow_trajectory([joint_angles[k][0]], [0.5])
            rospy.sleep(0.5)
            gripper.move(75, 50)
            self.finger_traj_pub.publish(Cmd())
            self.follow_trajectory(joint_angles[k], times[k])

            cur_time = 0
            for i in times[k]:
              rospy.sleep(i - cur_time)
              cur_time = i
            if(len(times[k]) == 1):
                rospy.sleep(10)
            self.finger_speed_pub.publish(Float32(-10.0))
            gripper.set_force(20)
            gripper.move(85, 50)
        self.follow_trajectory([joint_angles[k][-1] + self.non_drip_pos], [2])
        rospy.sleep(4.)   
    gripper.move(75, 50)
    # different colors



  def run(self):
    rospy.sleep(2.)
    gripper.move(75, 50)
    #self.move_to_joint(self.interp([np.array([0, 0])])[0], time = 5)
    data = image_decomp_main(self.adr)
    self.move_to_joint(self.interp([0,0])[0] + self.non_drip_pos, time = 2)
    rospy.sleep(2)
    flag = True
    velocity = flow_rate_to_time_matrix(self.flow_rate_adr)
    velocity2 = next(x[0] for x in enumerate(velocity) if x[1] > 20)
    velocity2 = velocity[velocity2-1:]
    
    for i in data:
        
        ys = -(i[0][0]/250.)*0.20 + 0.10
        xs = (i[0][1]/250.)*0.20 - 0.10
        ys = np.where(ys > 0.1, 0.1, ys)
        ys = np.where(ys < -0.1, -0.1, ys)
        xs = np.where(xs > 0.1, 0.1, xs)
        xs = np.where(xs < -0.1, -0.1, xs)
        joint_angles = []
        times = []
        time = []
        joint_angle = []
        
        reset_count = 0
        for j in range(len(xs)):
            if(velocity[j - reset_count]>20.5):
                
                times.append(time)
                joint_angles.append(joint_angle)
                time = []
                joint_angle = []
                reset_count=j
            joint_angle.append(self.interp([np.array([xs[j], ys[j]])])[0])
            if reset_count == 0:
                time.append(velocity[j - reset_count])
            else:
                time.append(velocity2[j - reset_count] - 20)
        times.append(time)
        joint_angles.append(joint_angle)
        while flag:
          command = raw_input("Is bottle placed? [y]")
          if command == 'y':
              flag = False
              break
          else:
              print("Command '%s' unrecognized"%command)
              continue
        # self.follow_trajectory([joint_angles[0][0] + self.non_drip_pos], [1])
        # rospy.sleep(1)
        self.follow_trajectory([joint_angles[0][0] + self.non_drip_pos], [2])
        rospy.sleep(2.) 
        for k in range(len(times)):
            #print(times[k])
            self.follow_trajectory([joint_angles[k][0]], [0.5])
            rospy.sleep(0.5)
            gripper.move(75, 50)
            self.finger_traj_pub.publish(Cmd())
            self.follow_trajectory(joint_angles[k], times[k])

            cur_time = 0
            for i in times[k]:
              rospy.sleep(i - cur_time)
              cur_time = i
            if(len(times[k]) == 1):
                rospy.sleep(10)
            self.finger_speed_pub.publish(Float32(-10.0))
            gripper.set_force(20)
            gripper.move(85, 50)
        self.follow_trajectory([joint_angles[k][-1] + self.non_drip_pos], [2])
        rospy.sleep(4.)   
    gripper.move(75, 50)
    # different colors
  

    
if __name__ == "__main__":
  # Start the dish garneshing process
  rospy.init_node("pancake")
  dish_garneshing = DishGarneshing()
  dish_garneshing.run()
  #dish_garneshing.line_squeeze()
  #dish_garneshing.finger_traj_pub.publish(Cmd())