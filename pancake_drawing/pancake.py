import cv2
import os
import json
import argparse
import numpy as np
import rospy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.signal import butter, filtfilt, lfilter
from ur5_lib import gripper
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from wsg_50_common.msg import Status, Cmd
from squeeze_bot.utils import load_csv_as_dict, deconvolve, undistort
from squeeze_bot.bottle_constants import BOTTLE_WEIGHT, DENSITY_DICT, VISCOSITY_DICT
from squeeze_bot.learning.mlp import (
    MLPRegressor, get_frequency_features, get_rotation_features,
    get_viscosity_volume_features, get_viscosity_features)
from scipy.interpolate import interp1d
from scipy import interpolate

"""
Using haptics to predict flow and thickness.
Then given the stroke we want to draw, draw a line.
"""

ALGORITHMS = ["haptics", "viscosities", "mean"]
KNOT_TIMES = np.linspace(0., 17., 18)
KNOT_THICKNESSES = np.linspace(5., 20., 10)

class DishGarneshing(object):
  def __init__(self, args):
    # Arguments
    self.parent_dir = args.parent_dir
    self.liquid_type = args.liquid_type
    self.thickness_parent_dir = args.thickness_parent_dir
    self.flow_parent_dir = args.flow_parent_dir
    self.algorithm = args.algorithm
    self.drawing_filename = args.drawing_filename
    self.drawing_path = os.path.join(self.parent_dir, self.drawing_filename)
    # Create save direcory
    catalog_path = os.path.join(self.parent_dir, "catalog.csv")
    with open(catalog_path, 'r') as f:
      lines = f.readlines()
      if len(lines) == 1:
        self.experiment_index = 0
      else:
        self.experiment_index = int(os.path.basename(lines[-1].split()[0])) + 1
    self.experiment_dir = os.path.join(self.parent_dir, str(self.experiment_index))
    self.frame_dir = os.path.join(self.experiment_dir, "frames")
    if not os.path.isdir(self.frame_dir):
      os.makedirs(self.frame_dir)
    # Camera calibration and warping
    data_path = "/home/rocky/joehuang/active_shake_ws/src/squeeze_bot/camera_calibration.npz"
    data = np.load(data_path)
    self.mtx = data["mtx"]
    self.dist = data["dist"]
    data_path = "/home/rocky/joehuang/active_shake_ws/src/squeeze_bot/warp.npy"
    self.M = np.load(data_path)
    # Grid poses
    data = np.load('/home/rocky/joehuang/active_shake_ws/src/squeeze_bot/joint_poses.npz')
    self.interp = RegularGridInterpolator((data["xs"], data["ys"]), data["joint_poses"])
    self.xmin = np.min(data["xs"])
    self.xmax = np.max(data["xs"])
    self.ymin = np.min(data["ys"])
    self.ymax = np.max(data["ys"])
    # Key poses
    # self.dripping_pose = np.array([2.1213, -1.5316, 1.3533, -0.5717, -0.2125, -0.0182])
    self.dripping_pose = np.array([1.7605, -1.3508, 1.4975, -0.3379, -0.0695, -0.0677])
    # States
    self.start_time = None
    self.is_drawing = False
    self.joint_state = None
    self.total_volume = 0.0
    self.sauce_count = 0
    # Haptics collecting states
    self.is_wrist_ft_initialized = True
    self.is_recording_haptics = False
    self.initial_wrist_ft = None
    self.stamped_wrist_fts = []

    # Subscribers
    rospy.Subscriber('/joint_states', JointState, self.cb_joint_states)
    rospy.Subscriber('/wrist_sensor/wrench', WrenchStamped, self.cb_wrist_sensor)
    # Publishers
    self.pos_controller = rospy.Publisher('/scaled_pos_joint_traj_controller/command',
                                          JointTrajectory, queue_size=20)
    self.finger_traj_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=1)
    self.finger_speed_pub = rospy.Publisher('/wsg_50_driver/goal_speed', Float32, queue_size=1)

  def undistort_and_warp(self, frame):
    dst_frame = cv2.warpPerspective(undistort(frame, self.mtx, self.dist), self.M, (500,500))
    return dst_frame

  def get_joint_angle(self, s):
    x = max(min(self.s2x(s), self.xmax), self.xmin)
    y = max(min(self.s2y(s), self.ymax), self.ymin)
    return self.interp([np.array([x, y])])[0]


  def cb_joint_states(self, msg):
    self.joint_state = np.array(msg.position)

  def cb_wrist_sensor(self, msg):
    """ Wrist sensor force readings. """
    time = msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs
    stamped_wrist_ft = (time, msg.wrench.force.x, msg.wrench.torque.z)
    if not self.is_wrist_ft_initialized:
      if len(self.stamped_wrist_fts) <= 500:
        self.stamped_wrist_fts.append(stamped_wrist_ft)
      else:
        self.initial_wrist_ft = np.mean(self.stamped_wrist_fts, axis = 0)[1:]
        self.is_wrist_ft_initialized = True
        self.stamped_wrist_fts = []
    elif self.is_recording_haptics:
      self.stamped_wrist_fts.append(stamped_wrist_ft)

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


  def reset_squeeze(self):
    # Reset Squeeze
    gripper.set_force(20)
    gripper.move(75, 50)
    self.finger_speed_pub.publish(Float32(-10.0))
    rospy.sleep(2.)
    self.move_to_joint(
        self.dripping_pose + np.array([0., 0., 0., 0., 0., -np.pi]), time=1.5)
    rospy.sleep(4.0)
    self.move_to_joint(self.dripping_pose, time=1.5)
    rospy.sleep(2.0)
    gripper.move(75, 50)
    rospy.sleep(30.0)xs = np.linspace(-0.12, 0.12, 2)

    gripper.set_force(80)
    gripper.homing()

  def run(self):
    rospy.sleep(2.)
    gripper.homing()
    self.move_to_joint(self.dripping_pose, time = 2)
    rospy.sleep(4.)
    # different colors
    data = np.load(self.drawing_path)
    line_infos = data["line_infos"]
    colors = data["colors"]
    unique_colors = np.unique(colors, axis = 0)
    for color in unique_colors:
      # Change bottle
      print("Remove the bottle from the holder")
      while True:
        command = raw_input("Continue? [y]")
        if command == 'y':
          break
        else:
          print("Command '%s' unrecognized"%command)
          continue
      self.is_wrist_ft_initialized = False
      rospy.sleep(1.0)
      initial_force = self.initial_wrist_ft[0]
      print("Please put a sauce with %s color on the robot"%str(color))
      while True:
        command = raw_input("Continue? [y]")
        if command == 'y':
          break
        else:
          print("Command '%s' unrecognized"%command)
          continue
      self.is_wrist_ft_initialized = False
      rospy.sleep(1.0)
      # draw each line
      line_idxs = np.where(np.all(colors == color, axis=1))[0]
      for i, line_idx in enumerate(line_idxs):
        line_info = line_infos[line_idx]
        self.run_once(line_info)
        if i != len(line_idxs) - 1:
          self.reset_squeeze()
      self.sauce_count += 1

  def run_once(self, line_info):
    """ Draw a line. """
    # Use the line info
    ss = line_info[:, 0]
    xs = line_info[:, 1]
    ys = line_info[:, 2]
    thicknesses = np.ones_like(ss) * 7.
    rhos = #TODO just some random value will do
    vs = [0.]
    for idx in range(len(ss) - 1):
      ds = ss[idx + 1] - ss[idx]
      vs.append(vs[-1] + ds * rhos[idx])
    v2s = interp1d(vs, ss, kind = 'linear')
    self.s2x = interp1d(ss, xs, kind='linear')
    self.s2y = interp1d(ss, ys, kind='linear')

    flows = #TODO just some random value will do
    # Obtain the squeeze trajectory
    dt = times[1] - times[0]
    volumes = []
    volume = 0.
    for flow in flows:
      volume += flow * dt
      volumes.append(volume)
    volumes = np.array(volumes) * 0.001
    t2v = interp1d(times, volumes, kind='cubic')
    self.times = times
    self.volumes = volumes
    self.flows = flows
    # Homing the gripper
    gripper.set_force(80)
    gripper.homing()
    # Move to starting pose
    self.is_done = False
    while not self.is_done:
      self.move_to_joint(self.dripping_pose, time = 2)
      rospy.sleep(2.)
      initial_joint_angle = self.get_joint_angle(v2s(self.total_volume))
      self.follow_trajectory([initial_joint_angle], [2.])
      rospy.sleep(2.1)
      # Calculate the open-loop plan
      joint_angles = []
      times = []
      stop_time = 20.0
      for time in np.linspace(0.0, 17.0, 1701):
        volume = np.maximum(t2v(time) + self.total_volume, 1e-8)
        if volume > np.max(vs):
          stop_time = time
          break
        joint_angle = self.get_joint_angle(v2s(volume))
        joint_angles.append(joint_angle)
        times.append(time)
      self.total_volume = volume
      gripper.move(75, 50)
      self.finger_traj_pub.publish(Cmd())
      self.follow_trajectory(joint_angles, times)
      print("squeeze and move")
      # Wait for terminal condition
      rate = rospy.Rate(20)
      self.is_drawing = True
      self.start_time = rospy.get_time()
      while self.is_drawing:
        # stop squeezing after a certain duration
        if rospy.get_time() > self.start_time + 17.0:
          self.follow_trajectory([self.joint_state], [0.])
          self.finger_speed_pub.publish(Float32(50.0))
          self.is_drawing = False
          print("Resetting")
        elif rospy.get_time() > self.start_time + stop_time:
          self.follow_trajectory([self.joint_state], [0.])
          self.finger_speed_pub.publish(Float32(50.0))
          self.is_drawing = False
          self.is_done = True
          print("Completed")
        # Sleep
        rate.sleep()
      gripper.homing()
      self.move_to_joint(self.dripping_pose, time = 2)
      rospy.sleep(2.)
      if not self.is_done:
        self.reset_squeeze()
    # Reset states
    self.start_time = None
    self.is_drawing = False
    self.total_volume = 0.0

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description = "Garnishing dishes with sauces")
  parser.add_argument('-p', '--parent_dir',
      type=str, help='The path of the dish garnishing',
      default='/media/rocky/extraHDD1/joehuang/dish_garnishing')
  parser.add_argument('-tp', '--thickness_parent_dir',
      default='/media/rocky/extraHDD1/joehuang/squeezebot_data/thickness_data',
      type=str, help='The path of thickness trained model')
  parser.add_argument('-fp', '--flow_parent_dir',
      default='/media/rocky/extraHDD1/joehuang/squeezebot_data/flow_data',
      type=str, help='The path of flow trained model')
  parser.add_argument('-l', '--liquid_type', default='ketchup', type=str,
      help='The type of liquid')
  parser.add_argument('-a', '--algorithm', default='haptics', type=str,
      help='The type of algorithm')
  parser.add_argument('-d', '--drawing_filename', type=str,
      help='The filename of drawing')

  # Start the dish garneshing process
  rospy.init_node("dish_garneshing")
  dish_garneshing = DishGarneshing(args)
  dish_garneshing.run()