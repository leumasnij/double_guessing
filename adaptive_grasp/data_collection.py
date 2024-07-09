#Adaptive Grasping
#!/usr/bin/env python
from re import I
import cv2
import os
import argparse
import numpy as np
import rospy
from ur5_lib import gripper
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from wsg_50_common.msg import Status, Cmd
from scipy.interpolate import interp1d
from scipy import interpolate
import gelsight_test as gs
import threading
from ur_ikfast import ur_kinematics
import math
import numpy as np
from kinematic import forward_kinematic, inverse_kinematic, inverse_kinematic_orientation
from apriltag_helper.tag import detect_tag, img2robot, CoM_calulation
import torch
import neural_networks.nn_helpers as nnh
from cam_read import CameraCapture, RealSenseCam

class Grasp(object):
  def __init__(self, record=False):
    
    self.ur5e_arm = ur_kinematics.URKinematics('ur5e')
    self.model = nnh.Net()
    # self.model.load_state_dict(torch.load('/home/okemo/samueljin/stablegrasp_ws/src/best_model.pth'))
    # torch.load('/home/okemo/samueljin/stablegrasp_ws/src/best_model.pth')
    self.model.eval()
    # self.reset_joint = [ 1.21490335, -1.32038331,  1.51271999, -1.76500773, -1.57009947,  1.21490407]
    self.reset_joint = [ 1.21490335, -1.283166,  1.6231562, -1.910088, -1.567829,  -0.359537]
    self.start_loc = np.array([0.0, 0.6, 0.245])
    # self.start_yaw = -np.pi/2 
    self.start_yaw = 2e-3
    self.start_pitch = 2e-3
    self.start_roll = np.pi
    # self.q_array = euler_to_quaternion(self.start_roll, self.start_pitch, self.start_yaw)
    self.start_rot = np.array([[np.cos(self.start_yaw)*np.cos(self.start_pitch), np.cos(self.start_yaw)*np.sin(self.start_pitch)*np.sin(self.start_roll)-np.sin(self.start_yaw)*np.cos(self.start_roll), np.cos(self.start_yaw)*np.sin(self.start_pitch)*np.cos(self.start_roll)+np.sin(self.start_yaw)*np.sin(self.start_roll)],
                               [np.sin(self.start_yaw)*np.cos(self.start_pitch), np.sin(self.start_yaw)*np.sin(self.start_pitch)*np.sin(self.start_roll)+np.cos(self.start_yaw)*np.cos(self.start_roll), np.sin(self.start_yaw)*np.sin(self.start_pitch)*np.cos(self.start_roll)-np.cos(self.start_yaw)*np.sin(self.start_roll)],
                               [-np.sin(self.start_pitch), np.cos(self.start_pitch)*np.sin(self.start_roll), np.cos(self.start_pitch)*np.cos(self.start_roll)]])
    self.start_mat = np.zeros((3,4))
    self.start_mat[:3,:3] = self.start_rot
    self.start_mat[:3,3] = self.start_loc
    self.start_joint = self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.reset_joint)
    self.start_up_joint = self.up_joint(self.start_joint)
    self.start_angle_1 = self.start_joint[-1]
    self.start_angle_2 = self.start_joint[-2]
    
    # States
    self.joint_state = None
    self.gripper_width = None
    self.gripper_force = None
    self.force_calibrated = False
    self.force_torque = None
    # Data
    self.stamped_haptic_data = []
    self.stamped_weight_data = []
    self.stamped_gripper_data = []
    self.stamped_force_data = []
    self.force_ref = [0,0,0,0,0,0]
    self.ftwindow = []
    rospy.Subscriber('/joint_states', JointState, self.cb_joint_states)
    # rospy.Subscriber('/weight', Float32, self.cb_weight)
    rospy.Subscriber('/wsg_50_driver/status', Status, self.cb_gripper)
    # rospy.Subscriber('/wrench', WrenchStamped, self.cb_force_torque, queue_size=1)
    rospy.Subscriber('/wrist_sensor/wrench', WrenchStamped, self.cb_force_torque, queue_size=1)
    rospy.sleep(1)
    # Publishers
    self.pos_controller = rospy.Publisher('/scaled_pos_joint_traj_controller/command',
                                          JointTrajectory, queue_size=20)
    self.finger_traj_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=1)
    self.finger_speed_pub = rospy.Publisher('/wsg_50_driver/goal_speed', Float32, queue_size=1)
    import datetime
    cur_date = datetime.datetime.now()
    self.saving_adr = '/media/okemo/extraHDD31/samueljin/' + str(cur_date.month) + '_' + str(cur_date.day) + '/'
    if not os.path.exists(self.saving_adr):
        os.makedirs(self.saving_adr)
    exp_run = len([name for name in os.listdir(self.saving_adr) ]) + 1
    self.saving_adr = self.saving_adr + 'run' + str(exp_run) + '/'
    # self.tracker = CameraCapture1()
    # self.tac_img_size = self.tracker.tac_img_size

    #cameras
    # self.realsense = RealSenseCam()
    self.overhead_cam = None
    # self.marker_gelsight = None
    self.init_cam()


  def init_cam(self):
    for i in range(0,9,2):
      cap = cv2.VideoCapture(i)
      ret, frame = cap.read()
      if ret:
          cv2.imwrite(str(i) + '.jpg', frame)
          file = 'video' + str(i)
          real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
          with open(real_file, "rt") as name_file:
              name = name_file.read()
          cap.release()
          if name == 'HD Pro Webcam C920\n':
            self.overhead_cam = CameraCapture(i)
            return
          if name == 'GelSight Mini R0B 2G4W-G32G: Ge\n':
            self.marker_gelsight = CameraCapture(i)
            continue
      cap.release()


  def cb_joint_states(self, msg):
    # self.joint_state = np.array(msg.position)
    while len(msg.position) < 6:
      continue
    self.joint_state = np.array([msg.position[2], msg.position[1], msg.position[0],
                   msg.position[3], msg.position[4], msg.position[5]])

  def cb_gripper(self, msg):
    # print(msg)
    self.gripper_width = msg.width
    self.gripper_force = msg.force

  def cb_force_torque(self, msg):
    """ Force Torque data callback. """
       
    force_torque = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                  msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
    # print(self.force_torque)
    if len(self.ftwindow) < 15:
        self.ftwindow.append(force_torque)
        return
    
    self.ftwindow.pop(0)
    self.ftwindow.append(force_torque)
    self.force_torque = np.mean(self.ftwindow, axis=0) - self.force_ref
    # print(self.ftwindow)
    # if self.force_calibrated == False:
    #     self.cali_force(self.force_torque)
    # else:
    #     self.force_torque -= self.force_offset
    

  def move_to_joint(self, joint_space_goal, time = 2.0):
    """ Move to joint state. """
    pos_message = JointTrajectory()
    pos_message.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    pos_message_point = JointTrajectoryPoint()
    pos_message_point.positions = joint_space_goal
    pos_message_point.time_from_start = rospy.Duration(time)
    pos_message.points.append(pos_message_point)
    self.pos_controller.publish(pos_message)
  
  def reset_gripper(self):
    gripper.move(50)

  def grasp_part(self, force):
    gripper.set_force(force)
    target_width = 40
    while self.gripper_force < 5 and target_width >=0:
      rospy.sleep(0.1)
      gripper.grasp(target_width, 60)
    #   rospy.sleep(0.1)
    #   print(self.gripper_force, self.gripper_width)
      target_width -= 20
    # gripper.set_force(60)
    rospy.sleep(1)
    width = max(0, self.gripper_width)
    gripper.grasp(width, 60)

  def pickup(self, force = 80, joint = None):
    """ Pickup the object. """
    # rospy.sleep(1)
    if joint is None:
       joint = self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.joint_state)
    self.reset_gripper()
    # print(self.joint_state)
    # print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))

    self.move_to_joint(joint, 2)
    rospy.sleep(5)
    # self.cali_force()

    self.grasp_part(force)
    up_mat = self.up_joint(joint=joint)
    self.move_to_joint(up_mat, 3)
    rospy.sleep(3)
    self.move_to_joint(self.start_up_joint, 2)
    rospy.sleep(2)

  
  def up_joint(self, joint):
      mat = self.ur5e_arm.forward(joint, 'matrix')
      mat[2,3] = 0.4
      return self.ur5e_arm.inverse(mat, False, q_guess=joint)


  def ypr_to_mat(self, yaw, pitch, roll):
    return np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                     [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                     [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
  
  def loc2mat(self, loc, rot = None):
    if rot is None:
        rot = self.start_rot
    mat = np.zeros((3,4))
    mat[:3,:3] = rot
    mat[:3,3] = loc
    return mat
  
  def move_away(self):

    self.reset_gripper()

    cur_loc = self.ur5e_arm.forward(self.joint_state, 'matrix')
    cur_loc[2,3] = 0.5
    new_joint = self.ur5e_arm.inverse(cur_loc, False, q_guess=self.joint_state)

    self.move_to_joint(new_joint, 5)
    rospy.sleep(5.5)

    away_joint = new_joint[:]
    away_joint[0] = 2.4

    self.move_to_joint(away_joint, 5)
    rospy.sleep(5.5)

  def save_data(self, force, rot, GT, testid = 0):
    for i in range(len(force)):
      x = force[i]
      x = np.append(x, rot[i])
      save_dirc = self.saving_adr + str(testid) + '/'
      save_dict = {'force': x, 'GT': GT}
      if not os.path.exists(save_dirc):
          os.makedirs(save_dirc)
      np.save(save_dirc + 'data' + str(i) + '.npy', save_dict)

  
  def rotate_to(self, angle1, angle2):
    joint = self.joint_state
    joint[-1] = angle1 + self.start_angle_1
    joint[-2] = angle2 + self.start_angle_2
    self.move_to_joint(joint, 3)
    rospy.sleep(8)
  
  def single_trial(self, size = 20, testid = 0):
    loc = self.ur5e_arm.forward(self.joint_state, 'matrix')[:3,3]
    starting_joint = self.joint_state.copy()
    FT_array = []
    rot_array = []
    for i in range(size):
      angle2 = np.random.uniform(5*np.pi/6, np.pi/6)
      for j in range(10):
          trial_num = i*10 + j + testid*size*10
          print('trial: ' + str(trial_num) + ' angle: ' + str(j))
          angle1 = np.random.uniform(-np.pi/2, np.pi/2)
          self.rotate_to(angle1, angle2)
          FT_array.append(self.force_torque)
          rot_array.append([angle1, angle2])
    self.move_to_joint(starting_joint, 8)
    rospy.sleep(11)
    FT_array.append(self.force_torque)
    rot_array.append([0, 0])
    return FT_array, rot_array

  def calibration(self, rot_array, FT_array):
    # self.reset_gripper()
    up_joint = self.up_joint(self.joint_state)
    self.move_to_joint(up_joint, 5)
    rospy.sleep(5)
    self.move_to_joint(self.start_up_joint, 5)
    rospy.sleep(5)
    FT_ref_array = []
    for i in rot_array:
      self.rotate_to(i[0], i[1])
      FT_ref_array.append(self.force_torque)
    # print(FT_array, FT_ref_array)
    return FT_array - FT_ref_array
  
            
  def yawpitchroll_from_joint(self):
      mat = self.ur5e_arm.forward(self.joint_state, 'matrix')
      yaw = np.arctan2(mat[1,0], mat[0,0])
      pitch = np.arctan2(-mat[2,0], np.sqrt(mat[2,1]**2 + mat[2,2]**2))
      roll = np.arctan2(mat[2,1], mat[2,2])
      print('Yaw: ' + str(yaw) + ' Pitch: ' + str(pitch) + ' Roll: ' + str(roll))

  def cali_force(self):
     self.force_ref = self.force_torque.copy()

     
  def single_grasp(self, num_tag, side, testid = 0, size = 20):
    rospy.sleep(1)
    self.move_away()
    main_tag, main_center, CoM, no_grasp_zone, moi = CoM_calulation(self.overhead_cam,num_tag)
    move_loc, rot = self.generate_rand_grasp(main_tag, main_center, CoM, side)
    if no_grasp_zone != []:
      flag = True
      while flag:
        for i in no_grasp_zone:
          if i[0] <= move_loc[0] <= i[1] and i[2] <= move_loc[1] <= i[3] and i[4] <= move_loc[2] <= i[5]:
            flag = True
            move_loc, rot = self.generate_rand_grasp(main_tag, main_center, CoM, side)
            break
          flag = False
    joint = self.ur5e_arm.inverse(self.loc2mat(move_loc, rot), False, q_guess=self.joint_state)
    GT = move_loc - CoM
    GT = np.append(GT, moi)
    up_joint = self.up_joint(joint)
    for i in range(size):
      self.move_to_joint(up_joint, 5)
      rospy.sleep(5)
      self.pickup(joint=joint)
      FT_array, rot_array = self.single_trial(1, testid)
      self.reset(main_center, move_loc)
      FT_array = np.array(FT_array)
      rot_array = np.array(rot_array)
      FT_final = self.calibration(rot_array, FT_array)
      self.save_data(FT_final, rot_array, GT, testid)
    
    
  
  def generate_rand_grasp(self, main_tag, main_center, CoM, side):
    height = np.random.uniform(0.232, 0.292)
    if main_tag == 0:
      offset = np.random.uniform(-0.07, 0.07)
      move_loc = np.array([main_center[0] + offset, main_center[1], height])
      rot = self.ypr_to_mat(2e-3, 2e-3, np.pi)
    elif main_tag == 1:
      if side == 0:
        offset = np.random.uniform(-0.055, 0.045)
        move_loc = np.array([main_center[0] + offset, main_center[1] - 0.075, height])
        rot = self.ypr_to_mat(2e-3, 2e-3, np.pi)
      elif side == 1:
        offset = np.random.uniform(-0.055, 0.045)
        move_loc = np.array([main_center[0] + offset, main_center[1] + 0.075, height])
        rot = self.ypr_to_mat(2e-3, 2e-3, np.pi)
    return move_loc, rot
  
  def up_joint(self, joint):
      mat = self.ur5e_arm.forward(joint, 'matrix')
      mat[2,3] = 0.4
      return self.ur5e_arm.inverse(mat, False, q_guess=joint)
  
  def reset(self, main_center, grasp_loc):
    rospy.sleep(1)
    overhead_h, overhead_w = self.overhead_cam.read()[1].shape[:2]
    mid_point = [overhead_w/2, overhead_h/2]
    robot_mid = img2robot(mid_point)
    robot_mid[2] = grasp_loc[2]
    grasp_offset = grasp_loc - main_center
    robot_mid[:2] += grasp_offset[:2]
    reset_joint = self.ur5e_arm.inverse(self.loc2mat(robot_mid), False, q_guess=self.joint_state)
    up_joint = self.up_joint(reset_joint)
    self.move_to_joint(up_joint, 5)
    rospy.sleep(5)
    self.move_to_joint(reset_joint, 5)
    rospy.sleep(5)
    self.reset_gripper()

  def data_collection_main(self, grasp_num, rot_num):
    while self.force_torque is None:
      rospy.sleep(1)
    num_tag = int(input('Number of tags: '))
    for i in range(grasp_num):
      if i % 2 == 0:
        self.single_grasp(num_tag, 0, i, rot_num)
      else:
        self.single_grasp(num_tag, 1, i, rot_num)
    

if __name__ == '__main__':
  rospy.init_node('grasp')
  rospy.sleep(1)
  rospy.Rate(15)
  # np.random.seed(42)
  Grasp_ = Grasp(record=False)
  rospy.sleep(1)
  # Grasp_.showcamera()
  Grasp_.data_collection_main(30,5)
  # Grasp_.pickup()
  # Grasp_.reset(np.array([0.0, 0.6, 0.245]), np.array([0.0, 0.6, 0.245]))
  # Grasp_.yawpitchroll_from_joint()

  # Grasp_.force_torque_data()
  # Grasp_.reset()
  # sum_nn = 0
  # sum_an = 0
  # for i in range(10):
  #   nn,an = Grasp_.CoM_verify()
  #   sum_nn += nn
  #   sum_an += an
  # print('NN: ' + str(sum_nn/10) + ' Analytical: ' + str(sum_an/10))
  # Grasp_.CoM_verify()
  # while True:
  #   pass
  # Grasp_.ft_test()
  # Grasp_.CoM_estimation()
#   Grasp_.test2()
#   print(forward_kinematic(Grasp_.joint_state))
# Grasp_.reset_gripper()
  # Grasp_.pickup()

#   Grasp_.test_rot()
  # Grasp_.test()
  # Grasp_.test2()