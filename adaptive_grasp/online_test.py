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
from apriltag_helper.tag import detect_tag, img2robot, CoM_calulation, robot2img
import torch
import neural_networks.nn_helpers as nnh
from cam_read import CameraCapture
from neural_networks.ActiveNet import DenseAFNet, deterministic_model, truedistributionfromobs, grid_search, DenserNet
from neural_networks.PyroNet import BNN_pretrained2Pos, BNN_pretrained, RegNet
from pyro.infer import Predictive

class Grasp(object):
  def __init__(self, active_model_adr, bnn_adr, bnn_model):
    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.ur5e_arm = ur_kinematics.URKinematics('ur5e')

    self.active_model = DenseAFNet()
    self.active_model.load_state_dict(torch.load(active_model_adr))
    self.active_model.eval().to(self.device)
    
    self.bnn_weight = torch.load(bnn_adr)
    self.bnn_model = bnn_model
    
    self.deterministic_model = RegNet(input_size=8, output_size=3)
    self.deterministic_model = deterministic_model(self.bnn_weight, self.deterministic_model, self.device)

    self.two_pos_model = RegNet(input_size=16, output_size=3)
    self.two_pos_weight = torch.load('/media/okemo/extraHDD31/samueljin/Model/bnn2pos5_best_model.pth', map_location=self.device)
    self.two_pos_model = deterministic_model(self.two_pos_weight, self.two_pos_model, self.device)

    
    
    # self.reset_joint = [ 1.21490335, -1.32038331,  1.51271999, -1.76500773, -1.57009947,  1.21490407]
    self.reset_joint = [ 1.21490335, -1.283166,  1.6231562, -1.910088, -1.567829,  -0.359537]
    self.start_loc = np.array([0.0, 0.6, 0.33])
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
          # cv2.imwrite(str(i) + '.jpg', frame)
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
    # gripper.move(20)
    # gripper.homing()
    s = input('Please Reset Gripper')

  # def grasp_part(self, force):
  #   gripper.set_force(force)
  #   target_width = self.gripper_width
  #   while self.gripper_force < 5 and target_width >=0:
  #     rospy.sleep(0.1)
  #     gripper.grasp(target_width, 60)
  #   #   rospy.sleep(0.1)
  #   #   print(self.gripper_force, self.gripper_width)
  #     target_width -= 20
  #   # gripper.set_force(60)
  #   rospy.sleep(1)
  #   width = max(0, self.gripper_width)
  #   gripper.grasp(width, 60)

  def grasp_part(self, force):
     input('Please Grasp Object with Force ' + str(force))

  def pickup(self, force = 80, joint = None):
    """ Pickup the object. """
    # rospy.sleep(1)
    if joint is None:
       joint = self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.joint_state)
    # self.reset_gripper()
    # print(self.joint_state)
    # print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))

    self.move_to_joint(joint, 2)
    rospy.sleep(3)
    # self.cali_force()
    height = self.ur5e_arm.forward(joint, 'matrix')[2,3]
    print('Height: ' + str(height))
    self.grasp_part(force)
    up_mat = self.up_joint(joint=joint)
    
    self.move_to_joint(up_mat, 3)
    rospy.sleep(3)
    self.move_to_joint(self.start_up_joint, 2)
    rospy.sleep(2)

  
  def up_joint(self, joint):
      mat = self.ur5e_arm.forward(joint, 'matrix')
      mat[2,3] = max(0.4, mat[2,3] + 0.1)
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
  
  def yawpitchroll_from_joint(self):
      mat = self.ur5e_arm.forward(self.joint_state, 'matrix')
      yaw = np.arctan2(mat[1,0], mat[0,0])
      pitch = np.arctan2(-mat[2,0], np.sqrt(mat[2,1]**2 + mat[2,2]**2))
      roll = np.arctan2(mat[2,1], mat[2,2])
      print('Yaw: ' + str(yaw) + ' Pitch: ' + str(pitch) + ' Roll: ' + str(roll))
  
  def up_joint(self, joint):
      mat = self.ur5e_arm.forward(joint, 'matrix')
      mat[2,3] = 0.4
      return self.ur5e_arm.inverse(mat, False, q_guess=joint)

  def calib_force(self, angle1 = 0.0, angle2 = 0.0):
      # self.reset_gripper()
      print('Calibrating Force')
      # self.move_to_joint(self.start_up_joint, 5)
      # rospy.sleep(5)
      new_joint = self.start_up_joint.copy()
      new_joint[-1] += angle1
      new_joint[-2] += angle2
      self.move_to_joint(new_joint, 3)
      rospy.sleep(8)
      rer_force = self.force_torque.copy()
      return rer_force
  
  def estimate_CoM(self, force, angle1 = 0.0, angle2 = 0.0, ret_std = False):
      input = np.concatenate((force, [angle1, angle2]))
      input = torch.tensor(input, dtype=torch.float32).float().to(self.device)
      input = input.view(1,-1)
      mean = self.deterministic_model(input).cpu().detach().numpy()
      std = [0,0,0]
      if ret_std:
          pred = Predictive(model = self.bnn_model, posterior_samples=self.bnn_weight)
          # print(input)
          output = pred(input)
          std = output['obs'].std(0).cpu().detach().numpy()
          print(std)
      mean = self.deterministic_model(input).cpu().detach().numpy()
      return mean[0], std[0]
    
  def estimate_action(self, mean, std):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      input = np.concatenate((mean, std))
      # input = torch.tensor(input, dtype=torch.float32).float().to(device)
      angle1, angle2 = grid_search(self.active_model, device=device, inputs=input, grid_size=100)
      return angle1, angle2
  
  def estimate_two_pos(self, input):
      self.two_pos_model.eval()
      input = torch.tensor(input, dtype=torch.float32).float().to(self.device)
      input = input.view(1,-1)
      mean = self.two_pos_model(input).cpu().detach().numpy()
      return mean[0]

  def move_away(self):

    self.reset_gripper()

    cur_loc = self.ur5e_arm.forward(self.joint_state, 'matrix')
    cur_loc[2,3] = 0.5
    new_joint = self.ur5e_arm.inverse(cur_loc, False, q_guess=self.joint_state)

    self.move_to_joint(new_joint, 5)
    rospy.sleep(5.5)

    away_joint = new_joint[:]
    away_joint[0] = 0.75

    self.move_to_joint(away_joint, 5)
    rospy.sleep(5.5)
    
  def test_main(self):
      rospy.sleep(1)
      self.reset_gripper()
      # self.estimate_CoM([0,0,0,0,0,0], 0.0, 0.0, True)
      # raise Exception('Test Done')
      ref_force = self.calib_force(0.0, 0.0)
      # self.move_to_joint(self.start_joint, 5)
      # rospy.sleep(5)
      self.pickup()
      self.move_to_joint(self.start_up_joint, 5)
      rospy.sleep(5)
      force = self.force_torque - ref_force
      print(force)
      print('First Estimation Start')
      mean, std = self.estimate_CoM(force, ret_std=True)
      print('First Estimation: ' + str(mean))
      angle1, angle2 = self.estimate_action(mean, std)
      print('New Angles: ' + str(angle1) + ' ' + str(angle2))
      new_joint = self.joint_state.copy()
      new_joint[-1] += angle1
      new_joint[-2] += angle2
      print('Second Estimation Start')
      self.move_to_joint(new_joint, 3)
      rospy.sleep(8)
      force2 = self.force_torque

      rand_angle1 = np.random.uniform(-np.pi/2, np.pi/2)
      rand_angle2 = np.random.uniform(5*np.pi/6, np.pi/6)
      print('Random Angles: ' + str(rand_angle1) + ' ' + str(rand_angle2))
      rand_joint = self.start_up_joint.copy()
      rand_joint[-1] += rand_angle1
      rand_joint[-2] += rand_angle2
      self.move_to_joint(rand_joint, 3)
      rospy.sleep(8)
      force3 = self.force_torque

      self.move_to_joint(self.start_up_joint, 5)
      print('Second Estimation End')
      rospy.sleep(5)
      self.move_to_joint(self.start_joint, 2)
      rospy.sleep(2)
      self.reset_gripper()
      self.move_to_joint(self.start_up_joint, 2)
      rospy.sleep(2)
      ref_force = self.calib_force(angle1, angle2)
      force2 = force2 - ref_force
      ref_force = self.calib_force(rand_angle1, rand_angle2)
      force3 = force3 - ref_force
      mean2, std2 = self.estimate_CoM(force2, angle1=angle1, angle2=angle2, ret_std=True)
      rand_mean, rand_std = self.estimate_CoM(force3, angle1=rand_angle1, angle2=rand_angle2, ret_std=True)
      print('Second Estimation: ' + str(mean2))
      mean_active, _ = truedistributionfromobs(mean, mean2,std, std2)
      mean_rand, _ = truedistributionfromobs(mean, rand_mean, std, rand_std)
      print('Active: ' + str(mean_active) + ' Random: ' + str(mean_rand))
      self.move_to_joint(self.start_up_joint, 5)
      # self.mark_CoM(mean_actual)
      # return mean_actual

      two_pos_input = np.concatenate((force, [0,0]))
      two_pos_input = np.concatenate((two_pos_input, force2))
      two_pos_input = np.concatenate((two_pos_input, [angle1, angle2]))
      two_pos_mean = self.estimate_two_pos(two_pos_input)
      print('Two Pos Mean: ' + str(two_pos_mean))



  def box_test(self):
    num_tag = int(input('Number of Tags: '))
    rospy.sleep(1)
    self.move_away()
    main_tag, main_center, CoM, no_grasp_zone, moi = CoM_calulation(self.overhead_cam,num_tag)
    side = np.random.choice([0,1])
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
    print('GT: ' + str(GT*100))
    ref_force = self.calib_force(0.0, 0.0)
    
    up_joint = self.up_joint(joint)
    self.move_to_joint(up_joint, 5)
    rospy.sleep(5)
    self.pickup(joint=joint)
    self.move_to_joint(self.start_up_joint, 5)
    rospy.sleep(10)
    force = self.force_torque - ref_force
    print(force)
    print('First Estimation Start')
    mean, std = self.estimate_CoM(force, ret_std=True)
    print('First Estimation: ' + str(mean))
    angle1, angle2 = self.estimate_action(mean, std)
    print('New Angles: ' + str(angle1) + ' ' + str(angle2))
    new_joint = self.joint_state.copy()
    new_joint[-1] += angle1
    new_joint[-2] += angle2
    print('Second Estimation Start')
    self.move_to_joint(new_joint, 3)
    rospy.sleep(8)
    force2 = self.force_torque

    rand_angle1 = np.random.uniform(-np.pi/2, np.pi/2)
    rand_angle2 = np.random.uniform(5*np.pi/6, np.pi/6)
    print('Random Angles: ' + str(rand_angle1) + ' ' + str(rand_angle2))
    rand_joint = self.start_up_joint.copy()
    rand_joint[-1] += rand_angle1
    rand_joint[-2] += rand_angle2
    self.move_to_joint(rand_joint, 3)
    rospy.sleep(8)
    force3 = self.force_torque

    up_joint = self.up_joint(joint)
    self.move_to_joint(up_joint, 5)
    print('Second Estimation End')
    rospy.sleep(5)
    self.move_to_joint(joint, 2)
    rospy.sleep(2)
    self.reset_gripper()
    self.move_to_joint(up_joint, 2)
    rospy.sleep(2)
    ref_force = self.calib_force(angle1, angle2)
    force2 = force2 - ref_force
    ref_force = self.calib_force(rand_angle1, rand_angle2)
    force3 = force3 - ref_force
    mean2, std2 = self.estimate_CoM(force2, angle1=angle1, angle2=angle2, ret_std=True)
    rand_mean, rand_std = self.estimate_CoM(force3, angle1=rand_angle1, angle2=rand_angle2, ret_std=True)
    print('Second Estimation: ' + str(mean2))
    mean_active, _ = truedistributionfromobs(mean, mean2,std, std2)
    mean_rand, _ = truedistributionfromobs(mean, rand_mean, std, rand_std)
    print('Active: ' + str(mean_active) + ' Random: ' + str(mean_rand))
    self.move_to_joint(self.start_up_joint, 5)
    two_pos_input = np.concatenate((force, [0,0]))
    two_pos_input = np.concatenate((two_pos_input, force2))
    two_pos_input = np.concatenate((two_pos_input, [angle1, angle2]))
    two_pos_mean = self.estimate_two_pos(two_pos_input)
    print('Two Pos Mean: ' + str(two_pos_mean))
    # return GT, mean_active, mean_rand, two_pos_mean


  def generate_rand_grasp(self, main_tag, main_center, CoM, side):
    height = np.random.uniform(0.232, 0.272)
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
  
  def mark_CoM(self, CoM):
      
      while True:
          frame = self.overhead_cam.read()
          grasp_loc = self.ur5e_arm.forward(self.start_joint, 'matrix')
          grasp_loc = grasp_loc[:3,3]
          grasp_loc = grasp_loc + CoM
          grasp_loc = grasp_loc[:2]/100
          frame_loc = robot2img(grasp_loc)
          cv2.circle(frame, (int(frame_loc[0]), int(frame_loc[1])), 5, (0,0,255), 1)
          cv2.imshow('frame', frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      
  
    

if __name__ == '__main__':
  rospy.init_node('grasp')
  rospy.sleep(1)
  # np.random.seed(42)
  active_model_adr = '/media/okemo/extraHDD31/samueljin/Model/GSNet1_best_model.pth'
  bnn_adr = '/media/okemo/extraHDD31/samueljin/Model/bnn1_best_model.pth'
  Grasp_ = Grasp(active_model_adr, bnn_adr, BNN_pretrained)
  rospy.sleep(1)
  print('Initialized')
  # Grasp_.estimate_CoM([0,0,0,0,0,0], 0.0, 0.0, True)
  # Grasp_.test_main()
  Grasp_.box_test()
  # Grasp_.estimate_action([0.0,0.0,0.0], [0.0,0.0,0.0])
