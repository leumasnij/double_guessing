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




# class OpFlowTracker:
#     def __init__(self, record=False, out_adr=None):
#         self.cap = CameraCapture1()
#         self.record = record
#         self.out_adr = out_adr
#         if self.record and self.out_adr is not None:
#             fourcc = cv2.VideoWriter_fourcc(*'XVID')
#             self.out = cv2.VideoWriter(self.out_adr, fourcc, 15.0, (640, 480))
#         ret, f0 = self.cap.read()
#         self.tac_img_size = (640, 480)

        
#         self.init_frame = gs.resize_crop_mini(f0, self.tac_img_size[0], self.tac_img_size[1])
#         f0gray = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2GRAY)
#         self.MarkerI = gs.find_markers(self.init_frame)
#         self.Ox = self.MarkerI[:, 0]
#         self.Oy = self.MarkerI[:, 1]
#         self.nct = len(self.MarkerI)

#         self.old_gray = f0gray.copy()
#         self.lk_params = dict(winSize=(50, 50), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#         self.color = np.random.randint(0, 255, (100, 3))

#         self.p0 = np.array([[self.Ox[0], self.Oy[0]]], np.float32).reshape(-1, 1, 2)
#         for i in range(self.nct - 1):
#             new_point = np.array([[self.Ox[i+1], self.Oy[i+1]]], np.float32).reshape(-1, 1, 2)
#             self.p0 = np.append(self.p0, new_point, axis=0)
#         self.is_running = True

#         thread = threading.Thread(target=self.update, args=())
#         thread.daemon = True
#         thread.start()
       
#         self.new = False
#     def update(self):
         
#         while self.is_running:
#             self.ret, frame = self.cap.read()
#             frame = gs.resize_crop_mini(frame, self.tac_img_size[0], self.tac_img_size[1])
#             self.frame = frame.copy()
#             cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, cur_gray, self.p0, None, **self.lk_params)
#             good_new = p1[st == 1]
#             good_old = self.p0[st == 1]
#             self.p0 = good_new.reshape(-1, 1, 2)

#             for i, (new, old) in enumerate(zip(good_new, good_old)):
#                 a, b = new.ravel()
#                 ix  = int(self.Ox[i])
#                 iy  = int(self.Oy[i])
#                 offrame = cv2.arrowedLine(frame, (ix,iy), (int(a), int(b)), (255,255,255), thickness=1, line_type=cv2.LINE_8, tipLength=.15)
#                 offrame = cv2.circle(offrame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
#             if self.record and self.out_adr is not None:
#                 self.out.write(offrame)
            
#             # print(offrame.shape)
#             self.return_frame = offrame.copy()
#             self.new = True
#             self.old_gray = cur_gray.copy()
#             # cv2.imshow('frame', frame)
#     def get(self):
#         while not self.new:
#             continue
#         self.new = False
#         return self.return_frame, self.frame
#     def set_init_frame(self):
#         self.init_frame = self.frame
#         self.MarkerI = gs.find_markers(self.init_frame)
#         self.Ox = self.MarkerI[:, 0]
#         self.Oy = self.MarkerI[:, 1]
#         self.nct = len(self.MarkerI)
#         self.old_gray = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2GRAY)
#         self.p0 = np.array([[self.Ox[0], self.Oy[0]]], np.float32).reshape(-1, 1, 2)
#         for i in range(self.nct - 1):
#             new_point = np.array([[self.Ox[i+1], self.Oy[i+1]]], np.float32).reshape(-1, 1, 2)
#             self.p0 = np.append(self.p0, new_point, axis=0)
#     def start_record(self, out_adr):
#         self.cap.start_record(out_adr)
#     def end_record(self):
#         self.cap.end_record()
#     def release(self):
#         self.is_running = False
#         rospy.sleep(1)
#         self.cap.release()
#         if self.record:
#             self.out.release()

class Grasp(object):
  def __init__(self, record=False):
    
    self.ur5e_arm = ur_kinematics.URKinematics('ur5e')
    self.model = nnh.Net()
    self.model.load_state_dict(torch.load('/home/okemo/samueljin/stablegrasp_ws/src/best_model.pth'))
    # torch.load('/home/okemo/samueljin/stablegrasp_ws/src/best_model.pth')
    self.model.eval()
    # self.reset_joint = [ 1.21490335, -1.32038331,  1.51271999, -1.76500773, -1.57009947,  1.21490407]
    self.reset_joint = [ 1.21490335, -1.283166,  1.6231562, -1.910088, -1.567829,  -0.359537]
    self.start_loc = np.array([0.08, 0.6, 0.245])
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
    
    # States
    self.joint_state = None
    self.gripper_width = None
    self.gripper_force = None
    self.force_calibrated = False
    # Data
    self.stamped_haptic_data = []
    self.stamped_weight_data = []
    self.stamped_gripper_data = []
    self.stamped_force_data = []
    self.ftwindow = []
    rospy.Subscriber('/joint_states', JointState, self.cb_joint_states)
    # rospy.Subscriber('/weight', Float32, self.cb_weight)
    rospy.Subscriber('/wsg_50_driver/status', Status, self.cb_gripper)
    rospy.Subscriber('/wrench', WrenchStamped, self.cb_force_torque, queue_size=1)
    # rospy.Subscriber('/wrist_sensor/wrench', WrenchStamped, self.cb_force_torque, queue_size=1)
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
    self.marker_gelsight = None
    self.unmarker_gelsight = None
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
            continue
          if name == 'GelSight Mini R0B 2G4W-G32G: Ge\n':
            self.marker_gelsight = CameraCapture(i)
            continue
          if name == 'GelSight Mini R0B 28UV-U5R7: Ge\n':
            self.unmarker_gelsight = CameraCapture(i)
            continue
      cap.release()



  def inti_model(self):
      self.gel_model = nnh.GelResNet()
      self.gel_model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/gel_best_model.pth'))
      self.gel_model.eval()
      self.hap_model = nnh.RegNet(input_size=6)
      self.hap_model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/hap_best_model.pth'))
      self.hap_model.eval()
      self.ref_model = nnh.GelRefResNet()
      self.ref_model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/ref_best_model.pth'))
      self.ref_model.eval()
      self.diff_model = nnh.GelResNet()
      self.diff_model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/dif_best_model.pth'))
      self.diff_model.eval()
      self.gelhap_model = nnh.GelHapResNet()
      self.gelhap_model.load_state_dict(torch.load('/media/okemo/extraHDD31/samueljin/Model/gelhap_best_model.pth'))
      self.gelhap_model.eval()



  def showcamera(self):
    while True:
        rs_frame = self.realsense.get_frame()
        cv2.imshow('rs_frame', rs_frame)
        if self.overhead_cam is not None:
          overhead_frame = self.overhead_cam.read()
          cv2.imshow('overhead_frame', overhead_frame)
        if self.marker_gelsight is not None:
          marker_frame = self.marker_gelsight.read()
          cv2.imshow('marker_frame', marker_frame)
        if self.unmarker_gelsight is not None:
          unmarker_frame = self.unmarker_gelsight.read()
          cv2.imshow('unmarker_frame', unmarker_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
  def __del__(self):
    # self.tracker.release()
    rospy.sleep(1)

  def get_joint_angle(self, s):
    x = max(min(self.s2x(s), self.xmax), self.xmin)
    y = max(min(self.s2y(s), self.ymax), self.ymin)
    return self.interp([np.array([x, y])])[0]


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
       
    self.force_torque = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                  msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
    # print(self.force_torque)
    if len(self.ftwindow) < 15:
        self.ftwindow.append(self.force_torque)
        return
    
    self.ftwindow.pop(0)
    self.ftwindow.append(self.force_torque)
    self.force_torque = np.mean(self.ftwindow, axis=0)
    # print(self.ftwindow)
    # if self.force_calibrated == False:
    #     self.cali_force(self.force_torque)
    # else:
    #     self.force_torque -= self.force_offset

    
  def cali_force(self, force):
     if not self.force_calibrated:
        self.force_calibrated = True
        self.force_offset = force
    

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
    rospy.sleep(1)
    target_width = 40
    while self.gripper_force < 5 and target_width >=0:
      rospy.sleep(0.1)
      gripper.grasp(target_width, 60)
    #   rospy.sleep(0.1)
    #   print(self.gripper_force, self.gripper_width)
      target_width -= 20
    # gripper.set_force(60)
    rospy.sleep(1)
    gripper.grasp(self.gripper_width, 60)


  # def random_rotate(self):
  #   # np.random.seed(0)
  #   new_state = self.generate_random_pos()
  #   ref_ang = 1.21485949
  #   rand_ang = np.random.uniform(-np.pi/2, np.pi/2)
  #   new_state[-1] = ref_ang + rand_ang
  #   self.move_to_joint(new_state, 10)
  #   rospy.sleep(15)

  # def test_rot(self):
  #    roll = np.random.uniform(3*np.pi/4, 5*np.pi/4)
  #    pitch = np.random.uniform(0, np.pi/2)
  #    yaw = np.random.uniform(np.pi/2, np.pi)
  #    rot = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
  #                   [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
  #                   [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
  #    cur_angle, cur_pos = forward_kinematic(self.joint_state)
  #    new_state = inverse_kinematic_orientation(self.joint_state, cur_pos, rot)
  #    print(self.joint_state)
  #    print(new_state)

    #  self.move_to_joint(new_state, 10)
    #  rospy.sleep(10)
  def pickup(self, force = 80, joint = None):
    """ Pickup the object. """
    # rospy.sleep(1)
    if joint is None:
       joint = self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.joint_state)
    self.reset_gripper()
    # print(self.joint_state)
    # print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))

    self.move_to_joint(joint, 2)
    rospy.sleep(7)

    self.grasp_part(force)
    ref_gel = self.marker_gelsight.read()
    up_mat = self.up_joint(joint=joint)
    self.move_to_joint(up_mat, 5)
    rospy.sleep(5)
    return ref_gel
    # return return_ft  
    # cap.release()
    # cv2.destroyAllWindows()
    # self.reset()
    # rospy.sleep(10)
    # while True:
    #     print(self.gripper_force, self.gripper_width)
    #     ret, frame = cap.read()
    #     frame = gs.resize_crop_mini(frame, self.tac_img_size[0], self.tac_img_size[1])
    #     markers = gs.find_markers(frame)
    #     center_now, markerU, markerV = gs.update_markerMotion(markers, old_markers, markersI)
    #     old_markers = center_now
    #     frame = gs.displaycentres(frame, center_now, markerU, markerV)
    #     average_movement = np.mean(np.sqrt(markerU**2 + markerV**2))
    #     cv2.putText(frame, 'Average Movement: ' + str(average_movement), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    #     cv2.putText(frame, 'gripper width: ' + str(self.gripper_width), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    #     out.write(frame)
    #     if count%5 == 0:
    #         gripper.move(self.gripper_width+0.1)
    #     count += 1
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     if self.gripper_width > 57.5:
    #         break
    #     cv2.imshow('frame', frame)
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
    # print('done')
    #   rospy.sleep(2)
  # def showcamera(self):
  #   while True:
  #       frame, og_frame = self.tracker.get()
  #       cv2.imshow('frame', frame)
  #       cv2.imshow('og_frame', og_frame)
  #       if cv2.waitKey(1) & 0xFF == ord('q'):
  #           break

  def reset(self):
    rospy.sleep(1)
    # gripper.homing()
    # print(self.joint_state)
    # print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))
    self.move_to_joint(self.reset_joint, 5)
    rospy.sleep(5)
    self.move_to_joint(self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.joint_state), 5)
    rospy.sleep(5)
    # gripper.homing()
    self.reset_gripper()

  # def save_np_data(self, force, pos_num):
  #   save_adr = self.saving_adr + 'raw_data/'+ str(pos_num) + '/'
  #   if not os.path.exists(save_adr):
  #       os.makedirs(save_adr)
  #   np.save(save_adr + 'force' + str(pos_num) + '.npy', np.array(force))
  #   # np.save(save_adr + 'movement' + str(pos_num) + '.npy', np.array(movement))

  # def save_raw_data(self, img, num_pos):
  #       save_adr = self.saving_adr + 'raw_data/'+ str(num_pos) + '/'
  #       if not os.path.exists(save_adr):
  #           os.makedirs(save_adr)
  #       count = len([name for name in os.listdir(save_adr) if os.path.isfile(os.path.join(save_adr, name))] ) + 1
  #       cv2.imwrite(save_adr + 'img' + str(count) + '.png', img)

  # def test(self):
  #   if not os.path.exists(self.saving_adr):
  #       os.makedirs(self.saving_adr)
  #   self.pickup()
  #   # frame, og_frame = self.tracker.get()
  #   # self.save_raw_data(og_frame, 0)
  
  #   # self.random_rotate()
  #   # count = 30
  #   pos_num = 1
  #   forces = []
  #   while pos_num<=10:
  #       forces.append(self.gripper_force)
  #       # frame, og_frame = self.tracker.get()
  #       # self.save_raw_data(og_frame, pos_num)
  #       # if count == 30:
  #           # self.save_np_data(forces, pos_num)
  #           # forces = []
  #           # movement = []
  #       self.random_rotate()
       
  #       self.test_motion(pos_num  )
       
  #       # count = 1
  #       pos_num += 1
  #       continue
  #       # cv2.imshow('frame', frame)
  #       # count += 1
  #       # print(count)
  #   self.reset()

  # def test_motion(self, pos_num):
  #     CurPos = self.ur5e_arm.forward(self.joint_state)
  #     print(CurPos)
  #     NewPos = CurPos.copy()
  #     if(NewPos[2] > 0.5):
  #       NewPos[2] -= 0.1
  #     else:
  #       NewPos[2] += 0.1
  #     if(NewPos[1] > 0.5):
  #       NewPos[1] -= 0.1
  #     else:
  #       NewPos[1] += 0.1
  #     if(NewPos[0] > 0.5):
  #       NewPos[0] -= 0.1
  #     else:
  #       NewPos[0] += 0.1
  #     # self.tracker.start_record(self.saving_adr + 'pos' + str(pos_num) + '.avi')
  #     new_state = self.ur5e_arm.inverse(NewPos, False, q_guess=self.joint_state)
  #     self.move_to_joint(new_state, 1)
  #     rospy.sleep(1.5)
  #     ori_state = self.ur5e_arm.inverse(CurPos, False, q_guess=self.joint_state)
  #     self.move_to_joint(ori_state, 1)
  #     rospy.sleep(1.5)
  #     # self.tracker.end_record()
  #     print('done')

  # def generate_random_pos(self):
    
  #   position = np.array([0.08, 0.6, 0.40])
  #   roll = np.pi
  #   # pitch = np.pi/4
  #   pitch = np.random.uniform(2e-3, np.pi)
  #   yaw = np.random.uniform(-np.pi/4, -np.pi/2)
  #   # yaw = -np.pi/2
  #   rot = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
  #                   [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
  #                   [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])

  #   new_mat = np.zeros((3,4))
  #   new_mat[:3,:3] = rot
  #   new_mat[:3,3] = position
  #   new_state = self.ur5e_arm.inverse(new_mat, False, q_guess=self.joint_state)
  #   if new_state is None:
  #     return self.generate_random_pos()
  #   # print(self.ur5e_arm.forward(new_state))
    
  #   ref_ang, _ = forward_kinematic(new_state)
  #   another_state = inverse_kinematic_orientation(self.joint_state, position, ref_ang)
  #   _, pos = forward_kinematic(another_state)
  #   pos[0] = pos[0] *-1
  #   pos[1] = pos[1] *-1


  #   another_mat = np.zeros((3,4))
  #   another_mat[:3,:3] = rot
  #   another_mat[:3,3] = pos
  #   another_state = self.ur5e_arm.inverse(another_mat, False, q_guess=self.joint_state)
  #   if another_state is None:
  #     return self.generate_random_pos()
  #   return another_state
  #   # print(ur5e_arm.forward(another_state))
  #   # print(self.joint_state)
  #   # print(new_state)
  #   # print(another_state)
  #   # self.move_to_joint(new_state, 15)
  #   # rospy.sleep(15)
  #   # # self.move_to_joint(another_state, 15)
  #   # # rospy.sleep(15)
  #   # self.move_to_joint(self.reset_joint, 15)
  #   # rospy.sleep(15)
  # def impluse_motion(self, direction, time):
  #   """ Impluse motion. """
  #   cur_ang, cur_pos = forward_kinematic(self.joint_state)
  #   new_pos = cur_pos + direction
  #   new_state = inverse_kinematic(self.joint_state, new_pos[:3], cur_ang)
  #   self.move_to_joint(new_state, time)
  #   rospy.sleep(time)

  # def rotate_to(self, roll, pitch, yaw, inhand):
  #   cur_pos = self.ur5e_arm.forward(self.joint_state, 'matrix')
  #   rot = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
  #                   [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
  #                   [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
  #   cur_pos[:3,:3] = rot
  #   new_state = self.ur5e_arm.inverse(cur_pos, False, q_guess=self.joint_state)
  #   ref_ang, ref_pos = forward_kinematic(new_state)
  #   another_state = inverse_kinematic_orientation(self.joint_state, ref_pos, ref_ang)
  #   _, pos = forward_kinematic(another_state)
  #   pos[0] = pos[0] *-1
  #   pos[1] = pos[1] *-1
  #   if pos[2] > 0.5:
  #     pos[2] = 0.5
  #   if pos[0] > 0.1:
  #     pos[0] = 0.1
  #   if pos[1] > 0.5:
  #     pos[1] = 0.5
  #   another_mat = np.zeros((3,4))
  #   another_mat[:3,:3] = rot
  #   another_mat[:3,3] = pos
  #   another_state = self.ur5e_arm.inverse(another_mat, False, q_guess=self.joint_state)
  #   another_state[-1] = self.joint_state[-1] + inhand + 0.3
  #   self.move_to_joint(another_state, 20)
  #   rospy.sleep(20)

  # def test2(self):
  #   self.start_roll = np.pi
  #   self.start_pitch = 2e-3
  #   self.start_yaw = -np.pi/2
    
    
  #   # self.pickup(3.5)
  #   self.pickup(75)
  #   rospy.sleep(1)
  #   self.rotate_to(np.pi, np.pi/2, -np.pi/2, 0)
  #   rospy.sleep(5)
  #   if not os.path.exists(self.saving_adr):
  #       os.makedirs(self.saving_adr)
  #   self.tracker.start_record(self.saving_adr + 'pos' + str(0) + '.avi')
  #   self.impluse_motion([0.1, 0, 0], 0.15)
  #   rospy.sleep(5)
  #   # self.impluse_motion([0, 0, -0.1], 0.1)
    
  #   # rospy.sleep(5)
  #   self.tracker.end_record()
  #   self.reset()

  def pre_grasp(self, mat):
    # gripper.homing()
    self.reset_gripper()
    homing = self.joint_state.copy()
    self.move_to_joint(self.ur5e_arm.inverse(mat, False, q_guess=self.joint_state), 5)
    rospy.sleep(10)
    pre_grasp = self.force_torque
    print(pre_grasp)
    self.move_to_joint(homing, 5)
    rospy.sleep(5)
    return pre_grasp
  
  def up_joint(self, joint):
      mat = self.ur5e_arm.forward(joint, 'matrix')
      mat[2,3] = 0.4
      return self.ur5e_arm.inverse(mat, False, q_guess=joint)

  def ft_test(self, force = 75, joint = None):
    # gripper.homing()
    self.reset_gripper()
    if joint is None:
       joint = self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.joint_state)
    up_joint = self.up_joint(joint)
    self.move_to_joint(up_joint, 5)
    rospy.sleep(5)
    
    # new_rot = self.ypr_to_mat(2e-3, 2e-3, np.pi)
    cur_pos = self.ur5e_arm.forward(up_joint, 'matrix')
    # cur_pos[:3,:3] = new_rot
    cur_pos[2,3]= 0.45
    new_state = self.ur5e_arm.inverse(cur_pos, False, q_guess=up_joint)

    pre_grasp = self.pre_grasp(cur_pos)
    ref_gel = self.pickup(75, joint = joint)[1]
    self.move_to_joint(new_state, 5)
    rospy.sleep(10)
    post_grasp = self.force_torque
    gel_marker = self.marker_gelsight.read()[1]
    print(post_grasp)
    # print(pre_grasp, post_grasp)
    # print(post_grasp - pre_grasp)
    total_force = np.linalg.norm(post_grasp[:3] - pre_grasp[:3])
    # print(total_force)
    
    rospy.sleep(5)
    self.move_to_joint(joint, 5)
    rospy.sleep(5)
    # gripper.homing()
    self.reset_gripper()
    return post_grasp - pre_grasp, gel_marker, ref_gel

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

  def save_data(self, force, loc, marker, marker_ref, testid = 0):
    save_dict = {'force': force, 'loc': loc}
    saving_adr = self.saving_adr + 'raw_data/'+ str(testid) + '/'
    if not os.path.exists(saving_adr):
        os.makedirs(saving_adr)
    np.save(saving_adr + 'data.npy', save_dict)
    np.savetxt(saving_adr + 'loc.txt', loc)
    cv2.imwrite(saving_adr + 'marker.png', marker)
    cv2.imwrite(saving_adr + 'marker_ref.png', marker_ref)

  def CoM_estimation(self, num_tag, testid = 0, save = True):
    rospy.sleep(1)
    self.move_away()
    main_tag, main_center, CoM = CoM_calulation(self.overhead_cam, num_tag)
    # height = np.random.uniform(0.232, 0.282)
    height = CoM[2] + 0.05
    if main_tag == 0:
      offset = np.random.uniform(-0.07, 0.07)
      
      move_loc = np.array([main_center[0] + offset, main_center[1], height])
      rot = self.ypr_to_mat(2e-3, 2e-3, np.pi)
    elif main_tag == 1:
      side = np.random.randint(0, 2)
      if side == 0:
        offset = np.random.uniform(-0.055, 0.045)
        move_loc = np.array([main_center[0] + offset, main_center[1] - 0.075, height])
        rot = self.ypr_to_mat(2e-3, 2e-3, np.pi)
      elif side == 1:
        offset = np.random.uniform(-0.055, 0.045)
        move_loc = np.array([main_center[0] + offset, main_center[1] + 0.075, height])
        rot = self.ypr_to_mat(2e-3, 2e-3, np.pi)
      elif side == 2:
        offset = np.random.uniform(-0.05, 0.05)
        move_loc = np.array([main_center[0] - 0.075, main_center[1] + offset, height])
        rot = self.ypr_to_mat(-np.pi/2, 2e-3, np.pi)
        CoM = CoM[:2][::-1]
      elif side == 3:
        offset = np.random.uniform(-0.05, 0.05)
        move_loc = np.array([main_center[0] +0.075, main_center[1] + offset, height])
        rot = self.ypr_to_mat(-np.pi/2, 2e-3, np.pi)
        CoM = CoM[:2][::-1]
      print('main_tag: ' + str(main_tag) + ' side: ' + str(side))

    GT = move_loc - CoM
    print(GT, CoM, main_tag, main_center)
    robot_mat = self.loc2mat(move_loc, rot)
    new_joint = self.ur5e_arm.inverse(robot_mat, False, q_guess=self.joint_state)
    x, gel_marker, ref_gel = self.ft_test(joint = new_joint)
    if save:
      self.save_data(x, GT, gel_marker,ref_gel, testid)
    return GT, x, gel_marker, ref_gel
    # img_tag, tagID = detect_tag()
    # robot_loc = img2robot(img_tag)
    # if tagID == 6:
    #   robot_loc[0] -= 0.075
    #   robot_loc[1] -= 0.075
    #   robot_loc[2] = 0.234 
    # GT = robot_loc.copy()
    # robot_mat = self.loc2mat(GT)
    # new_joint = self.ur5e_arm.inverse(robot_mat, False, q_guess=self.joint_state)
    # x = self.ft_test(joint = new_joint)
    # print(GT)
    # random_offset = np.random.uniform(-0.1, 0.1)
    # robot_loc[0] += random_offset
    # print(robot_loc)
    # robot_mat = self.loc2mat(robot_loc)
    # new_joint = self.ur5e_arm.inverse(robot_mat, False, q_guess=self.joint_state)
    # x = self.ft_test(joint = new_joint)
    # GT = robot_loc - GT
    # save_dict = {'GT': GT, 'force': x}
    # print(GT, x)
    # if not os.path.exists(self.saving_adr):
    #     os.makedirs(self.saving_adr)
    # np.save(self.saving_adr + 'data' + str(testid) + '.npy', save_dict)


  def CoM_verify(self, testid = 0):
    rospy.sleep(1)
    self.move_away()
    img_tag, tagID = detect_tag()
    robot_loc = img2robot(img_tag)
    if tagID == 1:
      robot_loc[2] = 0.245
    elif tagID == 2:
      robot_loc[2] = 0.23
    elif tagID == 3:
      robot_loc[2] = 0.23
    elif tagID == 7:
      robot_loc[2] = 0.23
    elif tagID == 0:
      robot_loc[2] = 0.25
    GT = robot_loc.copy()
    # print(GT)
    if tagID == 7:
      rand_idx = np.random.randint(0, 2)
      if rand_idx == 0:
        random_offset = np.random.uniform(0.03, 0.06)
      elif rand_idx == 1:
        random_offset = np.random.uniform(-0.03, -0.1)
    else:
      random_offset = np.random.uniform(-0.1, 0.1)
    robot_loc[0] += random_offset
    # print(robot_loc)
    robot_mat = self.loc2mat(robot_loc)
    new_joint = self.ur5e_arm.inverse(robot_mat, False, q_guess=self.joint_state)
    x = self.ft_test(joint = new_joint)
    GT = robot_loc - GT
    save_dict = {'GT': GT, 'force': x}
    print(save_dict)
    pred = self.model(torch.tensor(x).float())/1000
    pred = pred.detach().numpy()
    nn_diff = pred[0] - GT[0]
    analyical_sol = x[3] / x[2]
    analyical_diff = analyical_sol - GT[0]
    print('NN_diff: ' + str(nn_diff) + ' Analytical_diff: ' + str(analyical_diff))
    return abs(nn_diff), abs(analyical_diff)
    
  def online_test(self):
    self.inti_model()
    num_tag = int(input("How many tags are there?"))
    gel_offset = np.array([0.0, 0.0])
    haptic_offset = np.array([0.0, 0.0])
    ref_offset = np.array([0.0, 0.0])
    gelhap_offset = np.array([0.0, 0.0])
    for i in range(10):
      GT, hap, gel_marker, ref_gel = self.CoM_estimation(num_tag, i, save=False)
      GT = GT[:2]*100
      gel_marker = cv2.resize(gel_marker, (640,480))/255.0
      ref_gel = cv2.resize(ref_gel, (640,480))/255.0
      ref_gel = np.concatenate((gel_marker, ref_gel), axis=1)
      gel_marker = torch.tensor(gel_marker).float()
      gel_marker = gel_marker.permute(2,0,1)
      ref_gel = torch.tensor(ref_gel).float()
      ref_gel = ref_gel.permute(2,0,1)
      hap = torch.tensor(hap).unsqueeze(0).float()
      gel_offset += self.gel_model(gel_marker).detach().numpy()[0]
      haptic_offset += self.hap_model(hap).detach().numpy()[0]
      ref_offset += self.ref_model(ref_gel).detach().numpy()[0]
      gelhap_offset += self.gelhap_model(ref_gel, hap).detach().numpy()[0]
      print('Gel: ' + str(gel_offset/i+1) + ' Hap: ' + str(haptic_offset/i+1) + ' Ref: ' + str(ref_offset/i+1) + ' Gelhap: ' + str(gelhap_offset/i+1))





  def data_collection_main(self):
    num_tag = int(input("How many tags are there?"))
    for i in range(20):
      self.CoM_estimation(num_tag, i)
    
    # overall = []
    # for i in range(10):
    #   x = self.ft_test()
    #   overall.append(x)

    # stddv = np.std(overall, axis=0)
    # print('std ' + str(stddv))
    # avg_x = np.mean(overall, axis=0)
    # print('average ' + str(avg_x))



if __name__ == '__main__':
  rospy.init_node('grasp')
  rospy.sleep(1)
  rospy.Rate(15)
  # np.random.seed(42)
  Grasp_ = Grasp(record=False)
  # Grasp_.showcamera()
  # Grasp_.data_collection_main()
  Grasp_.online_test()


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