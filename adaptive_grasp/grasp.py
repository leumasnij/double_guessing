#Adaptive Grasping
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


class CameraCapture1:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.ret, self.frame = self.cap.read()
        self.is_running = True
        self.record = False
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()

    def update(self):
        while self.is_running:
            self.ret, self.frame = self.cap.read()
            while not self.ret:
               self.ret, self.frame = self.cap.read()
            if self.record:
                self.out.write(self.frame)

    def read(self):
        return self.ret, self.frame.copy()
    def start_record(self, out_adr):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(out_adr, fourcc, 10.0, self.frame.shape[1::-1])
        self.record = True

    def end_record(self):
        self.record = False
        rospy.sleep(1)
        if self.out.isOpened():
           self.out.release()

    def release(self):
        self.is_running = False
        rospy.sleep(1)
        self.cap.release()

class MarkerTracker:
    def __init__(self):
        self.cap = CameraCapture1()
        self.ret, self.frame = self.cap.read()
        self.tac_img_size = (960, 720)
        self.init_frame = gs.resize_crop_mini(self.frame, self.tac_img_size[0], self.tac_img_size[1])
        self.MarkerI = gs.find_markers(self.init_frame)
        self.old_markers = self.MarkerI
        self.is_running = True
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        rospy.sleep(1)
    def update(self):
        while self.is_running:
            self.ret, frame = self.cap.read()
            self.frame = gs.resize_crop_mini(frame, self.tac_img_size[0], self.tac_img_size[1])
            self.markers = gs.find_markers(self.frame)
            self.center_now, self.markerU, self.markerV = gs.update_markerMotion(self.markers, self.old_markers, self.MarkerI)
            self.old_markers = self.center_now
            frame2 = gs.displaycentres(self.frame, self.MarkerI, self.markerU, self.markerV)
            self.average_movement = np.mean(np.sqrt(self.markerU**2 + self.markerV**2))
            cv2.putText(frame, 'Average Movement: ' + str(self.average_movement), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            self.return_frame = frame2.copy()
            self.new = True
            # cv2.imshow('frame', frame)
    def get(self):
        while not self.new:
            continue
        self.new = False
        return self.return_frame, self.frame
    def move(self):
       return self.average_movement
    
    def start_record(self, out_adr):
        self.cap.start_record(out_adr)
    def end_record(self):
        self.cap.end_record()
    def release(self):
        self.is_running = False
        rospy.sleep(1)
        self.cap.release()


class OpFlowTracker:
    def __init__(self, record=False, out_adr=None):
        self.cap = CameraCapture1()
        self.record = record
        self.out_adr = out_adr
        if self.record and self.out_adr is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(self.out_adr, fourcc, 15.0, (640, 480))
        ret, f0 = self.cap.read()
        self.tac_img_size = (640, 480)

        
        self.init_frame = gs.resize_crop_mini(f0, self.tac_img_size[0], self.tac_img_size[1])
        f0gray = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2GRAY)
        self.MarkerI = gs.find_markers(self.init_frame)
        self.Ox = self.MarkerI[:, 0]
        self.Oy = self.MarkerI[:, 1]
        self.nct = len(self.MarkerI)

        self.old_gray = f0gray.copy()
        self.lk_params = dict(winSize=(50, 50), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = np.random.randint(0, 255, (100, 3))

        self.p0 = np.array([[self.Ox[0], self.Oy[0]]], np.float32).reshape(-1, 1, 2)
        for i in range(self.nct - 1):
            new_point = np.array([[self.Ox[i+1], self.Oy[i+1]]], np.float32).reshape(-1, 1, 2)
            self.p0 = np.append(self.p0, new_point, axis=0)
        self.is_running = True

        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
       
        self.new = False
    def update(self):
         
        while self.is_running:
            self.ret, frame = self.cap.read()
            frame = gs.resize_crop_mini(frame, self.tac_img_size[0], self.tac_img_size[1])
            self.frame = frame.copy()
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, cur_gray, self.p0, None, **self.lk_params)
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            self.p0 = good_new.reshape(-1, 1, 2)

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                ix  = int(self.Ox[i])
                iy  = int(self.Oy[i])
                offrame = cv2.arrowedLine(frame, (ix,iy), (int(a), int(b)), (255,255,255), thickness=1, line_type=cv2.LINE_8, tipLength=.15)
                offrame = cv2.circle(offrame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            if self.record and self.out_adr is not None:
                self.out.write(offrame)
            
            # print(offrame.shape)
            self.return_frame = offrame.copy()
            self.new = True
            self.old_gray = cur_gray.copy()
            # cv2.imshow('frame', frame)
    def get(self):
        while not self.new:
            continue
        self.new = False
        return self.return_frame, self.frame
    def set_init_frame(self):
        self.init_frame = self.frame
        self.MarkerI = gs.find_markers(self.init_frame)
        self.Ox = self.MarkerI[:, 0]
        self.Oy = self.MarkerI[:, 1]
        self.nct = len(self.MarkerI)
        self.old_gray = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2GRAY)
        self.p0 = np.array([[self.Ox[0], self.Oy[0]]], np.float32).reshape(-1, 1, 2)
        for i in range(self.nct - 1):
            new_point = np.array([[self.Ox[i+1], self.Oy[i+1]]], np.float32).reshape(-1, 1, 2)
            self.p0 = np.append(self.p0, new_point, axis=0)
    def start_record(self, out_adr):
        self.cap.start_record(out_adr)
    def end_record(self):
        self.cap.end_record()
    def release(self):
        self.is_running = False
        rospy.sleep(1)
        self.cap.release()
        if self.record:
            self.out.release()

class Grasp(object):
  def __init__(self, record=False):
    
    self.ur5e_arm = ur_kinematics.URKinematics('ur5e')
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
    rospy.Subscriber('/wrench', WrenchStamped, self.cb_force_torque)
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
    self.tracker = CameraCapture1()
    # self.tac_img_size = self.tracker.tac_img_size
    
  def __del__(self):
    self.tracker.release()
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
    if len(self.ftwindow) < 15:
        self.ftwindow.append(self.force_torque)
        return
    self.ftwindow.pop(0)
    self.ftwindow.append(self.force_torque)
    self.force_torque = np.mean(self.ftwindow, axis=0)
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

  def grasp_part(self, force):
    gripper.set_force(force)
    rospy.sleep(1)
    target_width = 90
    while self.gripper_force < 5 and target_width >=0:
      rospy.sleep(0.1)
      gripper.grasp(target_width, 60)
    #   rospy.sleep(0.1)
    #   print(self.gripper_force, self.gripper_width)
      target_width -= 20
    # gripper.set_force(60)
    rospy.sleep(1)
    gripper.grasp(self.gripper_width, 60)


  def random_rotate(self):
    # np.random.seed(0)
    new_state = self.generate_random_pos()
    ref_ang = 1.21485949
    rand_ang = np.random.uniform(-np.pi/2, np.pi/2)
    new_state[-1] = ref_ang + rand_ang
    self.move_to_joint(new_state, 10)
    rospy.sleep(15)

  def test_rot(self):
     roll = np.random.uniform(3*np.pi/4, 5*np.pi/4)
     pitch = np.random.uniform(0, np.pi/2)
     yaw = np.random.uniform(np.pi/2, np.pi)
     rot = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                    [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                    [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
     cur_angle, cur_pos = forward_kinematic(self.joint_state)
     new_state = inverse_kinematic_orientation(self.joint_state, cur_pos, rot)
     print(self.joint_state)
     print(new_state)

    #  self.move_to_joint(new_state, 10)
    #  rospy.sleep(10)
  def pickup(self, force = 20):
    """ Pickup the object. """
    # rospy.sleep(1)
    gripper.homing()
    # print(self.joint_state)
    # print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))

    self.move_to_joint(self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.joint_state), 2)
    rospy.sleep(7)

    self.grasp_part(force)
    # rospy.sleep(2)
    
    up_mat = self.start_mat.copy()
    up_mat[2,3] += 0.2
    self.move_to_joint(self.ur5e_arm.inverse(up_mat, False, q_guess=self.joint_state), 5)
    rospy.sleep(5)
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
  def showcamera(self):
    while True:
        frame, og_frame = self.tracker.get()
        cv2.imshow('frame', frame)
        cv2.imshow('og_frame', og_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

  def reset(self):
    rospy.sleep(1)
    # gripper.homing()
    # print(self.joint_state)
    # print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))
    self.move_to_joint(self.reset_joint, 5)
    rospy.sleep(5)
    self.move_to_joint(self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.joint_state), 5)
    rospy.sleep(5)
    gripper.homing()

  def save_np_data(self, force, pos_num):
    save_adr = self.saving_adr + 'raw_data/'+ str(pos_num) + '/'
    if not os.path.exists(save_adr):
        os.makedirs(save_adr)
    np.save(save_adr + 'force' + str(pos_num) + '.npy', np.array(force))
    # np.save(save_adr + 'movement' + str(pos_num) + '.npy', np.array(movement))

  def save_raw_data(self, img, num_pos):
        save_adr = self.saving_adr + 'raw_data/'+ str(num_pos) + '/'
        if not os.path.exists(save_adr):
            os.makedirs(save_adr)
        count = len([name for name in os.listdir(save_adr) if os.path.isfile(os.path.join(save_adr, name))] ) + 1
        cv2.imwrite(save_adr + 'img' + str(count) + '.png', img)

  def test(self):
    if not os.path.exists(self.saving_adr):
        os.makedirs(self.saving_adr)
    self.pickup()
    # frame, og_frame = self.tracker.get()
    # self.save_raw_data(og_frame, 0)
  
    # self.random_rotate()
    # count = 30
    pos_num = 1
    forces = []
    while pos_num<=10:
        forces.append(self.gripper_force)
        # frame, og_frame = self.tracker.get()
        # self.save_raw_data(og_frame, pos_num)
        # if count == 30:
            # self.save_np_data(forces, pos_num)
            # forces = []
            # movement = []
        self.random_rotate()
       
        self.test_motion(pos_num  )
       
        # count = 1
        pos_num += 1
        continue
        # cv2.imshow('frame', frame)
        # count += 1
        # print(count)
    self.reset()

  def test_motion(self, pos_num):
      CurPos = self.ur5e_arm.forward(self.joint_state)
      print(CurPos)
      NewPos = CurPos.copy()
      if(NewPos[2] > 0.5):
        NewPos[2] -= 0.1
      else:
        NewPos[2] += 0.1
      if(NewPos[1] > 0.5):
        NewPos[1] -= 0.1
      else:
        NewPos[1] += 0.1
      if(NewPos[0] > 0.5):
        NewPos[0] -= 0.1
      else:
        NewPos[0] += 0.1
      self.tracker.start_record(self.saving_adr + 'pos' + str(pos_num) + '.avi')
      new_state = self.ur5e_arm.inverse(NewPos, False, q_guess=self.joint_state)
      self.move_to_joint(new_state, 1)
      rospy.sleep(1.5)
      ori_state = self.ur5e_arm.inverse(CurPos, False, q_guess=self.joint_state)
      self.move_to_joint(ori_state, 1)
      rospy.sleep(1.5)
      self.tracker.end_record()
      print('done')

  def generate_random_pos(self):
    
    position = np.array([0.08, 0.6, 0.40])
    roll = np.pi
    # pitch = np.pi/4
    pitch = np.random.uniform(2e-3, np.pi)
    yaw = np.random.uniform(-np.pi/4, -np.pi/2)
    # yaw = -np.pi/2
    rot = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                    [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                    [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])

    new_mat = np.zeros((3,4))
    new_mat[:3,:3] = rot
    new_mat[:3,3] = position
    new_state = self.ur5e_arm.inverse(new_mat, False, q_guess=self.joint_state)
    if new_state is None:
      return self.generate_random_pos()
    # print(self.ur5e_arm.forward(new_state))
    
    ref_ang, _ = forward_kinematic(new_state)
    another_state = inverse_kinematic_orientation(self.joint_state, position, ref_ang)
    _, pos = forward_kinematic(another_state)
    pos[0] = pos[0] *-1
    pos[1] = pos[1] *-1


    another_mat = np.zeros((3,4))
    another_mat[:3,:3] = rot
    another_mat[:3,3] = pos
    another_state = self.ur5e_arm.inverse(another_mat, False, q_guess=self.joint_state)
    if another_state is None:
      return self.generate_random_pos()
    return another_state
    # print(ur5e_arm.forward(another_state))
    # print(self.joint_state)
    # print(new_state)
    # print(another_state)
    # self.move_to_joint(new_state, 15)
    # rospy.sleep(15)
    # # self.move_to_joint(another_state, 15)
    # # rospy.sleep(15)
    # self.move_to_joint(self.reset_joint, 15)
    # rospy.sleep(15)
  def impluse_motion(self, direction, time):
    """ Impluse motion. """
    cur_ang, cur_pos = forward_kinematic(self.joint_state)
    new_pos = cur_pos + direction
    new_state = inverse_kinematic(self.joint_state, new_pos[:3], cur_ang)
    self.move_to_joint(new_state, time)
    rospy.sleep(time)

  def rotate_to(self, roll, pitch, yaw, inhand):
    cur_pos = self.ur5e_arm.forward(self.joint_state, 'matrix')
    rot = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                    [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                    [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
    cur_pos[:3,:3] = rot
    new_state = self.ur5e_arm.inverse(cur_pos, False, q_guess=self.joint_state)
    ref_ang, ref_pos = forward_kinematic(new_state)
    another_state = inverse_kinematic_orientation(self.joint_state, ref_pos, ref_ang)
    _, pos = forward_kinematic(another_state)
    pos[0] = pos[0] *-1
    pos[1] = pos[1] *-1
    if pos[2] > 0.5:
      pos[2] = 0.5
    if pos[0] > 0.1:
      pos[0] = 0.1
    if pos[1] > 0.5:
      pos[1] = 0.5
    another_mat = np.zeros((3,4))
    another_mat[:3,:3] = rot
    another_mat[:3,3] = pos
    another_state = self.ur5e_arm.inverse(another_mat, False, q_guess=self.joint_state)
    another_state[-1] = self.joint_state[-1] + inhand + 0.3
    self.move_to_joint(another_state, 20)
    rospy.sleep(20)

  def test2(self):
    self.start_roll = np.pi
    self.start_pitch = 2e-3
    self.start_yaw = -np.pi/2
    
    
    # self.pickup(3.5)
    self.pickup(75)
    rospy.sleep(1)
    self.rotate_to(np.pi, np.pi/2, -np.pi/2, 0)
    rospy.sleep(5)
    if not os.path.exists(self.saving_adr):
        os.makedirs(self.saving_adr)
    self.tracker.start_record(self.saving_adr + 'pos' + str(0) + '.avi')
    self.impluse_motion([0.1, 0, 0], 0.15)
    rospy.sleep(5)
    # self.impluse_motion([0, 0, -0.1], 0.1)
    
    # rospy.sleep(5)
    self.tracker.end_record()
    self.reset()

  def pre_grasp(self, mat):
    gripper.homing()
    homing = self.joint_state.copy()
    self.move_to_joint(self.ur5e_arm.inverse(mat, False, q_guess=self.joint_state), 5)
    rospy.sleep(10)
    pre_grasp = self.force_torque
    self.move_to_joint(homing, 5)
    rospy.sleep(5)
    return pre_grasp
     

  def ft_test(self):
    gripper.homing()
    self.move_to_joint(self.reset_joint, 5)
    rospy.sleep(5)
    
    new_rot = self.ypr_to_mat(2e-3, 2e-3, np.pi)
    cur_pos = self.ur5e_arm.forward(self.reset_joint, 'matrix')
    cur_pos[:3,:3] = new_rot
    cur_pos[2,3]+= 0.1
    new_state = self.ur5e_arm.inverse(cur_pos, False, q_guess=self.reset_joint)

    pre_grasp = self.pre_grasp(cur_pos)
    self.pickup(75)
    self.move_to_joint(new_state, 5)
    rospy.sleep(10)
    post_grasp = self.force_torque
    # print(pre_grasp, post_grasp)
    print(post_grasp - pre_grasp)
    total_force = np.linalg.norm(post_grasp[:3] - pre_grasp[:3])
    print(total_force)
    
    rospy.sleep(5)
    self.reset()
    return post_grasp - pre_grasp

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
  

  def CoM_estimation(self):
    overall = []
    for i in range(10):
      x = self.ft_test()
      overall.append(x)

    stddv = np.std(overall, axis=0)
    print('std ' + str(stddv))
    avg_x = np.mean(overall, axis=0)
    print('average ' + str(avg_x))



if __name__ == '__main__':
  rospy.init_node('grasp')
  rospy.sleep(1)
  np.random.seed(42)
  Grasp_ = Grasp(record=False)
  Grasp_.CoM_estimation()
  # Grasp_.ft_test()
  
#   Grasp_.test2()
#   print(forward_kinematic(Grasp_.joint_state))
#   Grasp_.pickup()
#   cap.release()
#   print(1)
  # Grasp_.showcamera()
  # Grasp_.reset()
  
  # rospy.sleep(2)
#   Grasp_.test_rot()
  # Grasp_.test()
  # Grasp_.test2()
  