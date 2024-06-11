import cv2
import os
import apriltag
import numpy as np
from ur_ikfast import ur_kinematics
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


class Grasp(object):
  def __init__(self, record=False):
    
    self.ur5e_arm = ur_kinematics.URKinematics('ur5e')
    self.reset_joint = [ 1.21490335, -1.283166,  1.6231562, -1.910088, -1.567829,  -0.359537]
    self.start_loc = np.array([0.08, 0.6, 0.22])
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
    

    self.away_loc = np.array([0.5, 0.27, 0.34])
    self.away_yaw = 2e-3
    self.away_pitch = 2e-3
    self.away_roll = np.pi
    # self.q_array = euler_to_quaternion(self.start_roll, self.start_pitch, self.start_yaw)
    self.away_rot = np.array([[np.cos(self.away_yaw)*np.cos(self.away_pitch), np.cos(self.away_yaw)*np.sin(self.away_pitch)*np.sin(self.away_roll)-np.sin(self.away_yaw)*np.cos(self.away_roll), np.cos(self.away_yaw)*np.sin(self.away_pitch)*np.cos(self.away_roll)+np.sin(self.away_yaw)*np.sin(self.away_roll)],
                                [np.sin(self.away_yaw)*np.cos(self.away_pitch), np.sin(self.away_yaw)*np.sin(self.away_pitch)*np.sin(self.away_roll)+np.cos(self.away_yaw)*np.cos(self.away_roll), np.sin(self.away_yaw)*np.sin(self.away_pitch)*np.cos(self.away_roll)-np.cos(self.away_yaw)*np.sin(self.away_roll)],
                                [-np.sin(self.away_pitch), np.cos(self.away_pitch)*np.sin(self.away_roll), np.cos(self.away_pitch)*np.cos(self.away_roll)]])
    self.away_mat = np.zeros((3,4))
    self.away_mat[:3,:3] = self.away_rot
    self.away_mat[:3,3] = self.away_loc
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
    # self.tac_img_size = self.tracker.tac_img_size

  def __del__(self):
    rospy.sleep(1)

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

  def inverse_kinematic(self, mat):
    ik = self.ur5e_arm.inverse(mat, q_guess=self.joint_state)
    return ik
      
  def reset(self):
    rospy.sleep(1)
    # gripper.homing()
    # print(self.joint_state)
    # print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))
    self.move_to_joint(self.reset_joint, 10)
    rospy.sleep(11)
    self.move_to_joint(self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.joint_state), 3)
    rospy.sleep(5)
    gripper.homing()
    rospy.sleep(1)
    self.move_to_joint(self.reset_joint, 3)
    rospy.sleep(3)
    gripper.homing()
    self.move_away()

  def print_forwards(self):
    print(self.ur5e_arm.forward(self.joint_state))

  def move_to_start(self):
    self.move_to_joint(self.ur5e_arm.inverse(self.start_mat, False, q_guess=self.joint_state), 5)
    rospy.sleep(5)

  
  def move_away(self):

    self.move_to_joint(self.reset_joint, 5)
    rospy.sleep(5.5)

    away_joint = self.reset_joint
    away_joint[0] = 0.5
    print(away_joint)
    print(self.reset_joint)
    print('moving away')

    self.move_to_joint(away_joint, 5)
    rospy.sleep(5.5)
    # gripper.homing()

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

  def randomly_place(self, cur_mat = None):
    if cur_mat is None:
      print(self.start_mat)
      cur_mat = self.start_mat
    cur_mat[2,3] = 0.34
    self.move_to_joint(self.ur5e_arm.inverse(cur_mat, False, q_guess=self.joint_state), 10)
    rospy.sleep(11)
    cur_mat[2,3] = 0.22
    self.move_to_joint(self.ur5e_arm.inverse(cur_mat, False, q_guess=self.joint_state), 3)
    rospy.sleep(3)
    self.grasp_part(60)
    randx_offset = np.random.uniform(-0.25, 0.25)
    randy_offset = np.random.uniform(-0.25, 0.25)
    rand_loc = np.array([0.08 + randx_offset, 0.6 + randy_offset, 0.34])
    print(rand_loc)
    rand_mat = np.zeros((3,4))
    rand_mat[:3,:3] = self.start_rot
    rand_mat[:3,3] = rand_loc
    self.move_to_joint(self.ur5e_arm.inverse(rand_mat, False, q_guess=self.joint_state), 5)
    rospy.sleep(5)
    rand_mat[2,3] = 0.22
    self.move_to_joint(self.ur5e_arm.inverse(rand_mat, False, q_guess=self.joint_state), 3)
    rospy.sleep(3)
    gripper.homing()
    self.move_away()
  
  
  def loc2mat(self, loc, rot = None):
    if rot is None:
        rot = self.start_rot
    mat = np.zeros((3,4))
    mat[:3,:3] = rot
    mat[:3,3] = loc
    return mat
    




if __name__ == '__main__':
  rospy.init_node('grasp')
  grasp = Grasp()
  rospy.sleep(1)
  grasp.move_away()
  # prev_loc = [0.03764878, 0.71243892, 0.34]
  # prev_mat = grasp.loc2mat(prev_loc)
  # grasp.randomly_place()
  # grasp.print_forwards()
  # grasp.move_to_start()