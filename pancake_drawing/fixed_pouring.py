import os
import json
import argparse
import numpy as np
import rospy
import matplotlib.pyplot as plt
from ur5_lib import gripper
from wsg_50_common.msg import Status, Cmd
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from scipy.interpolate import interp1d, RegularGridInterpolator
from pancake_drawing.image_decomp import smoothen_and_to_list
"""
This module aims to collect the squeezing trajectory by squeeze at a fixed point.
It squeezes the bottle and collect the followings:
  1. Tactile data
  2. Gripper State (effort, width)
  3. Weight Measure Readings
  4. Wrist FT Readings
After collecting, it will reset and recollect/

Before collection, follow the following steps:
  1. Shake the bottle before squeezing.
  2. Tare the weight measure without anything on it.
  3. Measure the total weight of the liquid + bottle + measuring cup.
  4. Make sure the liquid is kept at the bottom of the bottle before squeezing.
"""
import copy
DH_theta = np.array([0,0,0,0,0,0])
DH_a = np.array([0,-0.425,-0.39225,0,0,0])
DH_d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
DH_alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
def inverse_kinematic (current_joint_angle, target_pos, rotate_R):
    """
    calculate the inverse kinematic of UR5E
    :param current_joint_angle: ndarray; the current 6 joint angles
    :param target_pos: ndarray; the target position of the end-effector
    :param rotate_R: ndarray; the target orientation of the end-effector

    nx,ox,ax;
    ny,oy,ay;
    nz,oz,az;

    """

    theta =  current_joint_angle.copy()

    nx = rotate_R[0,0]
    ny = rotate_R[1,0]
    nz = rotate_R[2,0]
    ox = rotate_R[0,1]
    oy = rotate_R[1,1]
    oz = rotate_R[2,1]
    ax = rotate_R[0,2]
    ay = rotate_R[1,2]
    az = rotate_R[2,2]

    m = DH_d[5]*rotate_R[1,2]-target_pos[1]
    n = DH_d[5]*rotate_R[0,2] - target_pos[0]
    # print("m,n",m,n,m**2+n**2-DH_d[3]**2)
    theta1_1 = np.arctan2(m,n) - np.arctan2(DH_d[3], np.sqrt(m**2+n**2-DH_d[3]**2))
    theta1_2 = np.arctan2(m,n) - np.arctan2(DH_d[3], -np.sqrt(m**2+n**2-DH_d[3]**2))
    if theta1_1<-np.pi:
        theta1_1 += 2*np.pi
    if theta1_2 < np.pi:
        theta1_2 += 2*np.pi
    # print(theta1_1,theta1_2)
    if abs(theta[0]-theta1_1)<abs(theta[0]-theta1_2):
        theta[0] = theta1_1
    else:
        theta[0] = theta1_2

    theta5_1 = np.arccos(rotate_R[0,2]*np.sin(theta[0])-rotate_R[1,2]*np.cos(theta[0]))
    theta5_2 = -theta5_1
    # print(theta5_1,theta5_2)
    if abs(theta[4]-theta5_1)<abs(theta[4]-theta5_2):
        theta[4] = theta5_1
    else:
        theta[4] = theta5_2

    mm = rotate_R[0,0]*np.sin(theta[0]) - rotate_R[1,0]*np.cos((theta[0]))
    nn = rotate_R[0,1]*np.sin(theta[0]) - rotate_R[1,1]*np.cos((theta[0]))
    theta[5] = np.arctan2(mm,nn)-np.arctan2(np.sin(theta[4]),0)


    mmm = DH_d[4]*(np.sin(theta[5])*(nx*np.cos(theta[0])+ny*np.sin(theta[0]))+
                   np.cos(theta[5])*(ox*np.cos(theta[0])+oy*np.sin(theta[0])))+\
          target_pos[0]*np.cos(theta[0])- DH_d[5]*(ax*np.cos(theta[0])+ ay*np.sin(theta[0]))+\
          target_pos[1]*np.sin(theta[0])
    nnn = DH_d[4]* (oz*np.cos(theta[5])+nz*np.sin(theta[5])) + target_pos[2] - DH_d[0] - az*DH_d[5]
    theta3_1 = np.arccos((mmm**2+nnn**2-DH_a[1]**2-DH_a[2]**2)/(2*DH_a[1]*DH_a[2]))
    theta3_2 = -theta3_1
    # print(theta3_1,theta3_2)

    if abs(theta[2]-theta3_1)<abs(theta[2]-theta3_2):
        theta[2] = theta3_1
    else:
        theta[2] = theta3_2


    s2 = ((DH_a[2]*np.cos(theta[2])+DH_a[1])*nnn - DH_a[2]*np.sin(theta[2])*mmm)/(DH_a[1]**2+DH_a[2]**2+2*DH_a[1]*DH_a[2]*np.cos(theta[2]))
    c2 = (mmm+DH_a[2]*np.sin(theta[2])*s2)/(DH_a[2]*np.cos(theta[2])+DH_a[1])
    theta[1] =np.arctan2(s2,c2)
    # print("theta2",theta[1])
    # theta[1] = np.arctan(((DH_a[2]*np.cos(theta[2])+DH_a[1])*nnn - DH_a[2]*np.sin(theta[2])*mmm)/(mmm*(DH_a[1]+DH_a[2]*np.cos(theta[2]))+nnn*DH_a[2]*np.sin(theta[2])))
    # print("theta2",theta[1])

    theta[3] = np.arctan2(-np.sin(theta[5])*(nx*np.cos(theta[0])+ny*np.sin(theta[0]))
                          -np.cos(theta[5])*(ox*np.cos(theta[0])+oy*np.sin(theta[0])),
                          oz*np.cos(theta[5])+nz*np.sin(theta[5])) - theta[1] - theta[2]

    return theta

class FixedSqueeze(object):
  def __init__(self, args):
    # Arguments
    self.parent_dir = args.parent_dir
    self.n_repeats = args.n_repeats
    self.total_weight = args.total_weight
    self.liquid_type = args.liquid_type
    self.shaking = args.shaking
    self.catalog_path = os.path.join(args.parent_dir, "catalog.csv")
    # Get the start index and the save keys
    with open(self.catalog_path, 'r') as f:
      lines = f.readlines()
      if len(lines) == 1:
        self.experiment_start_index = args.experiment_start_index
      else:
        self.experiment_start_index = int(os.path.basename(lines[-1].split()[0])) + 1
      self.catalog_keys = lines[0].split()
    # Key poses
    # self.ready_pose = np.array([86.53, -65.85, 52.86, -16.22, -36.77, 118.42])/180 * np.pi
    self.ready_pose = np.array([74.57, -69.41, 52.86, 0.35, -36.88, 118.42])/180 * np.pi
    self.pouring_pose =  np.array([77.06, -59.64, 55.86, -12.59, -33.79, 138.42])/180 * np.pi
    self.pouring_pose2 =  np.array([74.57, -69.41, 52.86, 0.35, -36.88, 178.42])/180 * np.pi
    # States
    self.is_wrist_ft_initialized = True
    self.is_weight_initialized = True
    self.initial_wrist_ft = None
    self.initial_weight = None
    self.stamped_wrist_fts = []
    self.stamped_weights = []
    self.stamped_gripper_states = []
    self.is_recording = False
    # Data
    self.stamped_haptic_data = []
    self.stamped_weight_data = []
    self.stamped_gripper_data = []
    self.stamped_force_data = []
    self.time_lapse = None

    # Subscribers
    rospy.Subscriber('/wsg_50_driver/status', Status, self.cb_gripper_state)
    rospy.Subscriber('/wrist_sensor/wrench', WrenchStamped, self.cb_wrist_sensor)
    rospy.Subscriber('/weight', Float32, self.cb_weight)
    # Publishers
    self.pos_controller = rospy.Publisher('/scaled_pos_joint_traj_controller/command',
                                          JointTrajectory, queue_size=20)
    self.finger_speed_pub = rospy.Publisher('/wsg_50_driver/goal_speed', Float32, queue_size=1)
    self.finger_traj_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=1)

  def cb_gripper_state(self, msg):
    """ Gripper State callback. """
    time = rospy.get_time()
    if self.is_recording:
      stamped_gripper_state = [time, msg.width, msg.force, msg.busy]
      self.stamped_gripper_states.append(stamped_gripper_state)

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
    elif self.is_recording:
      self.stamped_wrist_fts.append(stamped_wrist_ft)

  def cb_weight(self, msg):
    time = rospy.get_time()
    stamped_weight = (time, msg.data)
    if not self.is_weight_initialized:
      self.initial_weight = stamped_weight[1]
      self.is_weight_initialized = True
    elif self.is_recording:
      self.stamped_weights.append(stamped_weight)

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

  def flush_states(self):
    # States data
    self.is_wrist_ft_initialized = True
    self.is_weight_initialized = True
    self.initial_wrist_ft = None
    self.initial_weight = None
    self.stamped_wrist_fts = []
    self.stamped_weights = []
    self.stamped_gripper_states = []
    self.is_recording = False

  def flush_data(self):
    self.stamped_haptic_data = []
    self.stamped_weight_data = []
    self.stamped_gripper_data = []
    self.stamped_force_data = []
    self.time_lapse = None
    self.total_squeeze = None
    self.bottle_weight = None

  def save_json(self, save_dir):
    # Save gripper states
    
    # Save weight readings
    data_path = os.path.join(save_dir, 'weights.npz')
    np.savez(
        data_path, times = self.stamped_weight_data[:, 0],
        weights = self.stamped_weight_data[:, 1])
    # Save haptic readings
   


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

  
  

  def v_theta_measure(self):
    # Create directory
    experiment_dir = '/home/rocky/samueljin/pancake_bot/v_theta/' + str(rospy.get_time()) + '/'
    if not os.path.isdir(experiment_dir):
      os.makedirs(experiment_dir)
    # Homing the gripper
    self.is_weight_initialized = False
    rospy.sleep(1.0)
    self.angless = []
    self.weightss = []
    init_weight = self.initial_weight
    # Squeeze
    start_time = rospy.get_time()
    self.is_recording = True
    rospy.sleep(0.2)
    angle = 0
    pos = self.ready_pose + np.array([0,0,0,0,0,angle])
    print(self.stamped_weights)
    while self.stamped_weights[-1][1] <= init_weight+1 and angle < 1.57:
      print(self.stamped_weights[-1][1])
      angle += 0.05
      pos = self.ready_pose + np.array([0,0,0,0,0,angle])
      self.move_to_joint(pos, time = 1)
      rospy.sleep(2)

    rospy.sleep(10)
    self.weightss.append(self.stamped_weights[-1][1] - init_weight)
    self.angless.append(angle)
    while True:
      pos = self.ready_pose + np.array([0,0,0,0,0,angle])
      angle += 0.005
      self.move_to_joint(pos, time = 0.5)
      rospy.sleep(10)
      if angle > 1.0:
          break
      self.weightss.append(self.stamped_weights[-1][1] - init_weight)
      self.angless.append(angle)


    self.time_lapse = rospy.get_time() - start_time
   
    self.move_to_joint(self.ready_pose, time = 0.5)
    print(self.angless)
    print(self.weightss)
    data_path = os.path.join(experiment_dir, 'weights.npz')
    np.savez(
        data_path, angles = self.angless,
        weights = self.weightss)
  

  def run_once(self):
    # Create directory
    experiment_dir = '/home/rocky/samueljin/pancake_bot/batter_flowrate/with_spout/' + str(rospy.get_time())
    if not os.path.isdir(experiment_dir):
      os.makedirs(experiment_dir)
    rospy.sleep(1)
    init_pose = self.joit
    # Homing the gripper
    self.is_weight_initialized = False
    rospy.sleep(1.0)
    # Squeeze
    start_time = rospy.get_time()
    self.is_recording = True
    rate = rospy.Rate(100)
    while True:
      if rospy.get_time() - start_time > 100.:
          break
      rate.sleep()

    self.time_lapse = rospy.get_time() - start_time
  
    self.move_to_joint(self.ready_pose, time = 0.5)
    rospy.sleep(5)
    # Stop the flow and record for another 3 seconds
    self.is_recording = False
    # Change state to data
    self.stamped_weight_data = np.array(self.stamped_weights)
    self.stamped_weight_data[:, 0] -= start_time
    self.stamped_weight_data[:, 1] -= self.initial_weight
    self.total_squeezed = self.stamped_weight_data[-1, 1]
    # Record data
    self.save_json(experiment_dir)
    
    print("Save Experiment Data at %s, total squeezed: %.2f, total squeeze time: %.2f"
        %(experiment_dir, self.total_squeezed, self.time_lapse))

  
  def run(self):
    # Move to squeeze pose
    # rospy.sleep(1)
    # self.move_to_joint(self.low_squeeze_pose+self.non_drip_pos, time = 3)
    # rospy.sleep(3.0)
    rospy.sleep(2)
    self.move_to_joint(self.ready_pose, time = 10)
    rospy.sleep(10)
    #self.run_once()
    #self.v_theta_measure()
    
    return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description = "Collect Robot Grasping Data Without Moving the Arm")
  parser.add_argument('-p', '--parent_dir', type=str,
      default = "/home/rocky/samueljin/pancake_bot")
  parser.add_argument('-e', '--experiment_start_index', type=int, default = 0)
  parser.add_argument('-n', '--n_repeats', type=int, default = 1,
      help="ask to remove sauce from measuring cup.")
  parser.add_argument('-s', '--shaking', action='store_true')
  parser.add_argument('-w', '--total_weight', type=float,
      help='The total weight of bottle + liquid + measuring cup')
  parser.add_argument('-l', '--liquid_type', default='batter_wide_2', type=str,
      help='The type of liquid')
  args = parser.parse_args()
  # Start the squeezing process without moving the arm
  rospy.init_node("fixed_squeeze")
  fixed_squeeze = FixedSqueeze(args)
  fixed_squeeze.run()
