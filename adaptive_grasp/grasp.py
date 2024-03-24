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
from kinematic import forward_kinematic, inverse_kinematic
import gelsight_test as gs

class Grasp(object):
  def __init__(self):

    self.start_loc = np.array([-0.08, -0.6, 0.25])
    self.start_yaw = np.pi/2
    self.start_pitch = 0
    self.start_roll = np.pi
    self.start_rot = np.array([[np.cos(self.start_yaw)*np.cos(self.start_pitch), np.cos(self.start_yaw)*np.sin(self.start_pitch)*np.sin(self.start_roll)-np.sin(self.start_yaw)*np.cos(self.start_roll), np.cos(self.start_yaw)*np.sin(self.start_pitch)*np.cos(self.start_roll)+np.sin(self.start_yaw)*np.sin(self.start_roll)],
                               [np.sin(self.start_yaw)*np.cos(self.start_pitch), np.sin(self.start_yaw)*np.sin(self.start_pitch)*np.sin(self.start_roll)+np.cos(self.start_yaw)*np.cos(self.start_roll), np.sin(self.start_yaw)*np.sin(self.start_pitch)*np.cos(self.start_roll)-np.cos(self.start_yaw)*np.sin(self.start_roll)],
                               [-np.sin(self.start_pitch), np.cos(self.start_pitch)*np.sin(self.start_roll), np.cos(self.start_pitch)*np.cos(self.start_roll)]])
    
    self.tac_img_size = (1280, 960)
    # States
    self.joint_state = None
    self.gripper_width = None
    self.gripper_force = None
    # Data
    self.stamped_haptic_data = []
    self.stamped_weight_data = []
    self.stamped_gripper_data = []
    self.stamped_force_data = []
    rospy.Subscriber('/joint_states', JointState, self.cb_joint_states)
    # rospy.Subscriber('/weight', Float32, self.cb_weight)
    rospy.Subscriber('/wsg_50_driver/status', Status, self.cb_gripper)
    # Publishers
    self.pos_controller = rospy.Publisher('/scaled_pos_joint_traj_controller/command',
                                          JointTrajectory, queue_size=20)
    self.finger_traj_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=1)
    self.finger_speed_pub = rospy.Publisher('/wsg_50_driver/goal_speed', Float32, queue_size=1)
    self.saving_adr = '/media/okemo/extraHDD31/samueljin/'


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
    
    self.force_torque = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                  msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
    if self.offset != []:
      for i in range(6):
        self.force_torque[i] = self.force_torque[i] - self.offset[i]
      
    self.force_torque_cache.append(self.force_torque)
    if len(self.force_torque_cache) > 30:
      self.force_torque_cache = self.force_torque_cache[-30:]
    self.force_torque_avg = np.mean(self.force_torque_cache, axis=0)
        
    if not self.start_recording:
      return

    self.force_x = msg.wrench.force.x
    
    time = msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs
    
    self.force_torque_stamped_datas.append(
        (time, self.force_torque))

    self.fr_list.append(np.sqrt(self.force_torque[0]**2 + self.force_torque[1]**2))

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
  def grasp_part(self):

    rospy.sleep(1)
    target_width = 100
    while self.gripper_force < 10:
      rospy.sleep(0.1)
      gripper.grasp(target_width, 20)
    #   rospy.sleep(0.1)
    #   print(self.gripper_force, self.gripper_width)
      target_width -= 20

  def pickup(self):
    """ Pickup the object. """
    rospy.sleep(1)
    gripper.homing()
    print(self.joint_state)
    print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))
    self.move_to_joint(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))
    rospy.sleep(2)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frameI = gs.resize_crop_mini(frame, self.tac_img_size[0], self.tac_img_size[1])
    markersI = gs.find_markers(frameI)
    old_markers = markersI
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    index = len([name for name in os.listdir(self.saving_adr) if os.path.isfile(os.path.join(self.saving_adr, name))] ) + 1
    out = cv2.VideoWriter(self.saving_adr + 'output' + str(index) + '.avi', fourcc, 10.0, self.tac_img_size)
    self.grasp_part()
    up_loc = self.start_loc + np.array([0, 0, 0.1])
    self.move_to_joint(inverse_kinematic(self.joint_state, up_loc, self.start_rot))
    rospy.sleep(2)
    count = 0
    while True:
        print(self.gripper_force, self.gripper_width)
        ret, frame = cap.read()
        frame = gs.resize_crop_mini(frame, self.tac_img_size[0], self.tac_img_size[1])
        markers = gs.find_markers(frame)
        center_now, markerU, markerV = gs.update_markerMotion(markers, old_markers, markersI)
        old_markers = center_now
        frame = gs.displaycentres(frame, center_now, markerU, markerV)
        average_movement = np.mean(np.sqrt(markerU**2 + markerV**2))
        cv2.putText(frame, 'Average Movement: ' + str(average_movement), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
        if count%5 == 0:
            gripper.move(self.gripper_width+0.1)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if self.gripper_width > 58:
            break
        cv2.imshow('frame', frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('done')
    #   rospy.sleep(2)





if __name__ == '__main__':
  rospy.init_node('grasp')
  Grasp_ = Grasp()
  rospy.sleep(1)
#   print(forward_kinematic(Grasp_.joint_state))
  Grasp_.pickup() 