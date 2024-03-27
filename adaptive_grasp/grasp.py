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
from kinematic import forward_kinematic, inverse_kinematic, inverse_kinematic_orientation
import gelsight_test as gs
import threading

class CameraCapture1:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.ret, self.frame = self.cap.read()
        self.is_running = True
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        rospy.sleep(1)

    def update(self):
        while self.is_running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.is_running = False
        self.cap.release()
class Grasp(object):
  def __init__(self):

    self.start_loc = np.array([-0.08, -0.6, 0.30])
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
    import datetime
    cur_date = datetime.datetime.now()
    self.saving_adr = '/media/okemo/extraHDD31/samueljin/' + str(cur_date.month) + '_' + str(cur_date.day) + '/'
    if not os.path.exists(self.saving_adr):
        os.makedirs(self.saving_adr)
    exp_run = len([name for name in os.listdir(self.saving_adr) ]) + 1
    self.saving_adr = self.saving_adr + 'run' + str(exp_run) + '/'
    


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
  def random_rotate(self):
    # np.random.seed(0)
    self.start_yaw = np.pi/2
    self.start_pitch = 0
    self.start_roll = np.pi
    pitch = np.random.uniform(0, np.pi/2)
    yaw = np.random.uniform(np.pi/2, np.pi)
    roll = np.random.uniform(np.pi, 2*np.pi)
    rot = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                    [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                    [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
    cur_angle, cur_pos = forward_kinematic(self.joint_state)
    # new_state1 = inverse_kinematic(self.joint_state, cur_pos, rot)
    new_state2 = inverse_kinematic_orientation(self.joint_state, cur_pos, rot)
    # print(self.joint_state)
    # print(new_state1)
    # print(new_state2)
    # angle1, pos1 = forward_kinematic(new_state1)
    # angle2, pos2 = forward_kinematic(new_state2)
    # print(angle1, angle2) 
    self.move_to_joint(new_state2, 10)
    rospy.sleep(10)
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
  def pickup(self):
    """ Pickup the object. """
    rospy.sleep(1)
    gripper.homing()
    print(self.joint_state)
    print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))
    self.move_to_joint(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))
    rospy.sleep(2)
    cap = CameraCapture1()
    # rospy.sleep(1)
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    # print(frame.shape)
    self.init_frame = gs.resize_crop_mini(frame, self.tac_img_size[0], self.tac_img_size[1])
    self.grasp_part()
    up_loc = self.start_loc + np.array([0, 0, 0.2])
    self.move_to_joint(inverse_kinematic(self.joint_state, up_loc, self.start_rot))
    rospy.sleep(2)
    return cap
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
    cap = CameraCapture1()
    while True:
        ret, frame = cap.read()
        frame = gs.resize_crop_mini(frame, self.tac_img_size[0], self.tac_img_size[1])
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  def reset(self):
    rospy.sleep(1)
    # gripper.homing()
    # print(self.joint_state)
    # print(inverse_kinematic(self.joint_state, self.start_loc, self.start_rot))
    self.move_to_joint(inverse_kinematic_orientation(self.joint_state, self.start_loc, self.start_rot), 10)
    rospy.sleep(10)
    gripper.homing()


  def save_raw_data(self, img, num_pos):
        save_adr = self.saving_adr + 'raw_data/'+ str(num_pos) + '/'
        if not os.path.exists(save_adr):
            os.makedirs(save_adr)
        count = len([name for name in os.listdir(save_adr) if os.path.isfile(os.path.join(save_adr, name))] ) + 1
        cv2.imwrite(save_adr + 'img' + str(count) + '.png', img)
  def test(self):
    if not os.path.exists(self.saving_adr):
        os.makedirs(self.saving_adr)
    cap = self.pickup()
    frameI = self.init_frame
    self.save_raw_data(frameI, 0)
    markersI = gs.find_markers(frameI)
    old_markers = markersI
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    index = len([name for name in os.listdir(self.saving_adr) if os.path.isfile(os.path.join(self.saving_adr, name))] ) + 1
    out = cv2.VideoWriter(self.saving_adr + 'output' + str(index) + '.avi', fourcc, 5.0, self.tac_img_size)
    self.random_rotate()
    count = 1
    pos_num = 1
    while pos_num<20:
        ret, frame = cap.read()
        frame = gs.resize_crop_mini(frame, self.tac_img_size[0], self.tac_img_size[1])
        self.save_raw_data(frame, pos_num)
        markers = gs.find_markers(frame)
        center_now, markerU, markerV = gs.update_markerMotion(markers, old_markers, markersI)
        old_markers = center_now
        frame = gs.displaycentres(frame, center_now, markerU, markerV)
        average_movement = np.mean(np.sqrt(markerU**2 + markerV**2))
        cv2.putText(frame, 'Average Movement: ' + str(average_movement), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
        if count == 10:
            self.random_rotate()
            count = 1
            pos_num += 1
        # cv2.imshow('frame', frame)
        count += 1
    out.release()
    self.reset()
     




if __name__ == '__main__':
  rospy.init_node('grasp')
  np.random.seed(42)
  Grasp_ = Grasp()
  rospy.sleep(1)
#   print(forward_kinematic(Grasp_.joint_state))
#   Grasp_.pickup()
#   Grasp_.showcamera()
  Grasp_.reset()
#   Grasp_.test_rot()
#   Grasp_.test()
  