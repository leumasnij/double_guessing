#!/usr/bin/env python 

'''
Shubham Kanitkar (skanitka@andrew.cmu.edu) April, 2021
Hung-Jui Huang (hungjuih@andrew.cmu.edu) Sept, 2021
'''
import os
import csv
import cv2
import rospy
import utils
import numpy as np
from skimage.io import imsave
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState, CompressedImage
from geometry_msgs.msg import Wrench

class Sensor:
  """ The sensor interface. """
  def __init__(self, data_dir):
    self.data_dir = data_dir

  def __str__(self):
    return 'General'

class Azure(Sensor):
  """
  The Azure Kinect sensor [side view].
  Data: RGB Compressed and RGB-D Raw
  Data format: .jpg
  """
  def __init__(self, data_dir):
    """ Setup the data directory and subscribers. """
    self.data_dir = data_dir
    self.bridge = CvBridge()
    # RGB Data
    self.is_rgb_recording = False
    self.rgb_count = 0
    self.rgb_dir = os.path.join(self.data_dir, 'rgb')
    os.mkdir(self.rgb_dir)
    self.rgb_subscriber = rospy.Subscriber(
        '/rgb/image_raw/compressed', CompressedImage, self.cb_rgb)
    # Depth Data
    self.is_depth_recording = False
    self.depth_count = 0
    self.depth_dir = os.path.join(self.data_dir, 'depth')
    os.mkdir(self.depth_dir)
    self.depth_subscriber = rospy.Subscriber(
        '/depth/image_raw', Image, self.cb_depth)

  def cb_rgb(self, msg):
    """ The RGB image callback function. """
    if self.is_rgb_recording:
      try:
        np_image = np.fromstring(msg.data, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        timestamp = utils.get_current_timestamp()
        filename = 'rgb_'+str(self.rgb_count)+'_'+str(timestamp)+'.jpg'
        cv2.imwrite(os.path.join(self.rgb_dir, filename), image)
        self.rgb_count += 1
      except CvBridgeError, e:
        print(e)

  def cb_depth(self, msg):
    """ The Depth image callback function. """
    if self.is_depth_recording:
      try:
        image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        timestamp = utils.get_current_timestamp()
        filename = 'depth_'+str(self.depth_count)+'_'+str(timestamp)+'.tif'
        imsave(os.path.join(self.depth_dir, filename), image)
        self.depth_count += 1
      except CvBridgeError, e:
        print(e)

  def start_rgb_recording(self):
    self.is_rgb_recording = True

  def stop_rgb_recording(self):
    self.is_rgb_recording = False

  def start_depth_recording(self):
    self.is_depth_recording = True

  def stop_depth_recording(self):
    self.is_depth_recording = False

  def __str__(self):
    return 'Azure'

class SideCamera(Sensor):
  """
  The Logitech USB Camera sensor.
  Data: Side angle view
  Data format: .jpg
  """
  def __init__(self, data_dir):
    """ Setup the data directory and subscribers. """
    self.is_recording = False
    self.data_dir = data_dir
    self.bridge = CvBridge()
    self.side_camera_count = 0
    self.side_camera_dir = os.path.join(self.data_dir, 'side_camera')
    os.mkdir(self.side_camera_dir)
    self.side_camera_subscriber = rospy.Subscriber(
        '/side_camera/usb_cam/image_raw', Image, self.cb_side_camera)

  def cb_side_camera(self, msg):
    """ The side camera image callback function. """
    if self.is_recording:
      try:
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        timestamp = utils.get_current_timestamp()
        filename = 'side_camera_'+str(self.side_camera_count)+'_'+str(timestamp)+'.jpg'
        cv2.imwrite(os.path.join(self.side_camera_dir, filename), image)
        self.side_camera_count += 1
      except CvBridgeError, e:
        print(e)

  def start_recording(self):
    self.is_recording = True

  def stop_recording(self):
    self.is_recording = False

  def __str__(self):
    return 'SideCamera'

class TopCamera(Sensor):
  """
  The Azure Kinect Camera sensor using as USB Camera.
  Data: Top angle view
  Data format: .jpg
  """
  def __init__(self, data_dir):
    """ Setup the data directory and subscribers. """
    self.is_recording = False
    self.data_dir = data_dir
    self.bridge = CvBridge()
    self.top_camera_count = 0
    self.top_camera_dir = os.path.join(self.data_dir, "top_camera")
    os.mkdir(self.top_camera_dir)
    self.top_camera_subscriber = rospy.Subscriber(
        '/top_camera/usb_cam/image_raw', Image, self.cb_top_camera)

  def cb_top_camera(self, msg):
    """ The top camera image callback function. """
    if self.is_recording:
      try:
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        timestamp = utils.get_current_timestamp()
        filename = 'top_camera_'+str(self.top_camera_count)+'_'+str(timestamp)+'.jpg'
        cv2.imwrite(os.path.join(self.top_camera_dir, filename), image)
        self.top_camera_count += 1
      except CvBridgeError, e:
        print(e)

  def start_recording(self):
    self.is_recording = True

  def stop_recording(self):
    self.is_recording = False

  def __str__(self):
    return 'TopCamera'

class GelSight(Sensor):
  """
  The GelSight Tactile sensor [side view].
  Data: Tactile Images
  Data format: .jpg
  """
  def __init__(self, data_dir):
    """ Setup the data directory and subscribers. """
    self.is_recording = False
    self.data_dir = data_dir
    self.bridge = CvBridge()
    self.gelsight_count = 0
    self.gelsight_dir = os.path.join(self.data_dir, 'gelsight')
    os.mkdir(self.gelsight_dir)
    self.gelsight_subscriber = rospy.Subscriber(
        '/gelsight/usb_cam/image_raw', Image, self.cb_gelsight)

  def cb_gelsight(self, msg):
    """ The GelSight image callback function. """
    if self.is_recording:
      try:
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        timestamp = utils.get_current_timestamp()
        filename = 'gelsight_'+str(self.gelsight_count)+'_'+str(timestamp)+'.jpg'
        cv2.imwrite(os.path.join(self.gelsight_dir, filename), image)
        self.gelsight_count += 1
      except CvBridgeError, e:
        print(e)

  def start_recording(self):
    self.is_recording = True

  def stop_recording(self):
    self.is_recording = False

  def __str__(self):
    return 'GelSight'

class WSG50(Sensor):
  """
  The WSG 50 Gripper sensor.
  Data: WSG 50 Joint States (Position)
  Data format: .csv
  """
  def __init__(self, data_dir):
    """ Setup the data file and subscribers. """
    self.is_recording = False
    self.data_dir = data_dir
    self.wsg50_path = os.path.join(self.data_dir, 'gripper.csv')
    self.data_array = [0, 0]
    self.wsg50_count = 0

    with open(self.wsg50_path, 'w') as csvfile:
      filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      hdr = ['Time', 'Left', 'Right']
      filewriter.writerow(hdr)

    self.wsg50_subscriber = rospy.Subscriber(
        '/wsg_50_driver/joint_states', JointState, self.cb_wsg50)

  def cb_wsg50(self, msg):
    """ The WSG50 callback function. """
    if self.is_recording:
      self.data_array[0] = msg.position[0]
      self.data_array[1] = msg.position[1]
      self._dump_wsg50_data_to_csv()

  def _dump_wsg50_data_to_csv(self):
    """ Dump wsg50 data to csv file. """
    timestamp = utils.get_current_timestamp()
    with open(self.wsg50_path, 'a+') as csvfile:
      filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      filewriter.writerow([timestamp] + self.data_array)
      self.wsg50_count += 1

  def start_recording(self):
    self.is_recording = True

  def stop_recording(self):
    self.is_recording = False

  def __str__(self):
    return 'WSG50'

class UR5e(Sensor):
  """
  The UR5e sensor.
  Data: UR5e Joint States and Joint Velocity
  Data format: .csv
  """
  def __init__(self, data_dir):
    """ Setup the data file and subscribers. """
    self.is_recording = False
    self.data_dir = data_dir
    self.ur5e_path = os.path.join(self.data_dir, 'robot.csv')
    self.n_joints = 6
    self.data_array = [0] * self.n_joints * 2
    self.ur5e_count = 0

    with open(self.ur5e_path, 'w') as csvfile:
      filewriter = csv.writer(
          csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      hdr = ['Time', 'Elbow', 'Shoulder Lift', 'Shoulder Pan',
             'Wrist1', 'Wrist2', 'Wrist3', 'Velocity1', 'Velocity2',
             'Velocity3', 'Velocity4', 'Velocity5', 'Velocity6']
      filewriter.writerow(hdr)

    self.ur5e_subscriber = rospy.Subscriber(
        '/joint_states', JointState, self.cb_ur5e)

  def cb_ur5e(self, msg):
    """ The UR5e callback function. """
    if self.is_recording:
      for i in range(self.n_joints):
        self.data_array[i] = msg.position[i]
        self.data_array[i + self.n_joints] = msg.velocity[i]
      self._dump_ur5e_data_to_csv()

  def _dump_ur5e_data_to_csv(self):
    """ Dump UR5e data to csv file. """
    timestamp = utils.get_current_timestamp()
    with open(self.ur5e_path, 'a+') as csvfile:
      filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      filewriter.writerow([timestamp] + self.data_array)
      self.ur5e_count += 1

  def start_recording(self):
    self.is_recording = True

  def stop_recording(self):
    self.is_recording = False

  def __str__(self):
    return 'UR5e'

class OnRobot(Sensor):
  """
  The OnRobot force torque sensor.
  DataL Force-Torque 6D Data Vector
  Data format: .csv
  """
  def __init__(self, data_dir):
    """ Setup the data file and subscribers. """
    self.is_recording = False
    self.data_dir = data_dir
    self.onrobot_path = os.path.join(self.data_dir, 'force_torque.csv')
    self.data_array = [0] * 6
    self.onrobot_count = 0

    with open(self.onrobot_path, 'w') as csvfile:
      filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      hdr = ['Time', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
      filewriter.writerow(hdr)

    self.onrobot_subscriber = rospy.Subscriber(
        '/ur5e/onrobot_force_torque_usb_1', Wrench, self.cb_onrobot)

  def cb_onrobot(self, msg):
    """ The OnRobot callback function. """
    if self.is_recording:
      self.data_array[0] = msg.force.x
      self.data_array[1] = msg.force.y
      self.data_array[2] = msg.force.z
      self.data_array[3] = msg.torque.x
      self.data_array[4] = msg.torque.y
      self.data_array[5] = msg.torque.z
      self._dump_onrobot_data_to_csv()

  def _dump_onrobot_data_to_csv(self):
    """ Dump OnRobot data to csv file. """
    timestamp = utils.get_current_timestamp()
    with open(self.onrobot_path, 'a+') as csvfile:
      filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      filewriter.writerow([timestamp] + self.data_array)
      self.onrobot_count += 1

  def start_recording(self):
    self.is_recording = True

  def stop_recording(self):
    self.is_recording = False

  def __str__(self):
    return 'OnRobot'
