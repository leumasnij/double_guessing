
from re import I
import cv2 as cv
from sklearn.cluster import KMeans
import os
import argparse
import numpy as np
import rospy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from ur5_lib import gripper
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from wsg_50_common.msg import Status, Cmd
from scipy.interpolate import interp1d
from scipy import interpolate
from pancake_drawing.image_decomp import image_decomp_main
import copy
DH_theta = np.array([0,0,0,0,0,0])
DH_a = np.array([0,-0.425,-0.39225,0,0,0])
DH_d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
DH_alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
MAT = np.array([[-2.09871615e-02,  2.36852099e+00,  2.03717928e+03],
       [ 2.39103046e+00,  8.50207530e-03,  9.41272360e+02],
       [-1.46489995e-06, -3.10660706e-05,  1.00000000e+00]])
"""
Using haptics to predict flow and thickness.
Then given the stroke we want to draw, draw a line.
"""
def forward_kinematic(joint_angle):

    T = np.eye (4)
    for theta_idx in range(0,6):
        stheta = np.sin(joint_angle[theta_idx])
        ctheta = np.cos(joint_angle[theta_idx])
        salpha  =np.sin(DH_alpha[theta_idx])
        calpha = np.cos(DH_alpha[theta_idx])
        T = np.dot(T,np.array([[ctheta, -stheta*calpha, stheta*salpha, DH_a[theta_idx]*ctheta],
                        [stheta, ctheta*calpha, -ctheta*salpha, DH_a[theta_idx]*stheta],
                        [0, salpha, calpha, DH_d[theta_idx]],
                        [0,0,0,1]]))
    # print(T)
    angle = T[:3,:3]
    position = T[:3,3]
    return angle, position
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

    theta =  copy.deepcopy(current_joint_angle)
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
    


    if np.isnan(theta[0]) or np.isnan(theta[1]) or np.isnan(theta[2]) or np.isnan(theta[3]) or np.isnan(theta[4]) or np.isnan(theta[5]):
      raise ValueError("Input position is unreachable")

    return theta

def find_width(mask):
    mask = mask.T
    widths = []

    # Iterate through each row of the mask
    for i in range(mask.shape[0]):  # For each row
        row = mask[i]
        
        # Find indices of non-zero elements (object pixels in this row)
        object_indices = np.where(row == 255)[0]
        
        if object_indices.size > 0:  # If the current row contains part of the object
            # Calculate width as difference between max and min indices of the object in this row
            width = object_indices[-1] - object_indices[0] + 1  # +1 to include both endpoints
            widths.append(width)

    # Now calculate the average, minimum, and maximum width if there are any widths calculated
    if widths:
        average_width = sum(widths) / len(widths) 
        standard_deviation = np.std(widths)

    else:
        average_width, standard_deviation = 0, 0  # Defaults in case of no object found

    # Print the results
    return average_width, standard_deviation

def kmeans_mask(image, location, K=3, experiment_dir = '', record = False):
    
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(pixel_values)
    labels = kmeans.labels_

    # Map the segmented labels to colors
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    if record:
      cv.imwrite(experiment_dir + str(rospy.get_time()) + '_seg.jpg', segmented_image)

    cv.imshow('seg', segmented_image)
    cluster_label = labels[location[1]*image.shape[1]+location[0]]
    #cv.imshow('seg', segmented_image)
    mask = labels.reshape(image.shape[:2])
    mask = (mask == cluster_label).astype(np.uint8) * 255

    ret, labels = cv.connectedComponents(mask)
    mask = np.zeros_like(mask)

    mask[labels == labels[location[1], location[0]]] = 255

    kernel = np.ones((3,3),np.uint8)
    #mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    return mask


class Pourring(object):
  def __init__(self):

    # Grid poses
    
    self.adr = '/home/rocky/samueljin/pancake_bot/active_shake_ws/src/pancake_drawing/file4.jpg'
    self.flow_rate_adr = '/home/rocky/samueljin/pancake_bot/batter_wide/1/weights.npz'
    data = np.load('/home/rocky/samueljin/pancake_bot/active_shake_ws/src/pouring.npz')
    # self.interp = RegularGridInterpolator((data['angles'], data["xs"], data["ys"]), data["joint_poses"])
    # self.xmin = np.min(data["xs"])
    # self.xmax = np.max(data["xs"])
    # self.ymin = np.min(data["ys"])
    # self.ymax = np.max(data["ys"])
    # Key poses
    # self.dripping_pose = np.array([2.1213, -1.5316, 1.3533, -0.5717, -0.2125, -0.0182])
    self.dripping_pose = data["joint_poses"][0][0]
    self.non_drip_pos = np.array([0., 0., 0., 0., 0., np.pi/2+0.05])
    self.non_drip_pos2 = np.array([0., 0., 0., 0., 0., np.pi])
    self.start_pose = np.array([36.78, -73.88,122.98, -45.78, -52.97, 33.52])/180 * np.pi
    self.start_pose2 = np.array([132.28, -72.5,124.37, -53.02, 127.9, 28.52])/180 * np.pi
    self.grab_pose = np.array([44.42, -65.88,113.29, -43.78, -45.07, 33.52])/180 * np.pi
    self.grab_pose2 = np.array([110.45, -82.54,135.57, -51.4, 116.6, 27.52])/180 * np.pi
    self.ready_pose = np.array([52.86, -69.41,74.57 , 0.35, -36.88, 31.52])/180 * np.pi
    self.prepour_pose = np.array([36.22, -42.29,  57.59, -12.01, -53.53, 33.52])/180 * np.pi
    self.prepour_pose2  = np.array([ 0.20516998, -1.84140978,  2.16585283, -0.46076835,  0.12861453,  0.68840752])

    self.angle = np.array([[ 0.79775209,-0.59817881,0.07598498],
                            [0.01715456,-0.10344896,-0.99448682],
                            [0.60274151,  0.79465743, -0.07226513]])

    #self.up_pose = np.array()
    
    # States
    self.start_time = None
    self.is_drawing = False
    self.joint_state = None
    self.total_volume = 0.0
    self.sauce_count = 0
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
    #rospy.Subscriber('/NordboLRS6_node/wrench', WrenchStamped, self.cb_force_torque)
    rospy.Subscriber('/joint_states', JointState, self.cb_joint_states)
    rospy.Subscriber('/weight', Float32, self.cb_weight)
    # Publishers
    self.pos_controller = rospy.Publisher('/scaled_pos_joint_traj_controller/command',
                                          JointTrajectory, queue_size=20)
    self.finger_traj_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=1)
    self.finger_speed_pub = rospy.Publisher('/wsg_50_driver/goal_speed', Float32, queue_size=1)
    
  def get_joint_angle(self, s):
    x = max(min(self.s2x(s), self.xmax), self.xmin)
    y = max(min(self.s2y(s), self.ymax), self.ymin)
    return self.interp([np.array([x, y])])[0]


  def cb_joint_states(self, msg):
    # self.joint_state = np.array(msg.position)
    self.joint_state = np.array([msg.position[2], msg.position[1], msg.position[0],
                   msg.position[3], msg.position[4], msg.position[5]])
  def cb_weight(self, msg):
    time = rospy.get_time()
    stamped_weight = (time, msg.data)
    if not self.is_weight_initialized:
      self.initial_weight = stamped_weight[1]
      self.is_weight_initialized = True
    elif self.is_recording:
      self.stamped_weights.append(stamped_weight)



  
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


  def follow_trajectory(self, joint_space_goals, times):
    """ follow joint trajectory. """
    pos_message = JointTrajectory()
    pos_message.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    for joint_space_goal, time in zip(joint_space_goals, times):
      pos_message_point = JointTrajectoryPoint()
      pos_message_point.positions = joint_space_goal
      pos_message_point.time_from_start = rospy.Duration(time)
      pos_message.points.append(pos_message_point)
    self.pos_controller.publish(pos_message)
    rospy.sleep(times[-1])

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

  def measure_weight(self):
    # Create directory
    experiment_dir = '/home/rocky/samueljin/pancake_bot/batter_flowrate/with_spout/' + str(rospy.get_time())
    if not os.path.isdir(experiment_dir):
      os.makedirs(experiment_dir)
    rospy.sleep(1)
    init_pose = self.joint_state
    poses = []
    times = []
    print(1)
    speed = 0.00125
    for i in range(min(int(1/speed), 4000)):
       temp = init_pose.copy()
       temp[5] += speed*i
       poses.append(temp)
       times.append(i+1)
    #print(poses)
    start_time = rospy.get_time()
    self.is_weight_initialized = False
    self.is_recording = True
    print(min(int(1/speed), 4000))
    self.follow_trajectory(poses, times)
    rate = rospy.Rate(100)
    self.time_lapse = rospy.get_time() - start_time
    
    self.move_to_joint(init_pose, time = 1)
    rospy.sleep(3)
    # Stop the flow and record for another 3 seconds
    self.is_recording = False
    # Change state to data
    self.stamped_weight_data = np.array(self.stamped_weights)
    self.stamped_weight_data[:, 0] -= start_time
    self.stamped_weight_data[:, 1] -= self.initial_weight
    self.total_squeezed = self.stamped_weight_data[-1, 1]
    # Record data
    data_path = os.path.join(experiment_dir, 'weights.npz')
    np.savez(
        data_path, weights = self.stamped_weight_data[:,1],
        times = self.stamped_weight_data[:,0])
    
    print("Save Experiment Data at %s, total squeezed: %.2f, total squeeze time: %.2f"
        %(experiment_dir, self.total_squeezed, self.time_lapse))
    

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
  
  def check_start_flow(self, record = False, wf = 1.1):
    sample_time = int((1.6-wf)*100)
    sample_time = max(10, sample_time)
    rotate_rate = 0.001/50
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    rospy.sleep(0.5)
    count = 0
    first_3_flag = True
    initial_points = []
    check_points = []
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    cur_pos = forward_kinematic(self.joint_state)[1]
    cur_pos = cur_pos[:2]
    cur_pos = (cur_pos*1000).astype(np.float32)

    if cur_pos[1] < -0.48145*1000:
      y_compensation = (cur_pos[1] + (0.48145)*1000)/2.36852099
    else:
      y_compensation = 0
    estimate_center = np.dot(MAT, np.array([cur_pos[0]-50, cur_pos[1]-y_compensation, 1]))
    estimate_center = estimate_center.astype(np.int32)
    
    size = (frame_width, frame_height)
    experiment_dir = ''
    if record:
      experiment_dir = '/home/rocky/samueljin/pancake_bot/video/' + str(rospy.get_time()) + '/'
      if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
      result = cv.VideoWriter(experiment_dir + 'frame_flow.avi',  
                          cv.VideoWriter_fourcc(*'MJPG'), 
                          10, size)
      result2 = cv.VideoWriter(experiment_dir + 'frame_mask.avi',
                            cv.VideoWriter_fourcc(*'MJPG'), 
                            3, (400, 350))
    take = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        #print(frame.shape)
      frame1 = cv.putText(frame, 'Not Flowing', (50, 50) , cv.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 0, 0) , 3, cv.LINE_AA) 
      if record:
        result.write(frame1)
      count += 1
      if count%sample_time != 0:
        continue
      hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
      
      height = 250
      width = 200
      left = np.max([estimate_center[0].astype(np.int32)-width, 0])
      right = estimate_center[0].astype(np.int32)+width
      top = np.max([estimate_center[1].astype(np.int32)-height, 0])
      bottom = estimate_center[1].astype(np.int32)+height-150
      bowl = hsv[top:bottom, left:right]
      if left == 0 or right == frame_width or top == 0 or bottom == frame_height:
          bowl = cv.copyMakeBorder(bowl, 350-bottom+top, 0, 400-right+left, 0, cv.BORDER_CONSTANT, value = [0,0,0])

      
      seg1 = kmeans_mask(bowl,(200,300), 5, experiment_dir=experiment_dir, record = record)

      seg1 = seg1.astype(np.uint8)
      points = np.where(seg1 == 255)
      h = 350 - np.min(points[0])
      
      if record:
          seg1 = cv.merge([seg1, seg1, seg1])
          seg11 = cv.putText(seg1, 'Not Flowing', (50, 50) , cv.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 0, 0) , 3, cv.LINE_AA) 
          result2.write(seg11)
      if first_3_flag:
        initial_points.append(h)
        if len(initial_points) == 3:
          if np.max(initial_points) - np.min(initial_points) > 15:
             initial_points = initial_points[1:]
             continue
          first_3_flag = False
          initial_points = np.mean(initial_points)
        continue
      check_points.append(h)
      if len(check_points) == 3:
        if np.max(check_points) - np.min(check_points) > 50:
          check_points = []
          continue
        check_points = np.mean(check_points)
        if check_points > max(initial_points*1.05,initial_points+8):
          break
        else:
          if check_points > initial_points*0.975 and check_points < initial_points:
             initial_points = check_points
          print("Flow not started, the current height is %d and initial height is %d"%(check_points, initial_points))
          check_points = []
          next_joint = self.joint_state.copy()
          next_joint[5] += rotate_rate*200
          self.move_to_joint(next_joint, time = 0.5)
          take += 1
          rospy.sleep(1)
          
          if take > 50:
            #print("Flow takes too long to started, the current height is %d and initial height is %d"%(check_points, initial_points))
            return False
    # cap.release()
    # cv.destroyAllWindows()
    # result2.release()
    # result.release()
    # return True

    last_5_frames = []
    print("Flow started, the current height is %d and initial height is %d"%(check_points, initial_points))
    while True:
      ret, frame = cap.read()
      count += 1
      frame1 = cv.putText(frame, 'Flowing', (50, 50) , cv.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 0, 0) , 3, cv.LINE_AA) 
      if record:
        result.write(frame1)
      if count%sample_time != 0:
        continue
      hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

      height = 250
      width = 200
      left = np.max([estimate_center[0].astype(np.int32)-width, 0])
      right = estimate_center[0].astype(np.int32)+width
      top = np.max([estimate_center[1].astype(np.int32)-height, 0])
      bottom = estimate_center[1].astype(np.int32)+height-150
      bowl = hsv[top:bottom, left:right]
      if left == 0 or right == frame_width or top == 0 or bottom == frame_height:
          bowl = cv.copyMakeBorder(bowl, 350-bottom+top, 0, 400-right+left, 0, cv.BORDER_CONSTANT, value = [0,0,0])
      seg1 = kmeans_mask(bowl,(200,300), 5, experiment_dir=experiment_dir, record = record)
      
      points = np.where(seg1 == 255)
      h = 350 - np.min(points[0])
      last_5_frames.append(h)
      if record:
        seg1 = cv.merge([seg1, seg1, seg1])
        seg11 = cv.putText(seg1, 'Flowing', (50, 50) , cv.FONT_HERSHEY_SIMPLEX ,  
                    1, (255, 0, 0) , 3, cv.LINE_AA) 
        result2.write(seg11)
      if len(last_5_frames) == 3:
         if np.max(last_5_frames) - np.min(last_5_frames) < min(2,np.mean(last_5_frames)*0.05):
            break
         else:
            last_5_frames = []
      next_joint = self.joint_state.copy()
      next_joint[5] += rotate_rate*50
      self.move_to_joint(next_joint, time = 0.1)
    cap.release()
    cv.destroyAllWindows()
    if record:
      result2.release()
      result.release()
    return True
    
  
  
  def move_to_start(self):
    rospy.sleep(1)
    gripper.move(50, 50)
    self.move_to_joint(self.start_pose2, time = 20)
    rospy.sleep(20.5)
  
  def pickup(self): 
    rospy.sleep(1)
    gripper.move(50, 50)
    self.move_to_joint(self.start_pose2, time = 2)
    rospy.sleep(2.5)
    self.move_to_joint(self.grab_pose2, time = 5)
    rospy.sleep(5.5)
    gripper.set_force(80)
    gripper.grasp(10, 50)
    cur_ang,cur_pos = forward_kinematic(self.grab_pose2)
    cur_pos[2] = 0.4
    ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)

    self.move_to_joint(ik, time = 5)
    rospy.sleep(5.5)

  def drop(self):
    rospy.sleep(2)
    cur_ang,cur_pos = forward_kinematic(self.joint_state)
    cur_pos[2] = 0.4
    ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
    self.move_to_joint(ik, time = 5)
    rospy.sleep(5.5)
    cur_ang,cur_pos = forward_kinematic(self.grab_pose2)
    cur_pos[2] = 0.4
    ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
    self.move_to_joint(ik, time = 5)
    rospy.sleep(5.5)
    self.move_to_joint(self.grab_pose2, time = 2)
    rospy.sleep(2.5)
    gripper.move(50, 50)
    self.move_to_joint(self.start_pose2, time = 2)
    rospy.sleep(2.5)
    return


  def find_loc(self):
     rospy.sleep(1)
     cur_ang,cur_pos = forward_kinematic(self.joint_state)
     print(cur_pos)

  def pour_shape(self,depth, adrr, mode = 'outline'):
    start_angle = 111-11.4*depth
    rospy.sleep(1)
    cur_ang, _ = forward_kinematic(self.grab_pose2)
    ik = inverse_kinematic(self.joint_state, np.array([-0.17245124, -0.496458903, 0.25]), cur_ang)
    self.move_to_joint(ik, time = 5)
    rospy.sleep(5.5)
    #adrr = '/home/rocky/samueljin/pancake_bot/active_shake_ws/src/pancake_drawing/file10.jpg'
    ret_list = image_decomp_main(adrr, mode)
    print(len(ret_list))
    start_angle = (start_angle/180)*np.pi
    rotate_rate = (0.0015/50)
    GT_joint_state = self.joint_state.copy()
    way_points = []
    times = []
    
    for i in range(len(ret_list)):
      xs = ret_list[i][0]/250.0 * 0.15 - 0.075
      ys = ret_list[i][1]/250.0 * 0.15 - 0.075

      way_point = []
      time = []
      flag = True
      
      for j in range(len(xs)):
        cur_angle = j*rotate_rate + start_angle - self.prepour_pose[-1]
        cur_ang,cur_pos = forward_kinematic(GT_joint_state)
        cur_pos[0] += xs[j]+np.cos(cur_angle)*0.22-0.22+0.05
        cur_pos[1]+= ys[j]
        cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
        ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
        ik[5] = j*rotate_rate + start_angle
        way_point.append(ik)
        time.append(j*0.05)
      way_points.append(way_point)
      times.append(time)

    for i in range(len(way_points)):
        temp = way_points[i][0].copy()
        temp[-1]  = self.grab_pose2[-1]
        self.move_to_joint(temp, time = 2)
        rospy.sleep(2)
        self.move_to_joint(way_points[i][0], time = 2.5)
        rospy.sleep(4.5)

        if i == 0:
          if not self.check_start_flow(wf=1.25):
            return
          increment = self.joint_state[-1]- way_points[i][0][-1]
          for k in range(len(way_points)):
            if k!=0:
              increment -= (way_points[k-1][-1][5] - way_points[k][0][5])/2
            for j in range(len(way_points[k])):
              way_points[k][j][-1] += increment
              

        self.follow_trajectory(way_points[i], times[i])
        temp = way_points[i][-1].copy()
        temp[5] = self.grab_pose2[-1]
        self.move_to_joint(temp, time = 0.5)
        rospy.sleep(1)

       

  def line_pour_with_reset(self, depth, k):
      rospy.sleep(1)
      cur_ang, _ = forward_kinematic(self.grab_pose2)
      ik = inverse_kinematic(self.joint_state, np.array([-0.18245124, -0.496458903, 0.25]), cur_ang)
      self.move_to_joint(ik, time = 5)
      rospy.sleep(5.5)
      start_angle = 111-11.4*depth
      print(start_angle)
      velo_list = [0.05, 0.1, 0.2, 0.3, 0.4]
      rotate_rateList = [0.001/50, 0.003/50, 0.004/50]
      start_angle = start_angle/180*np.pi
      rotate_rate = 0.00175/50 * (velo_list[k]/0.05)

      flag = True
      GT_joint_state = self.joint_state
      
      way_point = []
      time = []
      length =  250
      for i in range(length):
        cur_angle = i*rotate_rate + start_angle - self.prepour_pose[-1]
        cur_ang,cur_pos = forward_kinematic(GT_joint_state)
        cur_pos[0] += (0.2*(float(i)/length))+np.cos(cur_angle)*0.22-0.22-0.05
        cur_pos[1] -= 0.1-0.05*k
        cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
        if(i == 0):
          print(cur_pos)
        ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
        ik[5] = i*rotate_rate + start_angle
        way_point.append(ik)
        time.append(i*velo_list[k])
      temp = way_point[0].copy()
      temp[-1]  = self.grab_pose2[-1]
      self.move_to_joint(temp, time = 2)
      rospy.sleep(2)
      self.move_to_joint(way_point[0], time = 2)
      rospy.sleep(2)
      if not self.check_start_flow(record= True,wf=1.25):
          return
      increment = self.joint_state[-1]- way_point[0][-1]
      for i in range(len(way_point)):
        way_point[i][-1] += increment
      self.follow_trajectory(way_point, time)
      # temp = way_point[-1].copy()
      # temp[5] = self.grab_pose2[-1]
      # self.move_to_joint(temp, time = 0.5)
      # rospy.sleep(1)
  

  def fixed_pour(self, depth, time):
      rospy.sleep(1)
      cur_ang, _ = forward_kinematic(self.grab_pose2)
      ik = inverse_kinematic(self.joint_state, np.array([-0.16245124, -0.491458903, 0.25]), cur_ang)
      self.move_to_joint(ik, time = 4)
      rospy.sleep(4.5)
      start_angle = 111-11.4*depth
      GT_joint_state = self.joint_state.copy()
      print(GT_joint_state)
      cur_angle = (start_angle/180)*np.pi - self.prepour_pose[-1]
      cur_ang,cur_pos = forward_kinematic(GT_joint_state)
      cur_pos[0] += (0.1)+np.cos(cur_angle)*0.22-0.22-0.05
      cur_pos[1] -= 0
      cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
      ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
      ik[5] = start_angle/180*np.pi
      self.move_to_joint(ik, time = 5)
      rospy.sleep(5.5)
      if not self.check_start_flow( wf=1.25):
          return
      rotate_rate = 0.001/50*20
      cur_angle = self.joint_state[-1] + rotate_rate*time - self.prepour_pose[-1]
      cur_ang,cur_pos = forward_kinematic(GT_joint_state)
      cur_pos[0] += (0.1)+np.cos(cur_angle)*0.22-0.22-0.05
      cur_pos[1] -= 0
      cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
      ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
      ik[5] = self.joint_state[-1] + rotate_rate*time
      self.move_to_joint(ik, time = time)
      rospy.sleep(time+0.5)
      print(GT_joint_state)
      self.move_to_joint(GT_joint_state, time = 0.5)
      rospy.sleep(0.5)
      
  def baseline_pour(self, depth, time):
      rospy.sleep(1)
      cur_ang, _ = forward_kinematic(self.grab_pose2)
      ik = inverse_kinematic(self.joint_state, np.array([-0.16245124, -0.491458903, 0.25]), cur_ang)
      self.move_to_joint(ik, time = 10)
      rospy.sleep(10.5)
      start_angle = 111-11.4*depth
      GT_joint_state = self.joint_state.copy()
      print(GT_joint_state)
      cur_angle = (start_angle/180)*np.pi - self.prepour_pose[-1]
      cur_ang,cur_pos = forward_kinematic(GT_joint_state)
      cur_pos[0] += (0.1)+np.cos(cur_angle)*0.22-0.22-0.05
      cur_pos[1] -= 0
      cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
      ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
      ik[5] = start_angle/180*np.pi
      self.move_to_joint(ik, time = 5)
      rospy.sleep(5.5)
      if not self.check_start_flow( wf=1.25):
          return
      rotate_rate = 0.001/50*20
      cur_angle = self.joint_state[-1] + rotate_rate*time*1.5 - self.prepour_pose[-1]
      cur_ang,cur_pos = forward_kinematic(GT_joint_state)
      cur_pos[0] += (0.1)+np.cos(cur_angle)*0.22-0.22-0.05
      cur_pos[1] -= 0
      cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
      ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
      ik[5] = self.joint_state[-1] + rotate_rate*time*1.5
      self.move_to_joint(ik, time = 0.5)
      rospy.sleep(30.5)
      print(GT_joint_state)
      self.move_to_joint(GT_joint_state, time = 0.5)
      rospy.sleep(0.5)
      
        
  def test(self, depth):
      start_angle = 111-11.4*depth
      rospy.sleep(1)
      cur_ang, _ = forward_kinematic(self.grab_pose2)
      ik = inverse_kinematic(self.joint_state, np.array([-0.15245124, -0.481458903, 0.20]), cur_ang)
      self.move_to_joint(ik, time = 10)
      rospy.sleep(10.5)
      start_angle = (start_angle/180)*np.pi
      start_pos = self.joint_state
      start_pos[5] = start_angle
      GT_joint_state = self.joint_state
      cur_angle = (72./180)*np.pi - self.prepour_pose[-1]
      cur_ang,cur_pos = forward_kinematic(GT_joint_state)
      cur_pos[0] += (0.1)+np.cos(cur_angle)*0.22-0.22-0.05
      cur_pos[1] -= 0
      cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
      print(cur_pos)
      print(GT_joint_state)
      ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
      ik[5] = 72./180*np.pi
      print(ik)
      self.move_to_joint(ik, time = 10)
      rospy.sleep(10)
      # if not self.check_start_flow():
      #   print("Flow not started")
      #   return
      # print("Flow started")

  def line_width_measurement(self, experiment_dir, run):
      ratio = (587.+537.)/(36.5+33.5)
      cap = cv.VideoCapture(0)
      cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
      cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
      while True:
          ret, frame = cap.read()
          if not ret:
              print("No frame")
              break
          frame = frame[220:520, 780:1080]
          hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
          
          # mask = kmeans_mask(hsv, (380, 400),5)
          if cv.waitKey(1) == ord('q'):
              file_name = experiment_dir + 'mask' + str(run) + '.jpg'
              cv.imwrite(file_name, mask)
              file_name = experiment_dir + 'frame' + str(run) + '.jpg'
              cv.imwrite(file_name, frame)
              message = ''
              average_width, std = find_width(mask)
              average_width = average_width/ratio
              std = std/ratio
              message += 'File No.'+str(run) + ' Average: ' + str(average_width) + ' Standard Deviation: ' + str(std) + '\n'
              file_name = experiment_dir + 'mask' + str(run) + '.jpg'
              with open(experiment_dir + 'results.txt', 'a') as f:
                  f.write(message)
              break
          mask = kmeans_mask(hsv, (150,160), 3)
          ret, labels = cv.connectedComponents(mask)
          cv.imshow('mask', mask)
          cv.imshow('hsv', hsv)
      cap.release()
      cv.destroyAllWindows()

  def single_line_drawing(self, velo, depth):
      rospy.sleep(1)
      cur_ang, _ = forward_kinematic(self.grab_pose2)
      ik = inverse_kinematic(self.joint_state, np.array([-0.17245124, -0.496458903, 0.25]), cur_ang)
      self.move_to_joint(ik, time = 5)
      rospy.sleep(5.5)
      start_angle = 111-11.4*depth
      print(start_angle)
      start_angle = start_angle/180*np.pi
      rotate_rate = 0.00125/50 * (velo/0.05)
      GT_joint_state = self.joint_state
      way_point = []
      time = []
      length =  250
      for i in range(length):
        cur_angle = i*rotate_rate + start_angle - self.prepour_pose[-1]
        cur_ang,cur_pos = forward_kinematic(GT_joint_state)
        cur_pos[0] += 0.1+np.cos(cur_angle)*0.22-0.22-0.05
        cur_pos[1] -= -0.1 + (0.2*(float(i)/length))
        cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
        if(i == 0):
          print(cur_pos)
        ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
        ik[5] = i*rotate_rate + start_angle
        way_point.append(ik)
        time.append(i*velo)
      temp = way_point[0].copy()
      temp[-1]  = self.grab_pose2[-1]
      self.move_to_joint(temp, time = 2)
      rospy.sleep(2)
      self.move_to_joint(way_point[0], time = 2)
      rospy.sleep(3)
      if not self.check_start_flow(record= False,wf=1.25):
          return
      increment = self.joint_state[-1]- way_point[0][-1]
      for i in range(len(way_point)):
        way_point[i][-1] += increment
      self.follow_trajectory(way_point, time)


  def run_single_line(self, time_list, depth, wfr):
      from datetime import datetime
      import os
      experiment_dir = '/home/rocky/samueljin/pancake_bot/experiments/Actual/' + str(wfr) + '1'
      while os.path.exists(experiment_dir):
          experiment_dir = experiment_dir + '1'
      os.makedirs(experiment_dir)
      experiment_dir = experiment_dir + '/'
      
      for i in range(len(time_list)):
        # flag = True
        # while flag:
        #   command = raw_input("Continue? [y]")
        #   if command == 'y':
        #       flag = False
        #       break
        #   else:
        #       print("Command '%s' unrecognized"%command)
        #       continue
        self.pickup()
        self.single_line_drawing(time_list[i], depth)
        cur_ang,cur_pos = forward_kinematic(self.joint_state)
        cur_pos[2] = 0.4
        ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
        ik[5] = self.grab_pose2[-1]
        self.move_to_joint(ik, time = 5)
        rospy.sleep(5.5)
        cur_ang,cur_pos = forward_kinematic(self.grab_pose2)
        cur_pos[2] = 0.4
        ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
        self.move_to_joint(ik, time = 5)
        rospy.sleep(5.5)
        self.line_width_measurement(experiment_dir, i)
        self.move_to_joint(self.grab_pose2, time = 2)
        rospy.sleep(2.5)
        gripper.move(50, 50)
        self.move_to_joint(self.start_pose2, time = 2)
        rospy.sleep(2.5)
      
  def tele_op(self, depth):
    rospy.sleep(1)
    self.pickup()
    rospy.sleep(1)
    cur_ang, _ = forward_kinematic(self.grab_pose2)
    ik = inverse_kinematic(self.joint_state, np.array([-0.18245124, -0.496458903, 0.25]), cur_ang)
    self.move_to_joint(ik, time = 5)
    rospy.sleep(5.5)
    start_angle = 111-11.4*depth
    start_angle = start_angle/180*np.pi
    GT_joint_state = self.joint_state.copy()


    cur_angle = start_angle - self.prepour_pose[-1]
    cur_ang,cur_pos = forward_kinematic(GT_joint_state)
    cur_pos[0] += 0.1+np.cos(cur_angle)*0.22-0.22-0.05
    cur_pos[1] -= -0.075                               
    cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
    ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
    ik[5] = start_angle
    self.move_to_joint(ik, time = 5)
    while True:
      command = raw_input("Press 'A' to increase angle, press 'S' to start. Press 'D' to decrease angle")
      if command == 'A' or command == 'a':
          start_angle += 0.005
          cur_angle = start_angle - self.prepour_pose[-1]
          cur_ang,cur_pos = forward_kinematic(GT_joint_state)
          cur_pos[0] += 0.1+np.cos(cur_angle)*0.22-0.22-0.05
          cur_pos[1] -= -0.075
          cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
          ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
          ik[5] = start_angle
          self.move_to_joint(ik, time = 0.5)
          rospy.sleep(0.5)
      elif command == 'D' or command == 'd':
          start_angle -= 0.005
          cur_angle = start_angle - self.prepour_pose[-1]
          cur_ang,cur_pos = forward_kinematic(GT_joint_state)
          cur_pos[0] += 0.1+np.cos(cur_angle)*0.22-0.22-0.05
          cur_pos[1] -= -0.075
          cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
          ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
          ik[5] = start_angle
          self.move_to_joint(ik, time = 0.5)
          rospy.sleep(0.5)
      elif command == 'S' or command == 's':
          break
      else:
          print("Command '%s' unrecognized"%command)
          continue
    distance = 0
    while distance < 0.15:
        command = raw_input("Press 'A' to increase angle, Press 'D' to move, Press 'Q' to quit. A line is drawn with 30 times of move")
        if command == 'D' or command == 'd':
          distance += 0.005
          cur_ang,cur_pos = forward_kinematic(GT_joint_state)
          cur_pos[0] += 0.1+np.cos(cur_angle)*0.22-0.22-0.05
          cur_pos[1] -= -0.075+distance
          cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
          ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
          ik[5] = start_angle
          self.move_to_joint(ik, time = 1)
          rospy.sleep(1)
        elif command == 'A' or command == 'a':
          start_angle += 0.01
          cur_angle = start_angle - self.prepour_pose[-1]
          cur_ang,cur_pos = forward_kinematic(GT_joint_state)
          cur_pos[0] += 0.1+np.cos(cur_angle)*0.22-0.22-0.05
          cur_pos[1] -= -0.075+distance
          cur_pos[2] = 0.15+np.sin(cur_angle)*0.22
          ik = inverse_kinematic(self.joint_state, cur_pos, cur_ang)
          ik[5] = start_angle
          self.move_to_joint(ik, time = 0.5)
          rospy.sleep(0.5)
        elif command == 'Q' or command == 'q':
          break
    self.drop()


  def calc_round_area(self):
        ratio = (587.+537.)/(35.4+33.4)
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame")
                break
            frame = frame[30:730, 550:1250]
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            
            mask = kmeans_mask(hsv, (372, 356),5)
            list_points = np.where(mask == 255)
            area = len(list_points[0])/ratio/ratio
            cv.imshow('mask', mask)
            print(area)
            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

  def camera_capture(self):
      cap = cv.VideoCapture(0)
      cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
      cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
      rospy.sleep(0.5)
     
      ret, frame = cap.read()
      if not ret:
          print("No frame")
          return
      frame = frame[30:730, 550:1250]
      cv.imwrite('frame22.jpg', frame) 
          
      cap.release()
      cv.destroyAllWindows()



if __name__ == "__main__":
  argparse = argparse.ArgumentParser()
  # Add arguments
  argparse.add_argument("--depth", default = 4.5, help = "The depth of the pancake")
  argparse.add_argument("--time", default = 10, help = "The time of the pancake")
  args = argparse.parse_args()
  depth = args.depth
  time = args.time
  depth = float(depth)
  time = float(time)
  # Start the dish garneshing process
  rospy.init_node("pancake2")
  runner = Pourring()
  #runner.tele_op(depth)
  #runner.move_to_start()
  # runner.pickup()
  # runner.single_line_drawing(0.05, depth)
  #runner.fixed_pour(depth=depth, time=time)
  # # # #runner.find_loc()
  # # # runner.line_pour_without_reset(3.8)
  # # #runner.line_pour_with_reset(3.8, 0)
  # # #runner.test(3.9)
  #runner.calc_round_area()
  # runner.move_to_start()
  # for i in range(5):
  #   flag = True
  #   while flag:
  #     command = raw_input("Continue? [y]")
  #     if command == 'y':
  #         flag = False
  #         break
  #     else:
  #         print("Command '%s' unrecognized"%command)
  #         continue
  #   runner.pickup()
  # # runner.test(4.3)

  # # runner.pour_shape(1.6)
  #   runner.line_pour_with_reset(depth-0.05*i, i)
    
  #   # runner.run()
  #   # runner.measure_weight()
  runner.drop()
    

 
