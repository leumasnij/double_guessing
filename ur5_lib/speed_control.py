import numpy as np
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import copy

DH_theta = np.array([0,0,0,0,0,0])
DH_a = np.array([0,-0.425,-0.39225,0,0,0])
DH_d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
DH_alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])



def forward_kinematic (joint_angle):

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
    return T
