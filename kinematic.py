import numpy as np
import copy
DH_theta = np.array([0,0,0,0,0,0])
DH_a = np.array([0,-0.425,-0.39225,0,0,0])
DH_d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
DH_alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
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
    m = DH_d[5]*ay-target_pos[1]
    n = DH_d[5]*ax - target_pos[0]
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

    theta5_1 = np.arccos(ax*np.sin(theta[0])-ay*np.cos(theta[0]))
    theta5_2 = -theta5_1
    # print(theta5_1,theta5_2)
    if abs(theta[4]-theta5_1)<abs(theta[4]-theta5_2):
        theta[4] = theta5_1
    else:
        theta[4] = theta5_2

    mm = nx*np.sin(theta[0]) - ny*np.cos((theta[0]))
    nn = ox*np.sin(theta[0]) - oy*np.cos((theta[0]))
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



def inverse_kinematic_orientation (current_joint_angle, target_pos, rotate_R):
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
    # print("m,n",m,n,m**2+n**2-DH_d[3]**2)
    # theta1_1 = np.arctan2(m,n) - np.arctan2(DH_d[3], np.sqrt(m**2+n**2-DH_d[3]**2))
    # theta1_2 = np.arctan2(m,n) - np.arctan2(DH_d[3], -np.sqrt(m**2+n**2-DH_d[3]**2))
    # if theta1_1<-np.pi:
    #     theta1_1 += 2*np.pi
    # if theta1_2 < np.pi:
    #     theta1_2 += 2*np.pi
    # # print(theta1_1,theta1_2)
    # if abs(theta[0]-theta1_1)<abs(theta[0]-theta1_2):
    #     theta[0] = theta1_1
    # else:
    #     theta[0] = theta1_2

    theta5_1 = np.arccos(ax*np.sin(theta[0])-ay*np.cos(theta[0]))
    theta5_2 = -theta5_1
    # print(theta5_1,theta5_2)
    if abs(theta[4]-theta5_1)<abs(theta[4]-theta5_2):
        theta[4] = theta5_1
    else:
        theta[4] = theta5_2

    mm = nx*np.sin(theta[0]) - ny*np.cos((theta[0]))
    nn = ox*np.sin(theta[0]) - oy*np.cos((theta[0]))
    theta[5] = np.arctan2(mm,nn)-np.arctan2(np.sin(theta[4]),0)


    # mmm = DH_d[4]*(np.sin(theta[5])*(nx*np.cos(theta[0])+ny*np.sin(theta[0]))+
    #                np.cos(theta[5])*(ox*np.cos(theta[0])+oy*np.sin(theta[0])))+\
    #       target_pos[0]*np.cos(theta[0])- DH_d[5]*(ax*np.cos(theta[0])+ ay*np.sin(theta[0]))+\
    #       target_pos[1]*np.sin(theta[0])
    # nnn = DH_d[4]* (oz*np.cos(theta[5])+nz*np.sin(theta[5])) + target_pos[2] - DH_d[0] - az*DH_d[5]
    # theta3_1 = np.arccos((mmm**2+nnn**2-DH_a[1]**2-DH_a[2]**2)/(2*DH_a[1]*DH_a[2]))
    # theta3_2 = -theta3_1
    # # print(theta3_1,theta3_2)

    # if abs(theta[2]-theta3_1)<abs(theta[2]-theta3_2):
    #     theta[2] = theta3_1
    # else:
    #     theta[2] = theta3_2


    # s2 = ((DH_a[2]*np.cos(theta[2])+DH_a[1])*nnn - DH_a[2]*np.sin(theta[2])*mmm)/(DH_a[1]**2+DH_a[2]**2+2*DH_a[1]*DH_a[2]*np.cos(theta[2]))
    # c2 = (mmm+DH_a[2]*np.sin(theta[2])*s2)/(DH_a[2]*np.cos(theta[2])+DH_a[1])
    # theta[1] =np.arctan2(s2,c2)
    # print("theta2",theta[1])
    # theta[1] = np.arctan(((DH_a[2]*np.cos(theta[2])+DH_a[1])*nnn - DH_a[2]*np.sin(theta[2])*mmm)/(mmm*(DH_a[1]+DH_a[2]*np.cos(theta[2]))+nnn*DH_a[2]*np.sin(theta[2])))
    # print("theta2",theta[1])

    theta_temp_3 = np.arctan2(-np.sin(theta[5])*(nx*np.cos(theta[0])+ny*np.sin(theta[0]))
                          -np.cos(theta[5])*(ox*np.cos(theta[0])+oy*np.sin(theta[0])),
                          oz*np.cos(theta[5])+nz*np.sin(theta[5])) - theta[1] - theta[2]
    if theta_temp_3 - theta[3] > np.pi:
        theta_temp_3 -= 2*np.pi
    if theta_temp_3 - theta[3] < -np.pi:
        theta_temp_3 += 2*np.pi
    theta[3] = theta_temp_3

    


    if np.isnan(theta[0]) or np.isnan(theta[1]) or np.isnan(theta[2]) or np.isnan(theta[3]) or np.isnan(theta[4]) or np.isnan(theta[5]):
      raise ValueError("Input position is unreachable")

    return theta