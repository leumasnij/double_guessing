
import rospy
from robot import Robot


#center of table
table_center = [0, 0.55, 0.25]

#initialize ROS node
rospy.init_node('sample_robot_move', anonymous=True)

#initialize robot object
robot = Robot()

#move the robot to home position to start with
#override is optional argument and if not provided/set as False,
#   it will ask for y/n confirmation before executing the movement
#robot.joint_angle_homing(override=True)

print("moving to home position")


#move to another location
robot.move_to_location(	loc_x=table_center[0], loc_y=table_center[1], loc_z=table_center[2]+0.1, override=True)