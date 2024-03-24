import rospy
from ur_lib.robots import UR5E


TABLE_CENTER = [-0.1, 0.5, 0.25]

rospy.init_node('main', anonymous=True)
robot = UR5E()

#robot.joint_angle_homing()


#robot.move_to_location(
#        TABLE_CENTER[0], TABLE_CENTER[1], TABLE_CENTER[2],
#        ask_confirmation=False)

#set low velocity
#robot.set_scale_factor()
robot.move_offset_distance(0, 0, -0.05, False)

#robot.joint_angle_horizontal_homing()
