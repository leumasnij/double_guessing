import rospy
from wsg_lib.grippers import WSG50
import time

#NOTE: Do not use the MOVE command to grip or release parts. Use the GRIP and RELEASE command instead

if __name__ == "__main__":
    gripper = WSG50()
    #initial homing
    gripper.homing()
    
    gripper.grip(force = 5, position=0, velocity = 50)
    time.sleep(20)
    #release
    gripper.release()
    
