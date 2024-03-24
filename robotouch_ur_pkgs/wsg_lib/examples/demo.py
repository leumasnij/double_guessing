import rospy
from wsg_lib.grippers import WSG50
import time

#NOTE: Do not use the MOVE command to grip or release parts. Use the GRIP and RELEASE command instead

if __name__ == "__main__":
    gripper= WSG50()
    #initial homing
    gripper.homing()
    print("home")

    #move to position
    gripper.move(20)

    #move to position at specified velocity
    gripper.move(70,20)

    #open the gripper
    gripper.open()

    #grip with default parameters ( force = 10, position = 0, velocity = 50mm)
    gripper.grip()
    time.sleep(5)
    #release the grip
    #NOTE: Do not use the MOVE command to grip or release parts. Use the GRIP and RELEASE command instead
    gripper.release()
    
    #open
    gripper.open()

    #grip at different force and speed
    gripper.grip(force = 25, position=0, velocity = 100)
    time.sleep(5)
    #release object, with custon opening width
    gripper.release(distance = 10)

    gripper.open()

    #disconenct when done
    gripper.bye()
    
