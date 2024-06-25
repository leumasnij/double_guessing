#!/usr/bin/env python3

import rospy

def main():
    print("Hello World!")
    rospy.sleep(1)
    rospy.init_node('test_node')
    # rospy.loginfo("Test node has been initialized")
    # rospy.spin()
    print("Goodbye World!")

if __name__ == '__main__':
    main()

