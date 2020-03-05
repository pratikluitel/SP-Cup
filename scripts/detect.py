#!/usr/bin/env python

import rospy
import sys
import random

from std_msgs.msg import String
from sensor_msgs.msg import Imu

from algo import algo


# Main function of the node
def main():
    # Show the interpreter version that the node is using
    print("--version", sys.version)
    algo.start_detection("/home/n-is/tmp/test")

    rospy.init_node('AbonormalitiesDetector', anonymous=True)
    rospy.Subscriber("mavros/imu/data", Imu, cbImu)

    while not rospy.is_shutdown():
        continue

    print("Detection Completed")
    algo.stop_detection()


# Callback function if an Imu message was received
def cbImu(data):
    score, abnormal = algo.detect_abnormalities(data)

    info = str(score) + ', ' + str(abnormal)
    # print(info)


if __name__ == '__main__':
    main()
