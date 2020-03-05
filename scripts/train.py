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
    algo.start_train()

    rospy.init_node('AbonormalitiesTrainer', anonymous=True)
    rospy.Subscriber("mavros/imu/data", Imu, cbImu)

    while not rospy.is_shutdown():
        continue

    algo.stop_train("/home/n-is/tmp/test")


# Callback function if an Imu message was received
def cbImu(data):
    algo.train(data)


def train(data):
    pass

if __name__ == '__main__':
    main()
