#!/usr/bin/env python

import rospy
import sys
import random
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image

from algo import algo


# Main function of the node
def main():
    # Show the interpreter version that the node is using
    print("--version", sys.version)
    algo.start_detection("/home/n-is/tmp/test")

    rospy.init_node('AbonormalitiesDetector', anonymous=True)
    rospy.Subscriber("mavros/imu/data", Imu, cbImu)
    rospy.Subscriber("pylon_camera_node/image_raw", Image, cbImg)

    while not rospy.is_shutdown():
        continue

    print()
    print("Detection Completed")
    algo.stop_detection()


# Callback function if an Imu message was received
def cbImu(data):
    score, abnormal = algo.detect_abnormalities(data)

    info = str(score) + ', ' + str(abnormal)
    # print(info)


# Callback function if an Image message was received
def cbImg(data):

    image = data.data
    img_raw = np.frombuffer(image, dtype=np.uint8)
    # print(len(img_raw))
    algo.process_image(img_raw)


if __name__ == '__main__':
    main()
