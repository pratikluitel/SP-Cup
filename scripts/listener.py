#!/usr/bin/env python

import rospy
import time
import numpy as np
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from sensor_msgs.msg import Image
from theora_image_transport.msg import Packet

import sys
print(sys.version)

def cbImage(data):
        print(data.header.stamp)

        image = data.data

        img_raw = np.frombuffer(image, dtype=np.uint8)
        print(img_raw)

        # time.sleep(10)
        print()

def cbImu(data):
        # print(data.header.stamp)

        time.sleep(10)

        print()


def listener():

        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('listener', anonymous=True)

        rospy.Subscriber("pylon_camera_node/image_raw/theora", Packet, cbImage)
        rospy.Subscriber("mavros/imu/data", Imu, cbImu)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

if __name__ == '__main__':
        listener()