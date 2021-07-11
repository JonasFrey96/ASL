#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as RosImage
from kimera_interfacer.msg import SyncSemantic
import cv_bridge
import cv2 as cv
import os

import numpy as np
from sensor_msgs.msg import CameraInfo
import imageio
import tf_conversions
import tf2_ros
import geometry_msgs.msg
import rospkg
from time import sleep

import message_filters


class LabDataSyncNode:
  def __init__(self):
    rospy.init_node("labdata_node")

    image_sub = message_filters.Subscriber("image", RosImage)
    depth_sub = message_filters.Subscriber("depth", RosImage)
    seg_sub = message_filters.Subscriber("seg", RosImage)
    sync_topic = rospy.get_param("~/labdata_node/sync_topic")
    self.sync_pub = rospy.Publisher(sync_topic, SyncSemantic, queue_size=1)
    self.sync_msg = SyncSemantic()

    ts = message_filters.TimeSynchronizer([image_sub, depth_sub, seg_sub], 100)
    ts.registerCallback(self.callback)

    rospy.spin()

  def callback(self, image, depth, seg):
    self.sync_msg.image = image
    self.sync_msg.depth = depth
    self.sync_msg.seg = seg

    self.sync_pub.publish(self.sync_msg)


if __name__ == "__main__":
  try:
    LabDataSyncNode()
  except rospy.ROSInterruptException:
    pass
