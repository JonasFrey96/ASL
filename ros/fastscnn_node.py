#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as RosImage

sys.path.append(os.path.join(os.getcwd() + "/src"))

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

import torch

import sys
import os
import cv_bridge
import cv2 as cv

sys.path.append(os.path.join(os.getcwd() + "/src"))
from models_asl import FastSCNN


class FastSCNNNode:
  def __init__(self):
    rospy.init_node("fastscnn_node")
    seg_topic = rospy.get_param("~/scannet_node/seg_topic")
    self.fast_scnn
    self.model = FastSCNN(num_classes=40)
    self.bridge = cv_bridge.CvBridge()
    self.seg_pub = rospy.Publisher(seg_topic, RosImage, queue_size=1)

    rospack = rospkg.RosPack()
    kimera_interfacer_path = rospack.get_path("kimera_interfacer")
    mapping = np.genfromtxt(
      f"{kimera_interfacer_path}/cfg/nyu40_segmentation_mapping.csv", delimiter=","
    )
    self.rgb = mapping[1:, 1:4]
    rospy.spin()

  def callback(self, image):
    img = self.bridge.imgmsg_to_cv2(image)
    img = np.array(img)
    img = torch.from_numpy(img)[None]
    out = self.model(img)
    label = out[0]
    sem = torch.argmax(label, dim=2)
    sem_new = np.zeros((sem.shape[0], sem.shape[1], 3))
    for i in range(0, 41):
      sem_new[sem == i, :3] = self.rgb[i]
    sem_new = np.uint8(sem_new)

    sem_new = self.bridge.cv2_to_imgmsg(sem_new, encoding="rgb8")

    sem_new.header = image.header
    sem_new.header.frame_id = "base_link"  # base_link_gt
    self.sync_pub.publish(self.sync_msg)


if __name__ == "__main__":
  try:
    FastSCNNNode()
  except rospy.ROSInterruptException:
    pass
