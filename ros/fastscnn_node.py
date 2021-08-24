#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as RosImage

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

from torchvision import transforms


class FastSCNNNode:
  def __init__(self, device="cpu"):
    """
    The node is currently a minimal implementation without any configs.

    TODO: ADD config.
    Integrate CUDA support.
    """
    rospy.init_node("fastscnn_node")
    seg_topic = "/fastscnn_node/semantic"  # rospy.get_param("~/scannet_node/seg_topic")
    self.model = FastSCNN(
      num_classes=40, extraction={"active": False, "layer": "learn_to_down"}
    )
    self.device = device
    self.model.eval()
    self.model.to(device)

    p = os.path.join(
      "/home/jonfrey/Results",
      "scannet25k_24h_lr_decay_from_scratch/2021-06-05T14:36:26_scannet25k_24h_lr_decay_from_scratch/task0-epoch=64--step=158340.ckpt",
    )
    if os.path.isfile(p):
      res = torch.load(p, map_location=lambda storage, loc: storage)
      new_statedict = {}
      for k in res["state_dict"].keys():
        if k.find("model.") != -1:
          new_statedict[k[6:]] = res["state_dict"][k]
      res = self.model.load_state_dict(new_statedict, strict=True)
      print("Restoring weights: " + str(res))
    else:
      raise Exception("Checkpoint not a file")

    self.bridge = cv_bridge.CvBridge()
    self.seg_pub = rospy.Publisher(seg_topic, RosImage, queue_size=1)
    self.debug_pub = rospy.Publisher("debug", RosImage, queue_size=1)

    rospack = rospkg.RosPack()
    kimera_interfacer_path = rospack.get_path("kimera_interfacer")
    mapping = np.genfromtxt(
      f"{kimera_interfacer_path}/cfg/nyu40_segmentation_mapping.csv", delimiter=","
    )
    self.rgb = torch.from_numpy(mapping[2:, 1:4]).type(torch.float32)

    undistorted_topic = "camera/rgb/image_raw"
    rospy.Subscriber(undistorted_topic, RosImage, self.callback)
    print(self.rgb)
    rospy.spin()

  def callback(self, image):
    img = self.bridge.imgmsg_to_cv2(image)
    img = np.array(img)
    img = torch.from_numpy(img[:, :, :3].astype(np.float32))[None].permute(0, 3, 1, 2)
    with torch.no_grad():
      img = img.to(self.device)
      img = torch.nn.functional.interpolate(img, (320, 640), mode="bilinear") / 255
      tra = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      out = self.model(tra(img))
      label = out[0]

      label = torch.nn.functional.interpolate(label, (480, 640), mode="bilinear")

      sem = torch.argmax(label, dim=1)[0]
      sem_new = torch.zeros((sem.shape[0], sem.shape[1], 3), device=self.device)
      cand = torch.unique(sem)
      for i in cand.type(torch.uint8).tolist():
        m = sem == int(i)
        sem_new[m, :] = self.rgb[int(i)]

      sem_new = np.uint8(sem_new.cpu())

    sem_new = self.bridge.cv2_to_imgmsg(sem_new, encoding="rgb8")
    sem_new.header = image.header
    sem_new.header.frame_id = "base_link"
    self.seg_pub.publish(sem_new)

    # sem_new = self.bridge.cv2_to_imgmsg(
    #   np.uint8(sem.cpu()[:, :, None].repeat(1, 1, 3) * 4), encoding="rgb8"
    # )
    # sem_new.header = image.header
    # sem_new.header.frame_id = "base_link"
    # self.debug_pub.publish(sem_new)


if __name__ == "__main__":
  try:
    FastSCNNNode()
  except rospy.ROSInterruptException:
    pass
