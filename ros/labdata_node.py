#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as RosImage

from kimera_interfacer.msg import SyncSemantic
import cv_bridge
import cv2 as cv
import os

from sensor_msgs.msg import CameraInfo
import message_filters
import cv_bridge
import numpy as np
import tf

import time
from scipy.spatial.transform import Rotation as R

import numpy as np
from pathlib import Path


class LabDataSyncNode:
  def __init__(
    self, r_sub=2, storage_dir="/home/jonfrey/Datasets/labdata/result/asl_loop"
  ):
    rospy.init_node("labdata_node")
    Path(storage_dir + "/color").mkdir(exist_ok=True, parents=True)
    Path(storage_dir + "/pose").mkdir(exist_ok=True, parents=True)
    Path(storage_dir + "/label-filt").mkdir(exist_ok=True, parents=True)
    Path(storage_dir + "/depth").mkdir(exist_ok=True, parents=True)

    self.storage_dir = storage_dir
    image_sub = message_filters.Subscriber("/camera/rgb/image_raw", RosImage)
    depth_sub = message_filters.Subscriber(
      "/camera/depth_registered/image_raw", RosImage
    )
    sem_sub = message_filters.Subscriber("/fastscnn_node/semantic", RosImage)

    # sync_topic = rospy.get_param("~/labdata_node/sync_topic")

    sync_topic = "sync_semantic"
    self.bridge = cv_bridge.CvBridge()
    self.sync_pub = rospy.Publisher(sync_topic, SyncSemantic, queue_size=5)
    self.sync_msg = SyncSemantic()
    self.r_sub = r_sub
    ts = message_filters.ApproximateTimeSynchronizer(
      [image_sub, depth_sub, sem_sub], 50, 0.2
    )
    ts.registerCallback(self.callback)
    self.count = 0
    self.listener = tf.TransformListener()
    rospy.spin()

  def callback(self, image, depth, sem):
    st = time.time()
    sync_msg = SyncSemantic()
    sync_msg.image = image

    depth_cv2 = self.bridge.imgmsg_to_cv2(depth)
    depth_cv2 = np.array(depth_cv2).astype(np.float32)

    depth_new = (np.array(depth_cv2)).astype(np.uint16)
    print("PRE", (depth_new != 0).sum())

    depth_new[depth_new > 5000] = 0
    depth_new[depth_new < 50] = 0
    mask = np.zeros_like(depth_new)
    mask[:: self.r_sub, :: self.r_sub] = 1
    depth_new[mask == 0] = 0
    print("POST", (depth_new != 0).sum())
    # print(depth_new.sum(), depth_new.max(), depth_new.min(), depth_new.shape)
    depth = self.bridge.cv2_to_imgmsg(depth_new, encoding="16UC1")
    # print(depth.encoding, depth.header)
    # depth.encoding = "mono8"

    sync_msg.depth = depth

    sync_msg.sem = sem

    sync_msg.image.header.seq = self.count
    sync_msg.depth.header.seq = self.count

    sync_msg.sem.header.seq = self.count

    sync_msg.depth.header.stamp = sync_msg.image.header.stamp
    sync_msg.sem.header.stamp = sync_msg.image.header.stamp
    sync_msg.image.header.frame_id = "base_link_forward"
    sync_msg.depth.header.frame_id = sync_msg.image.header.frame_id
    sync_msg.sem.header.frame_id = sync_msg.image.header.frame_id

    # print("Got messages")
    # print("Publish")

    # print("IMG", image.header)
    # print("DEPTH", depth.header)
    # print("SEM", sem.header)
    time.sleep(1)
    self.sync_pub.publish(sync_msg)
    image_cv2 = self.bridge.imgmsg_to_cv2(image)
    cv.imwrite(self.storage_dir + f"/color/{self.count}.jpg", image_cv2)
    sem_cv2 = self.bridge.imgmsg_to_cv2(sem)
    cv.imwrite(self.storage_dir + f"/label-filt/{self.count}.png", sem_cv2)
    cv.imwrite(self.storage_dir + f"/depth/{self.count}.png", depth_new)

    try:
      (trans, rot) = self.listener.lookupTransform(
        "map", "base_link_forward", sync_msg.image.header.stamp
      )
      rot = R.from_quat(rot)
      H = np.eye(4)
      H[:3, :3] = rot.as_matrix()
      H[:3, 3] = trans
      np.savetxt(self.storage_dir + f"/pose/{self.count}.txt", H)

      print(trans, rot)
    except Exception as e:
      print(e)
      pass
    print("total: ", time.time() - st)
    self.count += 1


if __name__ == "__main__":
  try:
    LabDataSyncNode()
  except rospy.ROSInterruptException:
    pass
