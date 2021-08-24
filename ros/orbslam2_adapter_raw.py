import yaml
import geometry_msgs
import sensor_msgs
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage as RosCompressedImage
from geometry_msgs.msg import Pose as RosPose
import rosbag
import rospy
import cv2
from sensor_msgs.msg import CameraInfo as RosCameraInfo
from sensor_msgs.msg import PointCloud2 as PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from PIL import Image
import copy
import sensor_msgs.point_cloud2
import time


class OrbslamRosbagAdapter:
  def __init__(self):
    rospy.init_node("orbslam_rosbag_adapter")
    self.bridge = CvBridge()
    self.pub_img = rospy.Publisher("/camera/rgb/image_raw", RosImage, queue_size=10)
    self.pub_depth = rospy.Publisher(
      "/camera/depth_registered/image_raw", RosImage, queue_size=10
    )
    self.pub_camera_info = rospy.Publisher(
      "/mono/camera_info", RosCameraInfo, queue_size=10
    )

    rospy.Subscriber("/rgb/camera_info", RosCameraInfo, self.callback_rgb_camera_info)
    self.map1, self.map2 = None, None
    self.camera_info = None
    self.frame = "pickelhaubergb_camera_link"
    rospy.Subscriber("/rgb/image_raw", RosImage, self.callback_rgb)
    rospy.Subscriber(
      "/depth_to_rgb/image_raw/compressedDepth", RosCompressedImage, self.callback_depth
    )

    self.r = rospy.Rate(20)

  def run(self):
    self.r.sleep()

  def callback_rgb(self, data):
    print("RGB")
    rgb = self.bridge.imgmsg_to_cv2(data)
    if not self.map1 is None:
      rgb_undistorted = cv2.remap(
        rgb,
        self.map1,
        self.map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
      )
      rgb_undistorted = cv2.resize(
        rgb_undistorted, (640, 480), interpolation=cv2.INTER_LINEAR
      )

      mono_msg = self.bridge.cv2_to_imgmsg(rgb_undistorted)
      mono_msg.header = data.header
      mono_msg.header.frame_id = self.frame

      self.pub_img.publish(mono_msg)
    else:
      print("Map not initalized")

  def callback_depth(self, data):
    print("DEPTH")
    msg = data
    depth_fmt, compr_type = msg.format.split(";")
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if compr_type != "compressedDepth":
      raise Exception(
        "Compression type is not 'compressedDepth'."
        "You probably subscribed to the wrong topic."
      )

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(
      np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED
    )
    if depth_img_raw is None:
      # probably wrong header size
      raise Exception(
        "Could not decode compressed depth image."
        "You may need to change 'depth_header_size'!"
      )

    if depth_fmt == "16UC1":
      # write raw image data
      cv2.imwrite(
        os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png"),
        depth_img_raw,
      )
    elif depth_fmt == "32FC1":
      raw_header = msg.data[:depth_header_size]
      # header: int, float, float
      [compfmt, depthQuantA, depthQuantB] = struct.unpack("iff", raw_header)
      depth_img_scaled = depthQuantA / (depth_img_raw.astype(np.float32) - depthQuantB)
      # filter max values
      depth_img_scaled[depth_img_raw == 0] = 0

      # depth_img_scaled provides distance in meters as f32
      # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
      depth_img_mm = (depth_img_scaled * 1000).astype(np.uint16)
    else:
      raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")

    # buf = np.ndarray(shape=(1, len(data.data)-12),
    #    dtype=np.uint8, buffer=data.data[12:])

    # im = cv2.imdecode(buf,  cv2.IMREAD_ANYDEPTH    )

    if not self.map1 is None:
      rgb_undistorted = cv2.remap(
        depth_img_mm,
        self.map1,
        self.map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
      )

      rgb_undistorted = rgb_undistorted  # depth factor is 1000
      rgb_undistorted[rgb_undistorted < 300] = 0  # cut off at 30 cm
      rgb_undistorted[rgb_undistorted > 25000] = 0
      rgb_undistorted = rgb_undistorted.astype(np.uint16)

      rgb_undistorted = cv2.resize(
        rgb_undistorted, (640, 480), interpolation=cv2.INTER_LINEAR
      )
      mono_msg = self.bridge.cv2_to_imgmsg(rgb_undistorted)
      mono_msg.header = data.header
      mono_msg.header.frame_id = self.frame
      mono_msg.encoding = "mono16"

      self.pub_depth.publish(mono_msg)
    else:
      print("Map not initalized")

  def callback_rgb_camera_info(self, data):
    if self.camera_info is None:
      self.camera_info = copy.deepcopy(data)
      h, w = data.height, data.width
      K = np.array(data.K).reshape(3, 3)
      D = np.array(data.D)

      new_K, validPixROI = cv2.getOptimalNewCameraMatrix(K, D[None, :], (w, h), alpha=0)
      self.map1, self.map2 = cv2.initUndistortRectifyMap(
        K, D[None, :], np.eye(3), new_K, (w, h), cv2.CV_16SC2
      )
      self.camera_info.K = new_K.reshape(-1)
      D.fill(0)
      self.camera_info.D = D
      self.new_K = self.camera_info.K.reshape(3, 3)

      self.new_K[0, 0] = self.new_K[0, 0] / (1280 / 640)
      self.new_K[0, 2] = self.new_K[0, 2] / (1280 / 640)

      self.new_K[1, 1] = self.new_K[1, 1] / (720 / 480)
      self.new_K[1, 2] = self.new_K[1, 2] / (720 / 480)

      print("Unrectified Kamera Intrinsics:", self.new_K)

    self.camera_info.header = data.header
    self.pub_camera_info.publish(self.camera_info)


if __name__ == "__main__":
  ora = OrbslamRosbagAdapter()
  print("OrbslamRosbagAdapter: Started Listening to /rgb/image_raw and /points2")
  while not rospy.is_shutdown():
    try:
      ora.run()
    except rospy.exceptions.ROSTimeMovedBackwardsException as e:
      print("Ignore time moved backwards!")
