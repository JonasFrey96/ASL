


from cv_bridge import CvBridge
import os
import rosbag
from sensor_msgs.msg import Image
import cv2
import numpy as np

output ='/media/scratch2/jonfrey/datasets/labdata'
bag = rosbag.Bag('/media/scratch2/jonfrey/datasets/labdata/output_file2.bag', 'r')
bridge = CvBridge()
topics = ['/versavis/cam0/image_raw', '/versavis/cam1/image_raw', '/versavis/cam2/image_raw']
for k, t in enumerate(topics):
    output_dir = output+'/'+ str( k )
    try:
        os.mkdir(output_dir)
    except:
        print("FAILED", output, output_dir)
        pass
    j = 0
    for topic, msg, to  in bag.read_messages(topics=[t]):
        
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % j), cv_img)
        
        j += 1
        
        if j% 100 == 0:
            print(j,t)
