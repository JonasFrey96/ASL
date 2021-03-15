import sys
import os
os.chdir("/home/jonfrey/ASL")
sys.path.append("""/home/jonfrey/ASL/src/""")
sys.path.append("""/home/jonfrey/ASL/src/pseudo_label""")

from pseudo_label.yolo import YoloHelper
from pseudo_label.deeplab import DeeplabHelper
from pseudo_label.fast_scnn import FastSCNNHelper

import numpy as np
from visu import Visualizer
import imageio

visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=False, num_classes=41)

yh = YoloHelper()
dlh = DeeplabHelper(device="cuda:1")
fsh = FastSCNNHelper(device='cuda:1')