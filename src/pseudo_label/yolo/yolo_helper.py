import numpy as np
import pickle
import torch

__all__ = ['YoloHelper']

class YoloHelper():
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        with open('cfg/dataset/mappings/coco2017_nyu.pkl', 'rb') as handle:
            self.mappings = pickle.load(handle)
        
    def get_label(self, img):
        # H,W,C 0-255 uint8 np
        H,W,C = img.shape
        results = self.model( img )
        label = np.zeros( (H,W) )
        label += -1
        for k in range( 1, results.xyxy[0].shape[0]+1):
            x1 = int( results.xyxy[0][-k][0])
            y1 = int( results.xyxy[0][-k][1])
            x2 = int( results.xyxy[0][-k][2])
            y2 = int( results.xyxy[0][-k][3])
            confidence = results.xyxy[0][-k][4]      
            coco_class = int( results.xyxy[0][-k][5])
            label[ int(y1):int(y2), int(x1):int(x2) ] =  self.mappings['coco2017_id_nyu_id'][coco_class]      
        return label

if __name__ == "__main__":
    import sys
    import os
    sys.path.append("""/home/jonfrey/ASL/src""")
    os.chdir("/home/jonfrey/ASL")

    import imageio 
    i1 = imageio.imread( "/home/jonfrey/datasets/scannet/scans/scene0033_00/color/500.jpg" )
    yh = YoloHelper()
    label = yh.get_label( i1 )
    from visu import Visualizer
    visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=False, num_classes=41)
    visu.plot_segmentation(seg=label+1,jupyter=True, method='right')
    visu.plot_image(img=i1,jupyter=True, method='left')