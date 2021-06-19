import torch
import numpy as np
from torchvision import transforms as tf

import torch
import torch.nn.functional as F
import pickle

__all__ = ['DeeplabHelper']


class DeeplabHelper():
    def __init__(self, device,nc=40):
            self.model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k', n_classes=182)
            self.model.eval()
            self.model.to(device)
            self.tra = tf.Resize((266,513))
            self.device = device

            with open('/home/jonfrey/ASL/cfg/dataset/mappings/coco_nyu.pkl', 'rb') as handle:
                mappings = pickle.load(handle)
            ls = [mappings['coco_id_nyu_id'][k] for k in mappings['coco_id_nyu_id'].keys() ]
            self.map_coco_nyu = torch.tensor(ls) 
            self._nc = nc

    def get_label(self, img):
        with torch.no_grad():
            # H,W,C 0-255 uint8 np
            
            H,W,C = img.shape
            img = torch.from_numpy( img.astype(np.float32) ).permute(2,0,1).to( self.device )[None]
            _, _, H, W = img.shape
            img = self.img_resize_normalize( img )
            # Image -> Probability map
            logits = self.model(img)
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)[0]
            
            
            label_coco = torch.argmax(probs, dim=0)
            sa = label_coco.shape
            label = label_coco.flatten()
            label_nyu = self.map_coco_nyu[label_coco.type(torch.int64)] # scannet to nyu40
            tra_up = tf.Resize((H,W))
            label_nyu = tra_up( label_nyu.reshape(sa)[None,:,:] )
            return label_nyu.cpu().numpy()[0]
    
    def get_label_prob(self, img):
        with torch.no_grad():
            # img: H,W,C 0-255 uint8 np
            # return: 41, H, W
            
            H,W,C = img.shape
            img = torch.from_numpy( img.astype(np.float32) ).permute(2,0,1).to( self.device )[None]
            _, _, H, W = img.shape
            img = self.img_resize_normalize( img )
            # Image -> Probability map
            logits = self.model(img)
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)[0]
            
            label_nyu = torch.zeros( self._nc+1,probs.shape[1],probs.shape[2], device=probs.device )
            for i in range( probs.shape[0] ) :
               label_nyu[ self.map_coco_nyu[i]+1 ] += probs[i]
            
            label_nyu = label_nyu / label_nyu.sum( dim=0 )
            # tra_up = tf.Resize((H,W))
            # label_nyu = tra_up( label_nyu.reshape(sa)[None,:,:] )
            return label_nyu.cpu().numpy()


    def img_resize_normalize(self, img):
        # Resize and normalize
        img = self.tra(img)
        img[:,0,:,:] -= 122.675
        img[:,1,:,:] -= 116.669
        img[:,2,:,:] -= 104.008
        return img

if __name__ == "__main__":
    import sys
    import os
    sys.path.append("""/home/jonfrey/ASL/src""")
    os.chdir("/home/jonfrey/ASL")

    import imageio 
    i1 = imageio.imread( "/home/jonfrey/datasets/scannet/scans/scene0033_00/color/500.jpg" )
    yh = DeeplabHelper(device="cuda:0")
    label = yh.get_label_prob( i1 )
    
    from visu import Visualizer
    visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=False, num_classes=41)
    visu.plot_segmentation(seg=label+1,jupyter=True, method='right')
    visu.plot_image(img=i1,jupyter=True, method='left')
