import os
import sys 
from torchvision import transforms as tf
import numpy as np
import pickle
import torch

from utils_asl import file_path, load_yaml
from models_asl import FastSCNN

__all__ = ['FastSCNNHelper']
class FastSCNNHelper():
    def __init__(self, device, eval_cfg_path="cfg/eval/eval.yml"):
        env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
        env_cfg = load_yaml(env_cfg_path)	
        eval_cfg = load_yaml(eval_cfg_path)
        self.device= device
        self.model = FastSCNN(**eval_cfg['model']['cfg'])
        
        p = os.path.join( env_cfg['base'], eval_cfg['checkpoint_load'])
        if os.path.isfile( p ):
            res = torch.load(p)
            new_statedict = {}
            for k in res['state_dict'].keys():
                if k.find('model.') != -1: 
                    new_statedict[ k[6:]] = res['state_dict'][k]
            res = self.model.load_state_dict( new_statedict, strict=True)
            print('Restoring weights: ' + str(res))
        else:
            raise Exception('Checkpoint not a file')
        self.model.to(device)
        self.model.eval()

        self.output_transform = tf.Compose([
          tf.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        
    def get_label(self, img):
        with torch.no_grad():
            # H,W,C 0-255 uint8 np
            img = (torch.from_numpy( img.astype(np.float32)/255).permute(2,0,1)[None]).to(self.device)
            img = self.output_transform( img )
            outputs = self.model( img )
            return torch.argmax(outputs[0], 1).cpu().numpy()[0]


if __name__ == '__main__':
    from pseudo_label import readImage 
    i1 = readImage("/home/jonfrey/datasets/scannet/scans/scene0033_00/color/500.jpg", H=640, W=1280, scale=True)
    fsh = FastSCNNHelper(device='cuda:1')
    label = fsh.get_label( i1 )
    from visu import Visualizer
    visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=False, num_classes=41)
    visu.plot_segmentation(seg=label+1,jupyter=True, method='right')
    visu.plot_image(img=i1,jupyter=True, method='left')