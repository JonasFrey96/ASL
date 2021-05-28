if __name__ == '__main__':
    import os
    import sys 
    os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
    sys.path.insert(0, os.path.join(os.getenv('HOME'), 'ASL'))
    sys.path.append(os.path.join(os.path.join(os.getenv('HOME'), 'ASL') + '/src'))

import os
import sys 
import numpy as np
import torch
from pseudo_label import FastSCNNHelper
from pseudo_label import FastSCNNHelperTorch
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"src"))
from PIL import Image
import argparse
from visu import Visualizer    
from pseudo_label import readImage
from pathlib import Path
from utils_asl import load_yaml
from torchvision import transforms as tf

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils_asl import label_to_png

class FastDataset(Dataset):
  def __init__(self, paths):
    self.paths = paths
    self.output_transform = tf.Compose([
      tf.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
  def __len__(self):
    return len(self.paths)
  
  def __getitem__(self,index):
    p = self.paths[index] 
    i1 = readImage(p, H=640, W=1280, scale=False)
    h,w,c = i1.shape
    img = torch.nn.functional.interpolate(torch.from_numpy(i1.astype(np.float32))[None].permute(0,3,1,2), (320,640), mode='bilinear')[0].permute(1,2,0)
    img = ( img/255 ).permute(2,0,1)[None]
    img = self.output_transform( img )
    return img, torch.tensor( index )


def label_generation(**kwargs):
    # idea:
    # load model .ckpt
    # set scenes to pred
    # store as png to fil 
    env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
    env = load_yaml(env_cfg_path)
    idfs = str( kwargs["identifier"])
    confidence = kwargs["confidence"]
    scenes = kwargs["scenes"]

    exp = kwargs["exp"]


    if os.environ['ENV_WORKSTATION_NAME'] == 'ptarmigan':
      base = os.path.join(env['scannet'], 'scans')
      scratchdir = "/media/scratch1/jonfrey/labels_generated"
    elif os.environ['ENV_WORKSTATION_NAME'] == 'ws': 
      base = os.path.join(env['scannet'], 'scans')
      scratchdir = "/home/jonfrey/Datasets/labels_generated"
    else:      
      scratchdir = os.getenv('TMPDIR')
      base = os.path.join(scratchdir, 'scannet', 'scans')


    fsh = FastSCNNHelperTorch(device='cuda:0', exp=exp)
    export = os.path.join(scratchdir, idfs )
    paths = [str(s) for s in Path(base).rglob('*.jpg') if str(s).find("color") != -1]
    # filter to get evey 10 image
    paths = [s for s in paths if int(s.split('/')[-1][:-4]) % 10 == 0]

    # filter only scenes of interrest
    pa = []
    for scene in scenes:
      pa += [s for s in paths if s.find(scene) != -1]

    dataset = FastDataset( pa)
    dataloader = DataLoader(dataset,
      shuffle = False,
      num_workers = 0,
      pin_memory = False,
      batch_size = 1)
    
    h,w,_= readImage(pa[0], H=640, W=1280, scale=False).shape
    for j, batch in enumerate( dataloader ):
        print(f"Progress: {j}/{len(dataloader)}")
        img = batch[0].to('cuda:0')[0]
        index = int(batch[1])

        label_probs = fsh.get_label_prob( img )
        label_probs = torch.nn.functional.interpolate( label_probs.type(torch.float32)[None] , (h,w), mode='bilinear')[0]
        l = label_probs.cpu()
        l = l.permute( (1,2,0) ).numpy()

        outpath = pa[index]
        outpath = outpath.replace("color", idfs)[:-4]+'.png'
        res = outpath[outpath.find('scans/'):]
        res = os.path.join(export, res)
        Path(res).parent.mkdir(exist_ok=True, parents= True)
        label_to_png( l[:,:,1:], res )
        
        # label = np.uint8( torch.argmax( label_probs, axis=0 ).cpu().numpy() )
        # mask = (torch.max( label_probs, axis=0 ).values < confidence).cpu().numpy()
        # label[ mask ] = 0 # set labels with little confidence to 0
        # Image.fromarray( label ).save(res)

    os.system( f"cd {scratchdir} && tar -cvf {scratchdir}/{idfs}.tar {idfs}" )

    
    if not env['workstation']:
      os.system(f"cd {scratchdir} && mv {idfs}.tar $DATASETS")
    else:
      print(f"On cluster execute: ", "cd {scratchdir} && mv {idfs}.tar $DATASETS")


def test():
  label_generation(
     activate = True, identifier ="scannet_retrain50",
     confidence = 0,
     scenes = ['scene0000'],
     exp = load_yaml("/home/jonfrey/ASL/cfg/test/test.yml")
  )

if __name__ == '__main__':
  test()
  