import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.insert(0, os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.append(os.path.join(os.path.join(os.getenv('HOME'), 'ASL') + '/src'))

import yaml
import coloredlogs
coloredlogs.install()
import time
import argparse

from pathlib import Path
import gc

# Frameworks
import torch
import numpy as np
import imageio
# Costume Modules
from datasets_asl import get_dataset
from torchvision import transforms
from torchvision import transforms as tf
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

DEVICE = 'cuda:1'

def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)

def load_yaml(path):
  with open(path) as file:  
    res = yaml.load(file, Loader=yaml.FullLoader) 
  return res

def img_resize_normalize(images):
  # Resize
  tra = tf.Resize((266,513))

  images = tra(images)*255
  
  images[:,0,:,:] -= 122.675
  images[:,1,:,:] -= 116.669
  images[:,2,:,:] -= 104.008
  return images

def inference(model, image, raw_image=None, postprocessor=None):
  _, _, H, W = image.shape
  # Image -> Probability map
  logits = model(image)
  logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
  probs = F.softmax(logits, dim=1)[0]
  labelmap = torch.argmax(probs, dim=0)
  return labelmap


with open('/home/jonfrey/ASL/cfg/dataset/mappings/coco_nyu.pkl', 'rb') as handle:
    mappings = pickle.load(handle)
print( mappings.keys() )
ls = [mappings['coco_id_nyu_id'][k] for k in mappings['coco_id_nyu_id'].keys() ]
map_coco_nyu = torch.tensor(ls) 

if __name__ == "__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument('--eval', type=file_path, default="/home/jonfrey/ASL/cfg/eval/eval.yml",
                      help='Yaml containing dataloader config')
  args = parser.parse_args()
  env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
  env_cfg = load_yaml(env_cfg_path)	
  eval_cfg = load_yaml(args.eval)

  # SETUP MODEL
  torch.set_grad_enabled(False)
  model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k', n_classes=182)
  model.eval()
  model.to(DEVICE)

  # SETUP DATALOADER
  dataset_test = get_dataset(
    **eval_cfg['dataset'],
    env = env_cfg,
    output_trafo = None,
    )
  dataloader_test = torch.utils.data.DataLoader(dataset_test,
    shuffle = False,
    num_workers = 0,
    pin_memory = eval_cfg['loader']['pin_memory'],
    batch_size = 1, 
    drop_last = True)

  # CREATE RESULT FOLDER
  base = os.path.join(env_cfg['base'], eval_cfg['name'], eval_cfg['dataset']['name'])
  globale_idx_to_image_path = dataset_test.image_pths
  
  tra_up = tf.Resize(eval_cfg['dataset']['output_size'], Image.NEAREST)
  
  st = time.time()

  # START EVALUATION
  for j, batch in enumerate( dataloader_test ):
    print(j)
    images = batch[0].to(DEVICE)
    target = batch[1].to(DEVICE)
    ori_img = batch[2].to(DEVICE)
    replayed = batch[3].to(DEVICE)
    BS = images.shape[0]
    global_idx = batch[4] 
        
    images = img_resize_normalize(images)
    pred_coco = inference(model, images)
    sa = pred_coco.shape
    label = pred_coco.flatten()
    label = map_coco_nyu[label.type(torch.int64)] # scannet to nyu40
    prediction = tra_up( label.reshape(sa)[None,:,:] )
    

    pred_image = np.uint8(prediction.detach().cpu().numpy() + 1)
    target_image = np.uint8(target.detach().cpu().numpy() + 1 )
    # stored as uint8 png -> 0 == invalid 1 == wall , 40 == other prob !!!
    valid_image = np.uint8(target.detach().cpu().numpy() == -1)

    img = np.stack( [pred_image, target_image, valid_image], axis=1)
    for b in range(BS):
      img_path = globale_idx_to_image_path[global_idx[b]]
      
      p = os.path.join(base,
        img_path.split('/')[-3],
        'segmentation_deeblab_v3',
        img_path.split('/')[-1][:-4]+'.png')
      print(j, '  ', p)
      Path(p).parent.mkdir(parents=True, exist_ok=True)
      imageio.imwrite( p, np.moveaxis( img[b], [0,1,2], [2,0,1] ) )


    
