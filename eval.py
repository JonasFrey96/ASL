import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))

import coloredlogs
coloredlogs.install()

import time
import shutil
import datetime
import argparse
import signal
import yaml
import logging
from pathlib import Path
import gc

# Frameworks
import torch
import numpy as np
import imageio
# Costume Modules
from utils import file_path, load_yaml

from models import FastSCNN
from datasets import get_dataset
from torchvision import transforms
from torchvision import transforms as tf


DEVICE = 'cuda:1'

if __name__ == "__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument('--eval', type=file_path, default="cfg/eval/eval.yml",
                      help='Yaml containing dataloader config')
  
  args = parser.parse_args()
  env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
  env_cfg = load_yaml(env_cfg_path)	
  eval_cfg = load_yaml(args.eval)

  # SETUP MODEL
  model = FastSCNN(**eval_cfg['model']['cfg'])
  p = os.path.join( env_cfg['base'], eval_cfg['checkpoint_load'])
  if os.path.isfile( p ):
    res = model.load_state_dict( torch.load(p,
      map_location=lambda storage, loc: storage)['state_dict'], 
      strict=False)
    print('Restoring weights: ' + str(res))
  else:
    raise Exception('Checkpoint not a file')
  model.to(DEVICE)
  model.eval()
    
  # SETUP DATALOADER
  output_transform = transforms.Compose([
          transforms.Normalize([.485, .456, .406], [.229, .224, .225])
  ])

  dataset_test = get_dataset(
    **eval_cfg['dataset'],
    env = env_cfg,
    output_trafo = output_transform,
  )
  dataloader_test = torch.utils.data.DataLoader(dataset_test,
    shuffle = False,
    num_workers =eval_cfg['loader']['num_workers'],
    pin_memory = eval_cfg['loader']['pin_memory'],
    batch_size = eval_cfg['loader']['batch_size'], 
    drop_last = True)
  
  # CREATE RESULT FOLDER
  base = os.path.join(env_cfg['base'], eval_cfg['name'], eval_cfg['dataset']['name'])
  
  globale_idx_to_image_path = dataset_test.image_pths

  # START EVALUATION
  for j, batch in enumerate( dataloader_test ):
    images = batch[0].to(DEVICE)
    target = batch[1].to(DEVICE)
    ori_img = batch[2].to(DEVICE)
    replayed = batch[3].to(DEVICE)
    BS = images.shape[0]
    global_idx = batch[4] 

    outputs = model(images) # C,H,W
    prediction = torch.argmax(outputs[0], 1)

    pred_image = np.uint8(prediction.detach().cpu().numpy() + 1)
    target_image = np.uint8(target.detach().cpu().numpy() + 1 )
    # stored as uint8 png -> 0 == invalid 1 == wall , 40 == other prob !!!
    valid_image = np.uint8(target.detach().cpu().numpy() == -1)

    
    img = np.stack( [pred_image, target_image, valid_image], axis=1)
    for b in range(BS):
      img_path = globale_idx_to_image_path[global_idx[b]]

      p = os.path.join(base,
        *img_path.split('/')[-3:-1], 
        img_path.split('/')[-1][:-4]+'.png')
      Path(p).parent.mkdir(parents=True, exist_ok=True)
      imageio.imwrite( p, np.moveaxis( img[b], [0,1,2], [2,0,1] ) )


    
