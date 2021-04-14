import os
import sys
import yaml
from pathlib import Path

ASL = os.path.join( str(Path.home()), "ASL" )
src = os.path.join( str(Path.home()), "ASL", "src" )
sys.path.append( "/home/jonfrey/RPOSE")
sys.path.append('/home/jonfrey/RPOSE/core')
sys.path.append( ASL )
sys.path.append( src )

from utils_asl import load_yaml
from task import TaskCreator

DEVICE = "cuda:0"
name = os.getenv('ENV_WORKSTATION_NAME')
env_cfg_path =os.path.join( ASL, f"cfg/env/{name}.yml")  
exp_cfg_path =os.path.join( ASL, "cfg/exp/debug.yml")
eva_cfg_path =os.path.join( ASL, "cfg/eval/eval.yml")

env = load_yaml(env_cfg_path)

use_eva = True
if use_eva:
  exp = load_yaml(eva_cfg_path)
else:
  exp = load_yaml(exp_cfg_path)

# SETUP DATALOADERS
from task import TaskCreator
from datasets_asl import get_dataset
from math import ceil
import torch
tc = TaskCreator(**exp['task_generator'] )


use_tc = True
if use_tc:
  cfgs = []
  for idx, out in enumerate(tc):
    task, eval_lists = out
    cfgs.append(  task.dataset_train_cfg )
else:
  cfgs  = [exp['dataset']]

for cfg in cfgs:

  dataset= get_dataset(
    **cfg,
    env = env,
    output_trafo = None,
  )
  dataloader = torch.utils.data.DataLoader(dataset,
    shuffle = False,
    num_workers = exp['loader']['num_workers'],
    pin_memory = exp['loader']['pin_memory'],
    batch_size = exp['loader']['batch_size'], 
    drop_last = True)

  print( len( dataset ) )
  # dataloader_list_test = eval_lists_into_dataloaders(eval_lists, env=env, exp=exp)
  print( f'<<<<<<<<<<<< All Datasets are loaded and set up >>>>>>>>>>>>>' )

  # SETUP RAFT
  import time
  from raft import RAFT
  import torch
  import numpy as np
  import cv2
  from datasets_asl import ScanNet, LabData
  def writeFlowKITTI(filename, uv):
      uv = 64.0 * uv + 2**15
      valid = np.ones([uv.shape[0], uv.shape[1], 1])
      uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
      cv2.imwrite(filename, uv[..., ::-1])

  di = {
      'model': '/home/jonfrey/RPOSE/models/raft-things.pth',
      'small': False,
      'mixed_precision': False,
      'alternate_corr': False,
    }    
  class DotDict(dict):
      """dot.notation access to dictionary attributes"""
      __getattr__ = dict.get
      __setattr__ = dict.__setitem__
      __delattr__ = dict.__delitem__
    
  args = DotDict(di)
  model = torch.nn.DataParallel(RAFT(args))
  model.load_state_dict(torch.load(args.model))
  model = model.module
  model.to(DEVICE)
  model.eval()

  global_to_local_idx = dataloader.dataset.global_to_local_idx
  global_to_image_pths = dataloader.dataset.image_pths
  sub = 10
  with torch.no_grad():		
    for j, batch in enumerate( dataloader ):
      if j % 10 == 0:
        print(j, len(dataloader))
      img = batch[0].to(DEVICE)
      label = batch[1].to(DEVICE)
      global_idx = batch[4].to(DEVICE)
      BS = img.shape[0]

      valid = torch.ones( (BS) ,device=DEVICE)
      # GET CORRECT PATHS
      image_pths = np.array( global_to_image_pths) [global_idx.cpu().numpy()].tolist()
      if BS == 1: image_pths = [image_pths]
      
      
      if type( dataset ) == ScanNet:
        image_pths_next = [
          i.replace( 
            str( int(i[i.rfind('/')+1:i.find('.jpg')]))+'.jpg',
            str( int(i[i.rfind('/')+1:i.find('.jpg')])+sub )+'.jpg' )
            for i in image_pths ]

        store_pths = [
          str(os.path.join( "/home/jonfrey/datasets/scannet/flow", i[i.find('scans'):])).replace('.jpg','.png').replace('color', f'flow_sub{sub}')
            for i in image_pths ]
      elif type( dataset ) == LabData:
        image_pths_next = [
          f"{i[:-10]}{ int(i[-10:-4])+sub:06d}.png"
          for i in image_pths ]            
        store_pths = [
          i.replace("/2/",f"/2_flow_sub{sub}/")
            for i in image_pths ]
            


      # FILL IMG NEXT
      img_next = img.clone()
      for k,i in enumerate(image_pths_next):
        if i in image_pths:
          # Already loaded in current batch
          loc = image_pths.index( i )
          img_next[k] = img[loc]
        else:
          # Query dataloader to get index
          try:
            glo_idx = global_to_image_pths.index( i )
          except:
            print( "Skipped", i, k)
            valid[k] = 0
            continue
          try:
            local_idx = dataset.global_to_local_idx.index(glo_idx) 
          except:
            print( "Skipped", i, k)
            valid[k] = 0
            continue
          ba = dataset[local_idx]
          img_next[k] = ba[0]

      img = img[valid==1]
      img_next = img_next[valid==1]
      list_idx = torch.where( valid==1 )[0].cpu().tolist()
      store_pths = np.array( store_pths)[list_idx].tolist()

      img_next *= 255
      img *= 255
      flow_low, flow_up = model( img, img_next , iters=12, test_mode=True)

      for i in range(img.shape[0]):	
          Path(store_pths[i]).parent.mkdir(parents=True, exist_ok=True)
          writeFlowKITTI( store_pths[i] , flow_up[i].permute(1,2,0).cpu())

      # global_idx_next = global_idx.clone()
      # global_idx_next += sub 
      

      
      



