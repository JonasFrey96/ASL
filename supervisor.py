import argparse
import os
import yaml
from src import *




import os 
import sys 
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))

import time
import shutil
import datetime
import argparse
import signal
import coloredlogs
import yaml
import logging
coloredlogs.install()
from math import ceil
import copy
from pathlib import Path
from contextlib import redirect_stdout
from contextlib import nullcontext
import pathlib
import pickle
# Frameworks
import neptune

import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profiler import AdvancedProfiler

from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger

# Costume Modules
from lightning import Network, fill_buffer
from task import TaskCreator
from visu import MainVisualizer
import numpy as np
from callbacks import TaskSpecificEarlyStopping
from datasets import get_dataset
from log import _create_or_get_experiment2

if __name__ == "__main__":
  parser = argparse.ArgumentParser()    
  parser.add_argument('--exp', type=file_path, default='/home/jonfrey/ASL/cfg/exp/coco/coco.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--env', type=file_path, default='cfg/env/env.yml',
                      help='The environment yaml file.')
  parser.add_argument('--mode', default='module', choices=['shell','module'],
                      help='The environment yaml file.')
  
  args = parser.parse_args()
  exp = load_yaml(args.exp)
  env = load_yaml(args.env)
  if exp['max_tasks'] > exp['task_generator']['total_tasks']:
    print('Max Tasks largen then total tasks -> Setting max_tasks to total_tasks')
    exp['max_tasks'] = exp['task_generator']['total_tasks']
  
  
  from pathlib import Path
  from pytorch_lightning.loggers.neptune import NeptuneLogger
  exp = load_yaml(args.exp)
  env = load_yaml(args.env)
  
  params = log_important_params( exp )
  cwd = os.getcwd()
  files = [str(p).replace(cwd+'/','') for p in Path(cwd).rglob('*.py') 
          if str(p).find('vscode') == -1]
  files.append( args.exp )
  files.append( args.env )
  if env['workstation']:
    t1 = 'workstation'
  else:
    t1 = 'leonhard'

  logger = NeptuneLogger(
    api_key=os.environ["NEPTUNE_API_TOKEN"],
    project_name="jonasfrey96/asl",
    experiment_name= exp['name'].split('/')[-2] +"_"+ exp['name'].split('/')[-1], # Optional,
    params=params, # Optional,
    tags=[t1, exp['name'].split('/')[-2], exp['name'].split('/')[-1]] + exp["tag_list"], # Optional,
    close_after_fit = False,
    offline_mode = exp.get('offline_mode', False),
    upload_source_files=files,
    upload_stdout=True,
    upload_stderr=True
  )
  for i in range( exp['max_tasks'] ):
    init = int(bool(i==0))
    close = int(bool(i==exp['max_tasks']))
    
    if args.mode == 'shell':
      if env['workstation']:
        mc = '/home/jonfrey/miniconda3/envs/track4'
      else:
        mc = '/cluster/home/jonfrey/miniconda3/envs/track4'
      cmd = f'cd $HOME/ASL && {mc}/bin/python main.py' 
      cmd += f' --exp={args.exp} --env={args.env} --init={init} --task_nr={i} --close={close}'
      print(cmd)
      os.system(cmd)
    else:
      torch.cuda.empty_cache() 
      train_task( init, close, args.exp, args.env, i, logger_pass=logger)
      import gc
      gc.collect()
      torch.cuda.empty_cache()
    