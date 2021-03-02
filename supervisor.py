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

# Costume Modules
from utils import file_path, load_yaml

if __name__ == "__main__":
  parser = argparse.ArgumentParser()    
  parser.add_argument('--exp', type=file_path, default='/home/jonfrey/ASL/cfg/exp/scannet/scannet.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--env', type=file_path, default='cfg/env/env.yml',
                      help='The environment yaml file.')
  parser.add_argument('--mode', default='module', choices=['shell','module'],
                      help='The environment yaml file.')
  
  args = parser.parse_args()
  exp = load_yaml(args.exp)
  env = load_yaml(args.env)
  if exp['max_tasks'] > exp['task_generator']['total_tasks']:
    print('Max Tasks larger then total tasks -> Setting max_tasks to total_tasks')
    exp['max_tasks'] = exp['task_generator']['total_tasks']

  exp = load_yaml(args.exp)
  env = load_yaml(args.env)
  if args.mode != 'sell':
    from src import train_task
    from utils import get_neptune_logger
    logger = get_neptune_logger(exp,env, args.exp, args.env)
    
  for i in range( exp['max_tasks'] ):
    init = int(bool(i==0))
    close = int(bool(i==exp['max_tasks']))
    
    if args.mode == 'shell':
      if env['workstation']:
        mc = '/home/jonfrey/miniconda3/envs/track4'
      else:
        mc = '/cluster/home/jonfrey/miniconda3/envs/track4'
      cmd = f'cd $HOME/ASL && {mc}/bin/python src/train_task.py' 
      cmd += f' --exp={args.exp} --env={args.env} --init={init} --task_nr={i} --close={close}'
      print("Execute script: " , cmd)
      os.system(cmd)
    else:
      train_task( init, close, args.exp, args.env, i, logger_pass=logger)

