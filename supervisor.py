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
from utils_asl import file_path, load_yaml

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp', type=file_path, default='cfg/exp/eval/random.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--mode', default='module', choices=['shell','module'],
                      help='The environment yaml file.')
  
  args = parser.parse_args()
  exp = load_yaml(args.exp)
  env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
  env = load_yaml(env_cfg_path)
  
  if exp['max_tasks'] > exp['task_generator']['total_tasks']:
    print('Max Tasks larger then total tasks -> Setting max_tasks to total_tasks')
    exp['max_tasks'] = exp['task_generator']['total_tasks']
  
  if args.mode != 'sell':
    sys.path.append(os.path.join(os.getcwd() + '/train_task.py'))
    from train_task import train_task
    from utils_asl import get_neptune_logger
    logger = get_neptune_logger(exp,env, args.exp, env_cfg_path)
 
  sta = exp['start_at_task']
  end = exp['max_tasks']-1
  print(f"SUPERVISOR: Execute Task {sta}-{end}")
  for i in range(int(sta), exp['max_tasks'] ):
    
    init = int(bool(i==exp['start_at_task']))
    close = int(bool(i==exp['max_tasks']-1))
    
    if args.mode == 'shell':
      if env['workstation']:
        mc = '/home/jonfrey/miniconda3/envs/track4'
      else:
        mc = '/cluster/home/jonfrey/miniconda3/envs/track4'
      cmd = f'cd $HOME/ASL && {mc}/bin/python train_task.py' 
      cmd += f' --exp={args.exp} --init={init} --task_nr={i} --close={close}'
      print("Execute script: " , cmd)
      os.system(cmd)
    else:
      print("SUPERVISOR: CALLING train_task:", init, close, args.exp, env_cfg_path, i)
      train_task( init, close, args.exp, env_cfg_path, i, logger_pass=logger)
      torch.cuda.empty_cache()

