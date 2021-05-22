#TODO: Jonas Frey check if new neptune AI api would do the job for us and ease the ddp problem away!
#TODO: Refactor the dataloader and replayed label that we get in a usefull way maybe with a wrappwer dataset
#TODO: Remove the continual learning part maybe fully to a plugin.
#TODO: Check if the monitoring that we are currently doing is worth in the lighning module here it got just to large
#TODO: Check if we can use smart callbacks to fullfill the loggin better 
# Light 1.2.4 Neptune 0.5.1 1.7.1+cu110
# Refactor the training task maybe aswell !


import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))

import coloredlogs
coloredlogs.install()

import argparse
# Frameworks
import torch

# Costume Modules
from utils_asl import file_path, load_yaml

from train_task import train_task
from utils_asl import get_neptune_logger

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp', type=file_path, default='cfg/exp/debug.yml',
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
  
  if args.mode != 'shell':
    sys.path.append(os.path.join(os.getcwd() + '/train_task.py'))

    logger = get_neptune_logger(exp,env, args.exp, env_cfg_path)
  
  sta = 0 #exp['start_at_task']
  end = exp['max_tasks']-1
  print(f"SUPERVISOR: Execute Task {sta}-{end}")
  for i in range(int(sta), exp['max_tasks'] ):
    
    init = int(bool(i== 0 ))
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
      train_task( init, close, args.exp, env_cfg_path, i, logger_pass=None)
      torch.cuda.empty_cache()

