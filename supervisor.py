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

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp', type=file_path, default='cfg/exp/debug/debug.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--mode', default='module', choices=['shell','module'],
                      help='The environment yaml file.')
  
  args = parser.parse_args()
  exp = load_yaml(args.exp)
  env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
  env = load_yaml(env_cfg_path)


  sta = exp['supervisor']['start_task']
  sto = exp['supervisor']['stop_task']
  print(f"SUPERVISOR: Execute Task {sta}-{sto}")
  for i in range( 0, sto+1 ):
    print(f"SUPERVISOR: Execute Task {i}/{sto}")
    init = int(bool(i== 0 ))
    close = 1 # int(bool(i==sto))
    skip = int(i < sta)

    if args.mode == 'shell':
      if env['workstation']:
        mc = '/home/jonfrey/miniconda3/envs/track4'
      else:
        mc = '/cluster/home/jonfrey/miniconda3/envs/track4'
      cmd = f'cd $HOME/ASL && {mc}/bin/python train_task.py' 
      cmd += f' --exp={args.exp} --init={init} --task_nr={i} --close={close} --skip={skip}'
      print("Execute script: " , cmd)
      os.system(cmd)
      torch.cuda.empty_cache()

    else:

      print("SUPERVISOR: CALLING train_task:", init, close, args.exp, env_cfg_path, i)
      
      train_task( init, close, args.exp, env_cfg_path, i, skip = skip, logger_pass=None)
      torch.cuda.empty_cache()

