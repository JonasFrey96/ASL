
import os 
import sys 
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))

import shutil
import datetime
import argparse
import signal
import coloredlogs
import yaml
coloredlogs.install()

import torch
from pytorch_lightning import seed_everything,Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning import Network
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profiler import AdvancedProfiler

def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)

def load_yaml(path):
  with open(path) as file:  
    res = yaml.load(file, Loader=yaml.FullLoader) 
  return res

if __name__ == "__main__":
  seed_everything(42)

  def signal_handler(signal, frame):
    print('exiting on CRTL-C')
    sys.exit(0)

  # this is needed for leonhard to use interactive session and dont freeze on
  # control-C !!!!
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  parser = argparse.ArgumentParser()    
  parser.add_argument('--exp', type=file_path, default='cfg/exp/exp.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--env', type=file_path, default='cfg/env/env.yml',
                      help='The environment yaml file.')

  args = parser.parse_args()
  exp_cfg_path = args.exp
  env_cfg_path = args.env

  exp = load_yaml(exp_cfg_path)
  env = load_yaml(env_cfg_path)
  print(exp)
  if exp.get('timestamp',True):
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    exp['name'] = str(timestamp)+'_'+exp['name']
    model_path = os.path.join(env['base'], exp['name'])
  else:
    model_path = os.path.join(env['base'], exp['name'])
    try:
      shutil.rmtree(model_path)
    except:
      pass
      
  if not os.path.exists(model_path):
    try:
      os.makedirs(model_path)
      print("Generating network run folder")
    except:
      print("Failed generating network run folder")
  else:
    print("Network run folder already exits")

  exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
  env_cfg_fn = os.path.split(env_cfg_path)[-1]
  print(f'Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}')
  shutil.copy(exp_cfg_path, f'{model_path}/{exp_cfg_fn}')
  shutil.copy(env_cfg_path, f'{model_path}/{env_cfg_fn}')
  exp['name'] = model_path
  model = Network(exp=exp, env=env)

  early_stop_callback = EarlyStopping(
    **exp['cb_early_stopping']['cfg']
    )
  filepath = os.path.join(model_path, '{epoch}')

  if len(exp['cb_checkpoint'].get('nameing',[])) > 0:
    for m in exp['cb_checkpoint']['nameing']: 
      filepath += '-{'+ m + ':.2f}'

  checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    **exp['cb_checkpoint']['cfg']
  )
  lr_monitor = LearningRateMonitor(
    **exp['lr_monitor']['cfg'])

  cb_ls = [early_stop_callback, lr_monitor]
  
  # Always use advanced profiler
  if exp['trainer'].get('profiler', False):
    exp['trainer']['profiler'] = AdvancedProfiler(output_filename=os.path.join(model_path, 'profile.out'))
  else:
    exp['trainer']['profiler']  = False
      
  if exp.get('checkpoint_restore', False): 
    trainer = Trainer( **exp['trainer'],
      checkpoint_callback=checkpoint_callback,
      default_root_dir = model_path,
      callbacks=cb_ls, 
      resume_from_checkpoint = exp['checkpoint_load'])       
  else:
    trainer = Trainer(**exp['trainer'],
      checkpoint_callback=checkpoint_callback,
      default_root_dir=model_path,
      callbacks=cb_ls)

  trainer.fit(model)
  trainer.test(model)

#  elif exp.get('model_mode', 'fit') == 'lr_finder':
#      lr_finder = trainer.tuner.lr_find(model, min_lr= 0.0000001, max_lr=0.01) # Run learning rate finder
#      fig = lr_finder.plot(suggest=True) # Plot
#      from matplotlib.pyplot import savefig
#       a = exp['flow_pose_cfg']['backbone']
#       b = exp['flow_pose_cfg']['mode']
#       sug = str(  lr_finder.suggestion() )
#       p = exp['model_path']+f'/visu/lr_{a}_{b}_{sug}.png'
#       savefig(p)
#       print( 'SUGGESTION', lr_finder.suggestion())
#   else:
#       print("Wrong model_mode defined in exp config")
#       raise Exception