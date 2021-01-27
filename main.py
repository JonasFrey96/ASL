
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

import torch
from pytorch_lightning import seed_everything,Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profiler import AdvancedProfiler
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.loggers import TensorBoardLogger

from lightning import Network
from task import TaskCreator
from visu import MainVisualizer
import numpy as np

from pytorch_lightning.loggers.neptune import NeptuneLogger
import datetime
import time
from datasets import get_dataset
from torchvision import transforms
from math import ceil
import copy 

def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)

def load_yaml(path):
  with open(path) as file:  
    res = yaml.load(file, Loader=yaml.FullLoader) 
  return res

def eval_lists_into_dataloaders( eval_lists, env):
  loaders = []
  for eval_task in eval_lists:
    
    loaders.append( get_dataloader_test(eval_task.dataset_test_cfg, env))
  return loaders

def get_dataloader_test(d_test, env):
  output_transform = transforms.Compose([
          transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
  ])
  # dataset and dataloader
  dataset_test = get_dataset(
    **d_test,
    env = env,
    output_trafo = output_transform,
  )
  dataloader_test = torch.utils.data.DataLoader(dataset_test,
    shuffle = exp['loader']['shuffle'],
    num_workers = ceil(exp['loader']['num_workers']/torch.cuda.device_count()),
    pin_memory = exp['loader']['pin_memory'],
    batch_size = exp['loader']['batch_size'], 
    drop_last = True)
  return dataloader_test
  
def get_dataloader_train_val(d_train, d_val, env, replay_state):

  output_transform = transforms.Compose([
          transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
  ])
  # dataset and dataloader
  dataset_train = get_dataset(
    **d_train,
    env = env,
    output_trafo = output_transform,
  )
  
  if replay_state is not None:
    # dont loss the buffer replay state if a new dataset is created
    dataset_train.set_full_state(**replay_state)
    
  dataloader_train = torch.utils.data.DataLoader(dataset_train,
    shuffle = exp['loader']['shuffle'],
    num_workers = ceil(exp['loader']['num_workers']/torch.cuda.device_count()),
    pin_memory = exp['loader']['pin_memory'],
    batch_size = exp['loader']['batch_size'], 
    drop_last = True)
  
  dataset_val = get_dataset(
    **d_val,
    env = env,
    output_trafo = output_transform,
  )
  dataloader_val = torch.utils.data.DataLoader(dataset_val,
    shuffle = exp['loader']['shuffle'],
    num_workers = ceil(exp['loader']['num_workers']/torch.cuda.device_count()),
    pin_memory = exp['loader']['pin_memory'],
    batch_size = exp['loader']['batch_size'], 
    drop_last = True)
    
  return dataloader_train, dataloader_val
    
    
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
  
  local_rank = int(os.environ.get('LOCAL_RANK', 0))
  if local_rank == 0:
    # Set in name the correct model path
    if exp.get('timestamp',True):
      timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
      
      model_path = os.path.join(env['base'], exp['name'])
      p = model_path.split('/')
      model_path = os.path.join('/',*p[:-1] ,str(timestamp)+'_'+ p[-1] )
    else:
      model_path = os.path.join(env['base'], exp['name'])
      try:
        shutil.rmtree(model_path)
      except:
        pass
    # Create the directory
    if not os.path.exists(model_path):
      try:
        os.makedirs(model_path)
        print("Generating network run folder")
      except:
        print("Failed generating network run folder")
    else:
      print("Network run folder already exits")
    

    # Only copy config files for the main ddp-task  
    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    env_cfg_fn = os.path.split(env_cfg_path)[-1]
    print(f'Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}')
    shutil.copy(exp_cfg_path, f'{model_path}/{exp_cfg_fn}')
    shutil.copy(env_cfg_path, f'{model_path}/{env_cfg_fn}')
    exp['name'] = model_path
    
    # write back the exp file with the correct name set to the model_path!
    # other ddp-task dont need to care about timestamps.
    if not env['workstation']: 
      with open(exp_cfg_path, 'w+') as f:
        yaml.dump(exp, f, default_flow_style=False, sort_keys=False)
  else:
    # the correct model path has already been written to the yaml file.
    
    model_path = os.path.join( exp['name'], f'rank_{local_rank}')
    # Create the directory
    if not os.path.exists(model_path):
      try:
        os.makedirs(model_path)
      except:
        pass
  # Setup logger for each ddp-task 
  logging.getLogger("lightning").setLevel(logging.DEBUG)
  logger = logging.getLogger("lightning")
  fh = logging.FileHandler( os.path.join(model_path, f'info{local_rank}.log'), 'a')
  logger.addHandler(fh)
      
  # Copy Dataset from Scratch to Nodes SSD
  if env['workstation'] == False:
    for dataset in exp['move_datasets']:
      scratchdir = os.getenv('TMPDIR')
      print( 'scratchdir:', scratchdir, 'dataset:', dataset['env_var'])
      env_var = dataset['env_var']
      tar = os.path.join( env[env_var],f'{env_var}.tar')
      name = (tar.split('/')[-1]).split('.')[0]
        
      if not os.path.exists(os.path.join(scratchdir,dataset['env_var']) ):
        
        try:  
          cmd = f"tar -xvf {tar} -C $TMPDIR >/dev/null 2>&1"
          st =time.time()
          rank_zero_info( f'Start moveing dataset-{env_var}: {cmd}')
          os.system(cmd)
          env[env_var] = str(os.path.join(scratchdir, name))
          rank_zero_info( f'Finished moveing dataset-{env_var} in {time.time()-st}s')
          
        except:
            rank_zero_warn( 'ENV Var'+ env_var )
            env[env_var] = str(os.path.join(scratchdir, name))
            rank_zero_warn('Copying data failed')
      else:
        env[env_var] = str(os.path.join(scratchdir, name))
        print('Path already exists. Updated ENV')
  
  if ( exp['trainer'] ).get('gpus', -1):
    nr = torch.cuda.device_count()
    exp['trainer']['gpus'] = nr
    rank_zero_info( f'Set GPU Count for Trainer to {nr}!' )
    

  model = Network(exp=exp, env=env)


  filepath = os.path.join(model_path, '{task_count}-{epoch}-')

  if len(exp['cb_checkpoint'].get('nameing',[])) > 0:
    #filepath += '-{task_name:10s}'
    for m in exp['cb_checkpoint']['nameing']: 
      filepath += '-{'+ m + ':.2f}'

  checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    **exp['cb_checkpoint']['cfg']
  )
  lr_monitor = LearningRateMonitor(
    **exp['lr_monitor']['cfg'])

  if exp['cb_early_stopping']['active']:
    early_stop_callback = EarlyStopping(
    **exp['cb_early_stopping']['cfg']
    )
    cb_ls = [early_stop_callback, lr_monitor]
  else:
    cb_ls = [lr_monitor]
  cb_ls.append( checkpoint_callback )
  
    
  # Always use advanced profiler
  if exp['trainer'].get('profiler', False):
    exp['trainer']['profiler'] = AdvancedProfiler(output_filename=os.path.join(model_path, 'profile.out'))
  else:
    exp['trainer']['profiler']  = False
      
  if exp.get('checkpoint_restore', False):
    p = os.path.join( env['base'], exp['checkpoint_load'])
    trainer = Trainer( **exp['trainer'],
      default_root_dir = model_path,
      callbacks=cb_ls, 
      resume_from_checkpoint = p)
    
  else:
    if exp.get('tramac_restore', False): 
      p = env['tramac_weights']
      if os.path.isfile( p ):
        name, ext = os.path.splitext( p )
        assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
        logger.info( f'Resuming training TRAMAC, loading {p}...' )
        model.model.load_state_dict(torch.load(p, map_location=lambda storage, loc: storage))

    trainer = Trainer(**exp['trainer'],
      default_root_dir=model_path,
      callbacks=cb_ls)

  if exp.get('weights_restore', False):
    # it is not strict since the latent replay buffer is not always available
    p = os.path.join( env['base'], exp['checkpoint_load'])
    if os.path.isfile( p ):
      res = model.load_state_dict( torch.load(p, 
        map_location=lambda storage, loc: storage)['state_dict'], 
        strict=False)
      rank_zero_info('Restoring weights: ' + str(res))
    else:
      raise Exception('Checkpoint not a file')
    

  main_tbl = SummaryWriter(
      log_dir = os.path.join(  trainer.default_root_dir, "MainLogger"))
  main_visu = MainVisualizer( p_visu = os.path.join( model_path, 'main_visu'), 
                            writer=main_tbl, epoch=0, store=True, num_classes=22 )
  tc = TaskCreator(**exp['task_generator'],output_size=exp['model']['input_size'])
  results = []
  training_results = []
  print(tc)
  _task_start_training = time.time()
  _task_start_time = time.time()
  
  mode = exp.get('paper_analysis', {}).get('mode','normal')
  
  replay_state = None
  if mode == 'normal':
    for idx, out in enumerate(tc) :  
 
      if idx != 0:
        t = time.time() - _task_start_time
        t = str(datetime.timedelta(seconds=t))
        t2 = time.time() - _task_start_training
        t2 = str(datetime.timedelta(seconds=t2))
        rank_zero_info(f'Time for task {idx}: '+ t)
        rank_zero_info(f'Time for Task 0- Task {idx}: '+ t2)
        _task_start_time = time.time()

      
      task, eval_lists = out
      main_visu.epoch = idx
      # New Logger
      rank_zero_info( f'<<<<<<<<<<<< TASK IDX {idx} TASK NAME : '+task.name+ ' >>>>>>>>>>>>>' )
      tbl = TensorBoardLogger(
        save_dir = trainer.default_root_dir,
        name = task.name,default_hp_metric=False)

      # only way to avoid PytorchLightning SummaryWriter nameing!
      tbl._experiment = SummaryWriter(
        log_dir=os.path.join(trainer.default_root_dir, task.name), 
        **tbl._kwargs)
      trainer.logger = tbl


      trainer.current_epoch = 0      
      model._task_name = task.name
      model._task_count = idx
      dataloader_train, dataloader_val = get_dataloader_train_val(d_train= task.dataset_train_cfg,
                                                                  d_val= task.dataset_val_cfg,
                                                                  env=env , replay_state = replay_state)
      dataloader_list_test = eval_lists_into_dataloaders( eval_lists, env)
      
      #Training the model
      train_res = trainer.fit(model = model,
                              train_dataloader= dataloader_train,
                              val_dataloaders= dataloader_list_test)
      if exp['task_generator'].get('replay',True):
        bins, valids = trainer.train_dataloader.dataset.get_full_state()
        replay_state = { 'bins':bins, 'valids': valids, 'bin': idx+1 }
        
      
      training_results.append( copy.deepcopy(trainer.logged_metrics) )
      
      for met in ['val_loss/dataloader_idx_', 'val_acc/dataloader_idx_', 'val_mIoU/dataloader_idx_']:
        data_matrix  = np.zeros( (len(training_results),len(dataloader_list_test) ) )
        for _i, res in enumerate( training_results  ):
          for _j in range( len(dataloader_list_test) ):
            data_matrix [_i,_j] = res[met+str(_j)] * 100
        
        data_matrix = np.round( data_matrix, decimals=1)
        if met.find('loss') != -1:
          higher_is_better = False
        else:
          higher_is_better = True
        
        main_visu.plot_matrix(
            tag = str( met.split('/')[0]),
            data_matrix = data_matrix,
            higher_is_better= higher_is_better,
            title=str( met.split('/')[0]))