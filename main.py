
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
  fh = logging.FileHandler( os.path.join(model_path, f'info{local_rank}.log') )
  fh.setLevel(logging.DEBUG)
  logger.addHandler(fh)
  
  if env['workstation'] == False:
    for dataset in exp['move_datasets']:
      
      env_var = dataset['env_var']
      tar = os.path.join( env[env_var],f'{env_var}.tar')
      
      name = (tar.split('/')[-1]).split('.')[0]
      scratchdir = os.getenv('TMPDIR')
      try:  
        cmd = f"tar -xvf {tar} -C $TMPDIR >/dev/null 2>&1"
        print( cmd )
        os.system(cmd)
        env[env_var] = str(os.path.join(scratchdir, name))   
        print(env[env_var] )
      except:
          env[env_var] = p_ycb_new
          print('Copying data failed')
    

  model = Network(exp=exp, env=env)

  early_stop_callback = EarlyStopping(
    **exp['cb_early_stopping']['cfg']
    )
  filepath = os.path.join(model_path, '{epoch}{task_name}')

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
    if exp.get('tramac_restore', False): 
      p = env['tramac_weights']
      if os.path.isfile( p ):
        name, ext = os.path.splitext( p )
        assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
        logger.info( f'Resuming training TRAMAC, loading {p}...' )
        model.model.load_state_dict(torch.load(p, map_location=lambda storage, loc: storage))

    trainer = Trainer(**exp['trainer'],
      checkpoint_callback=checkpoint_callback,
      default_root_dir=model_path,
      callbacks=cb_ls)

  
  #  log_dir=None, comment='', purge_step=None, max_queue=10,
  #                flush_secs=120, filename_suffix=''
                 
  main_tbl = SummaryWriter(
      log_dir = os.path.join(  trainer.default_root_dir, "MainLogger"))
  main_visu = MainVisualizer( p_visu = os.path.join( model_path, 'main_visu'), 
                            writer=main_tbl, epoch=0, store=True, num_classes=22 )
  tc = TaskCreator(**exp['task_generator'])
  results = []
  training_results = []
  for idx, out in enumerate(tc) :
    task, eval_lists = out

    main_visu.epoch = idx
    # New Logger
    rank_zero_info( 'Executing Training Task: '+task.name )
    tbl = TensorBoardLogger(
      save_dir = trainer.default_root_dir,
      name = task.name,default_hp_metric=False)
    # only way to avoid PytorchLightning SummaryWriter nameing!
    tbl._experiment = SummaryWriter(
      log_dir=os.path.join(trainer.default_root_dir, task.name), 
      **tbl._kwargs)
    trainer.logger = tbl
    # when a new fit or test is called in the trainer all dataloaders are initalized from scratch.
    # here check if the same goes for the optimizers and learning rate schedules in this case
    # both configues would need to be restored for advanced settings.
    trainer.current_epoch = 0
    suc = model.set_train_dataset_cfg(
      dataset_train_cfg=task.dataset_train_cfg,
      dataset_val_cfg=task.dataset_val_cfg,
      task_name=task.name
    )
    if not suc:
      rank_zero_warn( "Training Task "+ task.name + 
                      " not started. Not enough val/train data!")
      continue
    train_res = trainer.fit(model)
    training_results.append( train_res )
    test_results = []
    for eval_task in eval_lists:
      rank_zero_info( "Executing Evaluation Task: "+ eval_task.name + 
                      " of Training Task: " + task.name )
      suc = model.set_test_dataset_cfg( dataset_test_cfg=eval_task.dataset_test_cfg)
      if not suc:
        rank_zero_warn( "Evaluation Task "+ eval_task.name + 
                        " not started. Not enough test data!")
        continue
      
      # TODO: one task might call mutiple tests!
      test_res = trainer.test(model)
      new_res= {}
      for k in test_res[0].keys():
        try:
          new_res[k] = float( test_res[0][k])
        except:
          pass
          
      test_results.append(new_res)
      
    results.append({'Time': idx, 'Traning': train_res, 'Test': test_results})
    mIoU = []
    for i, task in enumerate( results):
      test_vals = []
      for j, test in enumerate(task['Test']):
        test_vals.append( test['test_mIoU_epoch'])
      mIoU.append(test_vals)
    
    max_tests = 0
    for task in mIoU:
      if len(task) > max_tests :
        max_tests  = len(task)
    
    for i in range(len( mIoU)):
      if len(mIoU[i]) < max_tests :
        mIoU[i] = mIoU[i] + [0.0] * (max_tests -len(mIoU[i]))
    data_matrix = np.array( mIoU) * 100
    data_matrix = np.around(data_matrix , decimals=0)
    main_visu.plot_matrix(
      tag = 'mIoU',
      data_matrix = data_matrix)
#  elif exp.get('model_mode', 'fit') == 'lr_finder':
#      lr_finder = trainer.tuner.lr_find(model, min_lr= 0.0000001, max_lr=0.01) # Run learning rate finder
#      fig = lr_finder.plot(suggest=True) # Plot
#      from matplotlib.pyplot import savefig
#       a = exp['flow_pose_cfg']['backbone']
#       b = exp['flow_pose_cfg']['mode']
#       sug = str(  lr_finder.suggestion() )
#       p = exp['model_path']+f'/visu/lr_{a}_{b}_{sug}.png'
#       savefig(p)
#       logger.info( 'SUGGESTION', lr_finder.suggestion())
#   else:
#       logger.info("Wrong model_mode defined in exp config")
#       raise Exception