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

__all__ = ['train_task', 'file_path', 'load_yaml', 'log_important_params',
           'plot_from_neptune', 'eval_lists_into_dataloaders', 'get_dataloader_test',
           'get_dataloader_train', 'plot_from_pkl', 'plot_from_neptune']

def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)

def load_yaml(path):
  with open(path) as file:  
    res = yaml.load(file, Loader=yaml.FullLoader) 
  return res

def eval_lists_into_dataloaders( eval_lists, env, exp):
  loaders = []
  for eval_task in eval_lists:
    
    loaders.append( get_dataloader_test(eval_task.dataset_test_cfg, env, exp))
  return loaders

def get_dataloader_test(d_test, env, exp):
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
    shuffle = False,
    num_workers = max(1, ceil(exp['loader']['num_workers']/torch.cuda.device_count()) ),
    pin_memory = exp['loader']['pin_memory'],
    batch_size = exp['loader']['batch_size'], 
    drop_last = True)
  return dataloader_test
  
def get_dataloader_train(d_train, env, exp):
  print( 'Number CUDA Devices: '+ str( torch.cuda.device_count()) )
  
  output_transform = transforms.Compose([
          transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
  ])
  # dataset and dataloader
  dataset_train = get_dataset(
    **d_train,
    env = env,
    output_trafo = output_transform,
  )
  
    
  dataloader_train = torch.utils.data.DataLoader(dataset_train,
    shuffle = exp['loader']['shuffle'],
    num_workers = ceil(exp['loader']['num_workers']/torch.cuda.device_count()),
    pin_memory = exp['loader']['pin_memory'],
    batch_size = exp['loader']['batch_size'], 
    drop_last = True)
  
  # dataset and dataloader
  dataset_buffer = get_dataset(
    **d_train,
    env = env,
    output_trafo = output_transform,
  )
  dataset_buffer.replay = False
  dataset_buffer.unique = True
  
  dataloader_buffer= torch.utils.data.DataLoader(dataset_buffer,
    shuffle = False,
    num_workers = ceil(exp['loader']['num_workers']/torch.cuda.device_count()),
    pin_memory = exp['loader']['pin_memory'],
    batch_size = max(1,ceil(exp['loader']['batch_size'])), 
    drop_last = True)
    
  return dataloader_train, dataloader_buffer

import collections

def flatten_list(d, parent_key='', sep='_'):
  items = []
  for num, element in enumerate(d):
      new_key = parent_key + sep + str(num) if parent_key else str(num)

      if isinstance(element, collections.MutableMapping):
          items.extend(flatten_dict(element, new_key, sep=sep).items())
      else:
          if isinstance(element, list):
              if isinstance(element[0], dict):
                  items.extend(flatten_list(element, new_key, sep=sep))
                  continue
          items.append((new_key, element))
  return items

def flatten_dict(d, parent_key='', sep='_'):
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.MutableMapping):
        items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
        if isinstance(v, list):
            if isinstance(v[0], dict):
                items.extend(flatten_list(v, new_key, sep=sep))
                continue
        items.append((new_key, v))
  return dict(items)

def log_important_params( exp ):

  dic = {}
  dic['total_tasks'] = exp['task_generator']['total_tasks']
  dic['task_cfg'] = exp['task_generator']['mode']
  dic['replay_buffer_size_per_bin'] = exp['task_generator']['cfg_replay']['elements']
  dic['replay_adaptive_replay'] = exp['task_generator']['replay_adaptive_add_p']
  dic['replay_active'] = exp['task_generator']['replay']
  dic['teaching_active'] = exp['teaching']['active'] 
  dic['teaching_soft_labels'] = exp['teaching']['soft_labels'] 
  dic['model_finetune'] = exp['model']['freeze']['active']
  dic = flatten_dict(exp)
  
  dic = flatten_dict(exp)
  
  return dic


def plot_from_pkl(main_visu, base_path, task_nr):
  with open(f"{base_path}/res0.pkl",'rb') as f:
    res = pickle.load(f)
  nr_val_tasks = 0
  for k in res.keys():
    if str(k).find('task_count')  != -1:
      nr_val_tasks += 1
  val_metrices = ['val_acc', 'val_loss', 'val_mIoU']
  for m in val_metrices: 
    data_matrix = np.zeros( ( task_nr+1, nr_val_tasks) )
    for i in range(task_nr+1):
      with open(f"{base_path}/res{i}.pkl",'rb') as f:
        res = pickle.load(f)
      for j in range(nr_val_tasks):
        v = res[f'{m}/dataloader_idx_{j}']
        data_matrix[i,j] = v*100
    data_matrix = np.round( data_matrix, decimals=1) 
    if m.find('loss') != -1:
      higher_is_better = False
    else:
      higher_is_better = True
    main_visu.plot_matrix(
      tag = m,
      data_matrix = data_matrix,
      higher_is_better= higher_is_better,
      title= m)

def validation_acc_plot(main_visu, logger):
  try:
    df_acc = logger.experiment.get_numeric_channels_values('val_acc/dataloader_idx_0','val_acc/dataloader_idx_1','val_acc/dataloader_idx_2','val_acc/dataloader_idx_3','task_count/dataloader_idx_0')
    x = np.array(df_acc['task_count/dataloader_idx_0'])
    
    
    task_nrs, where = np.unique( x, return_index=True )
    task_nrs = task_nrs.astype(np.uint8).tolist()
    where = where.tolist()[1:]
    where = [w-1 for w in where]
    where += [x.shape[0]-1]*int(4-len(where))
    
    names = [f'val_acc_{idx}' for idx in range(4)]
    x =  np.arange(x.shape[0])
    y = [np.array(df_acc[f'val_acc/dataloader_idx_{idx}']  ) for idx in range(4)]
    arr = main_visu.plot_lines_with_bachground(
        x, y, count=where,
        x_label='Epoch', y_label='Acc', title='Validation Accuracy', 
        task_names=names, tag='Validation_Accuracy_Summary')
  except:
    pass
  
def plot_from_neptune(main_visu,logger):
  try: 
    idxs = logger.experiment.get_numeric_channels_values('task_count/dataloader_idx_0')['x']
    task_numbers = logger.experiment.get_numeric_channels_values('task_count/dataloader_idx_0')['task_count/dataloader_idx_0']
    
    val_metrices = ['val_acc', 'val_loss', 'val_mIoU']
    
    dic = {}
    task_start = 0
    np.unique(task_numbers, return_index=True)
    
    training_task_indices = []
    s = 0
    for i in range(len(task_numbers)):
      if task_numbers[i] != s:
        s = task_numbers[i]
        training_task_indices.append(i-1)
    training_task_indices.append(len(task_numbers)-1)
    
    val_dataloaders = len( [d for d in logger.experiment.get_logs().keys() if d.find('task_count/dataloader_idx') != -1] )
    trained_tasks = np.unique(task_numbers).shape[0]
    data_matrix = {}
    for m in val_metrices:
      data_matrix[m] = np.zeros( ( trained_tasks, val_dataloaders) )
      for training_task in range(trained_tasks):
        for val_dataloader in range(val_dataloaders):
          v = logger.experiment.get_numeric_channels_values(f'{m}/dataloader_idx_{val_dataloader}')[f'{m}/dataloader_idx_{val_dataloader}'][training_task_indices[training_task]]
          data_matrix[m][training_task, val_dataloader] = v*100
      data_matrix[m] = np.round( data_matrix[m], decimals=1)
      
      if m.find('loss') != -1:
        higher_is_better = False
      else:
        higher_is_better = True
      main_visu.plot_matrix(
          tag = m,
          data_matrix = data_matrix[m],
          higher_is_better= higher_is_better,
          title= m)
  except:
    pass    
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
  parser.add_argument('--exp', type=file_path, default='/home/jonfrey/ASL/cfg/exp/3/metric_soft_max_distance.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--env', type=file_path, default='cfg/env/env.yml',
                      help='The environment yaml file.')
  parser.add_argument('--task_nr', type=int, default=0,
                      help='Task nr.')
  parser.add_argument('--init', type=int, default=1,
                      help='Task nr.')
  parser.add_argument('--close', type=int, default=0,
                      help='Task nr.')

  args = parser.parse_args()
  args.init = bool(args.init)
  args.close = bool(args.close)
  init = args.init
  close = args.close
  exp_cfg_path = args.exp
  env_cfg_path = args.env
  task_nr = args.task_nr
  print('Called with the following arguments: '+ str(args))  
  train_task( init, close, exp_cfg_path, env_cfg_path, task_nr)


def train_task( init, close, exp_cfg_path, env_cfg_path, task_nr, logger_pass=None):
  seed_everything(42)
  local_rank = int(os.environ.get('LOCAL_RANK', 0))
  if local_rank != 0 or not init:
    print(init, local_rank)
    rm = exp_cfg_path.find('cfg/exp/') + len('cfg/exp/')
    exp_cfg_path = os.path.join( exp_cfg_path[:rm],'tmp/',exp_cfg_path[rm:])
  
  

  exp = load_yaml(exp_cfg_path)
  env = load_yaml(env_cfg_path)

  if local_rank == 0 and init:
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
  else:
    # the correct model path has already been written to the yaml file.
    model_path = os.path.join( exp['name'], f'rank_{local_rank}_{task_nr}')
    # Create the directory
    if not os.path.exists(model_path):
      try:
        os.makedirs(model_path)
      except:
        pass
  
  

  # if local_rank == 0 and env['workstation'] == False:
  #     cm = open(os.path.join(model_path, f'info{local_rank}_{task_nr}.log'), 'w')
  # else:
  #     cm = nullcontext()
  # with cm as f:
  #   if local_rank == 0 and env['workstation'] == False:
  #     cm2 = redirect_stdout(f)
  #   else:
  #     cm2 = nullcontext()
  #   with cm2:
  # # Setup logger for each ddp-task 
  # logging.getLogger("lightning").setLevel(logging.DEBUG)
  # logger = logging.getLogger("lightning")
  # fh = logging.FileHandler( , 'a')
  # logger.addHandler(fh)
      
  # Copy Dataset from Scratch to Nodes SSD

  if env['workstation'] == False:
    # use proxy hack for neptunai !!!
    NeptuneLogger._create_or_get_experiment = _create_or_get_experiment2

    # move data to ssd
    if exp['move_datasets'][0]['env_var'] != 'none':
      for dataset in exp['move_datasets']:
        scratchdir = os.getenv('TMPDIR')
        env_var = dataset['env_var']
        tar = os.path.join( env[env_var],f'{env_var}.tar')
        name = (tar.split('/')[-1]).split('.')[0]
          
        if not os.path.exists(os.path.join(scratchdir,dataset['env_var']) ):
          
          try:  
            cmd = f"tar -xvf {tar} -C $TMPDIR >/dev/null 2>&1"
            st =time.time()
            print( f'Start moveing dataset-{env_var}: {cmd}')
            os.system(cmd)
            env[env_var] = str(os.path.join(scratchdir, name))
            print( f'Finished moveing dataset-{env_var} in {time.time()-st}s')
            
          except:
              rank_zero_warn( 'ENV Var'+ env_var )
              env[env_var] = str(os.path.join(scratchdir, name))
              rank_zero_warn('Copying data failed')
        else:
          env[env_var] = str(os.path.join(scratchdir, name))
    else:
      env['mlhypersim'] = str(os.path.join(env['mlhypersim'], 'mlhypersim'))
      
      
  if ( exp['trainer'] ).get('gpus', -1):
    nr = torch.cuda.device_count()
    exp['trainer']['gpus'] = nr
    print( f'Set GPU Count for Trainer to {nr}!' )
    

  model = Network(exp=exp, env=env)
  
  lr_monitor = LearningRateMonitor(
    **exp['lr_monitor']['cfg'])

  if exp['cb_early_stopping']['active']:
    early_stop_callback = EarlyStopping(
    **exp['cb_early_stopping']['cfg']
    )
    cb_ls = [early_stop_callback, lr_monitor]
  else:
    cb_ls = [lr_monitor]
  
  tses = TaskSpecificEarlyStopping(
    nr_tasks=exp['task_generator']['total_tasks'] , 
    **exp['task_specific_early_stopping']
  )
  cb_ls.append(tses)
  if local_rank == 0:
    for i in range(exp['task_generator']['total_tasks']):
      if i == task_nr:
        m = '/'.join( [a for a in model_path.split('/') if a.find('rank') == -1])
        
        dic = copy.deepcopy( exp['cb_checkpoint']['cfg'])
        # try:
        #   if len(exp['cb_checkpoint'].get('nameing',[])) > 0:
        #     #filepath += '-{task_name:10s}'
        #     for m in exp['cb_checkpoint']['nameing']: 
        #       filepath += '-{'+ m + ':.2f}'
        # except:
        #   pass
        # dic['monitor'] += str(i)
        checkpoint_callback = ModelCheckpoint(
          dirpath= m,
          filename= 'task'+str(i)+'-{epoch:02d}--{step:06d}',
          **dic
        )
        
        cb_ls.append( checkpoint_callback )
      
  params = log_important_params( exp )
    
  if env['workstation']:
    t1 = 'workstation'
  else:
    t1 = 'leonhard'
  
  # if local_rank == 0:
  cwd = os.getcwd()
  files = [str(p).replace(cwd+'/','') for p in Path(cwd).rglob('*.py') 
          if str(p).find('vscode') == -1]
  files.append( exp_cfg_path )
  files.append( env_cfg_path )
  
  if not exp.get('offline_mode', False):
    # if exp.get('experiment_id',-1) == -1:
      #create new experiment_id and write back
    if  logger_pass is None:
      logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name="jonasfrey96/asl",
        experiment_name= exp['name'].split('/')[-2] +"_"+ exp['name'].split('/')[-1], # Optional,
        params=params, # Optional,
        tags=[t1, exp['name'].split('/')[-2], exp['name'].split('/')[-1]] + exp["tag_list"], # Optional,
        close_after_fit = False,
        offline_mode = exp.get('offline_mode', False),
        upload_source_files=files,
        upload_stdout=False,
        upload_stderr=False
      )
      exp['experiment_id'] = logger.experiment.id
      print('created experiment id' +  str( exp['experiment_id']))
    else:
      logger = logger_pass
      
    # else:
      # print('loaded experiment id' +  str( exp['experiment_id']))
      # TODO
      # logger = NeptuneLogger(
      #   api_key=os.environ["NEPTUNE_API_TOKEN"],
      #   project_name="jonasfrey96/asl",
      #   experiment_name= exp['name'].split('/')[-2] +"_"+ exp['name'].split('/')[-1], # Optional,
      #   params=params, # Optional,
      #   tags=[t1, exp['name'].split('/')[-2], exp['name'].split('/')[-1]] + exp["tag_list"], # Optional,
      #   close_after_fit = False,
      #   offline_mode = exp.get('offline_mode', False),
      #   upload_source_files=files,
      #   upload_stdout=False,
      #   upload_stderr=False
      # )
      
      # logger = NeptuneLogger(
      #   api_key=os.environ["NEPTUNE_API_TOKEN"],
      #   project_name="jonasfrey96/asl",
      #   experiment_id=exp.get('experiment_id',-1),
      #   close_after_fit = False,
      # )
    print('Neptune Experiment ID: '+ str( logger.experiment.id)+" TASK NR "+str( task_nr ) )
  else:
    logger = TensorBoardLogger(
      save_dir=model_path,
      name= 'tensorboard', # Optional,
      default_hp_metric=params, # Optional,
    )
  # else:
  #   logger = TensorBoardLogger(
  #       save_dir=model_path+'/rank/'+str(local_rank),
  #       name= exp['name'].split('/')[-2] +"_"+ exp['name'].split('/')[-1], # Optional,
  #   )
  
  weight_restore = exp.get('weights_restore', False) 
  checkpoint_load = exp['checkpoint_load']
  
  if local_rank == 0 and init:
    # write back the exp file with the correct name set to the model_path!
    # other ddp-task dont need to care about timestamps
    # also storeing the path to the latest.ckpt that downstream tasks can restore the model state
    exp['weights_restore_2'] = False
    exp['checkpoint_restore_2'] = True
    exp['checkpoint_load_2'] = os.path.join( model_path,'last.ckpt')
    
    rm = exp_cfg_path.find('cfg/exp/') + len('cfg/exp/')
    exp_cfg_path = os.path.join( exp_cfg_path[:rm],'tmp/',exp_cfg_path[rm:])
    Path(exp_cfg_path).parent.mkdir(parents=True, exist_ok=True) 
    with open(exp_cfg_path, 'w+') as f:
      yaml.dump(exp, f, default_flow_style=False, sort_keys=False)
  
  if not init:
    # restore model state from previous task.
    exp['checkpoint_restore'] = exp['checkpoint_restore_2']
    exp['checkpoint_load'] = exp['checkpoint_load_2']
    exp['weights_restore'] = exp['weights_restore_2']
  
  # Always use advanced profiler
  if exp['trainer'].get('profiler', False):
    exp['trainer']['profiler'] = AdvancedProfiler(output_filename=os.path.join(model_path, 'profile.out'))
  else:
    exp['trainer']['profiler']  = False
  
  # print( exp['trainer'] )
  # print(os.environ.get('GLOBAL_RANK'))
  if exp.get('checkpoint_restore', False):
    p = os.path.join( env['base'], exp['checkpoint_load'])
    trainer = Trainer( **exp['trainer'],
      default_root_dir = model_path,
      callbacks=cb_ls, 
      resume_from_checkpoint = p,
      logger=logger)
  else:
    trainer = Trainer(**exp['trainer'],
      default_root_dir=model_path,
      callbacks=cb_ls,
      logger=logger)   
    

  if exp['weights_restore'] :
    # it is not strict since the latent replay buffer is not always available
    p = os.path.join( env['base'],exp['checkpoint_load'])
    if os.path.isfile( p ):
      res = model.load_state_dict( torch.load(p,
        map_location=lambda storage, loc: storage)['state_dict'], 
        strict=False)
      print('Restoring weights: ' + str(res))
    else:
      raise Exception('Checkpoint not a file')
  
  main_visu = MainVisualizer( p_visu = os.path.join( model_path, 'main_visu'), 
                            logger=logger, epoch=0, store=True, num_classes=22 )
  
  tc = TaskCreator(**exp['task_generator'],output_size=exp['model']['input_size'])
  print(tc)
  _task_start_training = time.time()
  _task_start_time = time.time()
  
  for idx, out in enumerate(tc):
    if idx == task_nr:
      break 
  
  if True:
  #for idx, out in enumerate(tc):
    task, eval_lists = out
    main_visu.epoch = idx
    # New Logger
    print( f'<<<<<<<<<<<< TASK IDX {idx} TASK NAME : '+task.name+ ' >>>>>>>>>>>>>' )

    model._task_name = task.name
    model._task_count = idx
    dataloader_train, dataloader_buffer= get_dataloader_train(d_train= task.dataset_train_cfg,
                                                                env=env,exp = exp)
    print(str(dataloader_train.dataset))
    print(str(dataloader_buffer.dataset))
    dataloader_list_test = eval_lists_into_dataloaders(eval_lists, env=env, exp=exp)
    print( f'<<<<<<<<<<<< All Datasets are loaded and set up >>>>>>>>>>>>>' )
    #Training the model
    trainer.should_stop = False
    # print("GLOBAL STEP ", model.global_step)
    for d in dataloader_list_test:
      print(str(d.dataset))
    
    
    if idx < exp['start_at_task']:
      # trainer.limit_val_batches = 1.0
      trainer.limit_train_batches = 1
      trainer.max_epochs = 1
      trainer.check_val_every_n_epoch = 1
      train_res = trainer.fit(model = model,
                              train_dataloader= dataloader_train,
                              val_dataloaders= dataloader_list_test)
      
      trainer.max_epochs = exp['trainer']['max_epochs']
      trainer.check_val_every_n_epoch =  exp['trainer']['check_val_every_n_epoch']
      trainer.limit_val_batches = exp['trainer']['limit_val_batches']
      trainer.limit_train_batches = exp['trainer']['limit_train_batches']
    else:
      print('Train', dataloader_train)
      print('Val', dataloader_list_test)
      train_res = trainer.fit(model = model,
                              train_dataloader= dataloader_train,
                              val_dataloaders= dataloader_list_test)
    res = trainer.logger_connector.callback_metrics
    res_store = {}
    for k in res.keys():
      try:
        res_store[k] = float( res[k] )
      except:
        pass
    base_path = '/'.join( [a for a in model_path.split('/') if a.find('rank') == -1])
    with open(f"{base_path}/res{task_nr}.pkl", "wb") as f:
      pickle.dump(res_store, f)
    
    print( f'<<<<<<<<<<<< TASK IDX {idx} TASK NAME : '+task.name+ ' Trained >>>>>>>>>>>>>' )

    if exp.get('buffer',{}).get('fill_after_fit', False):
      print( f'<<<<<<<<<<<< Performance Test to Get Buffer >>>>>>>>>>>>>' )
      
      trainer.test(model=model,
                  test_dataloaders= dataloader_buffer)
    
      if local_rank == 0:
        checkpoint_callback.save_checkpoint(trainer, model)
      print( f'<<<<<<<<<<<< Performance Test DONE >>>>>>>>>>>>>' )
    
    number_validation_dataloaders = len( dataloader_list_test ) 
    
    if model._rssb_active:
      # visualize rssb
      bins, valids = model._rssb.get()
      fill_status = (bins != 0).sum(axis=1)
      main_visu.plot_bar( fill_status, x_label='Bin', y_label='Filled', title='Fill Status per Bin', sort=False, reverse=False, tag='Buffer_Fill_Status')
    
    plot_from_pkl(main_visu, base_path, task_nr)
    
    validation_acc_plot(main_visu, logger)
  
  try:
    if close:
      logger.experiment.stop()
  except:
    pass