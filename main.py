
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
  dataset_train = get_dataset(
    **d_test,
    env = env,
    output_trafo = output_transform,
  )
  dataloader_train = torch.utils.data.DataLoader(dataset_train,
    shuffle = exp['loader']['shuffle'],
    num_workers = ceil(exp['loader']['num_workers']/torch.cuda.device_count()),
    pin_memory = exp['loader']['pin_memory'],
    batch_size = exp['loader']['batch_size'], 
    drop_last = True)
  return dataloader_train
  
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
    
    
  
  #  log_dir=None, comment='', purge_step=None, max_queue=10,
  #                flush_secs=120, filename_suffix=''
                
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
      
      model._task_name = task.name
      model._task_count = idx
      dataloader_train, dataloader_val = get_dataloader_train_val(d_train= task.dataset_train_cfg,
                                                                  d_val= task.dataset_val_cfg,
                                                                  env=env , replay_state = replay_state)
      dataloader_list_test = eval_lists_into_dataloaders( eval_lists, env)
      
      #Training the model
      train_res = trainer.fit(model = model,
                              train_dataloader= dataloader_train,
                              val_dataloaders= dataloader_list_test )
      bins, valids = trainer.train_dataloader.dataset.get_full_state()
      replay_state = { 'bins':bins, 'valids': valids, 'bin': idx }
      
      training_results.append( trainer.logged_metrics )
      # test_results = []
      # for eval_task in eval_lists:
      #   rank_zero_info( "Executing Evaluation Task: "+ eval_task.name + 
      #                   " of Training Task: " + task.name )
      #   suc = model.set_test_dataset_cfg( dataset_test_cfg=eval_task.dataset_test_cfg, task_name=eval_task.name)
      #   if not suc:
      #     rank_zero_warn( "Evaluation Task "+ eval_task.name + 
      #                     " not started. Not enough test data!")
      #     continue
        
      #   # TODO: one task might call mutiple tests!
      #   test_res = trainer.test(model)
      #   new_res= {}
      #   for k in test_res[0].keys():
      #     try:
      #       new_res[k] = float( test_res[0][k])
      #     except:
      #       pass
            
      #   test_results.append(new_res)
      
      # ## create diagonal metric for all test resutls
      # for met in training_results[0].keys():
      #   dia = np.zeros( (len(training_results),len(training_results)))
      #   suc = True
      #   try:
      #     for _i in range( len(training_results)):
      #       dia[_i,_i] = int( training_results[_i][met]* 100 )
               
      #   except:
      #     suc = False
      #   if suc: 
          
      #     main_visu.plot_matrix(
      #       tag = f'train_{met}',
      #       data_matrix = dia)
      
      # ## get test mIoU to right format to log as matrix 
      # results.append({'Time': idx, 'Traning': train_res, 'Test': test_results})
      # mIoU = []
      # for i, task in enumerate( results):
      #   test_vals = []
      #   for j, test in enumerate(task['Test']):
      #     test_vals.append( test['test_mIoU'])
      #   mIoU.append(test_vals)
      
      # max_tests = 0
      # for task in mIoU:
      #   if len(task) > max_tests :
      #     max_tests  = len(task)
      
      # for i in range(len( mIoU)):
      #   if len(mIoU[i]) < max_tests :
      #     mIoU[i] = mIoU[i] + [0.0] * (max_tests -len(mIoU[i]))
      # data_matrix = np.array( mIoU) * 100
      # data_matrix = np.around(data_matrix , decimals=0)
      # main_visu.plot_matrix(
      #   tag = 'mIoU',
      #   data_matrix = data_matrix)
      
  elif  mode == 'paper_analysis':
    # This mode should not be used for training. 
    # Here we train for a fixed amount of steps and perform a full eval of all tasks.
    # This is really time consuming but gives a nice graph.
    
    samples_per_training = exp.get('paper_analysis', {}).get('samples_per_training',25)
    steps_per_sample = exp.get('paper_analysis', {}).get('steps_per_sample',200)
    exp['trainer']['limit_train_batches'] = steps_per_sample
    exp['trainer']['limit_val_batches'] = 1
    exp['trainer']['limit_test_batches'] = 1.0
    
    task_data = []
    for idx, out in enumerate(tc) :  
      # reset model checkpoint storeing for each training task!
      checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        **exp['cb_checkpoint']['cfg']
      )
      # create a new Trainer. 
      trainer = Trainer(**exp['trainer'],
        default_root_dir=model_path,
        callbacks=cb_ls,
        max_steps=steps_per_sample
      )
      
      if idx != 0:
        print(f'Start training samples {j}')
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

      val_steps = []
      for _j, eval_task in enumerate( eval_lists):
        val_steps.append([])  
        
      for j in range(samples_per_training):
        trainer.current_steps = 0
        s =  f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Task name {task.name}, {j}/{samples_per_training}, Task IDX {idx} >>>>>>>>>>>>>>>>>>>>>>>>>>'
        rank_zero_info(s)
        
        train_res = trainer.fit(model)
        training_results.append( train_res )
        test_results = []
        for _j, eval_task in enumerate( eval_lists):
          rank_zero_info( "Executing Evaluation Task: "+ eval_task.name + 
                          " of Training Task: " + task.name )
          suc = model.set_test_dataset_cfg( dataset_test_cfg=eval_task.dataset_test_cfg, task_name=eval_task.name)
          if not suc:
            rank_zero_warn( "Evaluation Task "+ eval_task.name + 
                            " not started. Not enough test data!")
            continue
            
          # TODO: one task might call mutiple tests!
          test_res = trainer.test(model)
          test_results.append(test_res)
          
          val_steps[_j].append( [ steps_per_sample*j+ idx*(steps_per_sample*(samples_per_training-1)),test_res[0]['test_mIoU']] )
      
      arr = np.array(val_steps)
      ls_res = []
      ls_eval_names = []
      for _j, eval_task in enumerate( eval_lists):
        ls_eval_names.append( eval_task.name )
      for __i  in range(arr.shape[0]):
        ls_res.append( (arr[__i,:,0],arr[__i,:,1]) )
      T1 = {'name': task.name, 'val_task_results': ls_res, 'eval_names': ls_eval_names}
      task_data.append(T1)
      main_visu.plot_cont_validation_eval(task_data = task_data, tag='TaskData')
        # res1 =  np.linspace(0., 0.5, 6)
        # res2 =  np.linspace(0., 0.5, 6)*0.5
        # res3 =  np.linspace(0., 0.5, 6)**2
        # T1 = {'name': 'TrainTask1' ,'val_task_results': [(np.arange(0,6), res1), (np.arange(0,6), res2), (np.arange(0,6), res3) ] }
        # T2 = {'name': 'TrainTask2' ,'val_task_results': [(np.arange(5,11), res1), (np.arange(5,11),res2), (np.arange(5,11),res3) ] }
        # T3 = {'name': 'TrainTask3' ,'val_task_results': [(np.arange(10,16),res1), (np.arange(10,16),res2), (np.arange(10,16),res3) ] }
        # task_data = [T1, T2]
        
      #   new_res= {}
      #   for k in test_res[0].keys():
      #     try:
      #       new_res[k] = float( test_res[0][k])
      #     except:
      #       pass
            
      #   test_results.append(new_res)
        
      # results.append({'Time': idx, 'Traning': train_res, 'Test': test_results})
      # mIoU = []
      # for i, task in enumerate( results):
      #   test_vals = []
      #   for j, test in enumerate(task['Test']):
      #     test_vals.append( test['test_mIoU'])
      #   mIoU.append(test_vals)
      
      # max_tests = 0
      # for task in mIoU:
      #   if len(task) > max_tests :
      #     max_tests  = len(task)
      
      # for i in range(len( mIoU)):
      #   if len(mIoU[i]) < max_tests :
      #     mIoU[i] = mIoU[i] + [0.0] * (max_tests -len(mIoU[i]))
      # data_matrix = np.array( mIoU) * 100
      # data_matrix = np.around(data_matrix , decimals=0)
      # main_visu.plot_matrix(
      #   tag = 'mIoU',
      #   data_matrix = data_matrix)