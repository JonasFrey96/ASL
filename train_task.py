import os 
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))

import time
import shutil
import datetime
import argparse
import yaml
import copy
from pathlib import Path
import pickle

# Frameworks
import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.loggers.neptune import NeptuneLogger

# Costume Modules
from lightning import Network
from visu import MainVisualizer, plot_from_pkl, validation_acc_plot, plot_from_neptune
from callbacks import TaskSpecificEarlyStopping
from utils_asl import load_yaml, file_path
from utils_asl import get_neptune_logger, get_tensorboard_logger

from datasets_asl import eval_lists_into_dataloaders, get_dataloader_train
from task import TaskCreator
__all__ = ['train_task']

def train_task( init, close, exp_cfg_path, env_cfg_path, task_nr, logger_pass=None):
  
  seed_everything(42)
  local_rank = int(os.environ.get('LOCAL_RANK', 0))

  # LOADING THE CONFIGURATION
  if local_rank != 0 or not init:
    print(init, local_rank)
    rm = exp_cfg_path.find('cfg/exp/') + len('cfg/exp/')
    exp_cfg_path = os.path.join( exp_cfg_path[:rm],'tmp/',exp_cfg_path[rm:])

  exp = load_yaml(exp_cfg_path)
  env = load_yaml(env_cfg_path)

  # CREATE EXPERIMENTS FOLDER + MOVE THE CONFIG FILES
  if local_rank == 0 and init:
    # Set in name the correct model path
    if exp.get('timestamp',True):
      timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
      
      model_path = os.path.join(env['base'], exp['name'])
      p = model_path.split('/')
      model_path = os.path.join('/',*p[:-1] ,str(timestamp)+'_'+ p[-1] )
    else:
      model_path = os.path.join(env['base'], exp['name'])
      shutil.rmtree(model_path,ignore_errors=True)
    
    # Create the directory
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
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
    Path(model_path).mkdir(parents=True, exist_ok=True)

  # COPY DATASET
  if env['workstation'] == False:
    # use proxy hack for neptunai !!!
    # move data to ssd
    if exp['move_datasets'][0]['env_var'] != 'none':
      for dataset in exp['move_datasets']:
        scratchdir = os.getenv('TMPDIR')
        print( 'TMPDIR directory: ', scratchdir )
        env_var = dataset['env_var']
        tar = os.path.join( env[env_var],f'{env_var}.tar')
        name = (tar.split('/')[-1]).split('.')[0]
          
        if not os.path.exists(os.path.join(scratchdir,dataset['env_var']) ):
          try:
            if tar.find('labels') != -1:
              target = "$TMPDIR/scannet/scans/"
            else:
              target = "$TMPDIR"

            cmd = f"tar -xvf {tar} -C {target} >/dev/null 2>&1"
            st =time.time()
            print( f'Start moveing dataset-{env_var}: {cmd}')
            os.system(cmd)
            env[env_var] = str(os.path.join(scratchdir, name))
            new_env_var = env[env_var]
            print( f'Finished moveing dataset-{new_env_var} in {time.time()-st}s')
          except:
              rank_zero_warn( 'ENV Var'+ env_var )
              env[env_var] = str(os.path.join(scratchdir, name))
              rank_zero_warn('Copying data failed')
        else:
          env[env_var] = str(os.path.join(scratchdir, name))
    else:
      env['mlhypersim'] = str(os.path.join(env['mlhypersim'], 'mlhypersim'))
      
  # SET GPUS
  if ( exp['trainer'] ).get('gpus', -1) == -1:
    nr = torch.cuda.device_count()
    print( f'Set GPU Count for Trainer to {nr}!' )
    for i in range(nr):
      print( f"Device {i}: ", torch.cuda.get_device_name(i) )
    exp['trainer']['gpus'] = -1
  

  # MODEL
  model = Network(exp=exp, env=env)
  
  # COLLECT CALLBACKS
  lr_monitor = LearningRateMonitor(
    **exp['lr_monitor']['cfg'])
  if exp['cb_early_stopping']['active']:
    early_stop_callback = EarlyStopping(
    **exp['cb_early_stopping']['cfg']
    )
    cb_ls = [early_stop_callback, lr_monitor]
  else:
    cb_ls = [lr_monitor]
  if exp['task_specific_early_stopping']['active']:
    tses = TaskSpecificEarlyStopping(
      nr_tasks=exp['task_generator']['total_tasks'] , 
      **exp['task_specific_early_stopping']['cfg']
    )
    cb_ls.append(tses)
  if exp['cb_checkpoint']['active']:
    for i in range(exp['task_generator']['total_tasks']):
      if i == task_nr:
        m = '/'.join( [a for a in model_path.split('/') if a.find('rank') == -1])
        dic = copy.deepcopy( exp['cb_checkpoint']['cfg'])
        checkpoint_callback = ModelCheckpoint(
          dirpath= m,
          filename= 'task'+str(i)+'-{epoch:02d}--{step:06d}',
          **dic
        )
        cb_ls.append( checkpoint_callback )
  
  # GET LOGGER
  if not exp.get('offline_mode', False):
    if  logger_pass is None and exp.get('experiment_id',None) is None:
      logger = get_neptune_logger(exp=exp,env=env,
        exp_p =exp_cfg_path, env_p = env_cfg_path)
      exp['experiment_id'] = logger.experiment.id
      print('Created Experiment ID: ' +  str( exp['experiment_id']))
    else:
      logger = NeptuneLogger(
          api_key=os.environ["NEPTUNE_API_TOKEN"],
          project_name=env['neptune_project_name'],
          experiment_id= exp['experiment_id'],
          close_after_fit = False,
          upload_stdout=True,
          upload_stderr=True
      )
    print('Neptune Experiment ID: '+ str( logger.experiment.id)+" TASK NR "+str( task_nr ) )
  else:
    logger = get_tensorboard_logger(exp=exp,env=env, exp_p =exp_cfg_path, env_p = env_cfg_path)
  

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
  
  # CHECKPOINT
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
      state_dict_loaded = torch.load(p,
        map_location=lambda storage, loc: storage)['state_dict']
      if state_dict_loaded['_rssb.bins'].shape != model._rssb.bins.shape:
        state_dict_loaded['_rssb.bins'] = model._rssb.bins
        state_dict_loaded['_rssb.valid'] = model._rssb.valid

      res = model.load_state_dict( state_dict_loaded, 
        strict=False)
      print('Restoring weights: ' + str(res))
    else:
      raise Exception('Checkpoint not a file')
  
  main_visu = MainVisualizer( p_visu = os.path.join( model_path, 'main_visu'), 
                            logger=logger, epoch=0, store=True, num_classes=exp['model']['cfg']['num_classes']+1)
  
  tc = TaskCreator(**exp['task_generator'],output_size=exp['model']['input_size'])
  
  
  for idx, out in enumerate(tc):
    if idx == task_nr:
      break 
  
  task, eval_lists = out
  main_visu.epoch = idx
  # New Logger
  print( f'<<<<<<<<<<<< TASK IDX {idx} TASK NAME : '+task.name+ ' >>>>>>>>>>>>>' )

  model._task_name = task.name
  model._task_count = idx
  # Normal Validation datloader is ignored !
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
    # trainer.limit_val_batches = 1 # TODO: Jonas Frey remove this
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


  # SET THE REPLAY BUFFER FAST RANDOMLY !
  if (exp.get('buffer',{}).get('fill_after_fit', False) and 
    exp.get('buffer',{}).get('mode', 'random')  == 'random'):
    global_idx_list = torch.tensor( dataloader_buffer.dataset.global_to_local_idx, device=model.device)
    print("USE RSSB NOW IS ACTIVE")
    print( global_idx_list.shape, model._rssb.bins.shape)
    selected = torch.randperm( global_idx_list.shape[0], device = model.device )[:model._rssb.bins.shape[1]]
    ret_globale_indices = global_idx_list[selected]
    if exp['buffer'].get('sync_back', True):
          if model._rssb_active:
            model._rssb.bins[ model._task_count,:] = ret_globale_indices
            model._rssb.valid[ model._task_count,:] = True

            print( "BINS: ", model._rssb.bins[ model._task_count,:])
            print( "VALID: ", model._rssb.valid[ model._task_count,:])
            checkpoint_callback.save_checkpoint(trainer, model)
            

  # START THE COMPLEX REPLAY BUFFER FILLING BY ITERATING OVER THE FULL DATASET
  elif exp.get('buffer',{}).get('fill_after_fit', False):
    print( f'<<<<<<<<<<<< Performance Test to Get Buffer >>>>>>>>>>>>>' )
    trainer.test(model=model,
                test_dataloaders= dataloader_buffer)
  
    checkpoint_callback.save_checkpoint(trainer, model)
    print( f'<<<<<<<<<<<< Performance Test DONE >>>>>>>>>>>>>' )
  
  number_validation_dataloaders = len( dataloader_list_test )
  if model._rssb_active:
    # visualize rssb
    bins, valids = model._rssb.get()
    fill_status = (bins != 0).sum(axis=1)
    main_visu.plot_bar( fill_status, x_label='Bin', y_label='Filled', title='Fill Status per Bin', sort=False, reverse=False, tag='Buffer_Fill_Status')
  
  try:
    plot_from_pkl(main_visu, base_path, task_nr)
  except:
    print("Failde because not implemented when restarting training")
  
  validation_acc_plot(main_visu, logger)
  
  try:
    logger.experiment.stop()
  except:
    pass

if __name__ == "__main__":

  parser = argparse.ArgumentParser()    
  parser.add_argument('--exp', type=file_path, default='cfg/exp/scannet/exp.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--task_nr', type=int, default=0,
                      help='Task nr.')
  parser.add_argument('--init', type=int, default=1,
                      help='Task nr.')
  parser.add_argument('--close', type=int, default=1,
                      help='Task nr.')  
  args = parser.parse_args()

  print('Train Task called as MAIN with the following arguments: '+ str(args))  
  env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
  
  train_task( bool(args.init),  bool(args.close), args.exp, env_cfg_path, args.task_nr)
  