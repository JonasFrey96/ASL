# Naming Adapter TaskGenerator to Ensemble dataset
# Everything below is not aware of continual learning
# This sets the replay ratios !

from math import ceil

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from datasets_asl import get_dataset
from datasets_asl import Ensemble

__all__ = ['adapter_tg_to_dataloader']

def replay_cfg_to_probs(replay_cfg, nr):
  # get the rehearsel probabilieties
  # probs[-1] is probability of current task
  # probs[0] is the rehearsel probabiliety of the task firstly trained on
  probs = []
  if replay_cfg['active']:
    if type( replay_cfg['probs'] ) is float:
      probs = [replay_cfg['probs']] * (nr-1)
      val = 1 - np.array(probs).sum()
      assert val > 0 and val < 1
      probs.append( val )
      probs =  [replay_cfg['probs']] * nr
    else:
      if len(replay_cfg['probs']) < nr:
        raise ValueError("To few user defined probs in replay cfg! Give float or add entries to list")
      probs = replay_cfg['probs'][:nr]
  else:
    # dont use replay at all
    probs = [0] * nr
    probs[-1] = 1

  # normalize valid probability distribution
  probs = (np.array(probs) / np.array(probs).sum()).tolist() 
  return probs

def adapter_tg_to_en( tg, task_nr, replay_cfg, env):
  # TODO: Jonas Frey decide if we should stick with the generator implementation here

  # accumulate train datasets and then wrap them together
  train_dataset_list = []
  output_transform = transforms.Compose([
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
  ])

  train_dataset_list = [] 
  val_dataset_list = []
  for idx, task in enumerate(tg):
    if idx < task_nr+1:
      # add it train_dataset_list
      train_dataset_list.append( get_dataset(
        **task.dataset_train_cfg,
        env = env,
        output_trafo = output_transform,
      ))
    val_dataset_list.append( get_dataset(
      **task.dataset_val_cfg,
      env = env,
      output_trafo = output_transform,
    ))
  if len(train_dataset_list) == 1:
    # no ensemble needed
    train_dataset = train_dataset_list[0]
  else:
    probs = replay_cfg_to_probs( replay_cfg, len(train_dataset_list) )
    train_dataset = Ensemble( main_dataset = train_dataset_list[-1], 
                              replay_datasets = train_dataset_list[:-2],
                              probs = probs)

  return train_dataset, val_dataset_list

def adapter_tg_to_dataloader(tg, task_nr, loader_cfg, replay_cfg, env ):
  train_dataset, val_dataset_list = adapter_tg_to_en( tg, task_nr, replay_cfg, env)

  train_dataloader = DataLoader(train_dataset,
    shuffle = loader_cfg['shuffle'],
    num_workers = ceil(loader_cfg['num_workers']/torch.cuda.device_count()),
    pin_memory = loader_cfg['pin_memory'],
    batch_size = loader_cfg['batch_size'], 
    drop_last = True)

  val_dataloader_list = [
    DataLoader(d,
    shuffle = loader_cfg['shuffle'],
    num_workers = ceil(loader_cfg['num_workers']/torch.cuda.device_count()),
    pin_memory = loader_cfg['pin_memory'],
    batch_size = loader_cfg['batch_size'], 
    drop_last = True) for d in val_dataset_list
  ]

  return train_dataloader, val_dataloader_list 


def test():
  import sys, os
  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + '/src'))

  from utils_asl import load_yaml
  
  exp = load_yaml( os.path.join(os.getcwd() + '/cfg/test/test.yml'))
  env = load_yaml(os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml'))

  from task import TaskGeneratorScannet
  
  tg = TaskGeneratorScannet( 
    mode = exp['task_generator']['mode'], 
    cfg = exp['task_generator']['cfg'] )
  
  train, vals = adapter_tg_to_dataloader(tg, 0, exp['loader'], exp['replay']['cfg'], env )
  print(tg)
  print(train)
  print(vals)


  tg = TaskGeneratorScannet( 
    mode = exp['task_generator']['mode'], 
    cfg = exp['task_generator']['cfg'] )
  train, vals = adapter_tg_to_dataloader(tg, 2, exp['loader'], exp['replay']['cfg'], env )

  return True

if __name__ == "__main__":
  test()
  