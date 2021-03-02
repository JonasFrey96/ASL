from log import _create_or_get_experiment2
from pytorch_lightning.loggers.neptune import NeptuneLogger
import os
from pathlib import Path
import torch
try:
    from .utils import flatten_dict
except Exception:
    from utils import flatten_dict

__all__ = ['get_neptune_logger']

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
  return dic

def get_neptune_logger(exp,env,exp_p, env_p, project_name="jonasfrey96/asl"):
  params = log_important_params( exp )
  cwd = os.getcwd()
  files = [str(p).replace(cwd+'/','') for p in Path(cwd).rglob('*.py') 
          if str(p).find('vscode') == -1]
  files.append( exp_p )
  files.append( env_p )
    
  if env['workstation']:
    t1 = 'workstation'
  else:
    t1 = 'leonhard'
    NeptuneLogger._create_or_get_experiment = _create_or_get_experiment2

  gpus = 'gpus_'+str(torch.cuda.device_count())
  return NeptuneLogger(
    api_key=os.environ["NEPTUNE_API_TOKEN"],
    project_name=project_name,
    experiment_name= exp['name'].split('/')[-2] +"_"+ exp['name'].split('/')[-1], # Optional,
    params=params, # Optional,
    tags=[t1, exp['name'].split('/')[-2], exp['name'].split('/')[-1], gpus] + exp["tag_list"], # Optional,
    close_after_fit = False,
    offline_mode = exp.get('offline_mode', False),
    upload_source_files=files,
    upload_stdout=True,
    upload_stderr=True
  )

