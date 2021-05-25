import copy
from task import TaskGenerator, Task

__all__ = ['TaskGeneratorScannet']

scannet_template_dict = { 
    'name': 'scannet',
    'mode': 'train', 
    'output_size': [320,640],
    'scenes': [],    
    'data_augmentation': True,
    'label_setting': "default"
}

class TaskGeneratorScannet( TaskGenerator ):
  def __init__(self, mode, cfg, *args, **kwargs):
    # SET ALL TEMPLATES CORRECT
    super(TaskGeneratorScannet, self).__init__()
    
    for k in cfg['copy_to_template'].keys():
       scannet_template_dict[k] = cfg['copy_to_template'][k]

    mode_cfg = cfg.get(mode,{})
    if mode == 'scannet_scenes':
      self._scannet_scenes( **mode_cfg )
      
    elif mode == 'scannet_pretrain':
      self._scannet_pretrain( **mode_cfg )

    elif mode == 'scannet_auxilary_labels':
      self._scannet_auxilary_labels( **mode_cfg )

    else:
      raise AssertionError('TaskGeneratorScannet: Undefined Mode')
    self._current_task = 0
    self._total_tasks = len(self._task_list)
  
  def _scannet_auxilary_labels( self, label_setting="default" ):
    train = copy.deepcopy( scannet_template_dict )
    val = copy.deepcopy( scannet_template_dict )
    train['mode'] = 'train'
    val['mode'] = 'val'
    
    # Define the first pretrain task
    train['scenes'] = [f'scene{s:04d}' for s in range(10,60)]
    val['scenes'] = train['scenes']
    i = 0
    t = Task(name = f'Train_{i}',
              dataset_train_cfg= copy.deepcopy(train),
              dataset_val_cfg= copy.deepcopy(val))
    self._task_list.append(t)

    start_scene_train = 0
    scenes_per_task = 1
    for i in range( 1 ):
      # GENERATE TRAIN TASK        
      train['scenes'] = [f'scene{s:04d}' for s in range(start_scene_train, start_scene_train+scenes_per_task )]
      train['label_setting'] = label_setting

      val['scenes'] = train['scenes']
      val['label_setting'] = label_setting

      t = Task(name = f'Train_{i+1}',
                dataset_train_cfg= copy.deepcopy(train),
                dataset_val_cfg= copy.deepcopy(val))
      self._task_list.append(t)
      start_scene_train += scenes_per_task



  def _scannet_pretrain( self ):
    train = copy.deepcopy( scannet_template_dict )
    val = copy.deepcopy( scannet_template_dict )
    train['mode'] = 'train'
    val['mode'] = 'val'
    
    train['scenes'] = [f'scene{s:04d}' for s in range(10,60)]
    val['scenes'] = train['scenes']
    t = Task(name = f'Train_{i}',
              dataset_train_cfg= copy.deepcopy(train),
              dataset_val_cfg= copy.deepcopy(val))
    self._task_list.append(t)

  def _scannet_scenes(self, number_of_tasks, scenes_per_task):
    train = copy.deepcopy( scannet_template_dict )
    val = copy.deepcopy( scannet_template_dict )
    train['mode'] = 'train'
    val['mode'] = 'val'
    
    start_scene_train = 0
    for i in range( number_of_tasks ):
      # GENERATE TRAIN TASK        
      train['scenes'] = [f'scene{s:04d}' for s in range(start_scene_train, start_scene_train+scenes_per_task )]
      val['scenes'] = train['scenes']
      t = Task(name = f'Train_{i}',
                dataset_train_cfg= copy.deepcopy(train),
                dataset_val_cfg= copy.deepcopy(val))
      self._task_list.append(t)
      start_scene_train += scenes_per_task
    

def test():
  import sys, os
  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + '/src'))

  from utils_asl import load_yaml
  exp = load_yaml( os.path.join(os.getcwd() + '/cfg/test/test.yml'))
  env = load_yaml(os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml'))

  tg = TaskGeneratorScannet( 
    mode = exp['task_generator']['mode'], 
    cfg = exp['task_generator']['cfg'] )
  
  print(tg)
  return True

if __name__ == "__main__":
  test()
  