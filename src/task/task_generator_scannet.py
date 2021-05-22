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

    if mode == 'scannet_scenes':
      self._scannet_scenes( **cfg['scannet_scenes'])
    else:
      raise AssertionError('TaskGeneratorScannet: Undefined Mode')
    self._current_task = 0

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
  