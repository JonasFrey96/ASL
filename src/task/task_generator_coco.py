import copy
from task import TaskGenerator, Task

__all__ = ['TaskGeneratorCoco']

coco_template_dict = { 
    'name': 'coco',
    'mode': 'train', 
    'output_size': [320,640],
    'scenes': [],    
    'data_augmentation': True,
}

class TaskGeneratorCoco( TaskGenerator ):
  def __init__(self, mode, cfg, *args, **kwargs):
    # SET ALL TEMPLATES CORRECT
    super(TaskGeneratorCoco, self).__init__()

    mode_cfg = cfg.get(mode,{})
    if mode == 'coco_pretrain':
      self._coco_pretrain( **mode_cfg )

    else:
      raise AssertionError('TaskGeneratorCoco: Undefined Mode')
    self._current_task = 0
    self._total_tasks = len(self._task_list)
  

  def _coco_pretrain( self ):
    train = copy.deepcopy( coco_template_dict )
    val = copy.deepcopy( coco_template_dict )
    train['mode'] = 'train'
    val['mode'] = 'val'
    i = 0
    t = Task(name = f'Train_{i}',
              dataset_train_cfg= copy.deepcopy(train),
              dataset_val_cfg= copy.deepcopy(val))
    self._task_list.append(t)


def test():
  import sys, os
  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + '/src'))

  from utils_asl import load_yaml
  exp = load_yaml( os.path.join(os.getcwd() + '/cfg/test/test.yml'))
  env = load_yaml(os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml'))

  #TODO: Jonas Frey
  return True

if __name__ == "__main__":
  test()
  