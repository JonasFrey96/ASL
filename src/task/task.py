"""
Hardcode everything in the task and make it more flexible when needed
A task is not responsible how it is trained this is fully decided by the algorithm itself. 
It is simply responsible for providing data and the logging interface.
"""

if __name__ == "__main__": 
  import os 
  import sys 
  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + '/src'))

import torch
from torchvision import transforms as tf
from datasets_asl import NYUv2
from datasets_asl import MLHypersim
from datasets_asl import ScanNet
import numpy as np
import copy

__all__ = ['Task', 'TaskCreator']

class Task():
  "Might also be implemeted as a dataclass."
  def __init__(self, name, dataset_train_cfg, dataset_val_cfg):
    self.name = name
    self.dataset_train_cfg = dataset_train_cfg
    self.dataset_val_cfg = dataset_val_cfg
    
class EvalTask():
  "Might also be implemeted as a dataclass."
  def __init__(self, name, dataset_test_cfg):
    self.name = name
    self.dataset_test_cfg = dataset_test_cfg
    
nyu_template_dict = { 
    'name': 'nyuv2',
    'mode': 'val', 
    'output_size': 384,
    'load_all': True,
    'scenes': []
}

coco_template_dict = { 
    'name': 'coco',
    'mode': 'train', 
    'output_size': 384,
    'scenes': [],
    'replay': True,
    'squeeze_80_labels_to_40': True,
}

mlhypersim_template_dict = { 
    'name': 'mlhypersim',
    'mode': 'train', 
    'output_size': 384,
    'scenes': [],
    'replay': False,
    'cfg_replay':{'bins':4, 'elements':100, 'add_p': 0.5, 'replay_p':0.5, 'current_bin': 0},
    'data_augmentation': True,
    'data_augmentation_for_replay': True
}
mlhypersim_scene_names = MLHypersim.get_classes()

scannet_template_dict = { 
    'name': 'scannet',
    'mode': 'train', 
    'output_size': 384,
    'scenes': [],
    'replay': True,
    'cfg_replay':{'bins':4, 'elements':100, 'add_p': 0.5, 'replay_p':0.5, 'current_bin': 0},
    'data_augmentation': True,
    'data_augmentation_for_replay': True
}

template_list = [nyu_template_dict, coco_template_dict, mlhypersim_template_dict, scannet_template_dict]

nyu_scene_names, nyu_class_counts = NYUv2.get_classes('train')

class TaskCreator():
  """Iteratable object which returns Tasks
  Implements pythonic iterator protocol
  __iter__() and __next__()
  
  This object might be passed to an Task executor with the model
  and the trainer or simply iterated over in a for loop!
  """
  def __init__(self, mode= 'SingleScenesCountsDescending', output_size=384, *args, **kwargs):
    # SET ALL TEMPLATES CORRECT
    for t in template_list:
      ktt = kwargs.get('copy_to_template',{})
      for k in ktt.keys():
        t[k] = ktt[k]
    
    self._task_list = []
    self._eval_lists = []
    
    self.replay_adaptive_add_p = kwargs.get('replay_adaptive_add_p',False)
    self.scannet_strict_split = kwargs.get('scannet_strict_split',False)
    self.scannet_small_factor = kwargs.get('scannet_small_factor',1)

    if mode == 'SingleScenesCountsDescending':
      self._getTaskSingleScenesCountsDescending()  
    elif mode == 'All':
      self._getAll()
    elif mode == 'FourCategories':
      self._get4Categories()
    elif mode == 'pretrainCOCO':
      self._pretrainCOCO()
    elif mode == 'mlhypersim_random10':
      self._mlhypersim_random10()
    elif mode == 'mlhypersim_random10_test_all':
      self._mlhypersim_random10_test_all()
    elif mode == 'mlhypersim_all':
      self._mlhypersim_all()
    elif mode == 'mlhypersim_random4_tests':
      self._mlhypersim_random4_tests()
    elif mode == 'mlhypersim_random4_seeded':
      self._mlhypersim_random4_seeded(seed=kwargs['seed'])
    elif mode == 'scannet':
      self._scannet()
    elif mode == 'scannet_scenes':
      self._scannet()
    elif mode == 'scannet_continual_learning':
      self._scannet_continual_learning(kwargs['total_tasks'])
    else:
      raise AssertionError('TaskCreator: Undefined Mode')
    self._current_task = 0

  def _scannet_continual_learning(self, total_tasks):
    # Dont use the strict split !
    classes = ScanNet.get_classes()
    classes_per_task = int( int( len(classes) /4 ) / self.scannet_small_factor)

    train = copy.deepcopy( scannet_template_dict )
    val = copy.deepcopy( scannet_template_dict )
    
    train['mode'] = 'train'
    val['mode'] = 'val'
    
    seed = 0
    for i in range( total_tasks ):
      train_i = copy.deepcopy( train )
      train_i['scenes'] = classes[int( classes_per_task)*i:int( classes_per_task*(i+1))]
      
      if self.replay_adaptive_add_p:
        if i == seed:
          train_i['cfg_replay']['replay_p'] = 0.0
        else:
          train_i['cfg_replay']['replay_p'] = float(i-seed)/float(i+1-seed)

      t = Task(name = f'Scannet_Task_Train_All',
                dataset_train_cfg= train_i,
                dataset_val_cfg= val)
      self._task_list.append(t)
      
      eval_tasks = []
      for j in range(total_tasks):
        val_j = copy.deepcopy( val )

        val_j['scenes'] = classes[int( classes_per_task)*j:int( classes_per_task*(j+1))]
        eval_task = EvalTask(
          name = f'Scannet_Eval_All',
          dataset_test_cfg=val_j)
        eval_tasks.append( eval_task )
      self._eval_lists.append( eval_tasks )

  def _scannet(self):
    train = copy.deepcopy( scannet_template_dict )
    val = copy.deepcopy( scannet_template_dict )
    train['mode'] = 'train'
    val['mode'] = 'val'
    if self.scannet_strict_split:
      train['mode'] += '_strict'
      val['mode'] += '_strict'

    train['scenes'] = []
    val['scenes'] = []
    t = Task(name = f'Scannet_Task_Train_All',
              dataset_train_cfg= train,
              dataset_val_cfg= val)
    self._task_list.append(t)
    eval_tasks = []
    eval_task = EvalTask(
      name = f'Scannet_Eval_All',
      dataset_test_cfg=val)
    eval_tasks.append( eval_task )
    self._eval_lists.append( eval_tasks )
  
  def _mlhypersim_random10(self):
    spt = int( len(mlhypersim_scene_names)/10 ) # scenes_per_task spt

    for i in range(0,10):
      train = copy.deepcopy( mlhypersim_template_dict )
      val = copy.deepcopy( mlhypersim_template_dict )
      train['mode'] = 'train'
      val['mode'] = 'val'
      train['scenes'] = mlhypersim_scene_names[i*spt:(i+1)*spt]
      val['scenes'] = mlhypersim_scene_names[i*spt:(i+1)*spt]
      
      task_idx = str(i).zfill(2)
      t = Task(name = f'Task_{task_idx}_mlhyper_random10',
                dataset_train_cfg= train,
                dataset_val_cfg= val)
      self._task_list.append(t)
      
      # Get eval tasks
      eval_tasks = []
      for j in range(i+1):
        test = copy.deepcopy( mlhypersim_template_dict )
        test['mode'] = 'val'
        test['scenes'] = mlhypersim_scene_names[j*spt:(j+1)*spt]
        eval_idx = str(j).zfill(2)
        sc = f'{j*spt}-{(j+1)*spt}'
        eval_task = EvalTask(
          name = f'Eval_{eval_idx}_Scene_{sc}',
          dataset_test_cfg=test)
        eval_tasks.append( eval_task )
      self._eval_lists.append( eval_tasks )
      
  def _mlhypersim_random10_test_all(self):
    spt = int( len(mlhypersim_scene_names)/10 ) # scenes_per_task spt

    for i in range(0,10):
      train = copy.deepcopy( mlhypersim_template_dict )
      val = copy.deepcopy( mlhypersim_template_dict )
      train['mode'] = 'train'
      val['mode'] = 'val'
      train['scenes'] = mlhypersim_scene_names[i*spt:(i+1)*spt]
      val['scenes'] = mlhypersim_scene_names[i*spt:(i+1)*spt]
      
      task_idx = str(i).zfill(2)
      t = Task(name = f'Task_{task_idx}_mlhyper_random10_test_all',
                dataset_train_cfg= train,
                dataset_val_cfg= val)
      self._task_list.append(t)
      
      # Get eval tasks
      eval_tasks = []
      for j in range(0,10):
        test = copy.deepcopy( mlhypersim_template_dict )
        test['mode'] = 'val'
        test['scenes'] = mlhypersim_scene_names[j*spt:(j+1)*spt]
        eval_idx = str(j).zfill(2)
        sc = f'{j*spt}-{(j+1)*spt}'
        eval_task = EvalTask(
          name = f'Eval_{eval_idx}_Scene_{sc}',
          dataset_test_cfg=test)
        eval_tasks.append( eval_task )
      self._eval_lists.append( eval_tasks )
      
  def _mlhypersim_random4_tests(self):
    spt = int( len(mlhypersim_scene_names)/30 ) # scenes_per_task spt

    for i in range(0,4):
      train = copy.deepcopy( mlhypersim_template_dict )
      val = copy.deepcopy( mlhypersim_template_dict )
      train['mode'] = 'train'
      val['mode'] = 'val'
      train['scenes'] = mlhypersim_scene_names[i*spt:(i+1)*spt]
      val['scenes'] = mlhypersim_scene_names[i*spt:(i+1)*spt]
      
      if self.replay_adaptive_add_p:
        if i == 0:
          train['cfg_replay']['replay_p'] = 0.0
        else:
          train['cfg_replay']['replay_p'] = float(i)/float(i+1)
      
      task_idx = str(i).zfill(2)
      t = Task(name = f'Task_{task_idx}_mlhyper_random4_test_all',
                dataset_train_cfg= train,
                dataset_val_cfg= val)
      self._task_list.append(t)
      # Get eval tasks
      eval_tasks = []
      for j in range(0,4):
        test = copy.deepcopy( mlhypersim_template_dict )
        test['mode'] = 'val'
        test['scenes'] = mlhypersim_scene_names[j*spt:(j+1)*spt]
        eval_idx = str(j).zfill(2)
        sc = f'{j*spt}-{(j+1)*spt}'
        eval_task = EvalTask(
          name = f'Eval_{eval_idx}_Scene_{sc}',
          dataset_test_cfg=test)
        eval_tasks.append( eval_task )
      self._eval_lists.append( eval_tasks )
      
      
  def _mlhypersim_random4_seeded(self, seed):
    spt = int( len(mlhypersim_scene_names)/30 ) # scenes_per_task spt

     
    for i in range(seed,seed+4):
      train = copy.deepcopy( mlhypersim_template_dict )
      val = copy.deepcopy( mlhypersim_template_dict )
      train['mode'] = 'train'
      val['mode'] = 'val'
      train['scenes'] = mlhypersim_scene_names[i*spt:(i+1)*spt]
      val['scenes'] = mlhypersim_scene_names[i*spt:(i+1)*spt]
      
      if self.replay_adaptive_add_p:
        if i == seed:
          train['cfg_replay']['replay_p'] = 0.0
        else:
          train['cfg_replay']['replay_p'] = float(i-seed)/float(i+1-seed)
      
      task_idx = str(i).zfill(2)
      t = Task(name = f'Task_{task_idx}_mlhyper_random4_test_all',
                dataset_train_cfg= train,
                dataset_val_cfg= val)
      self._task_list.append(t)
      # Get eval tasks
      eval_tasks = []
      for j in range(seed,seed+4):
        test = copy.deepcopy( mlhypersim_template_dict )
        test['mode'] = 'val'
        test['scenes'] = mlhypersim_scene_names[j*spt:(j+1)*spt]
        eval_idx = str(j).zfill(2)
        sc = f'{j*spt}-{(j+1)*spt}'
        eval_task = EvalTask(
          name = f'Eval_{eval_idx}_Scene_{sc}',
          dataset_test_cfg=test)
        eval_tasks.append( eval_task )
      self._eval_lists.append( eval_tasks )
      
  
  def _mlhypersim_all(self):
    spt = int( len(mlhypersim_scene_names)/30 ) # scenes_per_task spt
    for i in range(0,1):
      train = copy.deepcopy( mlhypersim_template_dict )
      val = copy.deepcopy( mlhypersim_template_dict )
      train['mode'] = 'train'
      val['mode'] = 'val'
      train['scenes'] = mlhypersim_scene_names[:int(4*spt)]
      val['scenes'] = mlhypersim_scene_names[:int(4*spt)]
      
      task_idx = str(i).zfill(2)
      t = Task(name = f'Task_{task_idx}_mlhyper_all',
                dataset_train_cfg= train,
                dataset_val_cfg= val)
      self._task_list.append(t)
      
      # Get eval tasks
      eval_tasks = []
      for j in range(0,4):
        test = copy.deepcopy( mlhypersim_template_dict )
        test['mode'] = 'val'
        test['scenes'] = mlhypersim_scene_names[j*spt:(j+1)*spt]
        eval_idx = str(j).zfill(2)
        sc = f'{j*spt}-{(j+1)*spt}'
        eval_task = EvalTask(
          name = f'Eval_{eval_idx}_Scene_{sc}',
          dataset_test_cfg=test)
        eval_tasks.append( eval_task )
      self._eval_lists.append( eval_tasks )
      
  def _getAll(self):
    idx = np.argsort( nyu_class_counts)
    scene_names = nyu_scene_names[idx].tolist()
    scene_names
    counts = nyu_class_counts[idx].tolist()
    counts.reverse()
    min_counts = 50
    
    train = copy.deepcopy( nyu_template_dict )
    val = copy.deepcopy( nyu_template_dict )
    train['mode'] = 'train'
    val['mode'] = 'val'      
    train['scenes'] = []
    val['scenes'] = []
    
    task_idx = str(0).zfill(2)
    t = Task(name = f'Task_{task_idx}_Scene_ALL',
              dataset_train_cfg= train,
              dataset_val_cfg= val)
    self._task_list.append(t)
    eval_tasks = []
    for i in range(0,len(scene_names)):
      test = copy.deepcopy( nyu_template_dict )
      test['mode'] = 'val'
      test['scenes'] = [scene_names[i]]
      eval_idx = str(i).zfill(2)
      sc = test['scenes']
      eval_task = EvalTask(
        name = f'Eval_{eval_idx}_Scene_{sc}',
        dataset_test_cfg=test)
      eval_tasks.append( eval_task )
    self._eval_lists.append( eval_tasks )
  
  def _get4Categories(self):
    tasks = { 'bath': ['bathroom', 'bedroom','laundry_room'],
              'food': ['cafe', 'dining_room','office_kitchen','kitchen','dinette'],
              'work': ['home_office','conference_room','computer_lab', 'office',
                       'study', 'study_room','printer_room', 'bookstore','classroom'],
              'others': ['basement', 'excercise_room', 'foyer', 'furniture_store',
                         'home_storage', 'indoor_balcony', 'living_room','playroom',
                         'reception_room', 'student_lounge'] }
    
    
    idx = np.argsort( nyu_class_counts)
    scene_names = nyu_scene_names[idx].tolist()
    scene_names
    counts = nyu_class_counts[idx].tolist()
    counts.reverse()
    min_counts = 50
    
    train = copy.deepcopy( nyu_template_dict )
    val = copy.deepcopy( nyu_template_dict )
    train['mode'] = 'train'
    val['mode'] = 'val'      
    train['scenes'] = []
    val['scenes'] = []
    
    for j, name, scenes in zip(range(0,len(tasks)), tasks.keys(), tasks.values()):
      train = copy.deepcopy( nyu_template_dict )
      val = copy.deepcopy( nyu_template_dict )
      train['mode'] = 'train'
      val['mode'] = 'val'      
      train['scenes'] = scenes
      val['scenes'] = scenes
      
      task_idx = str(j).zfill(2)
      t = Task(name = f'Task_{task_idx}_Scenes_{name}',
               dataset_train_cfg= train,
               dataset_val_cfg= val)
      
      self._task_list.append(t)
      eval_tasks = []
      for i, name_eval, scenes_eval in zip(range(0,len(tasks)), tasks.keys(), tasks.values()):
        test = copy.deepcopy( nyu_template_dict )
        test['mode'] = 'val'
        test['scenes'] = scenes_eval
        
        eval_idx = str(i).zfill(2)
        eval_task = EvalTask(
          name = f'Eval_{eval_idx}_Scene_{name_eval}',
          dataset_test_cfg=test)
        eval_tasks.append( eval_task )
        
      self._eval_lists.append( eval_tasks ) 
  
  def _pretrainCOCO(self):
    train = copy.deepcopy( coco_template_dict )
    train['mode'] = 'train'
    val = copy.deepcopy( coco_template_dict )
    train['mode'] = 'val'
    
    t = Task(name = f'Task_TrainCOCO',
          dataset_train_cfg= train,
          dataset_val_cfg= val)
    self._task_list.append(t)
    
    eval_task = EvalTask(
          name = f'Eval_COCO_with_val_set',
          dataset_test_cfg=val)
    self._eval_lists.append( [eval_task ] ) 
  
  def _getTaskSingleScenesCountsDescending(self):
    # TODO: Refactor nameing and make getTask- Method more generic. 
    # dealing with different datasets
    
    idx = np.argsort( nyu_class_counts)
    scene_names = nyu_scene_names[idx].tolist()
    scene_names
    counts = nyu_class_counts[idx].tolist()
    counts.reverse()
    min_counts = 50
    
    for j, scene_name, count in zip(range(0,len(scene_names)), scene_names, counts):
      if count < min_counts:
        break
      train = copy.deepcopy( nyu_template_dict )
      val = copy.deepcopy( nyu_template_dict )
      train['mode'] = 'train'
      val['mode'] = 'val'      
      train['scenes'] = [scene_name]
      val['scenes'] = [scene_name]
      
      task_idx = str(j).zfill(2)
      t = Task(name = f'Task_{task_idx}_Scene_{scene_name}',
               dataset_train_cfg= train,
               dataset_val_cfg= val)
      self._task_list.append(t)
      eval_tasks = []
      for i in range(0,j+1):
        test = copy.deepcopy( nyu_template_dict )
        test['mode'] = 'val'
        test['scenes'] = [scene_names[i]]
        eval_idx = str(i).zfill(2)
        eval_task = EvalTask(
          name = f'Eval_{eval_idx}_Scene_{scene_name}',
          dataset_test_cfg=test)
        eval_tasks.append( eval_task )
        
      self._eval_lists.append( eval_tasks ) 
      
  def __iter__(self):
    return self

  def __next__(self):
    if self._current_task < len(self._task_list):
      task = self._task_list[self._current_task]
      eval_list = self._eval_lists[self._current_task]
      self._current_task += 1
      return task, eval_list
    
    else:
      raise StopIteration
  
  def __str__(self):
    p = '='*90 + '\n'
    p += f'Summary TaskCreator Tasks: ' + str(len(self._task_list))+'\n'
    for t, e_list  in zip( self._task_list, self._eval_lists):
      s = '  '+t.name
      s = '  Name: '+t.name + ' '*(40-len(s))
      p +=  s + '  EvalTasks: ' +  str(len(e_list)) +'\n'
      # for e in e_list:
      #   p += '    ' + e.name +'\n'
    p += '='*90
    return p

if __name__ == "__main__":
  tc = TaskCreator(mode= 'SingleScenesCountsDescending')
  print(tc)
  tc = TaskCreator(mode= 'All')
  print(tc)
  tc = TaskCreator(mode= 'FourCategories')
  print(tc) 
  tc = TaskCreator(mode= 'pretrainCOCO')
  print(tc) 
  tc = TaskCreator(mode= 'mlhypersim_random10')
  print(tc)
  