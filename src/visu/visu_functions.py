import pickle
import numpy as np
__all__ = ['validation_acc_plot_stored', "plot_from_pkl", "validation_acc_plot"]

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
        try:
          v = res[f'{m}/dataloader_idx_{j}']
        except:
          v = res[f'{m}']
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


def validation_acc_plot(main_visu, logger, nr_eval_tasks):
  n = [f'val_acc/dataloader_idx_{i}' for i in range(nr_eval_tasks)]
  n.append('task_count')
  df_acc = logger.experiment.get_numeric_channels_values(*n)
  x = np.array(df_acc['task_count'])
  
  
  task_nrs, where = np.unique( x, return_index=True )
  task_nrs = task_nrs.astype(np.uint8).tolist()
  where = where.tolist()[1:]
  where = [w-1 for w in where]
  where += [x.shape[0]-1]*int(nr_eval_tasks-len(where))
  
  names = [f'val_acc_{idx}' for idx in range(nr_eval_tasks)]
  x =  np.arange(x.shape[0])
  y = [np.array(df_acc[f'val_acc/dataloader_idx_{idx}']  ) for idx in range(nr_eval_tasks)]
  arr = main_visu.plot_lines_with_bachground(
      x, y, count=where,
      x_label='Epoch', y_label='Acc', title='Validation Accuracy', 
      task_names=names, tag='Validation_Accuracy_Summary')
  
def validation_acc_plot_stored(main_visu, res):
  task_nr = len(res)-2
  names = [f'val_acc_{idx}' for idx in range(task_nr)]
  
  task_nrs, task_indices = np.unique( np.array( res[-1]), return_index=True)
  
  count = (task_indices + 1).tolist()
  count = count + [res[-2][-1]]
  x =  np.array(res[-2])
  y = [np.array(res[i]) for i in range( task_nr) ]

  arr = main_visu.plot_lines_with_background(
      x, y, count=count,
      x_label='Epoch', y_label='Acc', title='Validation Accuracy', 
      task_names=names, tag='Validation_Accuracy_Summary')

  evaled_tasks = task_indices.shape[0]
  data_matrix = np.zeros( (evaled_tasks,task_nr) )
  
  for t in range( evaled_tasks ):
    try:
      nr = task_indices[t+1]
      time = nr - 1
    except:
      time = len(res[0]) - 1 
    for j in range(task_nr):
      
      data_matrix[t,j] = res[j][min( time,len( res[j] )-1)] * 100 

  data_matrix = np.round( data_matrix, decimals=1) 
    
  main_visu.plot_matrix(
    tag = "Data Matrix",
    data_matrix = data_matrix,
    higher_is_better= True,
    title= "Data matrix")
  

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
    print("VALIED TO PLOT FROM PICKLE")
    pass 