# STD
import os
import copy 
import io

# MISC
import numpy as np
import torch 
import imageio
import cv2
from PIL import Image, ImageDraw, ImageFont

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .colors import *

__all__ = ['Visualizer', 'MainVisualizer']

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
  fig.set_dpi(dpi)
  canvas = FigureCanvasAgg(fig)
  # Retrieve a view on the renderer buffer
  canvas.draw()
  buf = canvas.buffer_rgba()
  # convert to a NumPy array
  buf = np.asarray(buf)
  buf = Image.fromarray(buf)
  buf = buf.convert('RGB')
  return buf

def image_functionality(func):
  def wrap(*args, **kwargs):
    log = False
    if kwargs.get('method', 'def') == 'def':
      img = func(*args,**kwargs)
      log = True
    elif kwargs.get('method', 'def') == 'left':
      kwargs_clone = copy.deepcopy(kwargs)
      kwargs_clone['store'] = False
      kwargs_clone['jupyter'] = False
      res = func(*args,**kwargs_clone)
      args[0]._storage_left = res
    elif kwargs.get('method', 'def') == 'right':
      kwargs_clone = copy.deepcopy(kwargs)
      kwargs_clone['store'] = False
      kwargs_clone['jupyter'] = False
      res = func(*args,**kwargs_clone)
      args[0]._storage_right = res
      
    if args[0]._storage_right is not None and args[0]._storage_left is not None:
      img = np.concatenate( [args[0]._storage_left,  args[0]._storage_right] , axis=1 )
      args[0]._storage_left = None
      args[0]._storage_right = None
      log = True
    
    if log:
      log_exp = args[0].logger is not None
      tag = kwargs.get('tag', 'TagNotDefined')
      jupyter = kwargs.get('jupyter', False)
      # Each logging call is able to override the setting that is stored in the visualizer
      if kwargs.get('store', None) is not None:
        store = kwargs['store']
      else:
        store = args[0]._store

      if kwargs.get('epoch', None) is not None:
        epoch = kwargs['epoch']
      else:
        epoch = args[0]._epoch

      # Store & Log & Display in Jupyter
      if store:
        p = os.path.join( args[0].p_visu, f'{epoch}_{tag}.png')
        imageio.imwrite(p, img)
      
      if log_exp:
        H,W,C = img.shape
        ds = cv2.resize( img , dsize=(int(W/2), int(H/2)), interpolation=cv2.INTER_CUBIC)
        if args[0].logger is not None:
          try:
            # logger == neptuneai
            args[0].logger.log_image(
              log_name = tag, 
              image = np.float32( ds )/255 , 
              step=epoch)
          except:
            try:
              # logger == tensorboard
              args[0].logger.experiment.add_image(
                tag = tag, 
                img_tensor = ds, 
                global_step=epoch,
                dataformats='HWC')
            except:
              print('Tensorboard Logging and Neptune Logging failed !!!')
              pass 
        
      if jupyter:
          display( Image.fromarray(img))  
        
    return func(*args,**kwargs)
  return wrap

class MainVisualizer():
  def __init__(self, p_visu, logger= None, epoch=0, store=True, num_classes=22):
    self.p_visu = p_visu
    self.logger = logger
    
    if not os.path.exists(self.p_visu):
      os.makedirs(self.p_visu)
    
    self._epoch = epoch
    self._store = store
    self._storage_left =None
    self._storage_right =None
    
    jet = cm.get_cmap('jet')
    self.SEG_COLORS = (np.stack([jet(v)
      for v in np.linspace(0, 1, num_classes)]) * 255).astype(np.uint8)
    self.SEG_COLORS_BINARY = (np.stack([jet(v)
      for v in np.linspace(0, 1, 2)]) * 255).astype(np.uint8)
  
  @property
  def epoch(self):
    return self._epoch
  @epoch.setter
  def epoch(self, epoch):
    self._epoch = epoch

  @property
  def store(self):
    return self._store
  @store.setter
  def store(self, store):
    self._store = store
  
  @image_functionality
  def plot_segmentation(self, seg, *args,**kwargs):
    try:
      seg = seg.clone().cpu().numpy()
    except:
      pass

    if seg.dtype == np.bool:
      col_map = self.SEG_COLORS_BINARY
    else:
      col_map = self.SEG_COLORS
      seg = seg.round()
    
    H,W = seg.shape[:2]
    img = np.zeros((H,W,3), dtype=np.uint8)
    for i, color in enumerate( col_map ) :
      img[ seg==i ] = color[:3]

    return img
  
  @image_functionality
  def plot_image(self, img, *args,**kwargs):
    """
    ----------
    img : CHW HWC accepts torch.tensor or numpy.array
          Range 0-1 or 0-255
    """
    try:
      img = img.clone().cpu().numpy()
    except:
      pass
    
    if img.shape[2] == 3:
      pass
    elif img.shape[0] == 3:
      img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
    else:
      raise Exception('Invalid Shape')  
    if img.max() <= 1:
      img = img*255
    
    img = np.uint8(img)
    return img

  @image_functionality  
  def plot_matrix(self, data_matrix, higher_is_better= True, title='TitleNotDefined',max_tasks=None, max_tests= None,
                  label_x=None, label_y=None, *args,**kwargs):

    if max_tasks is None and max_tests is None:
            max_tasks = data_matrix.shape[0]
            max_tests = data_matrix.shape[1]
    else:
        d1 = data_matrix.shape[0]
        d2 = data_matrix.shape[1]
        assert d2 <= max_tests
        data = np.zeros( (max_tasks, max_tests))
        if max_tasks > d1:
            
            data[:d1,:d2] = data_matrix
        else:
            data[:max_tasks,:d2] = data_matrix[:max_tasks,:d2]

        data_matrix = data
    
    if label_y is None:
        label_y = ["Task  "+str( i) for i in range(max_tasks)]
    if label_x is None:
        label_x = ["Test "+str(i) for i in range(max_tests)]
    
    fig, ax = plt.subplots()
    if higher_is_better:
      im = ax.imshow(data_matrix,cmap=cm.get_cmap('PiYG')  )
    else:
      im = ax.imshow(data_matrix,cmap=cm.get_cmap('PiYG_r')  )

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(label_x)))
    ax.set_yticks(np.arange(len(label_y)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(label_x)
    ax.set_yticklabels(label_y)

    # Rotate the tick labels and set their alignment.

    #ax.invert_xaxis()
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
              rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(label_x)):
        for j in range(len(label_y)):
            text = ax.text(i,j, data_matrix[j,i], 
                           ha="center", va="center", color="w",
                          fontdict = {'backgroundcolor':(0,0,0,0.2)})

    ax.set_title(title)
    #fig.tight_layout()
    arr = get_img_from_fig(fig, dpi=600)
    plt.close()
    return np.uint8(arr)
  
  
  @image_functionality  
  def plot_cont_validation_eval(self, task_data, *args,**kwargs):
    """
    res1 =  np.linspace(0., 0.5, 6)
    res2 =  np.linspace(0., 0.5, 6)*0.5
    res3 =  np.linspace(0., 0.5, 6)**2
    T1 = {'name': 'TrainTask1' ,'val_task_results': [(np.arange(0,6), res1), (np.arange(0,6), res2), (np.arange(0,6), res3) ] }
    T2 = {'name': 'TrainTask2' ,'val_task_results': [(np.arange(5,11), res1), (np.arange(5,11),res2), (np.arange(5,11),res3) ] }
    T3 = {'name': 'TrainTask3' ,'val_task_results': [(np.arange(10,16),res1), (np.arange(10,16),res2), (np.arange(10,16),res3) ] }
    task_data = [T1, T2]
    """
    
    line_styles = ['-','--','-.',':'] 
    steps_min = 999
    steps_max = 0
    for t in task_data:
        for v in t['val_task_results']:
            if np.min( v[0]) <  steps_min:
                steps_min = np.min( v[0])
            if np.max( v[0]) > steps_max:
                steps_max = np.max( v[0])  
    span = steps_max - steps_min

    fig, axs = plt.subplots( len(task_data),sharex=True, sharey=True, figsize=(10,len(task_data)*2))
    if len(task_data) == 1:
      axs = [axs]
    plt.subplots_adjust(left = 0.125,
        right = 0.9, 
        bottom = 0.1,  
        top = 1,
        wspace = 0.2,  
        hspace = 0.8 )
    for nr, task in enumerate(task_data):
        name = task['name']
        axs[nr].set_title(name)
        axs[nr].set_xlabel('Step')
        axs[nr].set_ylabel('Acc')
        axs[nr].set_yticks( [0,0.2,0.4,0.6,0.8,1])
        axs[nr].grid(True, linestyle='-', linewidth=1)
        for j, i in enumerate( task['val_task_results']):
            k = list( col.keys())
            val = col[k[j]]
            val = [v/255 for v in val]
            axs[nr].plot(i[0], i[1], color=val, linestyle = line_styles[j], label=task['eval_names'][j])
        plt.legend(loc='upper left')
    arr = get_img_from_fig(fig, dpi=600)
    plt.close()
    return np.uint8(arr)
  
  @image_functionality
  def plot_bar(self, data, x_label='Sample', y_label='Value', title='Bar Plot', sort=True, reverse=True, *args,**kwargs):
    def check_shape(data):
        if len(data.shape)>1:
            if data.shape[0] == 0:
                data = data[0,:]
            elif data.shape[1] == 0:
                data = data[:,0]
            else:
                raise Exception('plot_hist: Invalid Data Shape')
        return data
    
    if type(data) == list:
        pass
    elif type(data) == torch.Tensor:
        data = check_shape(data)
        data = list( data.clone().cpu())
    elif type(data) == np.ndarray:
        data = check_shape(data)
        data = list(data)
    else:
        raise Exception("plot_hist: Unknown Input Type"+str(type(data)))
    
    if sort:
        data.sort(reverse=reverse)
    
    fig = plt.figure()
    plt.bar(list(range(len(data))), data, facecolor=COL_MAP(2) )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    arr = get_img_from_fig(fig, dpi=300)
    plt.close()
    return np.uint8(arr)
class Visualizer():
  def __init__(self, p_visu, logger=None, epoch=0, store=True, num_classes=22):
    self.p_visu = p_visu
    self.logger = logger

    if not os.path.exists(self.p_visu):
      os.makedirs(self.p_visu)
    
    self._epoch = epoch
    self._store = store
    self._storage_left =None
    self._storage_right =None
    
    jet = cm.get_cmap('jet')
    self.SEG_COLORS = (np.stack([jet(v)
      for v in np.linspace(0, 1, num_classes)]) * 255).astype(np.uint8)
    self.SEG_COLORS_BINARY = (np.stack([jet(v)
      for v in np.linspace(0, 1, 2)]) * 255).astype(np.uint8)
  
  @property
  def epoch(self):
    return self._epoch
  @epoch.setter
  def epoch(self, epoch):
    self._epoch = epoch

  @property
  def store(self):
    return self._store
  @store.setter
  def store(self, store):
    self._store = store
  
  @image_functionality
  def plot_segmentation(self, seg, *args,**kwargs):
    try:
      seg = seg.clone().cpu().numpy()
    except:
      try:
        seg = seg.numpy()
      except:
        print('Failed converting tensor to numpy')
        pass

    if seg.dtype == np.bool:
      col_map = self.SEG_COLORS_BINARY
    else:
      col_map = self.SEG_COLORS
      seg = seg.astype(np.float32)
      seg = seg.round()
        
    
    H,W = seg.shape[:2]
    img = np.zeros((H,W,3), dtype=np.uint8)
    for i, color in enumerate( col_map ) :
      img[ seg==i ] = color[:3]
    return img
  
  @image_functionality
  def plot_bar(self, data, x_label='Sample', y_label='Value', title='Bar Plot', sort=True, reverse=True, *args,**kwargs):
    def check_shape(data):
        if len(data.shape)>1:
            if data.shape[0] == 0:
                data = data[0,:]
            elif data.shape[1] == 0:
                data = data[:,0]
            else:
                raise Exception('plot_hist: Invalid Data Shape')
        return data
    
    if type(data) == list:
        pass
    elif type(data) == torch.Tensor:
        data = check_shape(data)
        data = list( data.clone().cpu())
    elif type(data) == np.ndarray:
        data = check_shape(data)
        data = list(data)
    else:
        raise Exception("plot_hist: Unknown Input Type"+str(type(data)))
    
    if sort:
        data.sort(reverse=reverse)
    
    fig = plt.figure()
    plt.bar(list(range(len(data))), data, facecolor=COL_MAP(2) )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    arr = get_img_from_fig(fig, dpi=300)
    plt.close()
    return np.uint8(arr)
  
  @image_functionality
  def plot_image(self, img, *args,**kwargs):
    try:
      img = img.clone().cpu().numpy()
    except:
      pass
    if img.shape[2] == 3:
      pass
    elif img.shape[0] == 3:
      img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
    else:
      raise Exception('Invalid Shape')  
    if img.max() <= 1:
      img = img*255
    img = np.uint8(img)
    return img