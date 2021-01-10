from PIL import Image, ImageDraw
import torch 
import numpy as np
from matplotlib import cm
import os
import imageio
import copy 
import matplotlib
import matplotlib.pyplot as plt
import io
import cv2

__all__ = ['Visualizer', 'MainVisualizer']

col = { "red":[255,89,94],
 "yellow":[255,202,58],
 "green":[138,201,38],
 "blue":[25,130,196],
 "purple":[106,76,147] }

li = [ [*(v),255] for v in col.values()]
li = (np.array(li)/255).tolist()
col_map = cm.colors.ListedColormap(li)

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=dpi)
  buf.seek(0)
  img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
  buf.close()
  img = cv2.imdecode(img_arr, 1)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img

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
      log_tensorboard = args[0].writer is not None
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
      
      if log_tensorboard:
        args[0].writer.add_image(
          tag, 
          img, 
          global_step=epoch, 
          dataformats='HWC')

      if jupyter:
          display( Image.fromarray(img))  
        
    return func(*args,**kwargs)
  return wrap

class MainVisualizer():
  def __init__(self, p_visu, writer=None, epoch=0, store=True, num_classes=22):
    self.p_visu = p_visu
    self.writer = writer

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
      
    img = np.uint8(img)
    return img

  @image_functionality  
  def plot_matrix(self, data_matrix, max_tasks=None, max_tests= None,
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
    im = ax.imshow(data_matrix,cmap=cm.get_cmap('PiYG')  )

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



    ax.set_title("Training Result")
    #fig.tight_layout()
    arr = get_img_from_fig(fig, dpi=600)
    return np.uint8(arr)
  
  
  
class Visualizer():
  def __init__(self, p_visu, writer=None, epoch=0, store=True, num_classes=22):
    self.p_visu = p_visu
    self.writer = writer

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
      
    img = np.uint8(img)
    return img