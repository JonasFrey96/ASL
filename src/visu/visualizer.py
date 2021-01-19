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

SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

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
        H,W,C = img.shape
        ds = cv2.resize( img , dsize=(int(W/4), int(H/4)), interpolation=cv2.INTER_CUBIC)
        
        args[0].writer.add_image(
          tag, 
          ds, 
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