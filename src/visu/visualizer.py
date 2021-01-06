from PIL import Image, ImageDraw
import torch 
import numpy as np
from matplotlib import cm
import os
import imageio
__all__ = ['Visualizer']

col = { "red":[255,89,94],
 "yellow":[255,202,58],
 "green":[138,201,38],
 "blue":[25,130,196],
 "purple":[106,76,147] }

def image_functionality(func):
  def wrap(*args, **kwargs):
    # return the standarized numpy array: H,W,C, np.uint8
    img = func(*args,**kwargs)

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
        display( Image.fromarray(img) )
    
    return func(*args,**kwargs)
  return wrap

class Visualizer():
  def __init__(self, p_visu, writer=None, epoch=0, store=True, num_classes=22):
    self.p_visu = p_visu
    self.writer = writer

    if not os.path.exists(self.p_visu):
      os.makedirs(self.p_visu)
    
    self._epoch = epoch
    self._store = store

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