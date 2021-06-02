import pandas 
import numpy as np
import imageio
from utils_asl import png_to_label
import torch
import os

__all__ = ['LabelLoader']
class LabelLoader():
  def __init__(self, mode, root_scannet = None):
    if mode == "scannet":
      assert root_scannet is not None
      self._get_mapping(root_scannet)
      self.get = self.get_scannet
    elif mode == "prob_label":
      self.get = self.get_probs
    elif mode == "simple":
      self.get = self.get_simple
      
  def get(self, path):
    pass
  
  def get_simple(self, path):
    return imageio.imread(path).astype(np.uint32)
  
  def get_probs( self, path):
    lab = png_to_label(path)
    return np.argmax( lab, axis = 2).astype(np.uint32) +1 
    
  def get_scannet( self, path):
    label = torch.from_numpy( imageio.imread( path ).astype(np.int32)).type(torch.float32)[None, :, :]
    sa = label.shape
    label = label.flatten()
    label = self.mapping[label.type(torch.int64)] 
    label = label.reshape(sa).numpy()
    return label[0].astype(np.uint32)
  
  def _get_mapping(self, root):
    tsv = os.path.join(root, "scannetv2-labels.combined.tsv")
    df = pandas.read_csv(tsv, sep='\t')
    mapping_source = np.array( df['id'] )
    mapping_target = np.array( df['nyu40id'] )
    
    self.mapping = torch.zeros( ( int(mapping_source.max()+1) ),dtype=torch.int64)
    for so,ta in zip(mapping_source, mapping_target):
      self.mapping[so] = ta
      
      
   