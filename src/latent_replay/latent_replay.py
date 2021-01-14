import torch
import torch.nn as nn
from random import randint
import numpy as np 

__all__ = ['LatentReplayBuffer']

class LatentReplayBin(nn.Module):
  def __init__(self, size, size_label, elements, dtype, device):
    super(LatentReplayBin, self).__init__()
    self.register_buffer('x', torch.zeros( (elements, *size), dtype=dtype)) # EL, S1, S2
    self.register_buffer('y', torch.zeros( (elements, *size_label), dtype=torch.int64))# EL, S1, S2
    self.register_buffer('valid', torch.zeros( (elements), dtype=torch.bool)) # EL
  

class LatentReplayBuffer(nn.Module):
  """
  stores latent activations to perform partial forward passing
  """
  def __init__(self, size=(100,100), size_label=(3,100,100), elements=100, mode='LIFO', bins=1, injection_rate=0.5, dtype=torch.float32, device='cpu'):
    """
    Implemented as a Module to have the base functionality of moveing between devices

    Total storage = size * bins * elements
    Storage is by default on GPU.
    TODO Implement this only for a single GPU (do we need to consider anything when using DDP?)
    Option: A single bin could be implemented as a Module

    Parameters
    ----------
    size : tuple(ints)
       [size of buffer elements]
    elements: int
      [Nr of elements in each bin]
    mode : str
      [indicates how elements are added to the buffer], by default 'LIFO'
    bins: int 
      [number of bins to add elements], by default 1
    injection_rate: float
      [ injection_rate <1: Probability of each element in the batch being replaced by a sample form the memory.
       injection_rate >= 1: # Elements to inject into batch]
    """
    super(LatentReplayBuffer, self).__init__()
    # Create memory bins
    self.bins = [LatentReplayBin(size, size_label, elements, dtype, device) for i in range(0,bins)]
    self.bins = nn.ModuleList( self.bins )
    self._max_elements = elements
    self._injections_rate = injection_rate
    
    self.register_buffer('_bin_counts', torch.zeros((bins)))
    
    self._size_label = size_label
    self._size = size
    self._dtype = dtype
    self._device = device
    
  def add(self, x,y, bin):
    
    free = self.bins[bin].valid == False
    if self._bin_counts[bin] < self._max_elements:
      # free_space
      self._bin_counts[bin] += 1
      
      idxs = torch.where( free )[0][0]
      
      print(idxs, self.bins[bin].x.shape, x.shape )
      self.bins[bin].x[idxs] = x
      self.bins[bin].y[idxs] = y
      self.bins[bin].valid[idxs] = True
    else:
      ran = randint(0,self._max_elements)
      self.bins[bin].x[ran] = x
      self.bins[bin].y[ran] = y
      self.bins[bin].valid[ran] = True
      
  def forward(self, BS, device):
    """Generate an injection into the neural network.
    There only on forward pass is needed. 
    
    The output of this function is an injections_mask which can be used to select where the injection should be injected. 
    And the injection information itself.
    
    There are two modes: 
      - fixed number of injections.
      - injection propability.
    
    Parameters
    ----------
    bs : [type]
        [description]
    """
    injection = torch.zeros((BS, *self._size), dtype=self._dtype, device=device)
    injection_labels = torch.zeros((BS, *self._size_label), dtype=torch.int64, device=device)
    injection_mask = torch.zeros( (BS), dtype=torch.bool, device=device)
    
    # create injection mask according to set injection_rate
    if self._bin_counts.sum() < BS:
      return injection, injection_labels, injection_mask
    else:
      if self._injections_rate < 1:
        injection_mask = torch.bernoulli( torch.ones(BS)*self._injections_rate , device=device)
      else:
        idx = torch.tensor( np.random.choice(BS, self._injections_rate, replace=False))
        injection_mask[idx] = True
        
    # fill the injection with random samples.
    # This could be implemented way faster in a batch manner.
    for i in range(BS):
      if injection_mask[i]:
        x,y = self.get_random_element()
        injection[i] = x
        injection_labels[i] = y 
    
    return  injection, injection_labels, injection_mask

  def get_random_element(self):
    # randomly select bin
    non_empty_bins_index = ( self._bin_counts != 0).nonzero()
    sel_bin = torch.randint(0, non_empty_bins_index.shape[0], (1,))
    # randomly select element in bin
    
    non_zero_elements = (self.bins[sel_bin].valid != 0).nonzero()
    sel_elm = torch.randint(0, non_zero_elements.shape[0], (1,))
    return self.bins[sel_bin].x[sel_elm], self.bins[sel_bin].y[sel_elm]  
  

def test():      
  lrb = LatentReplayBuffer(size=(100,100), size_label=(3,100,100), elements=100, mode='LIFO', bins=1, injection_rate=2, dtype=torch.float32, device='cpu')
  
  y = torch.ones( (3,100,100) )
  x = torch.ones( (100,100) )
  for i in range(0,10):
    
    lrb.add( x, y, 0)  
    
  injection, injection_labels, injection_mask = lrb.get_injections(4)
  # print(injection.shape, injection_labels.shape, injection_mask.shape)
  
if __name__ == "__main__":
    test()