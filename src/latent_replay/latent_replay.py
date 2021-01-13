import torch
import torch.nn as nn

__all__ = ['LatentReplayBuffer']

class LatentReplayBin(nn.Module):
  def __init__(self, size=(100,100), elements=100, dtype=torch.float32):
    super(LatentReplayBin, self).__init__()
    self.storage = torch.zeros( (elements, *size), dtype=dtype) # EL, S1, S2
    i = tuple( [0] * len(size)) 
    self.valid = (self.storage == 1)[:,i] # EL
  

class LatentReplayBuffer(nn.Module):
  """
  stores latent activations to perform partial forward passing
  """
  def __init__(self, size=(100,100), elements=100, mode='LIFO', bins=1, dtype=torch.float32):
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
    """
    super(LatentReplayBuffer, self).__init__()
    # Create memory bins
    self.bins = [LatentReplayBin(size, elements, dtype) for i in range(0,bins)]
    self.bins = nn.ModuleList( self.bins )
  
  def add(self, element, bin):
    free = self.bins[bin].valid == False
    if torch.sum( free ) != 0:
      # free_space
      idxs = torch.where( free )[0][0]
      self.bins[bin].storage[idxs] = element
    else:
      ran = torch.randint(size=( self.bins[bin].valid.shape[0],1), high=20)[0,0]
      self.bins[bin].storage[ran] = element
    