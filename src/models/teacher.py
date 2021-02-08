import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fast_scnn import FastSCNN
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from os.path import join
import os
import time
import copy
__all__ = ['Teacher']


"""
Goal: store the model for each task at the end of the training. 
The Teacher model can be used to create soft labels.
"""

class Teacher(nn.Module):
  def __init__(self, num_classes, n_teacher, soft_labels = True, verbose = True, fast_params = {}):
    super().__init__()
    self.models = nn.ModuleList( [FastSCNN(**fast_params) for i in range(n_teacher)] ) 
    self.soft_labels = soft_labels
    self.n_classes = num_classes
    self.softmax = torch.nn.Softmax( dim=1 )
    self.verbose = verbose
    
  def forward(self, x, teacher):
    # not used, replayed by get_latent_replay
    x = self.models[teacher](x)[0]

  def print_weight_summary(self):
    string = 'Summary Teacher:\n'
    for j in range(len(self.models)):
      sum = 0
      for i in self.models[j].parameters():
        sum += i[0].sum()
      string += f'   Teacher Level {j}: WeightSum == {sum}\n'
    rank_zero_info(string)
    
  def get_latent_replay(self, images, replayed):
    with torch.no_grad():
      res_targets, res_features = None, None
      for n in range(len(self.models)):
          mask = (replayed == n)[:,0]
          if mask.sum() != 0:
            if res_targets is not None:
              target, features = self.get_features( images, n) 
              res_targets[mask] = target[mask]
              res_features[mask] = features[mask]
            else:
              res_targets, res_features = self.get_features( images, n) 
  
    if self.soft_labels:
      res_targets = self.softmax(res_targets).detach()
    else:
      res_targets = torch.argmax(res_targets, 1).detach().type(torch.int64)      
    return res_targets.clone(), res_features.clone().detach()
  
  def get_features(self, x, teacher):
    return self.models[teacher](x) #ERROR 
  
  def absorbe_model(self, model, teacher, path=None):
	
    if teacher < len(self.models):
      if self.verbose:
        rank_zero_info( f'Storing the model {teacher} as a new teacher')
      
      para_copy = dict( model.named_parameters() )
      for name, params in self.models[teacher].named_parameters():
        # params.data.copy_( para_copy[name].clone)
        params.data = para_copy[name].clone()
     
      # if os.path.exists(path):
      #   self.models[teacher].load_state_dict( torch.load(path, map_location = list(model.parameters())[0].device ) )
      #   # self.models[teacher].load_state_dict( torch.load(path) )
      # else:
      #   rank_zero_info( f'Can not restore the teacher from the saved model parameters !!!! Problemo')
      
      # if os.path.exists(path):
      #   os.remove(path) 
      # else:
      #   print("The file does not exist")
    
    for i , mod in enumerate(self.models):
      mod.freeze_module( mask=[True, True, True, True] )