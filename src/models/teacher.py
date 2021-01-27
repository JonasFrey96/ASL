import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fast_scnn import FastSCNN
__all__ = ['Teacher']


"""
Goal: store the model for each task at the end of the training. 
The Teacher model can be used to create soft labels.
"""

class Teacher(nn.Module):
  def __init__(self, num_classes, n_teacher, soft_labels = True):
    super().__init__()
    self.models = nn.ModuleList( [FastSCNN(num_classes) for i in range(n_teacher)] ) 
    self.soft_labels = soft_labels
    self.n_classes = num_classes
    self.softmax = torch.nn.Softmax( dim=3 )
    
  def forward(self, x, teacher):
    x = self.models[teacher](x)[0]
    if self.soft_labels:
      return self.softmax(x).detach()
    else:
      x = torch.argmax(x, 1).detach()
      return torch.nn.functional.one_hot(x, 
          num_classes=self.n_classes)
      
  def absorbe_model(self, model, teacher):
    self.models[teacher].load_state_dict( model.state_dict() )
    
    for i , mod in enumerate(self.models):
      mod.freeze_module( mask=[True, True, True, True] )