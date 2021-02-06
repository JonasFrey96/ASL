import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fast_scnn import FastSCNN
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from os.path import join
import os
import time
__all__ = ['Teacher']


"""
Goal: store the model for each task at the end of the training. 
The Teacher model can be used to create soft labels.
"""

class Teacher(nn.Module):
  def __init__(self, num_classes, n_teacher, soft_labels = True, verbose = True):
    super().__init__()
    self.models = nn.ModuleList( [FastSCNN(num_classes) for i in range(n_teacher)] ) 
    self.soft_labels = soft_labels
    self.n_classes = num_classes
    self.softmax = torch.nn.Softmax( dim=1 )
    self.verbose = verbose
    
  def forward(self, x, teacher):
    x = self.models[teacher](x)[0]
    if self.soft_labels:
      return self.softmax(x).detach()
    else:
      x = torch.argmax(x, 1).detach()
      return x
      
      
  def print_weight_summary(self):
    string = 'Summary Teacher:\n'
    for j in range(len(self.models)):
      sum = 0
      for i in self.models[j].parameters():
        sum += i[0].sum()
      string += f'   Teacher Level {j}: WeightSum == {sum}\n'
    rank_zero_info(string)
    
    
  def absorbe_model(self, model, teacher, path=None):
	
    if teacher < len(self.models):
      if self.verbose:
        rank_zero_info( f'We are storing the model {teacher} as a new teacher')
        rank_zero_info( f'We are storing the model {teacher} as a new teacher')
        rank_zero_info( f'We are storing the model {teacher} as a new teacher')
        rank_zero_info( f'We are storing the model {teacher} as a new teacher')
        rank_zero_info( f'We are storing the model {teacher} as a new teacher')
      
      d = str(list(model.parameters())[0].device)
      if path is not None:
        pa = os.environ.get('TMPDIR')
        if type(pa) is str:
          if pa.find('tmp') != -1:
            pa = path
            
        path = join(path, f'weights_tmp_{d}.pth')
        
      else:
        path = f'weights_tmp_{d}.pth'
      string = 'I will save current model weights to ' + path
      rank_zero_info( string )

      # torch.save(model.state_dict(), path )
      # time.sleep(1)
      # for a,b  in zip( model.parameters(),self.models[teacher].parameters() ):
        # b=a.clone()
      para_copy = dict( model.named_parameters() )
      for name, params in self.models[teacher].named_parameters():
        params.data.copy_( para_copy[name])
     
        
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