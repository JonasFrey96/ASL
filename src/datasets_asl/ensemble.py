# Idee pass in mutiple datasets
# Pass in the ratios according to which they should be sampled
# Allows internally masking a dataset such that only a subset of elements is accessible via filtering
# Fully replace the memory thing

# TODO: Jonas Frey added atomatically switching on aux labels if needed
from torch.utils.data import Dataset
import numpy as np
import torch

eps = 0.0001

__all__ = ['Ensemble']

class Ensemble(Dataset):
  def __init__(self, main_dataset, replay_datasets, probs):
    """Initalizes the datasets

    Args:
        datasets (list of datasets): must return same sized elements
    """
    super(Ensemble, self).__init__()
    self.replay_datasets = replay_datasets
    self.main_dataset = main_dataset
    self.probs = np.array( probs )

    assert len(self.replay_datasets) == len(probs)-1
    # Quick Systematic Problem Explained:
    # Define a new length based on the probs !
    assert np.abs(self.probs.sum() -1 ) < eps


    aux_needed = self.main_dataset.aux_labels 
    for d in self.replay_datasets:
      aux_needed = aux_needed or d.aux_labels
    if aux_needed:
      if not self.main_dataset.aux_labels:
        self.main_dataset.aux_labels = True
        self.main_dataset.aux_labels_fake = True
      for d in self.replay_datasets:
        if not d.aux_labels:
          d.aux_labels = True
          d.aux_labels_fake = True
  

    self._length = int( len(main_dataset) *(1/self.probs[-1]) )
    
    self.replayed = [False]*len(main_dataset) # either_get_from_main_dataset
    
    if self._length-len(main_dataset) > 0:
      self.replayed += [True]*int( self._length-len(main_dataset) )  # either_get_from_main_dataset

  def __getitem__(self,index):
    if not self.replayed[index]:
      # Return value from main dataset
      res = self.main_dataset[index]
      return ( *res[:2], torch.tensor( -1 ), *res[2:] )
    
    else:
      # Return value from replayed datasets
      res, dataset_idx = self.get_random_replay_item()
      return ( *res[:2], torch.tensor( dataset_idx ), *res[2:] )

  def get_replay_datasets_globals(self):
    """Returns a list for each replay dataset with all available global_indices
    This list should be used for masking.
    """
    return [d.global_to_local_idx for d in self.replay_datasets ]

  def get_datasets_globals(self):
    """Returns a list for each replay dataset with all available global_indices
    This list should be used for masking.
    """
    if len( self.replay_datasets ) != 0:
      datasets = self.replay_datasets
      datasets.append( self.main_dataset)
    else:
      datasets = [ self.main_dataset ]
    return [d.global_to_local_idx for d in datasets ]


  def set_replay_datasets_globals(self, ls_global_indices):
    # check if for each replay dataset a list is given
    assert len(ls_global_indices) == len(self.replay_datasets)
    print("============== SET REPLAY DATASET GLOBALS ===============")
    c = 0
    for ls, dataset in zip( ls_global_indices, self.replay_datasets):
      # ckech if at least one element is set
      assert len(ls) != 0
      # check if all indices that we want to set are 
      # contained within the current configuration
      # allows only for contraction
      for idx in ls:
        assert idx in dataset.global_to_local_idx

      # if all checks pass set the new indices
      dataset.global_to_local_idx = ls
      dataset.length = len(ls)
      elements = min( len(ls), 6 )
      print(f"For Dataset {c} set replay index to: ", dataset.global_to_local_idx[:elements])
      c += 1
      

  def get_random_replay_item(self):
    dataset_idx = np.argmax( np.random.random( len(self.replay_datasets)) * self.probs[1:] )
    dataset_ele_idx = np.random.randint( 0, len(self.replay_datasets[dataset_idx]) )
    return self.replay_datasets[dataset_idx][dataset_ele_idx], dataset_idx

  def __len__(self):
    return self._length