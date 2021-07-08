import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# This model is used to store the state of the replay buffer in the dataloaders.
# This allows to easly continue training with the correct buffer mode from a checkpoint.
# Simply at the begining of each epoch write the buffer state to the dataloader shared memory.
# And at the end of the epoch copy the shared memory to the model.


__all__ = ["ReplayStateSyncBack"]


class ReplayStateSyncBack(nn.Module):
  def __init__(self, bins, elements):
    super().__init__()
    self.register_buffer(
      "bins", torch.zeros((bins, elements), dtype=torch.long), persistent=True
    )
    self.register_buffer(
      "valid", torch.zeros((bins, elements), dtype=torch.bool), persistent=True
    )
    self.nr_elements = elements
    self.nr_bins = bins

  def absorbe(self, bins, valid):
    self.bins = torch.from_numpy(bins).type(torch.long)
    self.valid = torch.from_numpy(valid).type(torch.bool)

  def get(self):
    return self.bins.to("cpu").numpy(), self.valid.to("cpu").numpy()
