import torch
from models_asl import FastSCNN
import os

__all__ = ["Teacher"]

"""
Cant be implemented as a callback given that we have to directly operate on the batch data.
1. Init: Setup a constant teacher model by providing the .ckpt path
2. Actively call the method modify_batch to overwrite the auxillary label in the batch with the teacher output.
"""

import torch.nn as nn


class Teacher(nn.Module):
  def __init__(self, active, base_path, cfg):
    self.active = active
    super().__init__()
    if active:
      self.teacher = FastSCNN(**cfg["model"]["cfg"])

      p = os.path.join(base_path, cfg["checkpoint_path"])

      if os.path.isfile(p):
        state_dict_loaded = torch.load(p, map_location=lambda storage, loc: storage)[
          "state_dict"
        ]
        state_dict_loaded = {
          k.replace("model.", ""): v for k, v in state_dict_loaded.items()
        }
        self.teacher.load_state_dict(state_dict_loaded, strict=False)
      self.teacher.eval()
      self.overwrite = cfg["overwrite"]
      self.soft = cfg["soft"]
      # TODO: MAPPING

  def modify_batch(self, batch):
    if not self.active:
      return batch

    assert (
      len(batch) < 5 and not self.overwrite
    ), "If len(batch) > 4 aux_label and aux_vaild are provided by the dataloader: Set overwrite in teacher, disable teacher or change dataloader"
    with torch.no_grad():
      BS = int(batch[0].shape[0])
      outputs = self.teacher(batch[0])

      if not self.soft:
        aux_label = torch.argmax(outputs[0], dim=1)
      else:
        aux_label = torch.nn.functional.softmax(outputs[0], dim=1)

      if len(batch) == 3:
        batch = batch[:3] + [
          aux_label,
          torch.tensor([True] * BS, device=outputs[0].device),
        ]
      if len(batch) == 4:
        batch = batch[:3] + [
          aux_label,
          torch.tensor([True] * BS, device=outputs[0].device),
          batch[3],
        ]

    return batch
