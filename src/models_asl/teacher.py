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
      self.only_replayed = cfg["only_replayed"]

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

      if self.only_replayed:
        aux_valid = batch[2].clone() != -1
      else:
        aux_valid = torch.tensor([True] * BS, device=outputs[0].device)

      if len(batch) == 3:
        batch = batch[:3] + [
          aux_label,
          aux_valid,
        ]
      elif len(batch) == 4:
        batch = batch[:3] + [
          aux_label,
          aux_valid,
          batch[3],
        ]
      else:
        raise ValueError("Batch length is not supported")

    return batch

  def absorb(self, reference_model):
    reference_state = reference_model.state_dict()
    s1 = 0
    for n, p in self.teacher.named_parameters():
      s1 += p.data.sum()

    s2 = 0
    for n, p in reference_model.named_parameters():
      s2 += p.data.sum()

    # COPYING WEIGHTS OVER
    for n, p in self.teacher.named_parameters():
      p.data.copy_(reference_state[n])

    s3 = 0
    for n, p in self.teacher.named_parameters():
      s3 += p.data.sum()
    print("Teacher Absorbing Result: ", s1, s2, s3)
    assert s2 == s3, "Failed absorbing waits at end of train to teacher"
