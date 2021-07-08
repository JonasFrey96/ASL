# TODO: Jonas Frey write test for this
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_warn

__all__ = ["FreezeCallback"]


class FreezeCallback(Callback):
  def __init__(self, active, mask):
    self.active = active
    self.mask = mask

  def on_train_start(self, trainer, pl_module):
    if self.active:
      pl_module.model.freeze_module(mask=self.mask)
      rank_zero_warn("ON_TRAIN_START: Freezed active")
