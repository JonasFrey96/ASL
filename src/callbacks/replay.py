# TODO: Jonas Frey write test for this

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_warn

__all__ = ['VisuCallback']

class ReplayCallback(Callback):
  def __init__(self ):
    pass

  def on_epoch_start(self, trainer, pl_module):
    ls_global_indices = trainer.train_dataloader.dataset.datasets.get_replay_datasets_globals()
    for i,ls in enumerate(ls_global_indices):
      bins_np, valid_np = pl_module._rssb.get()   
      elements = (bins_np[valid_np]).tolist()
      for e in elements:
        assert e in ls
      ls_global_indices[i] = elements
    
    trainer.train_dataloader.dataset.datasets.set_replay_datasets_globals( ls_global_indices )


  # def on_train_start(self, trainer, pl_module):
  #   pass
 
  # def training_step_end(self, trainer, pl_module, outputs):
  #   pass
 
  # def validation_step(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
  #   pass
 
  # def validation_step_end( self, trainer, pl_module, outputs ):
  #   pass