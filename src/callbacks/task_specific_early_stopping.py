import time
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info

__all__ = ['TaskSpecificEarlyStopping']

class TaskSpecificEarlyStopping(Callback):
  def __init__(
    self,
    nr_tasks, 
    patience = 5,
    timelimit_in_min = 60,
    verbose = True
  ):
    super().__init__()
    
    self.verbose = verbose
    self.timelimit_in_min = 60
    self.patience = 5
    
    self.time_buffer = [time.time()]*nr_tasks
    self.best_metric_buffer = [9999]*nr_tasks
    self.k_not_in_best_buffer = [0]*nr_tasks
    
    if self.verbose:
      rank_zero_info(f'TimeLimitCallback is set to {self.timelimit_in_min}min')

  def on_validation_end(self, trainer, pl_module):
    if trainer.running_sanity_check:
      return

    self._run_early_stopping_check(trainer, pl_module)

  def on_validation_epoch_end(self, trainer, pl_module):
    trainer.callback_metrics['task_count/dataloader_idx_0']
    if trainer.running_sanity_check:
      return

  def _run_early_stopping_check(self, trainer, pl_module):
    should_stop = False
    nr = pl_module._task_count
    if trainer.current_epoch == 0:
      # set task start time
      self.time_buffer[nr] = time.time()
   
    # check time 
    if (time.time() - self.time_buffer[nr])*60 > self.timelimit_in_min:
      # time limit reached
      should_stop = True

    metric = trainer.callback_metrics[f'val_acc/dataloader_idx_{nr}']
    
    if metric < self.best_metric_buffer[nr]:
      self.best_metric_buffer[nr] = metric
      self.k_not_in_best_buffer[nr] = 0
    else:
      self.k_not_in_best_buffer[nr] += 1
   
    if self.k_not_in_best_buffer[nr] > self.patience:
      should_stop = True
   
    if bool(should_stop):
        self.stopped_epoch = trainer.current_epoch
        trainer.should_stop = True
        
        # stop every ddp process if any world process decides to stop
        should_stop = trainer.accelerator_backend.early_stopping_should_stop(pl_module)
        trainer.should_stop = should_stop
   
    if self.verbose:
      string = 'Callback State\n'
      string += f'Trainger should stop: {should_stop}\n'
      for i in range( len(self.best_metric_buffer) ):
        n = self.k_not_in_best_buffer[i]
        v = self.best_metric_buffer[i]
        t = int( (time.time() - self.time_buffer[i])/60 )
        string += f'State {i}: Did not make top k n{n}-times. Best {v}. Time: {t}min!\n'
      rank_zero_info(string)
  # def _validate_condition_metric(self, logs):
  #   monitor_val = logs.get(self.monitor)

  #   error_msg = (f'Early stopping conditioned on metric `{self.monitor}`'
  #          f' which is not available. Pass in or modify your `EarlyStopping` callback to use any of the'
  #          f' following: `{"`, `".join(list(logs.keys()))}`')

  #   if monitor_val is None:
  #     if self.strict:
  #       raise RuntimeError(error_msg)
  #     if self.verbose > 0:
  #       rank_zero_warn(error_msg, RuntimeWarning)

  #     return False

  #   return True