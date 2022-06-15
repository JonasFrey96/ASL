import time
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
import logging

__all__ = ["TaskSpecificEarlyStopping"]


class TaskSpecificEarlyStopping(Callback):
  def __init__(
    self,
    nr_tasks,
    patience=5,
    timelimit_in_min=60,
    verbose=True,
    minimal_increase=0.0,
    max_epoch_count=-1,
  ):
    super().__init__()

    self.verbose = verbose
    self.timelimit_in_min = timelimit_in_min
    self.patience = patience
    self.epoch = -1
    self.time_buffer = [time.time()] * nr_tasks
    self.best_metric_buffer = [0] * nr_tasks
    self.k_not_in_best_buffer = [0] * nr_tasks
    self.minimal_increase = minimal_increase
    self.training_start_epoch = 0
    self.max_epoch_count = max_epoch_count

    if self.verbose:
      rank_zero_info(f"TimeLimitCallback is set to {self.timelimit_in_min}min")

  def on_validation_end(self, trainer, pl_module):
    if trainer.running_sanity_check:
      return

    self._run_early_stopping_check(trainer, pl_module)

  def on_validation_epoch_end(self, trainer, pl_module):
    # trainer.callback_metrics['task_count/dataloader_idx_0']
    if trainer.running_sanity_check:
      return

  def on_train_start(self, trainer, pl_module):
    """Called when the train begins."""
    nr = pl_module._task_count
    # set task start time
    self.time_buffer[nr] = time.time()

    self.training_start_epoch = pl_module.current_epoch

  def _run_early_stopping_check(self, trainer, pl_module):
    should_stop = False
    nr = pl_module._task_count

    if self.epoch != trainer.current_epoch:
      self.epoch = trainer.current_epoch
      # check time
      if (time.time() - self.time_buffer[nr]) / 60 > self.timelimit_in_min or (
        self.max_epoch_count != -1
        and self.epoch - self.training_start_epoch >= self.max_epoch_count
      ):
        if (
          self.max_epoch_count != -1
          and self.epoch - self.training_start_epoch >= self.max_epoch_count
        ):
          print("TSES: Stopped due to max epochs")
        else:
          print("TSES: Stopped due to timelimit reached!")
        should_stop = True

      try:
        metric = trainer.callback_metrics[f"val_acc/dataloader_idx_{nr}"]
      except:
        metric = trainer.callback_metrics[f"val_acc"]

      if metric > self.best_metric_buffer[nr] + self.minimal_increase:
        self.best_metric_buffer[nr] = metric
        self.k_not_in_best_buffer[nr] = 0
      else:
        self.k_not_in_best_buffer[nr] += 1

      if self.k_not_in_best_buffer[nr] > self.patience:
        should_stop = True
        print("TSES: Stopped due to not in best buffer!")

      if bool(should_stop):
        self.stopped_epoch = trainer.current_epoch
        trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        trainer.should_stop = trainer.training_type_plugin.reduce_boolean_decision(
          trainer.should_stop
        )

      if self.verbose:
        print("TSES: Callback State")
        print(str(f"TSES: Trainger should stop: {should_stop}"))
        for i in range(len(self.best_metric_buffer)):
          n = self.k_not_in_best_buffer[i]
          v = self.best_metric_buffer[i]
          t = int((time.time() - self.time_buffer[i]) / 60)
          print(
            str(
              f"TSES: S{i} not TopK n{n}-th - Best {round(float(v),2)} - Time: {t}min - Current {round(float(metric),2)}!"
            )
          )
    else:
      if self.verbose:
        logging.warning("TSES: Visited twice at same epoch")
