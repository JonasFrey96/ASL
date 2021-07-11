# STD
import copy
import time

# MISC
import numpy as np

# DL-framework
import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import metrics as pl_metrics
from torch.nn import functional as F
import datetime
import pickle
import os

# MODULES
from models_asl import FastSCNN, ReplayStateSyncBack, Teacher

# BUFFER FILLING
from uncertainty import get_softmax_uncertainty_max
from uncertainty import get_softmax_uncertainty_distance
from uncertainty import get_softmax_uncertainty_entropy

__all__ = ["Network"]


def two_line(a, b, length=40):
  return str(str(a) + " " * (length - len(str(a))) + str(b))


def wrap(s, length, hard=False):
  if len(s) < length:
    return s + " " * (length - len(s))
  if len(s) > length and hard:
    return s[:length]
  return s


class Network(LightningModule):
  def __init__(self, exp, env):
    super().__init__()
    self._epoch_start_time = time.time()
    self._exp = exp
    self._env = env
    self.hparams["lr"] = self._exp["lr"]

    if self._exp["model"]["name"] == "Fast-SCNN":
      self.model = FastSCNN(**self._exp["model"]["cfg"])
    else:
      raise Exception("Model name not implemented")

    self._rssb = ReplayStateSyncBack(**exp["replay"]["cfg_rssb"])

    self._mode = "train"

    self.train_acc = pl_metrics.classification.Accuracy()
    self.train_aux_acc = pl_metrics.classification.Accuracy()
    self.train_aux_vs_gt_acc = pl_metrics.classification.Accuracy()

    self.val_acc = torch.nn.ModuleList(
      [
        pl_metrics.classification.Accuracy()
        for i in range(exp["replay"]["cfg_rssb"]["bins"])
      ]
    )
    self.val_aux_acc = torch.nn.ModuleList(
      [
        pl_metrics.classification.Accuracy()
        for i in range(exp["replay"]["cfg_rssb"]["bins"])
      ]
    )
    self.val_aux_vs_gt_acc = torch.nn.ModuleList(
      [
        pl_metrics.classification.Accuracy()
        for i in range(exp["replay"]["cfg_rssb"]["bins"])
      ]
    )

    self.test_acc = pl_metrics.classification.Accuracy()

    self._task_name = "NotDefined"  # is used for model checkpoint nameing
    self._task_count = (
      0  # so this here might be a bad idea. Decide if we know the task or not
    )
    self._type = (
      torch.float16 if exp["trainer"].get("precision", 32) == 16 else torch.float32
    )
    self._train_start_time = time.time()

    self._replayed_samples = 0
    self._real_samples = 0
    self._val_results = {}

    self._visu_callback = None
    self._ltmene = self._exp["visu"].get("log_training_metric_every_n_epoch", 9999)

    self._val_epoch_results = []

    # If teacher is active used to generate teacher labels.
    # Current implementation only supports a fixed teacher network.
    self._teacher = Teacher(**exp["teacher"], base_path=env["base"])

  def append_training_epoch_results(self, results):
    if len(self._val_epoch_results) == 0:
      for r in results:
        self._val_epoch_results.append([r])
      self._val_epoch_results.append([self.current_epoch])
      self._val_epoch_results.append([self._task_count])
    else:
      assert len(self._val_epoch_results) - 2 == len(results)
      for j, r in enumerate(results):
        self._val_epoch_results[j].append(r)
      self._val_epoch_results[-2].append(self.current_epoch)
      self._val_epoch_results[-1].append(self._task_count)

  def forward(self, batch, **kwargs):
    if kwargs.get("replayed", None) is not None:
      injection_mask = kwargs["replayed"] != -1
      outputs = self.model.injection_forward(
        x=batch,
        injection_features=kwargs["injection_features"],
        injection_mask=injection_mask,
      )
    else:
      outputs = self.model(batch)
    return outputs

  def compute_loss(self, pred, label, aux_valid, replayed, aux_label=None, **kwargs):
    """
    Args:
        pred (torch.tensor): BSxCxHxW.
        label (torch.tensor]): BSxHxW.
        aux_label (torch.tensor): BSxHxW or BSxCxHxW.  -> aux_label might be provided by dataloader or by teacher model. Might be soft or hard
        replayed (torch.long): BS wheater a sampled is replayed or not.
        use_aux (torch.bool): Wheater to use aux_label or label to calculate the loss.
        not_reduce (bool, optional): reduce the loss or return for each element in batch.
    Returns:
        [type]: [description]
    """
    nr_replayed = (replayed != -1).sum()
    BS = replayed.shape[0]
    self._replayed_samples += int(nr_replayed)
    self._real_samples += int(BS - nr_replayed)

    # compute auxillary loss
    if aux_valid.sum() != 0:
      if len(aux_label.shape) == 4:
        # soft labels provided
        aux_loss = F.mse_loss(
          torch.nn.functional.softmax(pred, dim=1), aux_label, reduction="none"
        )
        aux_loss = aux_loss.mean(dim=[1, 2, 3])
        aux_loss *= self._exp.get("loss", {}).get("soft_aux_label_factor", 1)
      else:
        # hard labels provided
        aux_loss = F.cross_entropy(pred, aux_label, ignore_index=-1, reduction="none")
        aux_loss = aux_loss.mean(dim=[1, 2])
    else:
      aux_loss = torch.zeros((BS), device=pred.device)
    aux_loss *= self._exp.get("loss", {}).get("aux_label_factor", 1)

    # compute normal loss on labels
    if aux_valid.sum() != BS:
      non_aux_loss = F.cross_entropy(
        pred, label, ignore_index=-1, reduction="none"
      ).mean(dim=[1, 2])
    else:
      non_aux_loss = torch.zeros((BS), device=pred.device)

    # return the reduce mean
    return ((aux_loss * aux_valid).sum() + (non_aux_loss * ~aux_valid).sum()) / BS

  def parse_batch(self, batch):
    batch = self._teacher.modify_batch(batch)
    ba = {}
    if len(batch) == 1:
      raise Exception("Dataloader is set to unique and not implemented")
    ba["images"] = batch[0]
    ba["label"] = batch[1]
    ba["replayed"] = batch[2]
    if len(batch) == 4:
      ba["ori_img"] = batch[3]
    if len(batch) == 5:
      ba["aux_label"] = batch[3]
      ba["aux_valid"] = batch[4]
    if len(batch) == 6:
      ba["aux_label"] = batch[3]
      ba["aux_valid"] = batch[4]
      ba["ori_img"] = batch[5]
    return ba

  ##################
  #   TRAINING     #
  ##################

  def on_train_start(self):
    print("")
    print("================ ON_TRAIN_START ==================")
    print(two_line(" TASK NAME: ", self._task_name))
    print(two_line(" TASK COUNT: ", self._task_count))
    print(two_line(" CURRENT EPOCH: ", self.current_epoch))
    print(two_line(" CURRENT EPOCH: ", self.global_step))
    print(two_line(" RSSB STATE: ", self._rssb.valid.sum(dim=1)))
    for i in range(self._task_count):
      m = min(5, self._rssb.nr_elements)
      print(two_line("   RSSB INIDI TASK-" + str(i) + " :", self._rssb.bins[i, :m]))

    print(
      two_line(" TRAINING DATASET LENGTH:", len(self.trainer.train_dataloader.dataset))
    )

    for j, d in enumerate(self.trainer.val_dataloaders):
      print(two_line(f" VALIDATION DATASET {j} LENGTH:", len(d.dataset)))

    print(" =============  ON_TRAIN_START_DONE ===============")

  def on_train_epoch_start(self):
    self._mode = "train"

  def training_step(self, batch, batch_idx):
    ba = self.parse_batch(batch)
    outputs = self(batch=ba["images"])

    if not ("aux_valid" in ba.keys()):
      ba["aux_valid"] = torch.zeros(
        (ba["images"].shape[0]), device=ba["images"].device, dtype=torch.bool
      )
    loss = self.compute_loss(pred=outputs[0], **ba)
    self.log(f"{self._mode}_loss", loss, on_step=False, on_epoch=True)
    ret = {
      "loss": loss,
      "pred": outputs[0],
      "label": ba["label"],
      "ori_img": ba["ori_img"],
    }

    if "aux_label" in ba.keys():
      ret["aux_label"] = ba["aux_label"]
      ret["aux_vaild"] = ba["aux_valid"]

    return ret

  def training_step_end(self, outputs):
    with torch.no_grad():
      self._visu_callback.training_step_end(self.trainer, self, outputs)

    # LOG REPLAY / REAL
    self.logger.log_metrics(
      metrics={
        "real": torch.tensor(self._real_samples),
        "replayed": torch.tensor(self._replayed_samples),
      },
      step=self.global_step,
    )

    if self.current_epoch % self._ltmene == 0 and (
      self.current_epoch != 0 or self._ltmene == 0
    ):
      # LOG ACCURRACY
      self._acc_cal(
        outputs, self.train_acc, self.train_aux_acc, self.train_aux_vs_gt_acc
      )

    return {"loss": outputs["loss"]}

  ####################
  #   VALIDATION     #
  ####################

  def on_validation_epoch_start(self):
    self._mode = "val"

  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    images, label = batch[:2]

    outputs = self(images)

    loss = F.cross_entropy(outputs[0], label, ignore_index=-1)

    ret = {
      "pred": outputs[0],
      "label": label,
      "dataloader_idx": dataloader_idx,
      "loss_ret": loss,
    }

    if len(batch) == 3:
      ret["ori_img"] = batch[2]
    if len(batch) > 3:
      ret["aux_label"] = batch[2]
      ret["aux_valid"] = batch[3]
      ret["ori_img"] = batch[4]

    return ret

  def validation_step_end(self, outputs):
    with torch.no_grad():
      self._visu_callback.validation_step_end(self.trainer, self, outputs)
    dataloader_idx = outputs["dataloader_idx"]
    self._acc_cal(
      outputs,
      self.val_acc[dataloader_idx],
      self.val_aux_acc[dataloader_idx],
      self.val_aux_vs_gt_acc[dataloader_idx],
    )

  @torch.no_grad()
  def _acc_cal(self, outputs, acc, aux_acc, aux_vs_gt_acc):
    pred = torch.argmax(outputs["pred"], 1)

    m = outputs["label"] > -1
    self.log(
      f"{self._mode}_gt_label_valid_ratio",
      m.sum() / torch.numel(m),
      on_step=False,
      on_epoch=True,
    )

    acc(pred[m], outputs["label"][m])
    self.log(f"{self._mode}_acc", acc, on_step=False, on_epoch=True)

    if "aux_valid" in outputs.keys():
      aux_m = outputs["aux_label"] > -1
      self.log(
        f"{self._mode}_aux_label_valid_ratio",
        aux_m.sum() / torch.numel(aux_m),
        on_step=False,
        on_epoch=True,
      )

      aux_acc(pred[aux_m], outputs["aux_label"][aux_m])

      self.log(f"{self._mode}_aux_acc", aux_acc, on_step=False, on_epoch=True)

      aux_m2 = aux_m * m
      aux_vs_gt_acc(outputs["aux_label"][aux_m2], outputs["label"][aux_m2])
      self.log(
        f"{self._mode}_aux_vs_gt_acc", aux_vs_gt_acc, on_step=False, on_epoch=True
      )

  def validation_epoch_end(self, outputs):
    self.log(
      f"task_count", self._task_count, on_step=False, on_epoch=True, prog_bar=False
    )

    metrics = self.trainer.logger_connector.callback_metrics
    me = copy.deepcopy(metrics)
    for k in me.keys():
      try:
        me[k] = "{:10.4f}".format(me[k])
      except:
        pass

    t_l = me.get("train_loss", "NotDef")
    v_acc = me.get("val_acc", "NotDef")

    nr = 0
    for m in metrics.keys():
      if m.find("val_acc/dataloader") != -1:
        nr += 1

    results = [float(self.val_acc[i].compute()) for i in range(nr)]
    self.append_training_epoch_results(results)

    try:
      # only works when multiple val-dataloader are set!
      if len(self._val_results) == 0:
        for i in range(self._exp["replay"]["cfg_rssb"]["bins"]):
          self._val_results[f"val_acc/dataloader_idx_{i}"] = float(
            metrics[f"val_acc/dataloader_idx_{i}"]
          )
      else:
        val_results = {}
        for i in range(self._exp["replay"]["cfg_rssb"]["bins"]):
          val_results[f"val_acc/dataloader_idx_{i}"] = float(
            metrics[f"val_acc/dataloader_idx_{i}"]
          )
          res = (
            self._val_results[f"val_acc/dataloader_idx_{i}"]
            - val_results[f"val_acc/dataloader_idx_{i}"]
          )
          self.log(f"forgetting/acc_idx_{i}", res, on_epoch=True, prog_bar=False)

        if self._task_count > 0:
          res = 0
          for i in range(self._task_count):
            res += (
              self._val_results[f"val_acc/dataloader_idx_{i}"]
              - val_results[f"val_acc/dataloader_idx_{i}"]
            )

          res /= self._task_count
          self.log(f"forgetting/acc_avg_pervious", res, on_epoch=True, prog_bar=False)

        res = (
          val_results[f"val_acc/dataloader_idx_{self._task_count}"]
          - self._val_results[f"val_acc/dataloader_idx_{self._task_count}"]
        )
        self.log(f"learning/acc_current", res, on_epoch=True, prog_bar=False)
    except:
      pass

    epoch = str(self.current_epoch)

    t = time.time() - self._epoch_start_time
    t = str(datetime.timedelta(seconds=round(t)))
    t2 = time.time() - self._train_start_time
    t2 = str(datetime.timedelta(seconds=round(t2)))
    if not self.trainer.running_sanity_check:
      print("VALIDATION_EPOCH_END: Time for a complete epoch: " + str(t))
      n = self._task_name
      n = wrap(n, 20)
      t = wrap(t, 10, True)
      epoch = wrap(epoch, 3)
      t_l = wrap(t_l, 6)
      v_acc = wrap(v_acc, 6)

      print(
        str(
          f"VALIDATION_EPOCH_END: Exp: {n}, Epoch: {epoch}, TimeEpoch: {t}, TimeStart: {t2}"
        )
      )
      print(str(f"VALIDATION_EPOCH_END: Train-Loss: {t_l }, Val-Acc: {v_acc}"))

    self._epoch_start_time = time.time()
    print("VALIDATION_EPOCH_END: Should stop: " + str(self.trainer.should_stop))

  def on_test_epoch_start(self):
    """
    Memory Buffer Filling Explained:
      1. Perform normal trainer.fit
      2. Call trainer.test -> extracts with the desired method the correct global indices
      3. In test_epoch_end the computed indices are stored in the RSSB
      4. (skip to next task)
      5. Here tightly integrated ReplayCallback starts on_train_start
      6. Replay Callback: Fetch: RSSB state( Set Global Indices).
                          Fetch: For each Dataset in the Ensemble the Global Indices.
                          Verify that RSSB State is a valid subset of the Global Indices in the Ensemble.
                          Overwrite directly the global indices list in the dataset and reset the length.
      Overview:
        Ensembel dataset responsibly for sampling from the replay.
        trainer.test stores values to rssb (values are now a part of the models statedict and can be reloaded)
        Replay callback is responsibly for "contracting" the dataloader indices according to rssb state.

    """
    # PREPARE MODEL
    self._mode = "test"
    self._restore_extract, self._restore_extract_layer = (
      self.model.extract,
      self.model.extract_layer,
    )
    self.model.extract = True
    self.model.extract_layer = "fusion"

    # PREPARE LOGGING STRUCTURES
    gtli = self.trainer.test_dataloaders[0].dataset.global_to_local_idx

    nr = len(self.trainer.test_dataloaders[0].dataset)
    self.logs_test = {
      "loss": np.full((nr,), np.inf),
      "acc": np.zeros((nr,)),
      "softmax_max": np.zeros((nr,)),
      "softmax_distance": np.ones((nr,)),
      "softmax_entropy": np.zeros((nr,)),
      "indices": np.array(gtli),
      "features": np.zeros(
        (nr, 40, 128)
      ),  # for each class extract a 128 dimensional vector
      "label_count_pred": np.zeros((nr, 40)),
      "label_count_gt": np.zeros((nr, 40)),
    }
    self.count = 0

  @torch.no_grad()
  def test_step(self, batch, batch_idx):
    _ = self.validation_step(batch, batch_idx, dataloader_idx=0)
    images, label = batch[:2]
    BS = images.shape[0]

    outputs = self.model(images)
    pred = outputs[0]

    # EXTRACT LATENT FEATURE
    features = outputs[1]
    _BS, _C, _H, _W = features.shape
    label_features = F.interpolate(
      label[:, None].type(features.dtype), (_H, _W), mode="nearest"
    )[:, 0].type(label.dtype)
    NC = self._exp["model"]["cfg"]["num_classes"]
    latent_feature = torch.zeros(
      (_BS, NC, _C), device=self.device
    )  # 10kB per Image if 16 bit
    for b in range(BS):
      for n in range(NC):
        m = label_features[b] == n
        if m.sum() != 0:
          latent_feature[b, n] = features[b][:, m].mean(dim=1)
    self.logs_test["features"][
      self.count : self.count + BS
    ] = latent_feature.cpu().numpy()
    # EXTRACT UNCERTAINTY
    self.logs_test["softmax_max"][self.count : self.count + BS] = (
      get_softmax_uncertainty_max(pred).cpu().numpy()
    )  # confident 0 , uncertain 1
    self.logs_test["softmax_distance"][self.count : self.count + BS] = (
      get_softmax_uncertainty_distance(pred).cpu().numpy()
    )  # confident 0 , uncertain 1
    self.logs_test["softmax_entropy"][self.count : self.count + BS] = (
      get_softmax_uncertainty_entropy(pred).cpu().numpy()
    )  # confident 0 , uncertain 1

    # EXTRACT LOSS
    self.logs_test["loss"][self.count : self.count + BS] = (
      F.cross_entropy(pred, label, ignore_index=-1, reduction="none")
      .mean(dim=[1, 2])
      .cpu()
      .numpy()
    )

    # EXTRACT ACC + LABEL COUNT
    pred_onehot = torch.argmax(pred, 1)
    m = label > -1
    for b in range(BS):
      self.logs_test["acc"][self.count + b] = float(
        self.test_acc(pred_onehot[b, m[b]], label[b, m[b]])
      )
      unique, counts = torch.unique(pred_onehot[b], return_counts=True)
      self.logs_test["label_count_pred"][self.count + b][
        unique.cpu().numpy()
      ] = counts.cpu().numpy()
      unique, counts = torch.unique(label, return_counts=True)
      self.logs_test["label_count_gt"][self.count + b][
        unique.cpu().numpy()
      ] = counts.cpu().numpy()

  def test_epoch_end(self, outputs):

    p = os.path.join(self._exp["name"], f"test_result_task{self._task_count}.pkl")
    with open(p, "wb") as handle:
      pickle.dump(self.logs_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    self.model.extract, self.model.extract_layer = (
      self._restore_extract,
      self._restore_extract_layer,
    )

    stra = self._exp["replay"]["cfg_filling"]["strategy"]
    nr_indices = len(self.logs_test["indices"])
    if nr_indices < self._rssb.nr_elements:
      self._rssb.bins[self._task_count][:nr_indices] = torch.from_numpy(
        self.logs_test["indices"]
      )
      self._rssb.valid[self._task_count, :nr_indices] = True
      self._rssb.valid[self._task_count, nr_indices:] = False
    elif stra == "random":
      ind = np.random.permutation(self.logs_test["indices"])[: self._rssb.nr_elements]
      self._rssb.bins[self._task_count] = torch.from_numpy(ind)
      self._rssb.valid[self._task_count, :] = True

    elif (
      stra == "metric_softmax_distance"
      or stra == "metric_softmax_max"
      or stra == "metric_softmax_entropy"
      or stra == "loss"
      or stra == "acc"
    ):
      sel = np.argsort(self.logs_test[stra])
      metric_mode = self._exp["replay"]["cfg_filling"]["metric_mode"]

      if metric_mode == "max":
        sel = self.logs_test["indices"][sel[: self._rssb.nr_elements]]

      elif metric_mode == "min":
        sel = self.logs_test["indices"][sel[-self._rssb.nr_elements :]]

      elif metric_mode == "equal":
        sel2 = np.round(
          np.linspace(0, nr_indices - 1, self._rssb.nr_elements), 0
        ).astype(np.uint32)
        sel = self.logs_test["indices"][sel[sel2]]
      else:
        raise ValueError("Not defined")

      self._rssb.bins[self._task_count] = torch.from_numpy(sel)
      self._rssb.valid[self._task_count, :] = True
    elif stra == "cover_sequence":
      sel = np.round(np.linspace(0, nr_indices - 1, self._rssb.nr_elements), 0).astype(
        np.uint32
      )
      self._rssb.bins[self._task_count] = torch.from_numpy(
        self.logs_test["indices"][sel]
      )
      self._rssb.valid[self._task_count, :] = True

    else:
      raise Exception("Not implemented")

    val = min(self._rssb.nr_elements, 10)
    print(
      str(
        f"\nTEST_EPOCH_END: In Test overwritten bin {self._task_count} following indices: "
        + str(self._rssb.bins[self._task_count, :val])
        + "\n \n"
      )
    )

  def on_save_checkpoint(self, params):
    pass

  def configure_optimizers(self):
    if self._exp["optimizer"]["name"] == "ADAM":
      optimizer = torch.optim.Adam(
        [{"params": self.model.parameters()}], lr=self.hparams["lr"]
      )
    elif self._exp["optimizer"]["name"] == "SGD":
      optimizer = torch.optim.SGD(
        [{"params": self.model.parameters()}],
        lr=self.hparams["lr"],
        **self._exp["optimizer"]["sgd_cfg"],
      )
    else:
      raise Exception

    if self._exp.get("lr_scheduler", {}).get("active", False):
      n = self._exp["lr_scheduler"]["name"]
      if n == "POLY":
        # polynomial lr-scheduler
        init_lr = self.hparams["lr"]
        max_epochs = self._exp["lr_scheduler"]["poly_cfg"]["max_epochs"]
        target_lr = self._exp["lr_scheduler"]["poly_cfg"]["target_lr"]
        power = self._exp["lr_scheduler"]["poly_cfg"]["power"]
        lambda_lr = (
          lambda epoch: (
            ((max_epochs - min(max_epochs, epoch)) / max_epochs) ** (power)
          )
          + (1 - (((max_epochs - min(max_epochs, epoch)) / max_epochs) ** (power)))
          * target_lr
          / init_lr
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer, lambda_lr, last_epoch=-1, verbose=True
        )
        interval = "epoch"

      elif n == "ONE_CYCLE_LR":
        assert self.max_epochs != -1
        assert self._exp["task_specific_early_stopping"]["active"]

        cfg = self._exp["lr_scheduler"]["one_cycle_lr_cfg"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
          optimizer,
          self.hparams["lr"],
          epochs=int(self.max_epochs) + 1,
          steps_per_epoch=int(self.length_train_dataloader),
          pct_start=cfg["pct_start"],
          anneal_strategy="linear",
          final_div_factor=cfg["final_div_factor"],
        )
        # trainer.lr_schedulers[0]['scheduler']

        interval = "step"
      else:
        raise ValueError(f"The exp[lr_scheduler][name] is not well define {n}!")

      lr_scheduler = {"scheduler": scheduler, "interval": interval}

      ret = {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    else:
      ret = [optimizer]
    return ret
