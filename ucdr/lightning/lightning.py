# STD
import copy
import time

# MISC
import numpy as np
import datetime
import pickle
import os

# DL-framework
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F

# MODULES
from ucdr.models import FastSCNN
from ucdr.utils import SemanticsMeter, TorchSemanticsMeter

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
    def __init__(self, exp, env, total_tasks=2):
        super().__init__()

        self._epoch_start_time = time.time()
        self._exp = exp
        self._env = env
        self.hparams["lr"] = self._exp["lr"]

        if self._exp["model"]["name"] == "Fast-SCNN":
            self.model = FastSCNN(**self._exp["model"]["cfg"])
        else:
            raise Exception("Model name not implemented")

        self._mode = "train"

        keys = ["train_meter", "test_meter"]
        keys = keys + [f"{i}_val_meter" for i in range(total_tasks)]
        self.meters = torch.nn.ModuleDict(
            {k: TorchSemanticsMeter(self._exp["model"]["cfg"]["num_classes"]) for k in keys}
        )

        self._total_tasks = total_tasks

        self._task_name = "NotDefined"  # is used for model checkpoint nameing
        self._task_count = 0  # so this here might be a bad idea. Decide if we know the task or not
        self._train_start_time = time.time()

        self._replayed_samples = 0
        self._real_samples = 0

    def forward(self, batch, **kwargs):
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
            # if len(aux_label.shape) == 4:
            #     # soft labels provided
            #     aux_loss = F.mse_loss(torch.nn.functional.softmax(pred, dim=1), aux_label, reduction="none")
            #     aux_loss = aux_loss.mean(dim=[1, 2, 3])
            #     aux_loss *= self._exp.get("loss", {}).get("soft_aux_label_factor", 1)
            # else:

            # hard labels provided
            aux_loss = F.cross_entropy(pred, aux_label, ignore_index=-1, reduction="none")
            aux_loss = aux_loss.mean(dim=[1, 2])
        else:
            aux_loss = torch.zeros((BS), device=pred.device)
        aux_loss *= self._exp.get("loss", {}).get("aux_label_factor", 1)

        # compute normal loss on labels
        if aux_valid.sum() != BS:
            non_aux_loss = F.cross_entropy(pred, label, ignore_index=-1, reduction="none").mean(dim=[1, 2])
        else:
            non_aux_loss = torch.zeros((BS), device=pred.device)

        # return the reduce mean
        return ((aux_loss * aux_valid).sum() + (non_aux_loss * ~aux_valid).sum()) / BS

    def parse_batch(self, batch):
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
        print(two_line(" TRAINING DATASET LENGTH:", len(self.trainer.train_dataloader.dataset)))

        for j, d in enumerate(self.trainer.val_dataloaders):
            print(two_line(f" VALIDATION DATASET {j} LENGTH:", len(d.dataset)))
        print(" =============  ON_TRAIN_START_DONE ===============")

    def on_train_epoch_start(self):
        self._mode = "train"

        for k in self.meters.keys():
            self.meters[k].clear()

    def training_step(self, batch, batch_idx):
        self._mode = "train"
        ba = self.parse_batch(batch)
        outputs = self(batch=ba["images"])

        if not ("aux_valid" in ba.keys()):
            ba["aux_valid"] = torch.zeros((ba["images"].shape[0]), device=ba["images"].device, dtype=torch.bool)
        loss = self.compute_loss(pred=outputs[0], **ba)

        self.meters[f"{self._mode}_meter"].update(outputs[0].argmax(dim=1), ba["label"])

        self.log(f"train_loss", loss.item(), on_step=False, on_epoch=True)
        ret = {
            "loss": loss,
            "pred": outputs[0],
            "label": ba["label"],
            "ori_img": ba["ori_img"],
        }

        if "aux_label" in ba.keys():
            ret["aux_label"] = ba["aux_label"]
            ret["aux_valid"] = ba["aux_valid"]

        ret["replay"] = batch[2]

        self._visu_callback.training_step_end(self.trainer, self, ret)

        return ret

    def training_step_end(self, outputs):
        # LOG REPLAY / REAL
        self.logger.log_metrics(
            metrics={
                f"{self._task_count}_real": torch.tensor(self._real_samples),
                f"{self._task_count}_replayed": torch.tensor(self._replayed_samples),
            },
            step=self.global_step,
        )

        return {"loss": outputs["loss"]}

    def training_epoch_end(self, outputs):
        self._mode = "train"
        res = self.meters[f"{self._mode}_meter"].measure()

        self.logger.log_metrics(
            metrics={
                f"{self._task_count}_{self._mode}_mIoU": res[0].item(),
                f"{self._task_count}_{self._mode}_tAcc": res[1].item(),
                f"{self._task_count}_{self._mode}_cAcc": res[2].item(),
            },
            step=self.global_step,
        )

        self.log(f"{self._task_count}_{self._mode}_mIoU", res[0].item(), on_step=False, on_epoch=True)
        self.log(f"{self._task_count}_{self._mode}_tAcc", res[1].item(), on_step=False, on_epoch=True)
        self.log(f"{self._task_count}_{self._mode}_cAcc", res[2].item(), on_step=False, on_epoch=True)

    ####################
    #   VALIDATION     #
    ####################

    def on_validation_epoch_start(self):
        # maybe reset semantic meter
        self._mode = "val"

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._mode = "val"
        images, label = batch[:2]
        outputs = self(images)
        loss = F.cross_entropy(outputs[0], label, ignore_index=-1)

        self.meters[f"{dataloader_idx}_{self._mode}_meter"].update(outputs[0].argmax(dim=1), label)

        ret = {
            "pred": outputs[0].detach(),
            "label": label.detach(),
            "dataloader_idx": dataloader_idx,
            "loss_ret": loss.detach(),
            "aux_label": batch[2].detach(),
            "aux_valid": batch[3].detach(),
            "ori_img": batch[4].detach(),
        }
        with torch.no_grad():
            self._visu_callback.validation_step_end(self.trainer, self, ret)

        return loss.item()

    def validation_step_end(self, outputs):
        pass

    def on_validation_epoch_end(self):
        self._mode = "val"

        for i in range(self._total_tasks):
            res = self.meters[f"{i}_{self._mode}_meter"].measure()
            self.logger.log_metrics(
                metrics={
                    f"{i}_{self._mode}_mIoU": res[0].item(),
                    f"{i}_{self._mode}_tAcc": res[1].item(),
                    f"{i}_{self._mode}_cAcc": res[2].item(),
                },
                step=self.global_step,
            )

            self.log(f"{i}_{self._mode}_mIoU", res[0].item(), on_step=False, on_epoch=True)
            self.log(f"{i}_{self._mode}_tAcc", res[1].item(), on_step=False, on_epoch=True)
            self.log(f"{i}_{self._mode}_cAcc", res[2].item(), on_step=False, on_epoch=True)

    def on_train_end(self):
        pass

    def on_test_epoch_start(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

    def on_save_checkpoint(self, params):
        pass

    def configure_optimizers(self):
        if self._exp["optimizer"]["name"] == "ADAM":
            optimizer = torch.optim.Adam([{"params": self.model.parameters()}], lr=self.hparams["lr"])
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
                    lambda epoch: (((max_epochs - min(max_epochs, epoch)) / max_epochs) ** (power))
                    + (1 - (((max_epochs - min(max_epochs, epoch)) / max_epochs) ** (power))) * target_lr / init_lr
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1, verbose=True)
                interval = "epoch"

            elif n == "ONE_CYCLE_LR":
                assert self.max_epochs != -1
                assert self._exp["task_specific_early_stopping"]["active"]

                cfg = self._exp["lr_scheduler"]["one_cycle_lr_cfg"]
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    self.hparams["lr"],
                    total_steps=int((self.max_epochs * self.length_train_dataloader) + 100),
                    pct_start=cfg["pct_start"],
                    anneal_strategy="linear",
                    final_div_factor=cfg["final_div_factor"],
                    cycle_momentum=False,
                    div_factor=float(cfg.get("div_factor", 10000.0)),
                )
                interval = "step"
            else:
                raise ValueError(f"The exp[lr_scheduler][name] is not well define {n}!")

            lr_scheduler = {"scheduler": scheduler, "interval": interval}

            ret = [optimizer], [lr_scheduler]
        else:
            ret = [optimizer]
        return ret
