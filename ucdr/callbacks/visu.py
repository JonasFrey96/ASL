# TODO: Jonas Frey write test for this
from pytorch_lightning.callbacks import Callback

import torch
from torchvision.utils import make_grid
import os
from ucdr.visu import Visualizer

__all__ = ["VisuCallback"]


class VisuCallback(Callback):
    def __init__(self, exp):
        self.visualizer = Visualizer(
            p_visu=os.path.join(exp["name"], "visu"),
            logger=None,
            store=False,
            num_classes=exp["model"]["cfg"]["num_classes"] + 1,
        )

        self.logged_images = {"train": 0, "val": 0, "test": 0}
        self.visu_cfg = exp["visu"]

    def on_train_start(self, trainer, pl_module):
        # Set the Logger given that on init not initalized yet-
        self.visualizer.logger = pl_module.logger
        pl_module._visu_callback = self

    def on_epoch_start(self, trainer, pl_module):
        self.visualizer.epoch = pl_module.current_epoch
        # reset logged images counter
        self.logged_images = dict.fromkeys(self.logged_images, 0)

    def training_step_end(self, trainer, pl_module, outputs):
        # Logging + Visu
        if (
            self.visu_cfg["images"][pl_module._mode] > self.logged_images[pl_module._mode]
            and pl_module.current_epoch % self.visu_cfg["every_n_epochs"] == 0
        ):
            self._visu(trainer, pl_module, outputs, pl_module._task_count)

    def on_validation_start(self, trainer, pl_module):
        # Set the Logger given that on init not initalized yet-
        self.dataloader_idx_store = -1

    def validation_step_end(self, trainer, pl_module, outputs):
        dataloader_idx = outputs["dataloader_idx"]

        # reset for each dataloader the number of stored indices
        if self.dataloader_idx_store != dataloader_idx:
            self.dataloader_idx_store = dataloader_idx
            self.logged_images["val"] = 0

        if (
            self.visu_cfg["images"]["val"] > self.logged_images["val"]
            and pl_module.current_epoch % self.visu_cfg["every_n_epochs"] == 0
        ):
            self._visu(
                trainer,
                pl_module,
                outputs,
                pl_module._task_count,
                optional_key="Eval" + str(dataloader_idx),
            )

    def _visu(self, trainer, pl_module, outputs, task_nr, optional_key=""):
        pred = torch.argmax(outputs["pred"], 1).clone().detach()
        label = outputs["label"].clone().detach()
        pred[label == -1] = -1
        pred += 1
        label += 1

        BS = pred.shape[0]
        rows = int(BS ** 0.5)
        grid_pred = make_grid(
            pred[:, None].repeat(1, 3, 1, 1),
            nrow=rows,
            padding=2,
            scale_each=False,
            pad_value=0,
        )
        grid_label = make_grid(
            label[:, None].repeat(1, 3, 1, 1),
            nrow=rows,
            padding=2,
            scale_each=False,
            pad_value=0,
        )
        grid_image = make_grid(outputs["ori_img"], nrow=rows, padding=2, scale_each=False, pad_value=0)

        nr = self.logged_images[pl_module._mode]

        if "aux_label" in outputs:
            aux_label = outputs["aux_label"].clone().detach()
            if len(aux_label.shape) == 4:
                aux_label = torch.argmax(aux_label, dim=1)
            aux_pred = torch.argmax(outputs["pred"], 1).clone().detach()
            aux_pred[aux_label == -1] = -1
            aux_pred += 1
            aux_label += 1

            grid_aux_label = make_grid(
                aux_label[:, None].repeat(1, 3, 1, 1),
                nrow=rows,
                padding=2,
                scale_each=False,
                pad_value=0,
            )

            grid_aux_pred = make_grid(
                aux_pred[:, None].repeat(1, 3, 1, 1),
                nrow=rows,
                padding=2,
                scale_each=False,
                pad_value=0,
            )
            self.visualizer.plot_segmentation(tag=f"", seg=grid_aux_pred[0], method="right")
            self.visualizer.plot_segmentation(
                tag=f"{pl_module._mode}_Task{task_nr}_{optional_key}_AUX_left_pred_right_{nr}",
                seg=grid_aux_label[0],
                method="left",
            )
            self.visualizer.plot_segmentation(tag=f"", seg=grid_label[0], method="right")
            self.visualizer.plot_segmentation(
                tag=f"{pl_module._mode}_Task{task_nr}_{optional_key}__AUX_left_GT_right_{nr}",
                seg=grid_aux_label[0],
                method="left",
            )
        self.visualizer.plot_segmentation(tag=f"", seg=grid_pred[0], method="right")
        self.visualizer.plot_segmentation(
            tag=f"{pl_module._mode}_Task{task_nr}_{optional_key}__gt_left_pred_right_{nr}",
            seg=grid_label[0],
            method="left",
        )
        self.visualizer.plot_segmentation(tag=f"", seg=grid_pred[0], method="right")
        self.visualizer.plot_image(
            tag=f"{pl_module._mode}_Task{task_nr}_{optional_key}_img_ori_left_pred_right_{nr}",
            img=grid_image,
            method="left",
        )

        self.visualizer.plot_detectron(img=grid_image, label=grid_label[0], method="left", alpha=0.5)
        self.visualizer.plot_detectron(
            img=grid_image,
            label=grid_pred[0],
            method="right",
            alpha=0.5,
            tag=f"{pl_module._mode}_Task{task_nr}_{optional_key}_gt_left_pred_right_overlay_{nr}",
        )

        self.logged_images[pl_module._mode] += 1
