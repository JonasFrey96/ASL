import os
import sys

import time
import shutil
import datetime
import argparse
import yaml
import copy
from pathlib import Path
import pickle
import logging as logg

logging = logg.getLogger("lightning")

# Frameworks
import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from pytorch_lightning.loggers.neptune import NeptuneLogger

# Costume Modules
from ucdr.lightning import Network
from ucdr.visu import (
    MainVisualizer,
    validation_acc_plot,
    validation_acc_plot_stored,
)
from ucdr.callbacks import (
    TaskSpecificEarlyStopping,
    VisuCallback,
    ReplayCallback,
)
from ucdr.utils import load_yaml, file_path, load_env
from ucdr.utils import get_neptune_logger, get_tensorboard_logger
from ucdr.datasets import adapter_tg_to_dataloader
from ucdr.task import get_task_generator
from ucdr import UCDR_ROOT_DIR

__all__ = ["train_task"]


def train(exp_cfg_path):
    seed_everything(42)
    exp = load_yaml(exp_cfg_path)
    env = load_env()

    @rank_zero_only
    def create_experiment_folder():
        # Set in name the correct model path
        if exp.get("timestamp", True):
            timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
            model_path = os.path.join(env["base"], exp["name"])
            p = model_path.split("/")
            model_path = os.path.join("/", *p[:-1], str(timestamp) + "_" + p[-1])
        else:
            model_path = os.path.join(env["base"], exp["name"])
            shutil.rmtree(model_path, ignore_errors=True)

        # Create the directory
        Path(model_path).mkdir(parents=True, exist_ok=True)

        # Only copy config files for the main ddp-task
        exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
        print(f"Copy {exp_cfg_path} to {model_path}/{exp_cfg_fn}")
        shutil.copy(exp_cfg_path, f"{model_path}/{exp_cfg_fn}")
        return model_path

    model_path = create_experiment_folder()
    exp["name"] = model_path

    if not exp.get("offline_mode", False):
        logger = get_neptune_logger(exp=exp, env=env, exp_p=exp_cfg_path)
    else:
        logger = get_tensorboard_logger(exp=exp, env=env, exp_p=exp_cfg_path)

    # SET GPUS
    if (exp["trainer"]).get("gpus", -1) == -1:
        nr = torch.cuda.device_count()
        logging.debug(f"Set GPU Count for Trainer to {nr}!")
        for i in range(nr):
            logging.debug(f"Device {i}: " + str(torch.cuda.get_device_name(i)))
        exp["trainer"]["gpus"] = -1

    # TASK GENERATOR
    tg = get_task_generator(
        name=exp["task_generator"]["name"],
        mode=exp["task_generator"]["mode"],
        cfg=exp["task_generator"]["cfg"],
    )
    print(str(tg))

    # MODEL
    model = Network(exp=exp, env=env)

    # COLLECT CALLBACKS
    cb_ls = [LearningRateMonitor(**exp["lr_monitor"]["cfg"])]

    if exp["cb_early_stopping"]["active"]:
        early_stop_callback = EarlyStopping(**exp["cb_early_stopping"]["cfg"])
        cb_ls.appned(early_stop_callback)

    if exp["task_specific_early_stopping"]["active"]:
        tses = TaskSpecificEarlyStopping(nr_tasks=len(tg), **exp["task_specific_early_stopping"]["cfg"])
        cb_ls.append(tses)

    cb_ls.append(VisuCallback(exp, model))

    if exp["weights_restore"]:
        # it is not strict since the latent replay buffer is not always available
        p = os.path.join(env["base"], exp["checkpoint_load"])
        if os.path.isfile(p):
            state_dict_loaded = torch.load(p, map_location=lambda storage, loc: storage)["state_dict"]
            res = model.load_state_dict(state_dict_loaded, strict=False)
            # check if some key is missing
            missing_keys_in_dict = res[0]
            assert len(missing_keys_in_dict) == 0
            # assert if to many weights ("here filter for legacy modules")
            missing_keys_in_model = [k for k in res[1] if k.find("rssb") == -1 and k.find("teacher") == -1]
            assert len(missing_keys_in_model) == 0

        else:
            raise Exception("Checkpoint not a file")

    # add distributed plugin
    if exp["trainer"]["gpus"] > 1:
        if exp["trainer"]["accelerator"] == "ddp" or exp["trainer"]["accelerator"] is None:
            ddp_plugin = DDPPlugin(find_unused_parameters=exp["trainer"].get("find_unused_parameters", False))
        elif exp["trainer"]["accelerator"] == "ddp_spawn":
            ddp_plugin = DDPSpawnPlugin(find_unused_parameters=exp["trainer"].get("find_unused_parameters", False))
        elif exp["trainer"]["accelerator"] == "ddp2":
            ddp_plugin = DDP2Plugin(find_unused_parameters=exp["trainer"].get("find_unused_parameters", False))
        exp["trainer"]["plugins"] = [ddp_plugin]

    for task_nr in range(0, exp["supervisor"]["stop_task"]):
        # Reinitalizing of all datasets

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_path, filename="task" + str(i) + "-{epoch:02d}--{step:06d}", **exp["cb_checkpoint"]["cfg"]
        )

        train_dataloader, val_dataloaders, task_name = adapter_tg_to_dataloader(
            tg, task_nr, exp["loader"], exp["replay"]["cfg_ensemble"], env
        )

        # New Logger
        model._task_name = task_name
        model._task_count = task_nr

        skip = exp["supervisor"]["start_task"] > task_nr
        if skip:
            # VALIDATION
            cfg = copy.deepcopy(exp["trainer"])
            cfg["max_epochs"] = 1
            cfg["limit_train_batches"] = 10
            cfg["limit_val_batches"] = 10
            cfg["check_val_every_n_epoch"] = 1

            trainer = Trainer(
                **cfg, default_root_dir=model_path, callbacks=cb_ls + [checkpoint_callback], logger=logger
            )
            # set params to to determin lr_schedule
            model.length_train_dataloader = 10000
            model.max_epochs = 10000

            res = trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders)
        else:
            trainer = Trainer(
                **exp["trainer"], default_root_dir=model_path, callbacks=cb_ls + [checkpoint_callback], logger=logger
            )
            # FULL TRAINING
            if exp["trainer"]["limit_train_batches"] <= 1.0:
                model.length_train_dataloader = len(train_dataloader) * exp["trainer"]["limit_train_batches"]
            else:
                model.length_train_dataloader = exp["trainer"]["limit_train_batches"]

            model.max_epochs = exp["task_specific_early_stopping"]["cfg"]["max_epoch_count"]
            _ = trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="exp.yml",
        help="Experiment yaml file.",
    )

    args = parser.parse_args()

    exp_cfg_path = args.exp
    if not os.path.isabs(exp_cfg_path):
        exp_cfg_path = os.path.join(UCDR_ROOT_DIR, "cfg/exp", args.exp)

    train(exp_cfg_path)
    torch.cuda.empty_cache()
