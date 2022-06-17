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
    FreezeCallback,
    ReplayCallback,
)
from ucdr.utils import load_yaml, file_path
from ucdr.utils import get_neptune_logger, get_tensorboard_logger
from ucdr.datasets import adapter_tg_to_dataloader
from ucdr.task import get_task_generator

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

    # Reinitalizing of all datasets
    train_dataloader, val_dataloaders, task_name = adapter_tg_to_dataloader(
        tg, task_nr, exp["loader"], exp["replay"]["cfg_ensemble"], env
    )

    dataset_sizes = [int(len(val.dataset) * 5) for val in val_dataloaders]

    if exp["replay"]["cfg_rssb"]["bins"] == -1:
        exp["replay"]["cfg_rssb"]["bins"] = len(tg)

    # MODEL
    model = Network(exp=exp, env=env, dataset_sizes=dataset_sizes)

    # COLLECT CALLBACKS
    lr_monitor = LearningRateMonitor(**exp["lr_monitor"]["cfg"])
    if exp["cb_early_stopping"]["active"]:
        early_stop_callback = EarlyStopping(**exp["cb_early_stopping"]["cfg"])
        cb_ls = [early_stop_callback, lr_monitor]
    else:
        cb_ls = [lr_monitor]
    if exp["task_specific_early_stopping"]["active"]:
        tses = TaskSpecificEarlyStopping(nr_tasks=len(tg), **exp["task_specific_early_stopping"]["cfg"])
        cb_ls.append(tses)

    if exp["cb_checkpoint"]["active"]:
        for i in range(len(tg)):
            if i == task_nr:
                m = "/".join([a for a in model_path.split("/") if a.find("rank") == -1])
                dic = copy.deepcopy(exp["cb_checkpoint"]["cfg"])
                checkpoint_callback = ModelCheckpoint(
                    dirpath=m, filename="task" + str(i) + "-{epoch:02d}--{step:06d}", **dic
                )
                cb_ls.append(checkpoint_callback)
    cb_ls.append(VisuCallback(exp))
    cb_ls.append(ReplayCallback())

    if exp.get("checkpoint_restore", False):
        p = os.path.join(env["base"], exp["checkpoint_load"])
        trainer = Trainer(
            **exp["trainer"],
            default_root_dir=model_path,
            callbacks=cb_ls,
            resume_from_checkpoint=p,
            logger=logger,
        )
        res = model.load_state_dict(torch.load(p)["state_dict"], strict=True)
        print("Weight restore:" + str(res))
    else:
        trainer = Trainer(**exp["trainer"], default_root_dir=model_path, callbacks=cb_ls, logger=logger)

    if exp["weights_restore"]:
        # it is not strict since the latent replay buffer is not always available
        p = os.path.join(env["base"], exp["checkpoint_load"])
        if os.path.isfile(p):
            state_dict_loaded = torch.load(p, map_location=lambda storage, loc: storage)["state_dict"]
            if state_dict_loaded["_rssb.bins"].shape != model._rssb.bins.shape:
                state_dict_loaded["_rssb.bins"] = model._rssb.bins
                state_dict_loaded["_rssb.valid"] = model._rssb.valid

            res = model.load_state_dict(state_dict_loaded, strict=False)

            if len(res[1]) != 0:
                if res[1][0].find("teacher") != -1 and res[1][-1].find("teacher") != -1:
                    print("Restore weights: Got incompatiple teacher keys in file: " + p)
                else:
                    print("Restoring weights: Got incompatiple keys in file: " + p + str(res))
            if len(res[0]) != 0:
                if res[0][0].find("teacher") != -1 and res[0][-1].find("teacher") != -1:
                    print("Restore weights: Missing teacher keys in file: " + p)
                else:
                    print("Restoring weights: Missing keys in file: " + p + str(res))
        else:
            raise Exception("Checkpoint not a file")

    if exp.get("weights_restore_reset_buffer", False):
        model._rssb.valid[:, :] = False
        model._rssb.bins[:, :] = 0

    if model_path.split("/")[-1].find("rank") != -1:
        pa = os.path.join(str(Path(model_path).parent), "main_visu")
    else:
        pa = os.path.join(model_path, "main_visu")

    main_visu = MainVisualizer(
        p_visu=pa,
        logger=logger,
        epoch=0,
        store=True,
        num_classes=exp["model"]["cfg"]["num_classes"] + 1,
    )
    main_visu.epoch = task_nr

    # New Logger
    model._task_name = task_name
    model._task_count = task_nr

    # Training the model
    trainer.should_stop = False
    fn = os.path.join(exp["name"], "val_res.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as handle:
            val_res = pickle.load(handle)
            model._val_epoch_results = val_res

    if skip:
        # VALIDATION
        trainer.limit_train_batches = 10
        trainer.max_epochs = 1
        trainer.check_val_every_n_epoch = 1

        model.length_train_dataloader = 10000
        model.max_epochs = 10000
        _ = trainer.fit(model=model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)
        trainer.max_epochs = exp["trainer"]["max_epochs"]
        trainer.check_val_every_n_epoch = exp["trainer"]["check_val_every_n_epoch"]
        trainer.limit_val_batches = exp["trainer"]["limit_val_batches"]
        trainer.limit_train_batches = exp["trainer"]["limit_train_batches"]
    else:
        # FULL TRAINING
        if exp["trainer"]["limit_train_batches"] <= 1.0:
            model.length_train_dataloader = len(train_dataloader) * exp["trainer"]["limit_train_batches"]
        else:
            model.length_train_dataloader = exp["trainer"]["limit_train_batches"]
        model.max_epochs = exp["task_specific_early_stopping"]["cfg"]["max_epoch_count"]
        _ = trainer.fit(model=model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)

    checkpoint_callback._last_global_step_saved = -999
    checkpoint_callback.save_checkpoint(trainer, model)

    val_res = model._val_epoch_results
    with open(fn, "wb") as handle:
        pickle.dump(val_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        val_res[-2] = list(range(len(val_res[-2])))
        try:
            validation_acc_plot_stored(main_visu, val_res)
        except:
            print("Valied to generate ACC plot.")
            print("Currently not implemented if not started from task > 1 ?")
            pass

    res = trainer.logger_connector.callback_metrics
    res_store = {}
    for k in res.keys():
        try:
            res_store[k] = float(res[k])
        except:
            pass
    base_path = "/".join([a for a in model_path.split("/") if a.find("rank") == -1])
    with open(f"{base_path}/res{task_nr}.pkl", "wb") as f:
        pickle.dump(res_store, f)

    print(f"FINISHED TRAIN-TASK IDX: {task_nr} TASK NAME : " + task_name)

    if exp["replay"]["cfg_rssb"]["elements"] != 0 and exp["replay"]["cfg_filling"]["strategy"] != "random":
        from torch.utils.data import DataLoader

        test_dataloader = DataLoader(
            train_dataloader.dataset.main_dataset,
            shuffle=False,
            num_workers=train_dataloader.num_workers,
            pin_memory=True,
            batch_size=train_dataloader.batch_size,
            drop_last=False,
        )

        _ = trainer.test(model=model, test_dataloaders=test_dataloader)
        print(f"\n\nTEST IS DONE WITH TASK {task_nr}: ")
        for i in range(task_nr + 1):
            m = min(5, model._rssb.nr_elements)
            logging.debug(f" RSSB STATE {i}: " + str(model._rssb.bins[i, :m]))

    trainer.checkpoint_connector.save_checkpoint(exp["checkpoint_load_2"])

    if exp["replay"]["cfg_rssb"]["elements"] != 0:
        # visualize rssb
        bins, valids = model._rssb.get()
        fill_status = (bins != 0).sum(axis=1)
        main_visu.plot_bar(
            fill_status,
            x_label="Bin",
            y_label="Filled",
            title="Fill Status per Bin",
            sort=False,
            reverse=False,
            tag="Buffer_Fill_Status",
        )
    try:
        validation_acc_plot(main_visu, logger, nr_eval_tasks=len(val_dataloaders))
    except Exception as e:
        rank_zero_warn("FAILED while validation acc plot in train task: " + str(e))

    if task_nr == len(tg):
        try:
            logger.experiment.stop()
        except:
            pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=file_path,
        default="cfg/exp/scannet/exp.yml",
        help="Experiment yaml file.",
    )

    args = parser.parse_args()

    print("Train Task called as MAIN with the following arguments: " + str(args))

    train_task(
        bool(args.init),
        bool(args.close),
        args.exp,
        args.task_nr,
        skip=bool(args.skip),
    )
    torch.cuda.empty_cache()
