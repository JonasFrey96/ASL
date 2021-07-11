import os
from re import A
import sys

os.chdir(os.path.join(os.getenv("HOME"), "ASL"))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + "/src"))

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

from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.loggers.neptune import NeptuneLogger

# Costume Modules
from lightning import Network
from visu import (
  MainVisualizer,
  validation_acc_plot,
  validation_acc_plot_stored,
)
from callbacks import (
  TaskSpecificEarlyStopping,
  VisuCallback,
  FreezeCallback,
  ReplayCallback,
)
from utils_asl import load_yaml, file_path
from utils_asl import get_neptune_logger, get_tensorboard_logger
from datasets_asl import adapter_tg_to_dataloader
from task import get_task_generator

__all__ = ["train_task"]


def train_task(
  init, close, exp_cfg_path, env_cfg_path, task_nr, skip=False, logger_pass=None
):
  # skip flag indicates to not perform full fit but just validate the dataset

  seed_everything(42)
  local_rank = int(os.environ.get("LOCAL_RANK", 0))

  # LOADING THE CONFIGURATION
  if local_rank != 0 or not init:
    logging.debug(
      "TASK INIT FLAG: " + str(init) + "TASK LOCAL RANK: " + str(local_rank)
    )
    rm = exp_cfg_path.find("cfg/exp/") + len("cfg/exp/")
    exp_cfg_path = os.path.join(exp_cfg_path[:rm], "tmp/", exp_cfg_path[rm:])

  exp = load_yaml(exp_cfg_path)
  env = load_yaml(env_cfg_path)

  # CREATE EXPERIMENTS FOLDER + MOVE THE CONFIG FILES + STORE TMP FILE
  if local_rank == 0 and init:
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
    env_cfg_fn = os.path.split(env_cfg_path)[-1]
    print(f"Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}")
    shutil.copy(exp_cfg_path, f"{model_path}/{exp_cfg_fn}")
    shutil.copy(env_cfg_path, f"{model_path}/{env_cfg_fn}")
    exp["name"] = model_path
  else:
    # the correct model path has already been written to the yaml file.
    model_path = os.path.join(exp["name"], f"rank_{local_rank}_{task_nr}")
    # Create the directory
    Path(model_path).mkdir(parents=True, exist_ok=True)

  if not init:
    # Overwrite checkpoint and restore config !
    exp["checkpoint_load"] = exp["checkpoint_load_2"]
    if exp["lr_scheduler"]["active"]:
      exp["checkpoint_restore"] = False
      exp["weights_restore"] = True
    else:
      exp["checkpoint_restore"] = True
      exp["weights_restore"] = False
    # GET LOGGER
  if not exp.get("offline_mode", False):
    if logger_pass is None and exp.get("experiment_id", None) is None:
      logger = get_neptune_logger(
        exp=exp, env=env, exp_p=exp_cfg_path, env_p=env_cfg_path
      )
      exp["experiment_id"] = logger.experiment.id
      print("Created Experiment ID: " + str(exp["experiment_id"]))
    else:
      from log import _create_or_get_experiment2

      if not env["workstation"]:
        NeptuneLogger._create_or_get_experiment = _create_or_get_experiment2
      logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name=exp["neptune_project_name"],
        experiment_id=exp["experiment_id"],
        close_after_fit=False,
        upload_stdout=True,
        upload_stderr=True,
      )
    print(
      "Neptune Experiment ID: " + str(logger.experiment.id) + " TASK NR " + str(task_nr)
    )
  else:
    logger = get_tensorboard_logger(
      exp=exp, env=env, exp_p=exp_cfg_path, env_p=env_cfg_path
    )

  if local_rank == 0 and init:
    # Store checkpoint config and 'experiment_id'
    exp["weights_restore_2"] = False
    exp["checkpoint_restore_2"] = True
    exp["checkpoint_load_2"] = os.path.join(model_path, "last.ckpt")
    rm = exp_cfg_path.find("cfg/exp/") + len("cfg/exp/")
    exp_cfg_path = os.path.join(exp_cfg_path[:rm], "tmp/", exp_cfg_path[rm:])
    Path(exp_cfg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(exp_cfg_path, "w+") as f:
      yaml.dump(exp, f, default_flow_style=False, sort_keys=False)

  # COPY DATASET
  if not env["workstation"]:
    # use proxy hack for neptunai !!!
    # move data to ssd
    if exp["move_datasets"][0]["env_var"] != "none":
      for dataset in exp["move_datasets"]:
        scratchdir = os.getenv("TMPDIR")
        logging.debug(
          "Dataset: " + str(dataset) + " moved to TMPDIR directory: " + str(scratchdir)
        )
        env_var = dataset["env_var"]

        if not (env_var in env):
          if env_var.find("labels") != -1:
            bp = env["labels_generic"]
          else:
            assert Exception("Missing Environment Variable")
        else:
          bp = env[env_var]

        tar = os.path.join(bp, f"{env_var}.tar")
        name = (tar.split("/")[-1]).split(".")[0]
        # TODO: JONAS FREY is now not working for the labels in interactive session !!

        if dataset["env_var"] == "scannet_frames_25k":
          check_path = os.path.join(scratchdir, "scannet", "scannet_frames_25k")
        elif dataset["env_var"].find("labels") != -1:
          check_path = os.path.join(
            scratchdir, "scannet", "scans", "scene0000_00", dataset["env_var"]
          )
        else:
          check_path = os.path.join(scratchdir, dataset["env_var"])

        if not os.path.exists(check_path):
          try:
            if tar.find("label") != -1:
              target = "$TMPDIR/scannet/"
            elif tar.find("scannet_frames_25k") != -1:
              target = "$TMPDIR/scannet/"
            else:
              target = "$TMPDIR"

            cmd = f"tar -xvf {tar} -C {target} >/dev/null 2>&1"
            st = time.time()
            logging.debug(f"{dataset}: Start moveing dataset-{env_var}: {cmd}")

            os.system(cmd)

            env[env_var] = str(os.path.join(scratchdir, name))
            new_env_var = env[env_var]
            print(
              f"{dataset}: Finished moveing dataset-{new_env_var} in {time.time()-st}s"
            )

          except:
            rank_zero_warn("ENV Var" + env_var)
            env[env_var] = str(os.path.join(scratchdir, name))
            rank_zero_warn("Copying data failed")
        else:
          env[env_var] = str(os.path.join(scratchdir, name))
    else:
      env["mlhypersim"] = str(os.path.join(env["mlhypersim"], "mlhypersim"))

  # SET GPUS
  if (exp["trainer"]).get("gpus", -1) == -1:
    nr = torch.cuda.device_count()
    logging.debug(f"Set GPU Count for Trainer to {nr}!")
    for i in range(nr):
      logging.debug(f"Device {i}: " + str(torch.cuda.get_device_name(i)))
    exp["trainer"]["gpus"] = -1

  # TASK GENERATOR
  tg = get_task_generator(
    name=exp["task_generator"].get("name", "scannet"),  # select correct TaskGenerator
    mode=exp["task_generator"]["mode"],  # mode for TaskGenerator
    cfg=exp["task_generator"]["cfg"],
  )  # cfg for TaskGenerator

  print(str(tg))

  if exp["replay"]["cfg_rssb"]["bins"] == -1:
    exp["replay"]["cfg_rssb"]["bins"] = len(tg)

  if task_nr >= len(tg):
    logging.error("Specified supervisor stop_task is too high")
    return

  # MODEL
  model = Network(exp=exp, env=env)

  # COLLECT CALLBACKS
  lr_monitor = LearningRateMonitor(**exp["lr_monitor"]["cfg"])
  if exp["cb_early_stopping"]["active"]:
    early_stop_callback = EarlyStopping(**exp["cb_early_stopping"]["cfg"])
    cb_ls = [early_stop_callback, lr_monitor]
  else:
    cb_ls = [lr_monitor]
  if exp["task_specific_early_stopping"]["active"]:
    tses = TaskSpecificEarlyStopping(
      nr_tasks=len(tg), **exp["task_specific_early_stopping"]["cfg"]
    )
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
  cb_ls.append(FreezeCallback(**exp["model"]["freeze"]))

  # CHECKPOINT
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
    trainer = Trainer(
      **exp["trainer"], default_root_dir=model_path, callbacks=cb_ls, logger=logger
    )

  if exp["weights_restore"]:
    # it is not strict since the latent replay buffer is not always available
    p = os.path.join(env["base"], exp["checkpoint_load"])
    if os.path.isfile(p):
      state_dict_loaded = torch.load(p, map_location=lambda storage, loc: storage)[
        "state_dict"
      ]
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

  # What we can do now here is reinitalizing the datasets
  train_dataloader, val_dataloaders, task_name = adapter_tg_to_dataloader(
    tg, task_nr, exp["loader"], exp["replay"]["cfg_ensemble"], env
  )

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
  # ONE_CYCLE_LR
  # requires you to set model.length_train_dataloader
  #                     model.max_epochs
  # not accessible in config optimiuzer
  if skip:
    # VALIDATION
    trainer.limit_train_batches = 11
    trainer.max_epochs = 1
    trainer.check_val_every_n_epoch = 1
    model.length_train_dataloader = 11
    model.max_epochs = 1
    _ = trainer.fit(
      model=model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders
    )
    trainer.max_epochs = exp["trainer"]["max_epochs"]
    trainer.check_val_every_n_epoch = exp["trainer"]["check_val_every_n_epoch"]
    trainer.limit_val_batches = exp["trainer"]["limit_val_batches"]
    trainer.limit_train_batches = exp["trainer"]["limit_train_batches"]
  else:
    # FULL TRAINING
    if exp["trainer"]["limit_train_batches"] <= 1.0:
      model.length_train_dataloader = (
        len(train_dataloader) * exp["trainer"]["limit_train_batches"]
      )
    else:
      model.length_train_dataloader = exp["trainer"]["limit_train_batches"]
    model.max_epochs = exp["task_specific_early_stopping"]["cfg"]["max_epoch_count"]
    _ = trainer.fit(
      model=model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders
    )

  checkpoint_callback._last_global_step_saved = -999
  checkpoint_callback.save_checkpoint(trainer, model)

  val_res = model._val_epoch_results
  with open(fn, "wb") as handle:
    pickle.dump(val_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    validation_acc_plot_stored(main_visu, val_res)

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

  if (
    exp["replay"]["cfg_rssb"]["elements"] != 0
    and exp["replay"]["cfg_filling"]["strategy"] != "random"
  ):
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
    help="The main experiment yaml file.",
  )
  parser.add_argument("--task_nr", type=int, default=0, help="Task nr.")
  parser.add_argument("--init", type=int, default=1, help="Task nr.")
  parser.add_argument("--close", type=int, default=1, help="Task nr.")
  parser.add_argument("--skip", type=int, default=0, help="Task nr.")

  args = parser.parse_args()

  print("Train Task called as MAIN with the following arguments: " + str(args))
  env_cfg_path = os.path.join("cfg/env", os.environ["ENV_WORKSTATION_NAME"] + ".yml")

  train_task(
    bool(args.init),
    bool(args.close),
    args.exp,
    env_cfg_path,
    args.task_nr,
    skip=bool(args.skip),
  )
  torch.cuda.empty_cache()
