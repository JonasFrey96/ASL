from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pathlib import Path
import torch
import logging
from neptune.new.types import File

from ucdr.utils import flatten_dict

__all__ = ["get_neptune_logger", "get_tensorboard_logger"]


def log_important_params(exp):
    dic = {}
    dic = flatten_dict(exp)
    return dic


def get_neptune_logger(exp, env, exp_p):
    project_name = exp["neptune_project_name"]
    params = log_important_params(exp)
    cwd = os.getcwd()
    files = [str(p).replace(cwd + "/", "") for p in Path(cwd).rglob("*.py") if str(p).find("vscode") == -1]
    files.append(exp_p)

    t1 = str(os.environ["ENV_WORKSTATION_NAME"])

    gpus = "gpus_" + str(torch.cuda.device_count())
    if os.environ["ENV_WORKSTATION_NAME"] == "euler":
        proxies = {"http": "http://proxy.ethz.ch:3128", "https": "http://proxy.ethz.ch:3128"}
    else:
        proxies = None

    logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project=project_name,
        name=exp["name"],
        tags=[t1, exp["name"].split("/")[-2], exp["name"].split("/")[-1], gpus] + exp["tag_list"],
        # upload_source_files=files,
        # upload_stdout=True,
        # upload_stderr=True,
        proxies=proxies,
    )
    for k in params:
        logger.experiment["params/" + k] = params[k]
    logger.experiment["source_files"].upload_files(files)
    return logger


def get_tensorboard_logger(exp, env, exp_p):
    params = log_important_params(exp)
    cwd = os.getcwd()
    files = [str(p).replace(cwd + "/", "") for p in Path(cwd).rglob("*.py") if str(p).find("vscode") == -1]
    files.append(exp_p)
    gpus = "gpus_" + str(torch.cuda.device_count())

    logging.debug("Use Tensorboard Logger with exp['name]: " + exp["name"])
    return TensorBoardLogger(save_dir=exp["name"], name="tensorboard", default_hp_metric=params)
