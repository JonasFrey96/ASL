import os
import yaml
import torch
import imageio
import pandas
import numpy as np

from ucdr import UCDR_ROOT_DIR

__all__ = ["file_path", "load_yaml", "load_env", "load_label_scannet", "load_mapping_scannet"]


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def load_yaml(path):
    with open(path) as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
    return res


def load_env():
    env_cfg_path = os.path.join(UCDR_ROOT_DIR, "cfg/env", os.environ["ENV_WORKSTATION_NAME"] + ".yml")
    env = load_yaml(env_cfg_path)
    for k in env.keys():
        if k == "workstation":
            continue
        if not os.path.isabs(env[k]):
            env[k] = os.path.join(UCDR_ROOT_DIR, env[k])

    return env


def load_label_scannet(p, mapping_scannet):
    label_gt = imageio.imread(p)
    label_gt = torch.from_numpy(label_gt.astype(np.int32)).type(torch.float32)[:, :]  # H W
    sa = label_gt.shape
    label_gt = label_gt.flatten()

    label_gt = mapping_scannet[label_gt.type(torch.int64)]
    label_gt = label_gt.reshape(sa)  # 1 == chairs 40 other prop  0 invalid

    return label_gt.numpy()


def load_mapping_scannet(p):
    df = pandas.read_csv(p, sep="\t")
    mapping_source = np.array(df["id"])
    mapping_target = np.array(df["nyu40id"])
    mapping_scannet = torch.zeros((int(mapping_source.max() + 1)), dtype=torch.int64)
    for so, ta in zip(mapping_source, mapping_target):
        mapping_scannet[so] = ta
    return mapping_scannet
