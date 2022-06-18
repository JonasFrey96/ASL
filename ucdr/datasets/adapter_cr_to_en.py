# Naming Adapter TaskGenerator to Ensemble dataset
# Everything below is not aware of continual learning
# This sets the replay ratios !

from math import ceil
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from ucdr.datasets import get_dataset
from ucdr.datasets import Ensemble

import os

__all__ = ["adapter_tg_to_dataloader"]


def replay_cfg_to_probs(replay_cfg_ensemble, nr):
    # get the rehearsel probabilieties
    # probs[-1] is probability of current task
    # probs[0] is the rehearsel probabiliety of the task firstly trained on
    if nr == 1:
        return [1.0]

    probs = []
    if replay_cfg_ensemble["active"]:
        m = replay_cfg_ensemble.get("mode", "simple")
        cfg = replay_cfg_ensemble["cfg_" + m]

        if m == "simple":
            # replay each task eactly with the given ratio
            # inference the prob. for the current task
            assert cfg["ratio_per_task"] * (nr - 1) < 1
            probs = [cfg["ratio_per_task"] for i in range(nr - 1)]

        elif m == "fixed_total_replay_ratio":
            # ratio_replay defines the total replay probability
            # each past task is replayed with same prob
            assert cfg["ratio_replay"] < 1 and cfg["ratio_replay"] >= 0
            probs = [cfg["ratio_replay"] / (nr - 1) for i in range(nr - 1)]

        elif m == "focus_task_0":
            assert (cfg["ratio_replay_task_0"] + cfg["ratio_replay_task_1_N"] * (nr - 2)) < 1
            probs = [cfg["ratio_replay_task_0"]]
            probs += [cfg["ratio_replay_task_1_N"]] * (nr - 2)

        elif m == "individual_simple":
            assert sum(cfg["probs"]) < 1
            assert len(cfg["probs"]) >= (nr - 1)
            probs = cfg["probs"][: (nr - 1)]

        elif m == "individual_ratios":
            assert cfg["ratio_replay"] < 1
            assert len(cfg["importance"]) >= (nr - 1)
            imp = cfg["importance"][: (nr - 1)]
            # normalize and weight
            probs = [i / sum(imp) * cfg["ratio_replay"] for i in imp]

        elif m == "adaptive":
            probs = [1 / nr] * (nr - 1)

        else:
            raise ValueError("Not defined mode")

        probs += [1 - sum(probs)]

    else:
        # dont use replay at all
        probs = [0] * nr
        probs[-1] = 1

    # normalize valid probability distribution
    probs = (np.array(probs) / np.array(probs).sum()).tolist()
    return probs


def adapter_tg_to_en(tg, task_nr, replay_cfg_ensemble, env):
    # accumulate train datasets and then wrap them together
    output_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_dataset_list = []
    val_datasets = []
    for idx, task in enumerate(tg):
        if idx < task_nr + 1:
            # add it train_dataset_list
            task_name = task.name

            cfg_train_dataset = task.dataset_train_cfg
            if idx < task_nr:
                cfg_train_dataset["data_augmentation"] = replay_cfg_ensemble.get("replay_augmentation", True)

            train_dataset_list.append(
                get_dataset(
                    **cfg_train_dataset,
                    env=env,
                    output_trafo=output_transform,
                )
            )
        val_datasets.append(
            get_dataset(
                **task.dataset_val_cfg,
                env=env,
                output_trafo=output_transform,
            )
        )

    replay_datasets = train_dataset_list[:-1]

    probs = replay_cfg_to_probs(replay_cfg_ensemble, len(train_dataset_list))
    train_dataset = Ensemble(main_dataset=train_dataset_list[-1], replay_datasets=replay_datasets, probs=probs)

    return train_dataset, val_datasets, task_name


def adapter_tg_to_dataloader(tg, task_nr, loader_cfg, replay_cfg_ensemble, env):
    train_dataset, val_datasets, task_name = adapter_tg_to_en(tg, task_nr, replay_cfg_ensemble, env)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=loader_cfg["shuffle"],
        num_workers=ceil(loader_cfg["num_workers"] / torch.cuda.device_count()),
        pin_memory=loader_cfg["pin_memory"],
        batch_size=loader_cfg["batch_size"],
        drop_last=True,
    )

    val_dataloaders = [
        DataLoader(
            d,
            shuffle=False,
            num_workers=ceil(loader_cfg["num_workers"] / torch.cuda.device_count()),
            pin_memory=loader_cfg["pin_memory"],
            batch_size=loader_cfg["batch_size"],
            drop_last=False,
        )
        for d in val_datasets
    ]

    return train_dataloader, val_dataloaders, task_name


def test():
    import sys
    import os

    sys.path.insert(0, os.getcwd())
    sys.path.append(os.path.join(os.getcwd() + "/src"))

    from ucdr.utils import load_yaml, load_env

    exp = load_yaml(os.path.join(os.getcwd() + "/cfg/test/test.yml"))
    env = load_env()

    from ucdr.task import TaskGeneratorScannet

    tg = TaskGeneratorScannet(mode=exp["task_generator"]["mode"], cfg=exp["task_generator"]["cfg"])

    train, vals, task_name = adapter_tg_to_dataloader(tg, 0, exp["loader"], exp["replay"]["cfg"], env)
    print(tg)
    print(train)
    print(vals)
    print(task_name)

    tg = TaskGeneratorScannet(mode=exp["task_generator"]["mode"], cfg=exp["task_generator"]["cfg"])
    train, vals, task_name = adapter_tg_to_dataloader(tg, 2, exp["loader"], exp["replay"]["cfg"], env)
    print(task_name)

    return True


if __name__ == "__main__":
    test()
