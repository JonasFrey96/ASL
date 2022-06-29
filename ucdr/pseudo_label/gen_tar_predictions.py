import os
import sys
import argparse
from pathlib import Path
import time
from multiprocessing import Pool

import imageio
import numpy as np
from PIL import Image
import yaml

import torch
from torchvision import transforms as tf
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ucdr.utils import load_yaml, load_env
from ucdr.visu import Visualizer
from ucdr.pseudo_label import readImage
from ucdr.utils import label_to_png

from ucdr.pseudo_label import FastSCNNHelper
from ucdr.utils import LabelLoaderAuto
from ucdr.kimera_semantics import store_hard_label
from ucdr.datasets import ScanNet


@torch.no_grad()
def label_generation(identifier, scenes, model_cfg, checkpoint_load):
    env = load_env()
    base = os.path.join(env["scannet"], "scans")
    device = "cuda"
    fsh = FastSCNNHelper(device=device, model_cfg=model_cfg, checkpoint_load=checkpoint_load)
    export = os.path.join(env["labels_generic"], identifier)

    os.makedirs(export, exist_ok=True)
    with open(os.path.join(export, f"generation_cfg_{scenes[0]}_{identifier}.yaml"), "w") as stream:
        cfg = {
            "identifier": identifier,
            "scenes": scenes,
            "model_cfg": model_cfg,
            "checkpoint_load": checkpoint_load,
        }
        stream.write(yaml.dump(cfg, sort_keys=False))

    output_transform = tf.Compose([tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset_train = ScanNet(
        root=env["scannet"],
        mode="train",
        scenes=scenes,
        output_trafo=output_transform,
        output_size=(320, 640),
        degrees=0,
        data_augmentation=False,
        flip_p=0,
        jitter_bcsh=[0, 0, 0, 0],
        return_path=True,
    )
    dataset_val = ScanNet(
        root=env["scannet"],
        mode="val",
        scenes=scenes,
        output_trafo=output_transform,
        output_size=(320, 640),
        degrees=0,
        data_augmentation=False,
        flip_p=0,
        jitter_bcsh=[0, 0, 0, 0],
        return_path=True,
    )
    dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_val])
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4, pin_memory=False, batch_size=4)

    for j, batch in enumerate(dataloader):
        print(j, "/", len(dataloader))
        images, label = batch[:2]
        paths = batch[-1]
        images = images.to(device)
        preds = fsh.get_label(images)  # 0-39

        for i in range(preds.shape[0]):
            scene = paths[i].split("/")[-3]
            idx = paths[i].split("/")[-1][:-4]

            os.makedirs(os.path.join(export, scene, identifier), exist_ok=True)
            p = os.path.join(export, scene, identifier, idx + ".png")
            store_hard_label(preds[i] + 1, p)  # 1-40
