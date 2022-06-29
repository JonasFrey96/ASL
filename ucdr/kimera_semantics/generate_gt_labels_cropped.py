from pathlib import Path
import os
import imageio
import numpy as np
import torch


from ucdr import UCDR_ROOT_DIR
from ucdr.utils import LabelLoaderAuto
from ucdr.datasets import AugmentationList
from ucdr.kimera_semantics import store_hard_label, store_soft_label

p = "/media/Data/Datasets/scannet/scans"
lla = LabelLoaderAuto("/media/Data/Datasets/scannet")

images = [str(s) for s in Path(p).rglob("*.png") if str(s).find("label-filt") != -1]

key = "gt_new_format"
outdir = os.path.join(UCDR_ROOT_DIR, "results", "labels_generated", key, "scans")
os.makedirs(outdir, exist_ok=True)


augmenter = AugmentationList((320, 640), 0, 0, [0, 0, 0, 0])

for p in images:
    label, _ = lla.get(p)
    scene = p.split("/")[-3]
    store_p = os.path.join(outdir, scene, key, p.split("/")[-1])
    os.makedirs(os.path.join(outdir, scene, key), exist_ok=True)
    img = torch.zeros((3, label.shape[0], label.shape[1]))
    label = torch.from_numpy(label).type(torch.float32)
    img, label = augmenter.apply(img, [label[None]], only_crop=True)
    label = label[0].numpy()[0]
    store_hard_label(label, store_p)
