import argparse
import torch
import torch.nn.functional as F
import os
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import pandas as pd
import shutil
from pathlib import Path
import numpy as np

from ucdr import UCDR_ROOT_DIR
from ucdr.utils import SemanticsMeter
from ucdr.utils import load_env
from ucdr.datasets import ScanNet
from ucdr.visu import Visualizer
from ucdr.utils import LabelLoaderAuto


@torch.no_grad()
def eval_pseudo_label(scenes, mode, pseudo_label_idtf):
    # Use ScanNet to get list of labels
    env = load_env()
    only_loss_for_vaild_both = True
    dataset = ScanNet(
        root=env["scannet"],
        mode=mode,
        scenes=scenes,
        output_trafo=None,
        output_size=(320, 640),
        degrees=0,
        data_augmentation=False,
        flip_p=0,
        jitter_bcsh=[0, 0, 0, 0],
        return_path=True,
    )

    sm = SemanticsMeter(40)
    lla = LabelLoaderAuto(root_scannet=env["scannet"])
    paths = [dataset.label_pths[i] for i in dataset.global_to_local_idx]

    lg = env["labels_generic"]

    for j, path in enumerate(paths):
        if j % 10 == 0:
            print(f"{j}/{len(paths)}")

        try:
            label = lla.get(path)[0]
            label = torch.from_numpy(label).type(torch.float32)[None, :, :]

            pa = path.replace(env["scannet"], os.path.join(lg, pseudo_label_idtf))
            pa = pa.replace("label-filt", pseudo_label_idtf)
            pred, _ = lla.get(pa)
            pred = torch.from_numpy(pred).type(torch.float32)[None, :, :]
            img = pred.clone().repeat(3, 1, 1)

            labels = [pred, label]
            img, labels = dataset._augmenter.apply(img, labels, only_crop=True)
            if only_loss_for_vaild_both and (labels[0] == 0).sum() > 0:
                labels[1][labels[0] == 0] = 0
            sm.update(labels[0] - 1, labels[1] - 1)
        except:
            print("Problem with ", path)
    return sm.measure()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default="scene0000,scene0001,scene0002,scene0003,scene0004",
        help="The main experiment yaml file.",
    )
    parser.add_argument(
        "--mode",
        default="val",
        help="The main experiment yaml file.",
    )
    parser.add_argument(
        "--pseudo_label_idtf",
        default="labels_individual_scenes_map_2",
        help="The main experiment yaml file.",
    )
    args = parser.parse_args()

    scenes = args.scene.split(",")

    results = {}
    for scene in scenes:
        print(f"Evaluate scene {scene} with label {args.pseudo_label_idtf}!")
        result = eval_pseudo_label([scene], args.mode, args.pseudo_label_idtf)
        results[args.pseudo_label_idtf + scene] = result

    data = np.zeros((len(results.keys()), 3))
    tags = []
    cols = ["mIoU", "tAcc", "cAcc"]

    for j, tag in enumerate(results.keys()):
        tags.append(tag)
        for i in range(len(cols)):
            data[j, i] = results[tag][i]

    df = pd.DataFrame(data=data, index=tags, columns=cols)
    print(df)
