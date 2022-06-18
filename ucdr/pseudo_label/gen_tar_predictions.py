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


class FastDataset(Dataset):
    def __init__(self, paths, root_scannet):
        self.paths = paths
        self.output_transform = tf.Compose([tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.lla = LabelLoaderAuto(root_scannet=root_scannet)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        p = self.paths[index]

        # LOAD IMAGE AND SCALE
        i1 = readImage(p, H=640, W=1280, scale=False)
        h, w, c = i1.shape
        img = torch.nn.functional.interpolate(
            torch.from_numpy(i1.astype(np.float32))[None].permute(0, 3, 1, 2),
            (320, 640),
            mode="bilinear",
        )[0].permute(1, 2, 0)
        img = (img / 255).permute(2, 0, 1)[None]
        img = self.output_transform(img)

        # LOAD LABEL AND SCALE
        label = self.lla.get(p.replace("color", "label-filt").replace("jpg", "png"))[0]
        label = torch.from_numpy(label - 1)

        label = torch.round(
            torch.nn.functional.interpolate(label[None, None].type(torch.float32), (320, 640), mode="nearest")[:, 0]
        )
        label = label.type(torch.int32)

        return img[0], label, p, index


def imwr(a, b, c, d):
    imageio.imwrite(a, b, format=c, compression=d)


@torch.no_grad()
def label_generation(identifier, confidence, scenes, model_cfg, checkpoint_load):
    env = load_env()
    base = os.path.join(env["scannet"], "scans")
    device = "cuda"
    fsh = FastSCNNHelper(device=device, model_cfg=model_cfg, checkpoint_load=checkpoint_load)

    export = os.path.join(env["labels_generic"], identifier)

    with open(os.path.join(export, f"generation_cfg.yaml"), "w") as stream:
        cfg = {
            "identifier": identifier,
            "confidence": confidence,
            "scenes": scenes,
            "model_cfg": model_cfg,
            "checkpoint_load": checkpoint_load,
        }
        stream.write(yaml.dump(cfg, sort_keys=False))

    paths = [str(s) for s in Path(base).rglob("*.jpg") if str(s).find("color") != -1]
    # filter to get evey 10 image

    paths = [s for s in paths if int(s.split("/")[-1][:-4]) % 10 == 0]
    paths.sort(key=lambda x: int(x.split("/")[-3][-7:]) * 10000 + int(x.split("/")[-1][:-4]))
    # filter only scenes of interrest
    pa = []
    for scene in scenes:
        pa += [s for s in paths if s.find(scene) != -1]

    dataset = FastDataset(paths=pa, root_scannet=env["scannet"])
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4, pin_memory=False, batch_size=2)

    h, w, _ = readImage(pa[0], H=640, W=1280, scale=False).shape

    max_cores = 10
    scheduled = 0

    ringbuffer = []

    with Pool(processes=max_cores) as pool:
        for j, batch in enumerate(dataloader):
            print(f"Progress: {j}/{len(dataloader)}")
            st = time.time()
            img, label, path, index = batch
            img = img.to(device)
            index = index.tolist()

            pred, _ = fsh.model(img)
            pred = F.softmax(pred, dim=1)
            pred = torch.nn.functional.interpolate(pred, (h, w), mode="bilinear")
            pred = pred.permute((0, 2, 3, 1))

            ress = []
            for i in index:
                outpath = pa[i]
                outpath = outpath.replace("color", identifier)[:-4] + ".png"
                ress.append(os.path.join(export, outpath[outpath.find("scans/") :]))
                Path(ress[-1]).parent.mkdir(exist_ok=True, parents=True)

            pred = pred
            for i in range(len(ress)):
                label = pred[i]
                path = ress[i]
                max_classes = 40
                assert len(label.shape) == 3
                assert label.shape[2] == max_classes
                H, W, _ = label.shape
                idxs = torch.zeros((3, H, W), dtype=torch.uint8, device=label.device)
                values = torch.zeros((3, H, W), device=label.device)
                label_c = label.clone()
                max_val_10bit = 1023
                for i in range(3):
                    idx = torch.argmax(label_c, dim=2)
                    idxs[i] = idx.type(torch.uint8)

                    m = torch.eye(max_classes)[idx] == 1
                    values[i] = ((label_c[m] * max_val_10bit).reshape(H, W)).type(torch.int32)
                    values[i][values[i] > max_val_10bit] = max_val_10bit
                    label_c[m] = 0

                values = values.type(torch.int32)  # cpu().numpy().astype(np.uint16)
                idxs = idxs.type(torch.int32)  # .cpu().numpy().astype(np.uint8)

                png = torch.zeros((H, W, 4), dtype=torch.int32, device=values.device)
                for i in range(3):
                    png[:, :, i] = values[i]
                    png[:, :, i] = torch.bitwise_or(png[:, :, i], idxs[i] << 10)

                if scheduled > max_cores:
                    res = pool.apply_async(func=imwr, args=(path, png.cpu().numpy().astype(np.uint16), "PNG-FI", 9))
                    ringbuffer.append(res)
                    aro.get()
                    scheduled = 0
                else:
                    _res = pool.apply_async(func=imwr, args=(path, png.cpu().numpy().astype(np.uint16), "PNG-FI", 9))
                    ringbuffer.append(_res)
                    if scheduled == 0:
                        aro = _res
                    scheduled += 1

            if len(ringbuffer) > 25:
                ringbuffer = ringbuffer[1:]

            print("Forward Batch: ", time.time() - st)
        for r in ringbuffer:
            try:
                r.get()
            except:
                pass

    scratchdir = env["labels_generic"]
    os.system(f"cd {scratchdir} && tar -cvf {scratchdir}/{identifier}.tar {identifier}")
