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
from ucdr.utils import TorchSemanticsMeter
from ucdr.pseudo_label import readImage
from ucdr.datasets import ScanNet
from ucdr.models import FastSCNN
from ucdr.utils import load_yaml, load_env
from ucdr.datasets import ScanNet
from ucdr.visu import Visualizer
from ucdr.utils import LabelLoaderAuto


@torch.no_grad()
def eval_model(model_cfg, env, scenes, checkpoint, mode, device="cuda"):
    visu = Visualizer(p_visu=os.path.join(UCDR_ROOT_DIR, "results"), num_classes=40)
    output_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model = FastSCNN(**model_cfg)
    model.to(device)

    p = os.path.join(env["base"], checkpoint)

    sd = torch.load(p)
    try:
        # Normal Training Checkpoint
        sd = sd["state_dict"]
        sd = {k[6:]: v for (k, v) in sd.items() if k.find("teacher") == -1}
        try:
            sd.pop("bins"), sd.pop("valid")
        except:
            pass
    except:
        # UDA
        sd = {k.replace("fastscnn.", ""): v for k, v in sd.items() if k.find("aux_decoder") == -1}

    model.load_state_dict(sd)
    model.eval()
    dataset = ScanNet(
        root=env["scannet"],
        mode=mode,
        scenes=scenes,
        output_trafo=output_transform,
        output_size=(320, 640),
        degrees=0,
        data_augmentation=False,
        flip_p=0,
        jitter_bcsh=[0, 0, 0, 0],
        return_path=True,
    )
    dataloader = DataLoader(dataset, shuffle=False, num_workers=8, pin_memory=False, batch_size=8)
    sm = SemanticsMeter(40)
    lla = LabelLoaderAuto(root_scannet=env["scannet"])
    for j, batch in enumerate(dataloader):
        if j % 10 == 0:
            print(f"{j}/{len(dataloader)}")
        img, label, aux_label, aux_vaild, img_ori, path = batch
        img = img.to(device)
        pred, _ = model(img)
        pred = pred.argmax(1).cpu()
        sm.update(pred, label)

    return sm.measure()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        default="eval.yaml",
        help="The main experiment yaml file.",
    )

    args = parser.parse_args()

    eval_cfg_path = args.eval
    if not os.path.isabs(eval_cfg_path):
        eval_cfg_path = os.path.join(UCDR_ROOT_DIR, "cfg/eval", args.eval)

    env_cfg = load_env()
    eval_cfg = load_yaml(eval_cfg_path)

    results = {}

    for j, eva in enumerate(eval_cfg["evals"]):
        print(eva)
        checkpoint_load = eva.get("checkpoint_load", eval_cfg.get("global_checkpoint_load", "nan"))
        if not os.path.isabs(checkpoint_load):
            checkpoint_load = os.path.join(UCDR_ROOT_DIR, checkpoint_load)

        mode = eva.get("mode", "val")
        res = eval_model(
            eval_cfg["model"]["cfg"],
            env_cfg,
            scenes=eva["scenes"],
            checkpoint=checkpoint_load,
            mode=mode,
            device="cuda",
        )

        res = {"miou_valid_class": res[0], "total_accuracy": res[1], "class_average_accuracy": res[2]}
        tag = eva["tag"]
        print(f"{tag}", res)
        assert not (tag in list(results.keys()))
        results[f"{tag}"] = res

    data = np.zeros((len(results.keys()), 3))
    tags = []
    cols = ["mIoU", "tAcc", "cAcc"]
    cols_idx = ["miou_valid_class", "total_accuracy", "class_average_accuracy"]

    for j, tag in enumerate(results.keys()):
        tags.append(tag)
        for i, c in enumerate(cols_idx):
            data[j, i] = results[tag][c]

    df = pd.DataFrame(data=data, index=tags, columns=cols)
    p = os.path.join(Path(env_cfg["base"]).parent, "evals", eval_cfg["name"])
    os.makedirs(p, exist_ok=True)
    df.to_pickle(os.path.join(p, "df_results.pkl"))
    shutil.copy(eval_cfg_path, p)

    print(df)
