import argparse
import torch
import torch.nn.functional as F
import os
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

from ucdr import UCDR_ROOT_DIR
from ucdr.utils import SemanticsMeter
from ucdr.pseudo_label import readImage
from ucdr.datasets import ScanNet
from ucdr.models import FastSCNN
from ucdr.utils import load_yaml, load_env
from ucdr.pseudo_label.fast_scnn import FastDataset
from ucdr.visu import Visualizer
from ucdr.utils import LabelLoaderAuto


@torch.no_grad()
def eval_model(model_cfg, env, scenes, device):
    visu = Visualizer(p_visu=os.path.join(UCDR_ROOT_DIR, "results"), num_classes=40)
    output_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model = FastSCNN(**model_cfg)
    model.to(device)

    p = "/home/jonfrey/git/ASL/results/learning/2021-06-05T14:36:26_scannet25k_24h_lr_decay_from_scratch/task0-epoch=64--step=158340.ckpt"
    k = "iteration_2_individual_replay02_scene_0"

    sd = torch.load(p)
    sd = sd["state_dict"]

    sd = {k[6:]: v for (k, v) in sd.items() if k.find("teacher") == -1}
    sd.pop("bins"), sd.pop("valid")
    model.load_state_dict(sd)
    model.eval()
    dataset = ScanNet(
        root=env["scannet"],
        mode="val",
        scenes=scenes,
        output_trafo=output_transform,
        output_size=(320, 640),
        degrees=0,
        data_augmentation=False,
        flip_p=0,
        jitter_bcsh=[0, 0, 0, 0],
    )
    
    images = [dataset.image_pths[n] for n in dataset.global_to_local_idx]
    ds = FastDataset(images, root_scannet=env["scannet"])
    dataloader = DataLoader(ds, shuffle=False, num_workers=8, pin_memory=False, batch_size=1)
    sm = SemanticsMeter(40)

    lla = LabelLoaderAuto(root_scannet="/media/jonfrey/Fieldtrip1/Jonas_Frey_Master_Thesis/scannet")
    for j, batch in enumerate(dataloader):
        img, label, path, index = batch
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
    
    for j,eva in enumerate(eval_cfg[evals]):
        checkpoint_load = eva.get('checkpoint_load', eval_cfg["global_checkpoint_load"])
        res = eval_model(eval_cfg["model"]["cfg"], env_cfg, scenes=eva["scenes"], device="cuda", checkpoint_load)
        
        res = {
            "miou_valid_class": res[0]
            "total_accuracy": res[1]
            "class_average_accuracy": res[2]
        }
        print(f"{j}_{eva["tag"]} ", res)
        results[f"{j}_{eva["tag"]}"] = res
    
    