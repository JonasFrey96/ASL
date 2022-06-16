import argparse
import torch
import torch.nn.functional as F
import os
from torchvision import transforms
import torch


from ucdr import UCDR_ROOT_DIR
from ucdr.utils import SemanticsMeter
from torch.utils.data import DataLoader
from ucdr.pseudo_label import readImage
from ucdr.datasets import ScanNet
from ucdr.models import FastSCNN
from ucdr.utils import load_yaml
from ucdr.pseudo_label.fast_scnn import FastDataset
from ucdr. visu import Visualizer
from ucdr.utils import LabelLoaderAuto

@torch.no_grad()
def eval_model(eval, env, scenes, device):
    visu = Visualizer(p_visu=os.path.join(UCDR_ROOT_DIR,"results"), num_classes=40)
    
    output_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model = FastSCNN(**eval["model"]["cfg"])
    model.to(device)
    p = "/home/jonfrey/git/model-uncertainty-for-adaptation/debug/pretrained.pt"
    
    p = "/home/jonfrey/Results/paper/paper/2021-09-13T20:35:02_s0_iteration2_r02/last.ckpt"
    p = "/home/jonfrey/Results/paper2/paper2/2021-09-14T09:20:24_s0_iteration2_r02/last.ckpt"
    k = "iteration_2_individual_replay02_scene_0"
    if p[-3:] == ".pt":
        sd = torch.load(p)

    else:
        sd = torch.load(p)
        sd = sd["state_dict"]
    
    sd = {k[6:]: v for (k, v) in sd.items() if k.find("teacher") == -1}
    sd.pop("bins"), sd.pop("valid")   
    model.load_state_dict(sd)
    model.eval()

    dataset = ScanNet(
        root="/home/jonfrey/Datasets/scannet",
        mode="val_strict",
        scenes=scenes,
        output_trafo=output_transform,
        output_size=(320, 640),
        degrees=0,
        data_augmentation=False,
        flip_p=0,
        jitter_bcsh=[0, 0, 0, 0],
    )
    images = [dataset.image_pths[n] for n in dataset.global_to_local_idx]

    ds = FastDataset(images)
    dataloader = DataLoader(ds, shuffle=False, num_workers=8, pin_memory=False, batch_size=1)
    sm = SemanticsMeter(40)
    sm_stored = SemanticsMeter(40)
    
    # readImage
    H = 968
    W = 1296

    lla = LabelLoaderAuto(root_scannet="/media/jonfrey/Fieldtrip1/Jonas_Frey_Master_Thesis/scannet")


    # crop = transforms.CenterCrop((H,W))
    for j, batch in enumerate(dataloader):
        img, label, path = batch
        img = img.to(device)

        pred, _ = model(img)
        
        pred_s = pred.clone().argmax(1).cpu()
        label_s = torch.round(torch.nn.functional.interpolate(label[:,None].type(torch.float32), (320, 640), mode="nearest")[:,0])
        
        # visu.plot_detectron(img[0], pred_s[0]+1, tag=f"pred_{j}", store=True, text_off=True)
        # visu.plot_detectron(img[0], label_s[0]+1, tag=f"gt_{j}", store=True, text_off=True)
        
        sm.update(pred_s, label_s)
        p = path[0]
        
        p = p.replace("/home/jonfrey/Datasets/scannet/scans", 
                      f"/media/jonfrey/Fieldtrip1/Jonas_Frey_Master_Thesis/labels_generated/{k}/scans")
        p = p.replace("color", k)
        p = p.replace(".jpg", ".png")
        pred_loaded = torch.from_numpy( lla.get(p)[0] )
        pred_loaded_s = torch.round(torch.nn.functional.interpolate(pred_loaded[None,None].type(torch.float32), (320, 640), mode="nearest")[:,0])
        sm_stored.update( pred_loaded_s-1,label_s)
        print(j, "/", len(dataloader))
    
    print("320 Res", sm.measure(), "sm_stored", sm_stored.measure())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        default="nan",
        help="The main experiment yaml file.",
    )

    args = parser.parse_args()

    if args.eval != "nan":
        eval_cfg_path = args.eval
    else:
        eval_cfg_path = os.path.join(UCDR_ROOT_DIR, "ucdr/eval/eval.yaml")

    env_cfg_path = os.path.join("cfg/env", os.environ["ENV_WORKSTATION_NAME"] + ".yml")
    env_cfg = load_yaml(env_cfg_path)
    eval_cfg = load_yaml(eval_cfg_path)
    eval_model(eval_cfg, env_cfg, scenes=["scene0000"], device="cuda")