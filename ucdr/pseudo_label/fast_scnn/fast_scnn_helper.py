if __name__ == "__main__":
    import os
    import sys

    os.chdir(os.path.join(os.getenv("HOME"), "ASL"))
    sys.path.insert(0, os.path.join(os.getenv("HOME"), "ASL"))
    sys.path.append(os.path.join(os.path.join(os.getenv("HOME"), "ASL") + "/src"))

import os
import sys
from torchvision import transforms as tf
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from ucdr.utils import file_path, load_yaml, load_env
from ucdr.models import FastSCNN

__all__ = ["FastSCNNHelper"]


class FastSCNNHelper:
    def __init__(self, device, eval_cfg_path="cfg/eval/eval.yml"):
        env_cfg = load_env()
        eval_cfg = load_yaml(eval_cfg_path)
        self.device = device
        self.model = FastSCNN(**eval_cfg["model"]["cfg"])
        p = os.path.join(env_cfg["base"], eval_cfg["checkpoint_load"])
        if os.path.isfile(p):
            res = torch.load(p, map_location=lambda storage, loc: storage)
            new_statedict = {}
            for k in res["state_dict"].keys():
                if k.find("model.") != -1:
                    new_statedict[k[6:]] = res["state_dict"][k]
            res = self.model.load_state_dict(new_statedict, strict=True)
            print("Restoring weights: " + str(res))
        else:
            raise Exception("Checkpoint not a file")
        del res
        torch.cuda.empty_cache()
        self.model.to(device)
        self.model.eval()

        self.output_transform = tf.Compose([tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def get_label(self, img):
        with torch.no_grad():
            # H,W,C 0-255 uint8 np
            img = (torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)[None]).to(self.device)
            img = self.output_transform(img)
            outputs = self.model(img)
            return torch.argmax(outputs[0], 1).cpu().numpy()[0]

    def get_label_prob(self, img):
        with torch.no_grad():
            # H,W,C 0-255 uint8 np
            img = (torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)[None]).to(self.device)
            img = self.output_transform(img)
            outputs = self.model(img)

            pred = F.softmax(outputs[0], dim=1)[0]
            label = torch.zeros((41, pred.shape[1], pred.shape[2]), device=pred.device, dtype=pred.dtype)
            label[1:] = pred
            return label.cpu().numpy()  # 41,H,W


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.getcwd(), "src"))
    from PIL import Image
    import argparse
    from ucdr.visu import Visualizer
    from ucdr.pseudo_label import readImage
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument("--store_base", type=str, default="/media/scratch1/jonfrey/labels_generated", help="")
    parser.add_argument("--identifier", type=str, default="pretrain_scene_10-60_24h", help="")
    parser.add_argument("--confidence", type=float, default=0, help="")

    arg = parser.parse_args()
    idfs = str(arg.identifier)
    confidence = arg.confidence
    store_base = str(arg.store_base)

    fsh = FastSCNNHelper(device="cuda:0")
    paths = [str(s) for s in Path("/home/jonfrey/datasets/scannet/scans/").rglob("*.jpg") if str(s).find("color") != -1]
    paths = [s for s in paths if int(s.split("/")[-1][:-4]) % 10 == 0]

    for j, p in enumerate(paths):

        i1 = readImage(p, H=640, W=1280, scale=False)
        h, w, c = i1.shape
        i1 = torch.nn.functional.interpolate(
            torch.from_numpy(i1.astype(np.float32))[None].permute(0, 3, 1, 2), (320, 640), mode="bilinear"
        )[0].permute(1, 2, 0)

        label_probs = fsh.get_label_prob(i1.numpy())
        label_probs = torch.nn.functional.interpolate(
            torch.from_numpy(label_probs.astype(np.float32))[None], (h, w), mode="bilinear"
        )[0].numpy()

        label = np.uint8(np.argmax(label_probs, axis=0))

        mask = np.max(label_probs, axis=0) < confidence
        label[mask] = 0  # set labels with little confidence to 0

        dirs = p.split("/")
        scene = dirs[-3]
        frame = dirs[-1][:-4]

        out = "/".join([store_base, idfs, scene, idfs, frame + ".png"])
        Path(out).parent.mkdir(exist_ok=True, parents=True)

        print(j, "/", len(paths), " P: ", out, " Ratio: ", 1 - (mask.sum() / label.size))
        Image.fromarray(label).save(out)

    os.system(f"cd {store_base} && tar -cvf {store_base}/{idfs}.tar {idfs}")
