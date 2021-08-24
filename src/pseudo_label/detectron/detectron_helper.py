import torch
import detectron2

# import some common libraries
import numpy as np
from PIL import Image

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import pickle

__all__ = ["DetectronHelper"]


def map_function(outputs, mappings):
  seg_pan = outputs["panoptic_seg"][0]  # H,W
  sem_seg = torch.argmax(outputs["sem_seg"], dim=0)  # 640, 1280
  seg_seg_nyu = sem_seg.clone()
  seg_seg_nyu[:, :] = -1
  for i in range(53):
    if i == 0:
      m1 = sem_seg == 0
      for instance in outputs["panoptic_seg"][1]:
        ids = instance["id"]
        category_id = instance["category_id"]
        coco200 = COCO_CATEGORIES[instance["category_id"]]["id"]
        nyuid = mappings["coco_id_nyu_id"][coco200 - 1]
        mappings["coco_id_nyu_name"][coco200 - 1]
        m2 = seg_pan == ids
        seg_seg_nyu[m2 * m1] = nyuid
    else:
      coco200 = stuff_ids_coco_200_ids[i - 1]
      nyuid = mappings["coco_id_nyu_id"][coco200 - 1]
      seg_seg_nyu[sem_seg == i] = nyuid
  return seg_seg_nyu


def map_function_pre_argmax(outputs, mappings, stuff_ids_coco_200_ids):
  seg_pan = outputs["panoptic_seg"][0]  # H,W

  C1, H, W = outputs["sem_seg"].shape
  C2 = 41  # NYU
  nyu_probs = torch.zeros(
    (41, H, W), device=outputs["sem_seg"].device, dtype=outputs["sem_seg"].dtype
  )

  sem_seg = torch.argmax(outputs["sem_seg"], dim=0)
  for i in range(1, 53):
    # nyu_name = mappings['coco_id_nyu_name'][ coco200-1  ]
    # print("Mapped ", COCO_CATEGORIES[ instance['category_id'] ]['name'] ," to ", nyu_name )
    # Mapping stuff
    coco200 = stuff_ids_coco_200_ids[i - 1]
    nyuid = mappings["coco_id_nyu_id"][coco200 - 1]
    # +1 Given that 0 is now invalid
    nyu_probs[nyuid + 1, :, :] += outputs["sem_seg"][i, :, :]

    # nyu_name = mappings['coco_id_nyu_name'][ coco200-1  ]
    # for j in COCO_CATEGORIES:
    #   if j['id'] == coco200:
    #     name = j['name']
    #     break
    # print("Mapped ", name ," to ", nyu_name )
  masked_uncertain = torch.nn.functional.softmax(nyu_probs, dim=0).max(dim=0)[0] < 0.5
  nyu_probs[:, masked_uncertain] = 0
  nyu_probs[0, masked_uncertain] = 0.5

  nyu_probs = torch.nn.functional.softmax(nyu_probs, dim=0)

  m1 = sem_seg == 0
  for instance in outputs["panoptic_seg"][1]:
    ids = instance["id"]
    if instance["isthing"]:
      category_id = instance["category_id"]
      score = instance["score"]
      coco200 = COCO_CATEGORIES[instance["category_id"]]["id"]
      nyuid = mappings["coco_id_nyu_id"][coco200 - 1]
      m2 = seg_pan == ids
      # nyu_probs[:, m2 * m1 ] = 0

      # +1 Given that 0 is now invalid
      nyu_probs[nyuid + 1, m2 * m1] += score
      nyu_probs[:, m2 * m1] = torch.nn.functional.softmax(nyu_probs[:, m2 * m1], dim=0)

  return nyu_probs


class DetectronHelper:
  def __init__(self, device):

    cfg = get_cfg()
    cfg.merge_from_file(
      model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
      "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    )

    cfg.merge_from_file(
      model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
      )
    )

    self.model = DefaultPredictor(cfg)
    self.model.model.to(device)
    self.model.model.eval()

    self.device = device

    with open("cfg/dataset/mappings/coco200_nyu.pkl", "rb") as handle:
      self.mappings = pickle.load(handle)
    self.stuff_ids_coco_200_ids = list(
      _get_builtin_metadata("coco_panoptic_separated")[
        "stuff_dataset_id_to_contiguous_id"
      ].keys()
    )

  @torch.no_grad()
  def get_label(self, img):
    outputs = self.model(img)
    seg_seg_nyu = map_function_pre_argmax(
      outputs, self.mappings, self.stuff_ids_coco_200_ids
    )
    return torch.argmax(seg_seg_nyu, 0).cpu().numpy()

  @torch.no_grad()
  def get_label_prob(self, img):
    # H,W,C 0-255 uint8 np
    outputs = self.model(img)
    seg_seg_nyu = map_function_pre_argmax(
      outputs, self.mappings, self.stuff_ids_coco_200_ids
    )
    return torch.nn.functional.softmax(seg_seg_nyu, 0).cpu().numpy()


if __name__ == "__main__":
  import sys
  import os

  sys.path.append(os.path.join(os.getcwd(), "src"))

  from pseudo_label import readImage

  fsh = DetectronHelper(device="cuda:0")
  from pathlib import Path

  paths = [
    str(s)
    for s in Path("/home/jonfrey/Datasets/scannet/scans/").rglob("*.jpg")
    if str(s).find("color") != -1
  ]
  print(paths)

  for j, p in enumerate(paths):

    print(j, "/", len(paths))
    if int(p.split("/")[-1][:-4]) % 10 == 0:
      i1 = readImage(p, H=640, W=1280, scale=False)
      label = fsh.get_label(i1)
      torch.cuda.empty_cache()

      out = p.replace("color", "label_detectron2")
      out = out.replace(".jpg", ".png")

      Path(out).parent.mkdir(exist_ok=True, parents=True)
      Image.fromarray(np.uint8(label)).save(out)
      print(p, out)

  from visu import Visualizer

  visu = Visualizer(
    os.getenv("HOME") + "/tmp", logger=None, epoch=0, store=True, num_classes=41
  )
  visu.plot_detectron(img=i1, label=label, tag="test")
  # label = fsh.get_label( i1 )
  # from visu import Visualizer
  # visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=False, num_classes=41)
  # visu.plot_segmentation(seg=label+1,jupyter=True, method='right')
  # visu.plot_image(img=i1,jupyter=True, method='left')
