import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np

try:
  from .helper import AugmentationList
except Exception:  # ImportError
  from helper import AugmentationList
import imageio
from glob import glob
import pickle

__all__ = ["CocoStuff164k"]


class CocoStuffBase(Dataset):
  def __init__(
    self,
    root="/home/jonfrey/Datasets/cocostuff",
    mode="train",
    scenes=[],
    output_trafo=None,
    output_size=400,
    degrees=10,
    flip_p=0.5,
    jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
    nyu40_labels=True,
    data_augmentation=True,
  ):
    """
    Parameters
    ----------
    root : str, path to the COCO folder
    mode : str, option ['train','val]
    """
    super(CocoStuffBase, self).__init__()

    self._output_size = output_size
    if mode == "val":
      mode = "test"
    self._mode = mode
    self._load(root, mode)

    self._augmenter = AugmentationList(output_size, degrees, flip_p, jitter_bcsh)

    self._nyu40_labels = nyu40_labels
    self._output_trafo = output_trafo

    self._data_augmentation = data_augmentation

    self.unique = False
    self.aux_labels = False  # Flag accesed by Ensemble Dataset
    self.aux_labels_fake = False

    with open("cfg/dataset/mappings/coco200_nyu.pkl", "rb") as f:
      self.mapping = pickle.load(f)

  def __getitem__(self, index):
    global_idx = index
    idx = -1

    img = imageio.imread(self.image_pths[global_idx])
    if len(img.shape) == 2:
      img = np.repeat(img[:, :, None], 3, 2)

    img = (
      torch.from_numpy(img).type(torch.float32).permute(2, 0, 1) / 255
    )  # C H W range 0-1
    label = imageio.imread(self.label_pths[global_idx])
    label = torch.from_numpy(label).type(torch.float32)[None, :, :] + 1

    label[label == 256] = 0
    label = [label]

    if self._mode.find("train") != -1 and self._data_augmentation:
      img, label = self._augmenter.apply(img, label)
    else:
      img, label = self._augmenter.apply(img, label, only_crop=True)

    img_ori = img.clone()
    if self._output_trafo is not None:
      img = self._output_trafo(img)

    for k in range(len(label)):
      label[k] = label[k] - 1  # 0 == chairs 39 other prop  -1 invalid

    # map -> coco id -> nyu   COCO:  -1 = invalid  0 = person  NYU40 -1 = invalid 0=wall
    l_o = torch.full(label[0].shape, -1, dtype=label[0].dtype)

    uni = torch.unique(label[0])
    uni = uni[uni != -1]
    for u in uni:
      l_o[label[0] == u] = self.mapping["coco_id_nyu_id"][int(u)]

    label[0] = l_o

    # REJECT LABEL
    if (label[0] != -1).sum() < 10:
      idx = random.randint(0, len(self) - 1)
      if not self.unique:
        return self[idx]
      else:
        return False

    return (img, label[0].type(torch.int64)[0, :, :], img_ori)

  def __len__(self):
    return self._length

  def _load(self, root, mode):
    raise NotImplementedError()


# class CocoStuff10k(CocoStuffBase):
#   def __init__(self, **kwargs):
#     super(CocoStuff10k, self).__init__(**kwargs)

#   def _load(self, root, mode):
#     print(mode)
#     if mode in ["train", "test", "all"]:
#       self.image_pths = os.path.join(root, "imageLists", mode + ".txt")
#       self.image_pths = tuple(open(self.image_pths, "r"))
#       self.image_pths = [id_.rstrip() for id_ in self.image_pths]
#       self.image_pths = [ os.path.join(root, "images", image_id + ".jpg") for image_id in self.image_pths]
#       self.label_pths = [ img_p.replace("/images/","/annotations/" ).replace(".jpg",".mat") for img_p in self.image_pths]
#     else:
#         raise ValueError("Invalid mode name: {mode}")
#     self._length = len(self.image_pths)


class CocoStuff164k(CocoStuffBase):
  def __init__(self, **kwargs):
    super(CocoStuff164k, self).__init__(**kwargs)

  def _load(self, root, mode):
    print(mode)

    if mode in ["train", "test", "all"]:
      short = {"train": "train2017/*.jpg", "test": "val2017/*.jpg", "all": "*.jpg"}
      self.image_pths = sorted(glob(os.path.join(root, "images", short[mode])))
      self.label_pths = [
        img_p.replace("/images/", "/annotations/").replace(".jpg", ".png")
        for img_p in self.image_pths
      ]

    else:
      raise ValueError(f"Invalid mode name: {mode}")
    self.global_to_local_idx = np.arange(len(self.image_pths)).tolist()
    self._length = len(self.image_pths)


def test():
  # pytest -q -s src/datasets/coco.py
  ds = CocoStuff164k(
    root="/home/jonfrey/Datasets/cocostuff",
    mode="train",
    scenes=[],
    output_trafo=None,
    output_size=400,
    degrees=10,
    flip_p=0.5,
    jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
    nyu40_labels=True,
  )
  img, label, img_ori = ds[1100]
  from PIL import Image

  import os
  import sys

  os.chdir(os.path.join(os.getenv("HOME"), "ASL"))
  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + "/src"))
  from visu import Visualizer

  visu = Visualizer(
    os.getenv("HOME") + "/tmp", logger=None, epoch=0, store=True, num_classes=41
  )
  res = visu.plot_detectron(
    img=img_ori,
    label=label.numpy() + 1,
    tag="test",
    jupyter=False,
    store=False,
    alpha=0.6,
  )
  im = Image.fromarray(res)
  im.show()


if __name__ == "__main__":
  test()
