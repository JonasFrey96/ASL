import numpy as np
import torch
from torchvision import transforms as tf
import random
import os
import h5py
import copy
from torch.utils.data import Dataset

try:
  from .helper import AugmentationList
except Exception:
  from helper import AugmentationList

__all__ = ["MLHypersim"]


class MLHypersim(Dataset):
  def __init__(
    self,
    root="/media/scratch2/jonfrey/datasets/mlhypersim/",
    mode="train",
    scenes=[],
    output_trafo=None,
    output_size=(480, 640),
    degrees=10,
    flip_p=0.5,
    jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
    data_augmentation=True,
  ):

    super(MLHypersim, self).__init__()

    self._output_size = output_size
    self._mode = mode

    self._load(root, mode)
    self._filter_scene(scenes)

    self._augmenter = AugmentationList(output_size, degrees, flip_p, jitter_bcsh)

    self._output_trafo = output_trafo
    self._data_augmentation = data_augmentation

    self._scenes_loaded = scenes
    self.unique = False

  def __getitem__(self, index):
    """
    Returns
    -------
    img [torch.tensor]: CxHxW, torch.float
    label [torch.tensor]: HxW, torch.int64
    img_ori [torch.tensor]: CxHxW, torch.float
    replayed [torch.tensor]: 1 torch.float32
    global_idx [int]: global_index in dataset
    """
    replayed = torch.tensor([-1], dtype=torch.int32)
    idx = -1
    if self.replay:
      if self._mode == "train":
        idx, bin = self.idx(self.global_to_local_idx[index])
        if idx != -1:
          global_idx = idx
          replayed[0] = bin
        else:
          global_idx = self.global_to_local_idx[index]
      else:
        global_idx = self.global_to_local_idx[index]
    else:
      global_idx = self.global_to_local_idx[index]

    with h5py.File(self.image_pths[global_idx], "r") as f:
      img = np.array(f["dataset"])
    img[img > 1] = 1
    img = torch.from_numpy(img).type(torch.float32).permute(2, 0, 1)  # C H W
    with h5py.File(self.label_pths[global_idx], "r") as f:
      label = np.array(f["dataset"])
    label = torch.from_numpy(label).type(torch.float32)[
      None, :, :
    ]  # C H W    # label_max = 40 invalid = -1 0 is not used as an index

    if self._mode == "train" and (
      (self._data_augmentation and idx == -1)
      or (self._data_augmentation_for_replay and idx != -1)
    ):

      img, label = self._augmenter.apply(img, label)
    else:
      img, label = self._augmenter.apply(img, label, only_crop=True)

    # check if reject
    if (label != -1).sum() < 10:
      # reject this example
      idx = random.randint(0, len(self) - 1)

      if not self.unique:
        return self[idx]
      else:
        replayed[0] = -999

    img_ori = img.clone()
    if self._output_trafo is not None:
      img = self._output_trafo(img)

    label[label > 0] = label[label > 0] - 1

    # shifts label to -1 = invalid, 0 = wall, 39 = otherprop
    return (
      img,
      label.type(torch.int64)[0, :, :],
      img_ori,
      replayed.type(torch.float32),
      global_idx,
    )

  def __len__(self):
    return self.length

  def __str__(self):
    string = "=" * 90
    string += "\nML-HyperSim Dataset: \n"
    length = len(self)
    string += f"    Total Samples: {length}"
    string += f"  »  Mode: {self._mode} \n"
    string += f"    Replay: {self.replay}"
    string += f"  »  DataAug: {self._data_augmentation}"
    string += f"  »  DataAug Replay: {self._data_augmentation_for_replay}\n"
    string += f"    Replay P: {self.replay_p}"
    string += f"  »  Unique: {self.unique}"
    string += f"  »  Current_bin: {self._current_bin}"
    string += f"  »  Shape: {self._bins.shape}\n"
    filled_b = (self._bins != 0).sum(axis=1)
    filled_v = (self._valids != 0).sum(axis=1)
    string += f"    Bins not 0: {filled_b}"
    string += f"  »  Vals not 0: {filled_v} \n"
    string += "=" * 90
    return string

  def _load(self, root, mode, train_val_split=0.2):
    self.image_pths = np.load("cfg/dataset/mlhypersim/image_pths.npy").tolist()
    self.label_pths = np.load("cfg/dataset/mlhypersim/label_pths.npy").tolist()
    self.scenes = np.load("cfg/dataset/mlhypersim/scenes.npy").tolist()
    self.image_pths = [os.path.join(root, i) for i in self.image_pths]
    self.label_pths = [os.path.join(root, i) for i in self.label_pths]

    self.sceneTypes = list(set(self.scenes))
    self.sceneTypes.sort()

    self.global_to_local_idx = [i for i in range(len(self.image_pths))]

    self.filtered_image_pths = copy.deepcopy(self.image_pths)
    # Scene filtering checked by inspection
    for sce in self.sceneTypes:
      images_in_scene = [i for i in self.filtered_image_pths if i.find(sce) != -1]
      k = int((1 - train_val_split) * len(images_in_scene))
      if mode == "train":
        remove_ls = images_in_scene[k:]
      elif mode == "val":
        remove_ls = images_in_scene[:k]

      idx = self.filtered_image_pths.index(remove_ls[0])
      for i in range(len(remove_ls)):
        del self.filtered_image_pths[idx]

        del self.global_to_local_idx[idx]

    self.length = len(self.global_to_local_idx)

  @staticmethod
  def get_classes():
    scenes = np.load("cfg/dataset/mlhypersim/scenes.npy").tolist()
    sceneTypes = sorted(set(scenes))
    return sceneTypes

  def _filter_scene(self, scenes):
    if len(scenes) != 0:
      images_idx = []
      for sce in scenes:
        images_idx += [
          i
          for i in range(len(self.filtered_image_pths))
          if (self.filtered_image_pths[i]).find(sce) != -1
        ]
      idx = np.array(images_idx)
      self.global_to_local_idx = (np.array(self.global_to_local_idx)[idx]).tolist()
      self.length = len(self.global_to_local_idx)


def test():
  # pytest -q -s src/datasets/ml_hypersim.py
  # Testing

  output_transform = tf.Compose(
    [
      tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
  )
  dataset = MLHypersim(
    mode="train",
    scenes=["ai_001_002", "ai_001_003", "ai_001_004", "ai_001_005", "ai_001_006"],
    output_trafo=output_transform,
    output_size=400,
    degrees=10,
    flip_p=0.5,
    jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
    replay=True,
  )

  dataloader = torch.utils.data.DataLoader(
    dataset, shuffle=False, num_workers=16, pin_memory=False, batch_size=2
  )

  # dataloader.dataset.set_full_state(bins,valids, 2)
  import time

  st = time.time()
  print("Start")
  for j, data in enumerate(dataloader):
    t = data
    print(j, t[0].shape)

  print("Total time", time.time() - st)


if __name__ == "__main__":
  test()
