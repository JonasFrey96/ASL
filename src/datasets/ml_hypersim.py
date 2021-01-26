import torch.utils.data as data
import numpy as np
import torch
import pandas as pd
from torchvision import transforms as tf
from torchvision.transforms import functional as F
import PIL
import random
import scipy
import os
from pathlib import Path
from PIL import Image
import h5py
import time
try:
    from .helper import Augmentation
except Exception:  # ImportError
    from helper import Augmentation

import multiprocessing

__all__ = ['MLHypersim']


"""
Everyting might work out of the box Wait



This was a bad idea for the follwing reasons:
1. The replaybuffer will be destroyed when a new Dataset is initalized
(This could be avoided by not fully reinit the dataset but just updateing the lists)
2. The replay buffer will be filled differently for each ddp.

3. Better option would be to write a costume ddp-sampler which includes the replay buffer
4. Is this re-initit also ?
"""


class ReplayDataset(data.Dataset):
    def __init__(self, bins, elements, add_p=0.5, replay_p=0.5, current_bin=0):
        self._bins = [
            multiprocessing.Array(
                'i', (elements)) for i in range(bins)]
        self._valid = [
            multiprocessing.Array(
                'b', (elements)) for i in range(bins)]

        self.b = np.zeros(elements)
        self.v = np.zeros(elements)

        self._current_bin = current_bin
        self._replay_p = replay_p
        self._add_p = add_p

    def idx(self, index):

        self.v[:] = self._valid[0][:]

        if ((self.v).sum() != 0 and
                random.random() < self._replay_p):

            index = self.get_element()

        elif random.random() < self._add_p:
            self.add_element(index)
        return index

    def get_replay_state(self):
        el = self._bins[0][0]
        string = "ReplayDataset contains elements "
        return string

    def set_current_bin(self, bin):
        # might not be necessarry since we create the dataset with the correct
        # bin set
        if bin < self._bins.shape[0]:
            self._current_bin = bin
        else:
            raise Exception(
                "Invalid bin selected. Bin must be element 0-" +
                self._bins.shape[0])

    def get_element(self):
        if self._current_bin > 0:
            if self._current_bin > 1:
                b = int(np.random.randint(0, self._current_bin - 1, (1,)))
            else:
                b = 0

        self.v = self._valid[b]
        indi = np.nonzero(self.v)

        sel_ele = np.random.randint(0, indi.shape[0], (1,))

        return self._bins[b, int(indi[sel_ele, 1])]

    def add_element(self, index):

        self.b[:] = self._bins[self._current_bin][:]
        self.v[:] = self._valid[self._current_bin][:]

        if (index == self.b).sum() == 0:
            # not in buffer
            if self.v.sum() != self.v.shape[0]:
                # free space simply add
                indi = np.nonzero(self.v == 0)
                sel_ele = np.random.randint(0, indi.shape[0], (1,))
                sel_ele = int(indi[sel_ele])

                self._bins[self._current_bin][sel_ele] = index
                self._valid[self._current_bin][sel_ele] = True

            else:
                # replace
                sel_ele = np.random.randint(0, self.b.shape[0], (1,))
                self._bins[self._current_bin][sel_ele] = index
            return True

        return False


class MLHypersim(ReplayDataset):
    def __init__(
            self,
            root='/media/scratch2/jonfrey/datasets/mlhypersim/',
            mode='train',
            scenes=[],
            output_trafo=None,
            output_size=400,
            degrees=10,
            flip_p=0.5,
            jitter_bcsh=[
                0.3,
                0.3,
                0.3,
                0.05],
            current_bin=0):
        """
        Each dataloader loads the full .mat file into memory.
        For the small dataset size this is perfect.
        Both should work when file is located on SSD!

        Parameters
        ----------
        root : str, path to the ML-Hypersim folder
        mode : str, option ['train','val]
        """
        # super.__init__( )
        super(
            MLHypersim,
            self).__init__(
            5,
            1000,
            add_p=0.5,
            replay_p=0.5,
            current_bin=current_bin)

        self._output_size = output_size
        self._mode = mode

        self._load(root, mode)
        self._filter_scene(scenes)

        self._augmenter = Augmentation(output_size,
                                       degrees,
                                       flip_p,
                                       jitter_bcsh)

        self._output_trafo = output_trafo

        print(torch.utils.data.get_worker_info())
        # full training dataset with all objects
        # TODO
        #self._weights = pd.read_csv(f'cfg/dataset/ml-hypersim/test_dataset_pixelwise_weights.csv').to_numpy()[:,0]

    def __getitem__(self, index):
        index = self.idx(index)
        print(self.get_replay_state())
        print(torch.utils.data.get_worker_info())

        with h5py.File(self.image_pths[index], 'r') as f:
            img = np.array(f['dataset'])
        img[img > 1] = 1
        img = torch.from_numpy(img).type(
            torch.float32).permute(
            2, 0, 1)  # C H W
        with h5py.File(self.label_pths[index], 'r') as f:
            label = np.array(f['dataset'])
        label = torch.from_numpy(label).type(
            torch.float32)[None, :, :]  # C H W

        if self._mode == 'train':
            img, label = self._augmenter.apply(img, label)
        elif self._mode == 'val' or self._mode == 'test':
            img, label = self._augmenter.apply(img, label, only_crop=True)
        else:
            raise Exception('Invalid Dataset Mode')

        # check if reject
        if (label != -1).sum() < 10:
            # reject this example
            idx = random.randint(0, len(self) - 1)
            return self[idx]

        img_ori = img.clone()
        if self._output_trafo is not None:
            img = self._output_trafo(img)

        label[label > 0] = label[label > 0] - 1
        return img, label.type(torch.int64)[0, :, :], img_ori

    def __len__(self):
        return self.length

    def __str__(self):
        string = "HyperSim Dataset: \n"
        string += f"   Mode: {self._mode}"
        string += "Sequences active: \n"
        string += "Total Examples: \n"
        l = len(self.sceneTypes)
        string += f"    {l} \n"
        return string

    def _load(self, root, mode, train_val_split=0.2):
        self.image_pths = np.load(
            'cfg/dataset/mlhypersim/image_pths.npy').tolist()
        self.label_pths = np.load(
            'cfg/dataset/mlhypersim/label_pths.npy').tolist()
        self.scenes = np.load('cfg/dataset/mlhypersim/scenes.npy').tolist()
        self.image_pths = [os.path.join(root, i) for i in self.image_pths]
        self.label_pths = [os.path.join(root, i) for i in self.label_pths]

        # self.image_pths = [str(p) for p in Path(root).rglob('*final_hdf5/*color.hdf5')] # len = 74619
        # self.image_pths.sort()
        # self.label_pths = [i.replace('final_hdf5','geometry_hdf5').replace('color.hdf5','semantic.hdf5') for i in self.image_pths]
        # self.scenes = [p.split('/')[-4] for p in self.image_pths]

        self.sceneTypes = list(set(self.scenes))
        self.sceneTypes.sort()

        # Scene filtering checked by inspection
        for sce in self.sceneTypes:
            images_in_scene = [i for i in self.image_pths if i.find(sce) != -1]
            k = int((1 - train_val_split) * len(images_in_scene))
            if mode == 'train':
                remove_ls = images_in_scene[k:]
            elif mode == 'val':
                remove_ls = images_in_scene[:k]

            idx = self.image_pths.index(remove_ls[0])
            for i in range(len(remove_ls)):
                del self.image_pths[idx]
                del self.label_pths[idx]
                del self.scenes[idx]
        self.length = len(self.image_pths)

    @staticmethod
    def get_classes():
        scenes = np.load('cfg/dataset/mlhypersim/scenes.npy').tolist()
        sceneTypes = sorted(set(scenes))
        return sceneTypes

    def _filter_scene(self, scenes):
        images_idx = []
        for sce in scenes:
            images_idx += [i for i in range(len(self.image_pths))
                           if (self.image_pths[i]).find(sce) != -1]
        idx = np.array(images_idx)
        self.image_pths = (np.array(self.image_pths)[idx]).tolist()
        self.label_pths = (np.array(self.label_pths)[idx]).tolist()
        self.scenes = (np.array(self.scenes)[idx]).tolist()
        self.length = len(self.image_pths)


def test():
    # pytest -q -s src/datasets/ml_hypersim.py

    # Testing
    import imageio
    output_transform = tf.Compose([
        tf.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    dataset = MLHypersim(
        mode='train',
        scenes=[
            'ai_001_002',
            'ai_001_003',
            'ai_001_004',
            'ai_001_005',
            'ai_001_006'],
        output_trafo=output_transform,
        output_size=400,
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[
            0.3,
            0.3,
            0.3,
            0.05])

    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=False,
                                             batch_size=2)

    for j, data in enumerate(dataloader):
        print(j)
        # img, label, _img_ori= dataset[i]    # C, H, W

        # label = np.uint8( label.numpy() * (255/float(label.max())))[:,:]
        # img = np.uint8( img.permute(1,2,0).numpy()*255 ) # H W C
        # imageio.imwrite(f'/home/jonfrey/tmp/{i}img.png', img)
        # imageio.imwrite(f'/home/jonfrey/tmp/{i}label.png', label)


if __name__ == "__main__":
    test()
