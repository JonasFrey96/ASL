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
import copy 
import time

try:
    from .helper import Augmentation
except Exception:  # ImportError
    from helper import Augmentation

try:
    from .replay_base import StaticReplayDataset
except Exception:  # ImportError
    from replay_base import StaticReplayDataset

__all__ = ['MLHypersim']

class MLHypersim(StaticReplayDataset):
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
            replay = False,
            cfg_replay = {'bins':4, 'elements':100, 'add_p': 0.5, 'replay_p':0.5, 'current_bin': 0},
            data_augmentation= True, data_augmentation_for_replay=True):
        super(
            MLHypersim,
            self).__init__(
            ** cfg_replay, replay=replay)

        self._output_size = output_size
        self._mode = mode

        self._load(root, mode)
        self._filter_scene(scenes)

        self._augmenter = Augmentation(output_size,
                                       degrees,
                                       flip_p,
                                       jitter_bcsh)

        self._output_trafo = output_trafo
        self.replay = replay
        self._data_augmentation = data_augmentation
        self._data_augmentation_for_replay = data_augmentation_for_replay
        
        self.unique = False
        # full training dataset with all objects
        # TODO
        #self._weights = pd.read_csv(f'cfg/dataset/ml-hypersim/test_dataset_pixelwise_weights.csv').to_numpy()[:,0]

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
        replayed = torch.tensor( [-1] , dtype=torch.int32)
        idx = -1
        if self.replay:
            if self._mode == 'train':
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
            
        with h5py.File(self.image_pths[global_idx], 'r') as f:
            img = np.array(f['dataset'])
        img[img > 1] = 1
        img = torch.from_numpy(img).type(
            torch.float32).permute(
            2, 0, 1)  # C H W
        with h5py.File(self.label_pths[global_idx], 'r') as f:
            label = np.array(f['dataset'])
        label = torch.from_numpy(label).type(
            torch.float32)[None, :, :]  # C H W    # label_max = 40 invalid = -1 0 is not used as an index

        if (self._mode == 'train' and 
            ( ( self._data_augmentation and idx == -1) or 
              ( self._data_augmentation_for_replay and idx != -1) ) ):
            
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
        
        #shifts label to -1 = invalid, 0 = wall, 39 = otherprop 
        return img, label.type(torch.int64)[0, :, :], img_ori, replayed.type(torch.float32), global_idx

    def __len__(self):
        return self.length

    def __str__(self):
        string = "="*90
        string += "\nML-HyperSim Dataset: \n"
        l = len(self)
        string += f"    Total Samples: {l}"
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
        string += "="*90
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

        self.global_to_local_idx = [i for i in range(len(self.image_pths))]
        
        self.filtered_image_pths = copy.deepcopy( self.image_pths ) 
        # Scene filtering checked by inspection
        for sce in self.sceneTypes:
            images_in_scene = [i for i in self.filtered_image_pths if i.find(sce) != -1]
            k = int((1 - train_val_split) * len(images_in_scene))
            if mode == 'train':
                remove_ls = images_in_scene[k:]
            elif mode == 'val':
                remove_ls = images_in_scene[:k]

            idx = self.filtered_image_pths.index(remove_ls[0])
            for i in range(len(remove_ls)):
                del self.filtered_image_pths[idx]
                #del self.label_pths[idx]
                #del self.scenes[idx]
                del self.global_to_local_idx[idx]
                
        self.length = len(self.global_to_local_idx)

    @staticmethod
    def get_classes():
        scenes = np.load('cfg/dataset/mlhypersim/scenes.npy').tolist()
        sceneTypes = sorted(set(scenes))
        return sceneTypes

    def _filter_scene(self, scenes):
        if len(scenes) != 0:
            images_idx = []
            for sce in scenes:
                images_idx += [i for i in range(len(self.filtered_image_pths))
                            if (self.filtered_image_pths[i]).find(sce) != -1]
            idx = np.array(images_idx)
            self.global_to_local_idx =  (np.array(self.global_to_local_idx)[idx]).tolist()
            self.length = len(self.global_to_local_idx)

def test():
    # pytest -q -s src/datasets/ml_hypersim.py

    # Testing
    import imageio
    output_transform = tf.Compose([
        tf.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset = MLHypersim(
    #     mode='train',
    #     scenes=[
    #         'ai_001_002',
    #         'ai_001_003',
    #         'ai_001_004',
    #         'ai_001_005',
    #         'ai_001_006'],
    #     output_trafo=output_transform,
    #     output_size=400,
    #     degrees=10,
    #     flip_p=0.5,
    #     jitter_bcsh=[
    #         0.3,
    #         0.3,
    #         0.3,
    #         0.05])

    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          shuffle=False,
    #                                          num_workers=16,
    #                                          pin_memory=False,
    #                                          batch_size=2)
    # print('Start')
    # for j, data in enumerate(dataloader):
    #     t = data
    #     print(j)
    #     if j == 50: 
    #         print('Set bin to 1')
    #         dataloader.dataset.set_current_bin(bin=1)
            
    #     if j > 100:
    #         break
        
    # bins, valids = dataloader.dataset.get_full_state()
    
    # create new dataset 
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
            0.05],
        replay=True)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             num_workers=16,
                                             pin_memory=False,
                                             batch_size=2)
    
    # dataloader.dataset.set_full_state(bins,valids, 2)
    import time
    st = time.time()
    print("Start")
    for j, data in enumerate(dataloader):
        t = data
        print(j)

    print('Total time', time.time()-st)
        #print(j)
        # img, label, _img_ori= dataset[i]    # C, H, W

        # label = np.uint8( label.numpy() * (255/float(label.max())))[:,:]
        # img = np.uint8( img.permute(1,2,0).numpy()*255 ) # H W C
        # imageio.imwrite(f'/home/jonfrey/tmp/{i}img.png', img)
        # imageio.imwrite(f'/home/jonfrey/tmp/{i}label.png', label)


if __name__ == "__main__":
    test()
