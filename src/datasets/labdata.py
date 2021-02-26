import numpy as np
import torch
import PIL
import os
from pathlib import Path
from PIL import Image
import imageio
try:
    from .helper import Augmentation
except Exception:  # ImportError
    from helper import Augmentation

try:
    from .replay_base import StaticReplayDataset
except Exception:  # ImportError
    from replay_base import StaticReplayDataset

__all__ = ['LabData']

class LabData(StaticReplayDataset):
    def __init__(
            self,
            root='/media/scratch2/jonfrey/datasets/labdata/',
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
            LabData,
            self).__init__(** cfg_replay, replay=replay)

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
        idx = -1
        replayed = torch.zeros( [1] )
        
        
        global_idx = self.global_to_local_idx[index]
        
        img = imageio.imread(self.image_pths[global_idx])
        img = torch.from_numpy(img).type(
            torch.float32).permute(
            2, 0, 1)/255  # C H W range 0-1

        label = torch.ones( (1,img.shape[1],img.shape[2]), dtype=torch.float32) # C H W

        if (self._mode == 'train' and 
            ( ( self._data_augmentation and idx == -1) or 
              ( self._data_augmentation_for_replay and idx != -1) ) ):
            
            img, label = self._augmenter.apply(img, label)
        else:
            img, label = self._augmenter.apply(img, label, only_crop=True)

        img_ori = img.clone()
        if self._output_trafo is not None:
            img = self._output_trafo(img)
        
        return img, label.type(torch.int64)[0, :, :], img_ori, replayed.type(torch.float32), global_idx

    def __len__(self):
        return self.length

    def __str__(self):
        string = "="*90
        string += "\nLabData-Dataset: \n"
        l = len(self)
        string += f"    Total Samples: {l}"
        string += f"  »  Mode: {self._mode} \n"
        string += f"    Replay: {self.replay}"
        string += f"  »  DataAug: {self._data_augmentation}"
        if self.replay:
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
        sp = Path(os.path.join(root, '2'))
        self.image_pths =  [str(p) for p in sp.rglob('*.png')]
        self.global_to_local_idx = list( range( len(self.image_pths) ) )
        self.length = len(self.global_to_local_idx)

    def _filter_scene(self, scenes):
        pass

def test():
    # pytest -q -s src/datasets/ml_hypersim.py

    # Testing
    from torchvision import transforms as tf
    import time
    
    output_transform = tf.Compose([
        tf.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    
    dataset = LabData(
        mode='train',
        replay=True)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             num_workers=16,
                                             pin_memory=False,
                                             batch_size=2)
    
    st = time.time()
    print("Start")
    for j, data in enumerate(dataloader):
        if j > 50:
          break
        t = data
        print(j)
        img = data[0]
        print(img.shape)
        img = np.uint8( (img[0]).permute(1,2,0).numpy()*255 ) # H W C
        imageio.imwrite(f'/home/jonfrey/tmp/{j}img.png', img)

    print('Total time', time.time()-st)

if __name__ == "__main__":
    test()
