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
from  PIL import Image
import h5py
from .helper import Augmentation

__all__ = ['MLHypersim']

class MLHypersim(data.Dataset):
    def __init__(self, root='/media/scratch2/jonfrey/datasets/mlhypersim/', 
                 mode='train', scenes=[], output_trafo = None, 
                 output_size=400, degrees = 10, flip_p = 0.5, jitter_bcsh=[0.3, 0.3, 0.3, 0.05]):
        """
        Each dataloader loads the full .mat file into memory. 
        For the small dataset size this is perfect.
        Both should work when file is located on SSD!
        
        Parameters
        ----------
        root : str, path to the ML-Hypersim folder
        mode : str, option ['train','val]
        """
        self._output_size = output_size
        self._mode = mode
        
        self._load(root, mode)
        
        self._augmenter = Augmentation(output_size,
                                       degrees,
                                       flip_p,
                                       jitter_bcsh)
        
        self._output_trafo = output_trafo
        # full training dataset with all objects
        # TODO
        #self._weights = pd.read_csv(f'cfg/dataset/ml-hypersim/test_dataset_pixelwise_weights.csv').to_numpy()[:,0]
    
    @staticmethod
    def get_classes(mode):
        # TODO 
        return {}
            
    def __getitem__(self, index):

        with h5py.File(self.image_pths[index], 'r') as f: img = np.array( f['dataset'] )  
        img[img>1] = 1
        img = torch.from_numpy( img ).type(torch.float32).permute(2,0,1) # C H W
        with h5py.File(self.label_pths[index], 'r') as f: label = np.array( f['dataset'] ) 
        label = torch.from_numpy( label ).type(torch.float32)[None,:,:] # C H W
        
        if self._mode == 'train':
            img, label = self._augmenter.apply(img, label)
        elif self._mode == 'val' or self._mode == 'test':
            img, label = self._augmenter.apply(img, label, only_crop=True)
        else:
            raise Exception('Invalid Dataset Mode')
        
        img_ori = img.clone()
        if self._output_trafo is not None:
            img = self._output_trafo(img)
        
        return img, label.type(torch.int64)[0,:,:], img_ori
    
    def __len__(self):
        return self.length

    def _load(self, root, mode):
        self.image_pths = [str(p) for p in Path(root).rglob('*final_hdf5/*color.hdf5')]
        self.label_pths = [i.replace('final_hdf5','geometry_hdf5').replace('color.hdf5','semantic.hdf5') for i in self.image_pths]
        
        self.scenes = [str(p).split('/')[-2] for p in Path(root).rglob('*final_hdf5/*color.hdf5')]
        print(self.scenes)
        self.length = len(self.image_pths)
        
    def _filter_scene(self, scenes):
        pass

def test():
    # pytest -q -s src/datasets/ml_hypersim.py
    
    # Testing
    import imageio
    output_transform = tf.Compose([
      tf.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    dataset = MLHypersim(
        mode='train',
        scenes=[],
        output_trafo = None, 
        output_size=400, 
        degrees = 10, 
        flip_p = 0.5, 
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05])
    
    img, label = dataset[0]    # C, H, W
    
    label = np.uint8( label.numpy() * (255/float(label.max())))[:,:]
    img = np.uint8( img.permute(1,2,0).numpy()*255 ) # H W C
    imageio.imwrite('/home/jonfrey/tmp/img.png', img)
    imageio.imwrite('/home/jonfrey/tmp/label.png', label)