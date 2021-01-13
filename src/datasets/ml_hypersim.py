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
import time
try:
    from .helper import Augmentation
except Exception: #ImportError
    from helper import Augmentation
    
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
        self._filter_scene(scenes)
        
        self._augmenter = Augmentation(output_size,
                                       degrees,
                                       flip_p,
                                       jitter_bcsh)
        
        self._output_trafo = output_trafo
        # full training dataset with all objects
        # TODO
        #self._weights = pd.read_csv(f'cfg/dataset/ml-hypersim/test_dataset_pixelwise_weights.csv').to_numpy()[:,0]
    
            
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
        
        return img, label.type(torch.int64)[0,:,:] , img_ori
    
    def __len__(self):
        return self.length

    def _load(self, root, mode, train_val_split=0.2):
        self.image_pths = np.load( 'cfg/dataset/mlhypersim/image_pths.npy').tolist()
        self.label_pths = np.load( 'cfg/dataset/mlhypersim/label_pths.npy').tolist()
        self.scenes = np.load( 'cfg/dataset/mlhypersim/scenes.npy').tolist()
        
        # self.image_pths = [str(p) for p in Path(root).rglob('*final_hdf5/*color.hdf5')] # len = 74619
        # self.image_pths.sort()
        # self.label_pths = [i.replace('final_hdf5','geometry_hdf5').replace('color.hdf5','semantic.hdf5') for i in self.image_pths]
        # self.scenes = [p.split('/')[-4] for p in self.image_pths]
        
        self.sceneTypes = list(set(self.scenes))
        self.sceneTypes.sort()
              
        # Scene filtering checked by inspection          
        for sce in self.sceneTypes:
            images_in_scene = [i for i in self.image_pths if i.find(sce) != -1]
            k = int( (1-train_val_split)*len(images_in_scene) )
            if mode == 'train':
                remove_ls = images_in_scene[k:]
            elif mode == 'val':
                remove_ls = images_in_scene[:k]
            
            idx = self.image_pths.index( remove_ls[0] )
            for i in range(len(remove_ls)):
                del self.image_pths[idx]
                del self.label_pths[idx]
                del self.scenes[idx]
        self.length = len(self.image_pths)

    @staticmethod
    def get_classes():
        scenes = np.load( 'cfg/dataset/mlhypersim/scenes.npy').tolist()
        sceneTypes =  list(set(scenes))
        sceneTypes.sort()
        return sceneTypes
    
    def _filter_scene(self, scenes):
        images_idx = [] 
        for sce in scenes:
            images_idx += [i for i in range(len(self.image_pths)) if (self.image_pths[i]).find(sce) != -1] 
        idx = np.array(images_idx)
        self.image_pths = (np.array( self.image_pths )[ idx ]).tolist()
        self.label_pths = (np.array( self.label_pths )[ idx ]).tolist()
        self.scenes = (np.array( self.scenes )[ idx ]).tolist()
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
        scenes=['ai_001_002', 'ai_001_003', 'ai_001_004', 'ai_001_005', 'ai_001_006'],
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
    
if __name__ == "__main__":
    test()