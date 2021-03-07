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
import copy 
import time
try:
    from .helper import Augmentation
    from .replay_base import StaticReplayDataset
except Exception:  # ImportError
    from helper import Augmentation
    from replay_base import StaticReplayDataset
import imageio
import pandas

import pickle
__all__ = ['ScanNet']

class ScanNet(StaticReplayDataset):
    def __init__(
            self,
            root='/media/scratch2/jonfrey/datasets/scannet/',
            mode='train',
            scenes=[],
            output_trafo=None,
            output_size=(480,640),
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
        """
        Some images are stored in 640x480 other ins 1296x968

        Parameters
        ----------
        root : str, path to the ML-Hypersim folder
        mode : str, option ['train','val]
        """
        super(
            ScanNet,
            self).__init__(
            ** cfg_replay, replay=replay)

        if mode == 'val':
            mode = 'test'
            
        self._mode = mode

        self._load(root, mode)
        self._filter_scene(scenes)

        self._augmenter = Augmentation(output_size,
                                       degrees,
                                       flip_p,
                                       jitter_bcsh)

        self._output_trafo = output_trafo
        self._data_augmentation = data_augmentation
        self._data_augmentation_for_replay = data_augmentation_for_replay
        
        self.unique = False
        self.replay = replay
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

        idx = -1
        replayed = torch.zeros( [1] )
        
        global_idx = self.global_to_local_idx[index]
        
        # Read Image and Label
        label = imageio.imread(self.label_pths[global_idx])        
        #170 ms
        label = torch.from_numpy(label.astype(np.int32)).type(
            torch.float32)[None, :, :]  # C H W
        
        img = imageio.imread(self.image_pths[global_idx])
        img = torch.from_numpy(img).type(
            torch.float32).permute(
            2, 0, 1)/255  # C H W range 0-1
        if (self._mode == 'train' and 
            ( ( self._data_augmentation and idx == -1) or 
              ( self._data_augmentation_for_replay and idx != -1) ) ):
            
            img, label = self._augmenter.apply(img, label)
        else:
            img, label = self._augmenter.apply(img, label, only_crop=True)

                
        img_ori = img.clone()
        if self._output_trafo is not None:
            img = self._output_trafo(img)
        sa = label.shape
        label = label.flatten()
        label = self.mapping[label.type(torch.int64)] # scannet to nyu40
        label = label.reshape(sa)
        label = label - 1  # 0 == chairs 39 other prop  -1 invalid
        
        # check if reject
        if (label != -1).sum() < 10:
            # reject this example
            idx = random.randint(0, len(self) - 1)
            if not self.unique:
                return self[idx]
            else:
                replayed[0] = -999
        return img, label.type(torch.int64)[0, :, :], img_ori, replayed.type(torch.float32), global_idx

    def __len__(self):
        return self.length

    def __str__(self):
        string = "="*90
        string += "\nScannet Dataset: \n"
        l = len(self)
        string += f"    Total Samples: {l}"
        string += f"  »  Mode: {self._mode} \n"
        string += f"    Replay: {self.replay}"
        string += f"  »  DataAug: {self._data_augmentation}"
        string += f"  »  DataAug Replay: {self._data_augmentation_for_replay}\n"
        string += "="*90
        return string
        
    def _load(self, root, mode, train_val_split=0.2):
        tsv = os.path.join(root, "scannetv2-labels.combined.tsv")
        df = pandas.read_csv(tsv, sep='\t')
        self.df =df
        mapping_source = np.array( df['id'] )
        mapping_target = np.array( df['nyu40id'] )
        
        self.mapping = torch.zeros( ( int(mapping_source.max()+1) ),dtype=torch.int64)
        for so,ta in zip(mapping_source, mapping_target):
            self.mapping[so] = ta 
        
        
        self.train_test, self.scenes, self.image_pths, self.label_pths = self._load_cfg(root, train_val_split)
        self.image_pths = [ os.path.join(root,i) for i in self.image_pths]
        self.label_pths = [ os.path.join(root,i) for i in self.label_pths]
        
        self.valid_mode = np.array(self.train_test) == mode
        
        sub = 10
        sub_mask = np.zeros( self.valid_mode.shape )
        sub_mask[::sub] = 1
        self.valid_mode = self.valid_mode * (sub_mask == 1)
        
        self.global_to_local_idx = np.arange( self.valid_mode.shape[0] )
        self.global_to_local_idx = (self.global_to_local_idx[self.valid_mode]).tolist()
        self.length = len(self.global_to_local_idx)

    @staticmethod
    def get_classes():
        _, scenes, _, _ = self._load_cfg()
        
        
        return np.unique(scenes).tolist()
    
    def _load_cfg( self, root='not_def', train_val_split=0.2 ):
        # if pkl file already created no root is needed. used in get_classes
        try:
            data = pickle.load( open( f"cfg/dataset/scannet/scannet_trainval_{train_val_split}.pkl", "rb" ) )
            return data['train_test'], data['scenes'], data['image_pths'], data['label_pths']
        except:
            pass
        return self._create_cfg( root, train_val_split)
    
    def _create_cfg( self, root, train_val_split=0.2 ):
        """Creates a pickle file containing all releveant information. 
        For each train_val split a new pkl file is created
        """
        r = os.path.join( root,'scans')
        ls = [os.path.join(r,s[:9]) for s in os.listdir(r)]
        all_s = [os.path.join(r,s) for s in os.listdir(r)]

        scenes = np.unique( np.array(ls) ).tolist()
        scenes = [s for s in scenes if int(s[-4:]) <= 100] # limit to 100 scenes
        
        sub_scene = {s: [ a for a in all_s if a.find(s) != -1]  for s in scenes }
        for s in sub_scene.keys():
          sub_scene[s].sort()
        key = scenes[0] + sub_scene[scenes[0]][1] 

        image_pths = []
        label_pths = []
        train_test = []
        scenes_out = []
        for s in scenes:
          for sub in sub_scene[s]:
            # print(s)
            colors = [str(p) for p in Path(sub).rglob('*color/*.jpg')]
            labels = [str(p) for p in Path(sub).rglob('*label-filt/*.png')]
            fun = lambda x : int( x.split('/')[-1][:-4]) 
            colors.sort(key=fun)
            labels.sort(key=fun)
            
            for i,j  in zip(colors, labels):
              assert int( i.split('/')[-1][:-4]) == int( j.split('/')[-1][:-4]) 
            
            if len(colors) > 0:
              assert len(colors) == len(labels)
            
              nr_train = int(  len(colors)* (1-train_val_split)  )
              nr_test = int( len(colors)-nr_train)
              train_test += ['train'] * nr_train
              train_test += ['test'] * nr_test
              scenes_out += [s.split('/')[-1]]*len(colors)
              image_pths += colors
              label_pths += labels
            else:
                print( sub ,"Color not found" )
        image_pths = [ i.replace(root,'') for i in image_pths]
        label_pths = [ i.replace(root,'') for i in label_pths]
        data = {
           'train_test': train_test,
           'scenes': scenes_out,
           'image_pths': image_pths,
           'label_pths': label_pths,
        }
        pickle.dump( data, open( f"cfg/dataset/scannet/scannet_trainval_{train_val_split}.pkl", "wb" ) )
        return train_test, scenes_out, image_pths, label_pths
        
    def _filter_scene(self, scenes):
        self.valid_scene = copy.deepcopy( self.valid_mode )
        if len(scenes) != 0:
          for sce in scenes:
            tmp = np.array(self.scenes) == sce
            self.valid_scene = np.logical_and(tmp, self.valid_scene )
            
        self.global_to_local_idx = np.arange( self.valid_mode.shape[0] )
        self.global_to_local_idx = (self.global_to_local_idx[self.valid_scene]).tolist()
        self.length = len(self.global_to_local_idx)

def test():
    # pytest -q -s src/datasets/ml_hypersim.py

    # Testing
    import imageio
    output_transform = tf.Compose([
        tf.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    # create new dataset 
    dataset = ScanNet(
        mode='val',
        scenes=[],
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
    dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=False,
                                             batch_size=4)
    print(dataset)
    import time
    st = time.time()
    print("Start")
    for j, data in enumerate(dataloader):
        img = data[0]
        label = data[1]
        assert type(label) == torch.Tensor
        assert (label != -1).sum() > 10
        assert (label < -1).sum() == 0
        assert label.dtype == torch.int64
        if j % 10 == 0 and j != 0:
            
            print(j, '/', len(dataloader))
        if j == 100:
            break
        
    print('Total time', time.time()-st)
        #print(j)
        # img, label, _img_ori= dataset[i]    # C, H, W

        # label = np.uint8( label.numpy() * (255/float(label.max())))[:,:]
        # img = np.uint8( img.permute(1,2,0).numpy()*255 ) # H W C
        # imageio.imwrite(f'/home/jonfrey/tmp/{i}img.png', img)
        # imageio.imwrite(f'/home/jonfrey/tmp/{i}label.png', label)

if __name__ == "__main__":
    test()
