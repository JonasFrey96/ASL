from torchvision.utils import make_grid
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

import copy
import torch
import cv2
import numpy as np
import imageio
import os
try:
	from loader import PseudoLabelLoader
except:
	from .loader import PseudoLabelLoader
import matplotlib.pyplot as plt
"""
TODO:
- Depth
- BoundingBox Predicitons
"""

__all__ = ['PseudoLabelGenerator']

class PseudoLabelGenerator():
    def __init__(self, base_path, sub=10, confidence='equal', 
            flow_mode='sequential', H=640, W=1280, 
            nc=40, refine_superpixel=True,
            get_depth_superpixel=False,window_size=10,
            visu=None, pre_fusion_function=None, visu_active=True, cfg_loader={}):
        """  
        confidence:
          'equal': perfect optical flow -> all project labels are equally good
          'linear': linear rate -> per frame
          'exponential': exponential rate -> per frame 
        flow_mode:
          'sequential': #-> 0->1, 1->2, 2->3
          'target': 0->3 1->3 2->3
        """
        self._visu_active = visu_active
        self._sub = sub
        self._flow_mode = flow_mode #'sequential' #-> 0->1, 1->2, 2->3 # 'target' 0->3 1->3 2->3
        self._H,self._W = H,W
        self._confidence= confidence # equal, linear, exponential
        self._nc = nc
        self._refine_superpixel = refine_superpixel
        self._get_depth_superpixel = get_depth_superpixel
        self._window_size = window_size
        self._pll = PseudoLabelLoader(base_path = base_path, window_size=window_size, sub=10, h=H,w=W, **cfg_loader )
        self._ignore_depth = cfg_loader.get("ignore_depth",False)

        # Passed externally
        self._visu = visu
        self._pre_fusion_function = pre_fusion_function
    def __len__(self):
      return self._pll.length

    def get_gt_label(self, index):
        seg, depth, flow, paths = self._pll[index] 
        return seg[0][1]
    
    def get_img(self, index):
        return self._pll.getImage(index)
    
    def get_depth(self, index):
        seg, depth, flow, paths = self._pll[index]
        return depth[0]
    
    def calculate_label(self, index):
        seg_forwarded, depth_forwarded = self._forward_index(index, self._pre_fusion_function)
        if self._visu_active:
            print("Forward projected label")
            self._visu_seg_forwarded(seg_forwarded)

        # -1 39 -> 0 -> 40 We assume that the network is also able to predict the invalid class
        # In reality this is not the case but this way we can use the ground truth labels for testing
        for i in range(len( seg_forwarded) ):
            seg_forwarded[i] += 1 

        confidence_values_list = self._get_confidence_values(seq_length= len(seg_forwarded))

        one_hot_acc = np.zeros( (*seg_forwarded[0].shape,self._nc+1), dtype=np.float32) # H,W,C
        for conf, seg in zip(confidence_values_list, seg_forwarded):    
            one_hot_acc += (np.eye(self._nc+1)[seg.astype(np.int32)]).astype(np.float32) * conf

        invalid_labels = np.sum( one_hot_acc[:,:,1:],axis=2 ) == 0

        label = np.argmax( one_hot_acc[:,:,1:], axis=2 )
        label[ invalid_labels ] = -1 
        if self._visu_active:
            print("Aggregated Label")
            self._visu.plot_segmentation(seg= label+1, jupyter=True)

        if self._refine_superpixel:
            img = self._pll.getImage(index).astype(np.float32)/256
            label_super, img, segments = self._superpixel_label(img, label)
            if self._visu_active:
                print("Label Superpixel")
                self._visu.plot_segmentation(seg= label_super + 1, jupyter=True)  
                self._visu.plot_image(img=img, jupyter=True)  
                self._visu_superpixels(img, segments)
            label = label_super

        if self._get_depth_superpixel:
            self._superpixel_depth(depth_forwarded[-1], label)
        return depth_forwarded[-1], label, (seg_forwarded[-1]-1).astype(np.int32)
    
    def _get_confidence_values( self, seq_length ):
        if self._confidence == 'equal':
            return [float( 1/seq_length)] * seq_length 

        if self._confidence == 'linear':
            ret = []
            lin_rate = 0.1
            s = 0
            for i in range(seq_length):
                res = 1 - lin_rate* (seq_length-i)
                if res < 0: 
                    res = 0
                s += res

                ret.append(res)
            return [r/s for r in ret]

        elif self._confidence == 'exponential':
            ret = []
            exp_rate = 0.8
            s = 0
            for i in range(seq_length):
                res = exp_rate**(seq_length-i)
                if res < 0: 
                    res = 0
                s += res
                ret.append(res)
            return [r/s for r in ret]


    def _forward_index(self, index, pre_fusion_function=None ):
        """
        pre_fusion_function should be used to integrate the depth measurments 
        to the semseg before forward projection !

        seg_forwarded[0] -> oldest_frame
        seg_forwarded[len(seg_forwarded)] -> latest_frame not forwarded

        """
        i_seg = 0 # 0=pred 1=target 

        if pre_fusion_function is None:
            seg, depth, flow, _ = self._pll[index]
        else:
            seg, depth, flow, _ = pre_fusion_function( self._pll[index] )

        assert self._flow_mode == 'sequential'
        seg_forwarded = []
        depth_forwarded = []

        for i in range(0,len(seg)-1):
            i = len(seg)-1-i
            seg_forwarded.append( seg[i][i_seg].astype(np.float32) )
            if self._ignore_depth:
                depth_forwarded.append( seg[i][i_seg].astype(np.float32))
            else:
                depth_forwarded.append( depth[i].astype(np.float32) )

            # start at oldest frame

            if i != 0:
                f = flow[i][0]
            else:
                f = np.zeros(flow[i][0].shape, dtype=np.float32)

            h_, w_ = np.mgrid[0:self._H, 0:self._W].astype(np.float32)
            h_ -= f[:,:,1]
            w_ -= f[:,:,0]

            j = 0
            for s, d in zip ( seg_forwarded, depth_forwarded): #  seg_forwarded, depth_forwarded

                s = cv2.remap( s, w_, h_, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
                d = cv2.remap( d, w_, h_, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                seg_forwarded[j] = s
                depth_forwarded[j] = d

                j += 1

        seg_forwarded.append( seg[0][i_seg].astype(np.float32) )
        if self._ignore_depth:
            depth_forwarded.append(  seg[0][i_seg].astype(np.float32) )
        else:
            depth_forwarded.append( depth[0].astype(np.float32))

        return seg_forwarded, depth_forwarded 


    def _visu_seg_forwarded(self, seg):
        s = int( len(seg) ** 0.5 )
        ba = torch.zeros( (int(s*s),3, *seg[0].shape), dtype= torch.float32 )
        for i in range( int(s*s) ) :
            ba[i,:] = torch.from_numpy( seg[-(i+1)] )[None,:,:].repeat(3,1,1)
        grid_ba = make_grid( ba ,nrow = s ,padding = 2,
          scale_each = False, pad_value = -1)[0]
        self._visu.plot_segmentation(seg= grid_ba +1 , jupyter=True)

    def _superpixel_label(self, img, label, segments=250):
        assert segments < 256 #I think slic fails if segments > 256 given that a 8bit uint is returend!

        segments = slic(img, n_segments = segments, sigma = 5, start_label=0)
        # show the output of SLIC
        out_label = copy.copy(label)
        for i in range(0,segments.max()):
            m1 = segments == i
            m = m1 * ( label != -1 )
            unique_val, unique_counts = np.unique( label [m], return_counts=True)
            # fill a segment preferable not with invalid !
            if unique_counts.shape[0] == 0:
                val = -1
            else:
                ma = unique_counts == unique_counts.max()
                while ma.sum() != 1:
                    ma[np.random.randint(0,ma.shape[0])] = False
                val = unique_val[ma]
            out_label[m1] = val 

        return out_label, img, segments
    
    def _visu_superpixels(self, img, segments):
        import matplotlib.pyplot as plt
        from skimage.segmentation import mark_boundaries
        fig = plt.figure("Superpixels -- segments" )
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(img, segments))
        plt.axis("off")
        plt.show()


def test():
  from visu import Visualizer
  visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=False, num_classes=41)
  plg = PseudoLabelGenerator(visu=visu)
  label = plg.calculate_label(0)
  

if __name__ == "__main__":
  test()