import sys
import os
os.chdir("/home/jonfrey/ASL")
sys.path.append("""/home/jonfrey/ASL/src/""")
sys.path.append("""/home/jonfrey/ASL/src/pseudo_label""")

from visu import Visualizer
from pseudo_label.fast_scnn import FastSCNNHelper
from pseudo_label.detectron import DetectronHelper

import numpy as np
import imageio

visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=False, num_classes=41)

DEVICE = 'cuda:0'
fsh = FastSCNNHelper(device=DEVICE)
deh = DetectronHelper(device=DEVICE)
# -> visu, yh, dlh, fsh 
########################################################################################

import yaml
from pathlib import Path
import numpy as np 
from math import ceil
import torch

from utils_asl import load_yaml
from task import TaskCreator
from task import TaskCreator
from datasets_asl import get_dataset

ASL = os.path.join( str(Path.home()), "ASL" )
name = os.getenv('ENV_WORKSTATION_NAME')
env_cfg_path =os.path.join( ASL, f"cfg/env/{name}.yml")  
exp_cfg_path =os.path.join( ASL, "cfg/exp/debug.yml")
eva_cfg_path =os.path.join( ASL, "cfg/eval/eval.yml")

env = load_yaml(env_cfg_path)

use_eva = True
if use_eva:
  exp = load_yaml(eva_cfg_path)
else:
  exp = load_yaml(exp_cfg_path)

# SETUP DATALOADERS

tc = TaskCreator(**exp['task_generator'] )

use_tc = True
if use_tc:    
    cfgs = []
    for idx, out in enumerate(tc):
      task, eval_lists = out
      cfgs.append(  task.dataset_train_cfg )
else:
    cfgs  = [exp['dataset']]

for cfg in cfgs:
  dataset= get_dataset(
    **cfg,
    env = env,
    output_trafo = None,
  )
  dataloader = torch.utils.data.DataLoader(dataset,
    shuffle = False,
    num_workers = exp['loader']['num_workers'],
    pin_memory = exp['loader']['pin_memory'],
    batch_size = 1, 
    drop_last = True)
  break
    
# -> dataset, dataloader, exp
########################################################################################

def get_max_acc(label, gt , names = ['detectron', 'fastscnn'] ):
    m = gt != -1
    acc_indi = {}
    for l,n in zip(label[:len(names)],names):
        correct = ( l[m] == gt[m]).sum()
        m2 = m * (l != -1)
        total = m2.sum()
        acc_indi[n] = correct/total
        
    # Optimal upper bound
    m = gt != -1
    m_correct = np.zeros( label[0].shape )
    for l,n in zip(label,names):
        m_est = l == gt
        m_correct[m_est] = 1
        
    
    acc_upper_bound = m_correct[m].sum() / m.sum()
    acc_indi['upper_bound'] = acc_upper_bound
    return acc_indi
# -> get indiviual ACC per network but also upper bound

from pseudo_label import *

class PseudoLabelLoaderOnline():
    def __init__(self, image_paths, mapping, h=640, w=1280, sub=10):
        self.image_paths = image_paths
        self.sub = sub
        self.H, self.W= h,w 
        self.mapping = mapping
        
    def get_flow(self, global_idx):
        flow = []
        for idx in global_idx:
            fp = self.mapping( self.image_paths[idx] )
            
            try:
                print("Flow Path", fp)
                flow.append( readFlowKITTI( fp, H=self.H ,W=self.W))
            except:
                return False, False
        flow.reverse()
        return True, flow
    
def mapping_labdata(s, sub=10):
    return s.replace("/2/",f"/2_flow_sub{sub}/")
def mapping_scannet(s, sub=10):
    return str(os.path.join( "/home/jonfrey/datasets/scannet/flow", s[s.find('scans'):])).replace('.jpg','.png').replace('color', f'flow_sub{sub}')
    

mapping = mapping_scannet if use_tc else mapping_labdata
pllo = PseudoLabelLoaderOnline( 
    image_paths = dataset.image_pths,
    mapping = mapping,
    sub=10)

def index_check_labdata(global_idx_list, global_idx_pths, sub=10): 
    assert len(global_idx_list) > 1
    for a,b in zip(global_idx_list[:-1], global_idx_list[1:]):
        a = int(global_idx_pths[a].split('/')[-1].replace("undistorted_frame","")[:-4])
        b = int(global_idx_pths[b].split('/')[-1].replace("undistorted_frame","")[:-4])
        if a + sub != b:
            return False
    return True

########################################################################################

from torchvision.utils import make_grid
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

import copy
import torch
import cv2
import numpy as np
import imageio
import os

import matplotlib.pyplot as plt
from torch.nn.functional import one_hot

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
    
    def calculate_label(self, index=None, seg=[],flow=[], image= None ):
        if not index is None:
            seg_forwarded= self._forward_index(index, self._pre_fusion_function) #return H,W,C
        else:
            seg_forwarded = self._forward_index(index, seg, flow, self._pre_fusion_function)
        
        H,W,C = seg_forwarded[0].shape
        N = len(seg_forwarded)
        if self._visu_active:
            self._visu_seg_forwarded(seg_forwarded)
        
        returned_probs = C != 1
        # if not returned_probs: -1 invalid
        if not returned_probs:
            for i in range( N ):
                seg_forwarded[i] += 1
        # 0 = invalid
        
        confidence_values_list = self._get_confidence_values(seq_length=N)
        one_hot_acc = np.zeros( (H,W,self._nc+1), dtype=np.float32)
            
        for j, (conf, f_seg) in enumerate( zip(confidence_values_list, seg_forwarded)):    
            if not returned_probs:
                # Generate one hot encoding
                one_hot_acc += ((np.eye(self._nc+1)[f_seg.astype(np.int32)]).astype(np.float32) * conf )[:,:,0,:]
            else:
                one_hot_acc += ( f_seg * conf )
        
        invalid_labels = np.sum( one_hot_acc[:,:,1:], axis=2 ) == 0
        label = np.argmax( one_hot_acc[:,:,1:], axis=2 )
        label[ invalid_labels ] = -1 
        return label
    
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

    def _forward_index(self, index=None, seg=[],flow=[] ,pre_fusion_function=None ):
        """
        seg[0] , C,H,W
        seg[-1] oldest frame
        flow[-1] oldest frame

        # FORWARD INDEX TURNS SEQUENCE

        seg_forwarded[0] -> oldest_frame
        seg_forwarded[-1] -> latest_frame not forwarded
        """
        
        if not index is None:
            if pre_fusion_function is None:
                seg, _, flow, _ = self._pll[index]
            else:
                seg, _, flow, _ = pre_fusion_function( self._pll[index] )
        
        N = len(seg)
        C, H ,W = seg[0].shape 
        assert len(seg) == len(flow)
        assert self._flow_mode == 'sequential'

        soft = C != 1 
        seg_forwarded = []
        
        for j in range( N ):
            seg[j] = np.moveaxis( seg[j], [0,1,2], [2,0,1] ) #C,H,W -> H,W,C
        
        for i in range(0,N-1):
            i = N-1-i        
            # if len(seg) = 3 -> i=0 -> i = 1
            seg_forwarded.append( seg[i].astype(np.float32) )
            h_, w_ = np.mgrid[0:H, 0:W].astype(np.float32)
            h_ -= flow[i][0][:,:,1] # N x type x H x W x (0=H,1=W) 
            w_ -= flow[i][0][:,:,0]

            j = 0
            for k_, s in enumerate( seg_forwarded) : # seg_forwarded, depth_forwarded
                if soft:
                    s = cv2.remap( s, w_, h_, 
                        interpolation=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_CONSTANT, 
                        borderValue=0)
                else:
                    s = cv2.remap( s, w_, h_, 
                        interpolation=cv2.INTER_NEAREST, 
                        borderMode=cv2.BORDER_CONSTANT, 
                        borderValue=-1)[:,:,None]

                seg_forwarded[j] = s
                j += 1
    
        seg_forwarded.append( seg[0] )
        return seg_forwarded # H,W,C  -1 is invalid 

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


########################################################################################

from pseudo_label import readSegmentation
import time
from torchvision import transforms as tf
import copy
# from torch import one_hot_acc
st = time.time()
counts = 0
counts_flow = 0
length = len(dataloader)
globale_idx_to_image_path = dataset.image_pths

segmentation_list_fastscnn = []
global_idx_list = []

plot = False
# PARAMS LABEL GENERATION

window_size = 3

sub = 10

from PIL import ImageDraw, ImageFont, Image


acc_dict = {}

plg = PseudoLabelGenerator(base_path='/home/jonfrey/results/scannet_pseudo_label/scannet', 
                           visu=visu,
                           window_size=window_size,
                          cfg_loader = {"ignore_depth": True},
                          visu_active=plot,
                          refine_superpixel=False)

def plot_pseudo_labes( res, jupyter = False, label=False, save_indivdual=False, out_pth =""):
    key_list = list(res.keys())
    s = int( len( key_list ) ** 0.5 )
    if len( key_list ) - s*s != 0:
        s +=1
    ba = torch.zeros( (int(s*s),3, *res[key_list[0]].shape), dtype= torch.float32 )
    ref_img = copy.deepcopy( res['img'] )
    for i in range( len( key_list ) ):
        k = key_list[i]
        if k.find("img") == -1:
            if res[k].shape[0] == 41:
                la = np.argmax( res[k], axis = 0 )
                img = visu.plot_detectron( img = copy.deepcopy(ref_img ), label = la )
            else:
                img = visu.plot_detectron( img = copy.deepcopy(ref_img ), label = res[k]+1 )
        else:
            img = res[k]
            
        img = Image.fromarray(img)
        if label:
            img = img.convert("RGBA")
            d = ImageDraw.Draw(img)
            fnt = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSansMono-Bold.ttf", 50)   
            d.rectangle(((550, 500), (1400, 600)), fill=(254,10,10))
            d.text((600,530), k , font=fnt, fill=(254,254,254,254))
            img =  img.convert("RGB")
        if save_indivdual:
            out_pth = out_pth.replace("/2/", f"/2_{k}/")
            Path(out_pth).parent.mkdir(parents=True, exist_ok=True)
            print( out_pth)
            img.save( out_pth )
        
        img = np.array( img )
        ba[i,:] = torch.from_numpy( img[:,:,:3] ).permute(2,0,1)
    
    grid_ba = make_grid( ba ,nrow = s ,padding = 2, scale_each = False, pad_value = 0)
    return visu.plot_image(img = grid_ba, jupyter= jupyter)


down = tf.Resize((320,640))
up = tf.Resize((640,1280))
label_list = []

def print_acc(acc_dict):
    avg = {}
    for k in acc_dict.keys():
        if k.find('flow') != -1:
            avg  = acc_dict[k] / counts_flow
        else:
            avg = acc_dict[k] / counts
        print ( f'ACC {k}: {avg}' )
for j, batch in enumerate( dataloader ):
    # START EVALUATION  
    images = batch[0]
    target = batch[1]
    ori_img = batch[2]
    replayed = batch[3]
    BS = images.shape[0]
    global_idx = batch[4] 
    images *= 255
    images = images.permute(0,2,3,1).numpy().astype(np.uint8)
    
    
    pri = False
    
    
    for b in range( images.shape[0] ):
        # EVALUATE SEMANTIC SEGMENTATION NETWORKS
        label = {}
        st_ = time.time()
        prob_de = deh.get_label_prob( images[b] )
        
        inp = down(torch.from_numpy( images[b]).permute(2,0,1)).permute(1,2,0).numpy()
        prob_fastscnn = fsh.get_label_prob( inp )
        prob_fastscnn = up ( torch.from_numpy( prob_fastscnn[None]) )[0].numpy()
        
        label['fastscnn'] = np.int8 ( np.argmax( prob_fastscnn[:] , axis=0)-1 )
        label['detectron'] = np.int8 ( np.argmax( prob_de, axis=0)-1 )
        label['detectron_soft'] = prob_de
        
        
        # RINGBUFFER THE PREDICTIONS
        label_list.append( copy.deepcopy( label) )

        global_idx_list.append(int( global_idx[b] ))
        if len(label_list) >= window_size:
            label_list = label_list[-window_size:]
            global_idx_list = global_idx_list[-window_size:]
            
            # GET THE FLOW BETWEEN THE FRAMES -> global_idx_list = [10,20,30,40] 
            suc, flow_list = pllo.get_flow( global_idx = global_idx_list)
            # flow_list[0] == latest_frame;  flow_list[-1] oldest_frame
            # CHECK IF THE GLOBAL IDX LIST ALIGNS
            suc *= index_check_labdata(global_idx_list= global_idx_list, 
                                       global_idx_pths=globale_idx_to_image_path, 
                                       sub=10)    
            # CREATE PSEUDO LABEL
            if suc:
                flow_list.reverse()

                st_ = time.time()
                for k in list( ["detectron","fastscnn","detectron_soft"] ):
                    # seg[-1] == latest frame
                    if label_list[0][k].shape[0] != 41:   
                        seg = [ s[k][None] for s in label_list ] # seg[0].shape 1,H,W # -1 is invalid
                    else:
                        seg = [ s[k] for s in label_list ] # 41,H,W
                    # seg[0] == latest_frame
                    seg.reverse()
                    # seg[0] latest frame
                    # flow_list[0] oldest frame
                    pseudo_label = plg.calculate_label(index=None, 
                                                          seg= copy.deepcopy(seg),   
                                                          flow=copy.deepcopy(flow_list) )
                                                        
                    label[k+'_normal_flow'] = pseudo_label
                print("time to create all pseudo labels", time.time()-st_ )   
                pri =True
                counts_flow += 1

        # EVALUATE ALL LABELS
        ret = get_max_acc(
                label = list( label.values()) , 
                gt=target[b].numpy(), 
                names= list( label.keys())[:2] )


        for k in ret.keys():
            if k in acc_dict:
                acc_dict[k] += ret[k]
            else:
                acc_dict[k] = ret[k]
        counts += 1
    
    # LOGGING
    if j % 1 == 0 and j != 0:
        print_acc(acc_dict)
        mini =int((time.time()-st)/60)
        mini_left = int((time.time()-st)/j*(length-j)/60)
        print(f'{j}/{length} Total time elapsed: {mini}min; Projected finish in {mini_left}min')
        plot = label
        plot['img'] = images[b]
        plot['gt'] = target[b].numpy()  
        try:
            plot['img_flow'] = visu.plot_flow(flow= flow_list[1][0])
        except:
            print("No flow avail")
        out_pth = globale_idx_to_image_path[ global_idx_list[-1] ]    
        res = plot_pseudo_labes( plot, jupyter = False, label=True, save_indivdual=False, out_pth = out_pth )
        img = Image.fromarray(res)
        out_pth = globale_idx_to_image_path[ global_idx_list[-1] ].replace("/2/", "/2_detectron/")   

        out_pth = f"/home/jonfrey/tmp/result{j}.png"
        Path(out_pth).parent.mkdir(parents=True, exist_ok=True)
        img.save( out_pth )