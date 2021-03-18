import sys
import os
os.chdir("/home/jonfrey/ASL")
sys.path.append("""/home/jonfrey/ASL/src/""")
sys.path.append("""/home/jonfrey/ASL/src/pseudo_label""")

import numpy as np
import imageio
import time
from torchvision import transforms as tf
import copy
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot
from PIL import ImageDraw, ImageFont, Image
import yaml
import coloredlogs
coloredlogs.install()
import argparse
from pathlib import Path
import torch
from torchvision.utils import make_grid
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import cv2

# LOAD MODULES
from pseudo_label.yolo import YoloHelper
from pseudo_label.deeplab import DeeplabHelper
from pseudo_label.fast_scnn import FastSCNNHelper
from visu import Visualizer
from datasets_asl import get_dataset
from utils_asl import load_yaml
from pseudo_label import *
from pseudo_label import readSegmentation

def get_max_acc(label, gt , names = ['deeplab', 'yolo', 'fastscnn', 'gt'] ):
    m = gt != -1
    acc_indi = {}
    for l,n in zip(label,names):
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


class PseudoLabelLoaderOnline():
    def __init__(self, base_flow, image_paths, h=960, w=1280, sub=10):
        self.image_paths = image_paths
        self.sub = sub
        self.base_flow = base_flow
        self.H, self.W= h,w 
        
    def get_flow(self, global_idx):
        flow = []
        for idx in global_idx:
            fp = os.path.join( 
                self.base_flow, 
                dataset_test.image_pths[ idx].split('/')[-3], 
                f'flow_sub_{self.sub}',
                dataset_test.image_pths[ idx].split('/')[-1][:-4]+'.png' )
            try:
                flow.append( readFlowKITTI( fp, H=self.H ,W=self.W))
            except:
                return False, False 
        flow.reverse()
        return True, flow
    

class PseudoLabelGenerator():
    def __init__(self, base_path, sub=10, confidence='equal', 
            flow_mode='sequential', H=640, W=1280, 
            nc=40, window_size=10,
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
        self._window_size = window_size
        # Passed externally
        self._visu = visu
    
    def calculate_label(self, index=None, seg=[],flow=[], image= None ):
        if not index is None:
            seg_forwarded= self._forward_index(index) #return H,W,C
        else:
            seg_forwarded = self._forward_index(index, seg, flow)
        
        # -1 39 -> 0 -> 40 We assume that the network is also able to predict the invalid class
        # In reality this is not the case but this way we can use the ground truth labels for testing
        if seg_forwarded[0].shape[2] == 1:
            for i in range(len( seg_forwarded) ):
                seg_forwarded[i] += 1 
        
        confidence_values_list = self._get_confidence_values(seq_length= len(seg_forwarded))
        if seg_forwarded[0].shape[2] == 1:
            one_hot_acc = np.zeros( (*seg_forwarded[0].shape,self._nc+1), dtype=np.float32) # H,W,C
            for conf, seg in zip(confidence_values_list, seg_forwarded):    
                one_hot_acc += (np.eye(self._nc+1)[seg.astype(np.int32)]).astype(np.float32) * conf
            invalid_labels = np.sum( one_hot_acc[:,:,1:],axis=2 ) == 0
        
        
        
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
        
        pre_fusion_function should be used to integrate the depth measurments 
        to the semseg before forward projection !

        seg_forwarded[0] -> oldest_frame
        seg_forwarded[len(seg_forwarded)] -> latest_frame not forwarded

        """
        
        if len( seg[0].shape ) == 3 and seg[0].shape[0] != 1:
            soft = True
        else:
            soft = False
            

        seg_forwarded = []
        for j in range(len( seg )):
            seg[j] = np.moveaxis( seg[j], [0,1,2], [2,0,1] ) #C,H,W -> H,W,C
        
        for i in range(0,len(seg)-1):
            i = len(seg)-1-i
            seg_forwarded.append( seg[i].astype(np.float32) )

            
            # CREATE FLOW MAP
            if i != 0:
                f = flow[i][0]
            else:
                f = np.zeros(flow[i][0].shape, dtype=np.float32)
            h_, w_ = np.mgrid[0:self._H, 0:self._W].astype(np.float32)
            h_ -= f[:,:,1]
            w_ -= f[:,:,0]
            
            # FORWARD ALL PAST FRAMES
            j = 0
            for s in seg_forwarded :
                if soft:
                    s = cv2.remap( s, w_, h_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                else:
                    s = cv2.remap( s[None], w_, h_, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)[None]
                seg_forwarded[j] = s
                j += 1
        
        seg_forwarded.append( seg[0].astype(np.float32) )
        print( "shapes", [s.shape for s in seg_forwarded])
        return seg_forwarded


def plot_pseudo_labes( res ):
    key_list = list(res.keys())
    s = int( len( key_list ) ** 0.5 )
    if len( key_list ) - s*s != 0:
        s +=1
    ba = torch.zeros( (int(s*s),3, *res[key_list[0]].shape), dtype= torch.float32 )
    for i in range( len( key_list ) ):
        k = key_list[i]
        if k != 'img':
            img = visu.plot_segmentation( seg=res[k]+1 )
        else:
            img = res[k]
        
        img = Image.fromarray(img)
        img = img.convert("RGBA")
        d = ImageDraw.Draw(img)
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSansMono-Bold.ttf", 50)   
        d.rectangle(((550, 500), (1400, 600)), fill=(254,10,10))
        d.text((600,530), k , font=fnt, fill=(254,254,254,254))
        img =  img.convert("RGB")
        img = np.array( img )
        ba[i,:] = torch.from_numpy( img[:,:,:3] ).permute(2,0,1)
    
    grid_ba = make_grid( ba ,nrow = s ,padding = 2,
      scale_each = False, pad_value = -1)
    visu.plot_image(img = grid_ba +1 , jupyter=True)

def print_acc(acc_dict):
    avg = {}
    for k in acc_dict.keys():
        if k.find('flow') != -1:
            avg  = acc_dict[k] / counts_flow
        else:
            avg = acc_dict[k] / counts

    print(avg)
def valid_sequential_indi(global_idx_list, globale_idx_to_image_path ):
  v = global_idx_list[0]
  prev = int(globale_idx_to_image_path[0].split('/')[-1][:-4])
  for g in global_idx_list[1:]:
      g = int(globale_idx_to_image_path[g].split('/')[-1][:-4])
      if g != prev + sub:
          suc = False
          break
      prev = g
  return True

if __name__ == '__main__':
  DEVICE = 'cuda:1'
  visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=False, num_classes=41)

  yh = YoloHelper()
  dlh = DeeplabHelper(device="cuda:1")
  fsh = FastSCNNHelper(device='cuda:1')


  env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
  env_cfg = load_yaml(env_cfg_path)
  eval_cfg = load_yaml("/home/jonfrey/ASL/cfg/eval/eval.yml")

  # SETUP DATALOADER
  dataset_test = get_dataset(
      **eval_cfg['dataset'],
      env = env_cfg,
      output_trafo = None,
      )
  dataloader_test = torch.utils.data.DataLoader(dataset_test,
      shuffle = False,
      num_workers = 0,
      pin_memory = eval_cfg['loader']['pin_memory'],
      batch_size = 1, 
      drop_last = True)

  st = time.time()
  globale_idx_to_image_path = dataset_test.image_pths

  PseudoLabelLoaderOnline(base_flow = '/home/jonfrey/results/scannet_pseudo_label/scannet',
                          image_paths = dataset_test.image_pths,
                          sub=10)

  st = time.time()
  counts = 0
  counts_flow = 0
  length = len(dataloader_test)
  segmentation_list_fastscnn = []
  global_idx_list = []

  plot = False
  # PARAMS LABEL GENERATION
  weights = [2,0.5,1]
  weights_temperature = [1,0.5,1]
  window_size_2 = 3
  window_size = 6
  sub = 10
  acc_dict = {}

  plg = PseudoLabelGenerator(base_path='/home/jonfrey/results/scannet_pseudo_label/scannet', 
                            visu=visu,
                            window_size=window_size,
                            visu_active=plot)

  plg_linear_decay = PseudoLabelGenerator(base_path='/home/jonfrey/results/scannet_pseudo_label/scannet', 
                            visu=visu,
                            window_size=window_size,
                            visu_active=plot,
                            confidence = 'linear')


  plg_ws_2 = PseudoLabelGenerator(base_path='/home/jonfrey/results/scannet_pseudo_label/scannet', 
                            visu=visu,
                            window_size=window_size_2,
                            visu_active=plot)

  plg_super = PseudoLabelGenerator(base_path='/home/jonfrey/results/scannet_pseudo_label/scannet', 
                            visu=visu,
                            window_size=window_size,
                            visu_active=plot)

  pllo = PseudoLabelLoaderOnline(base_flow = '/home/jonfrey/results/scannet_pseudo_label/scannet',
                          image_paths = dataset_test.image_pths,
                          sub=10)


  down = tf.Resize((320,640))
  up = tf.Resize((640,1280))
  label_list = []


  for j, batch in enumerate( dataloader_test ):
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
          prob_dl = dlh.get_label_prob( images[b] )
          prob_yolo = yh.get_label_prob( images[b] )
          
          inp = down(torch.from_numpy( images[b]).permute(2,0,1)).permute(1,2,0).numpy()
          prob_fastscnn = fsh.get_label_prob( inp )
          prob_fastscnn = up ( torch.from_numpy( prob_fastscnn[None]) )[0].numpy()
          
          label['dl'] = np.argmax( prob_dl[:] , axis=0)-1
          label['yolo'] = np.argmax( prob_yolo[:] , axis=0)-1
          label['fastscnn'] = np.argmax( prob_fastscnn[:] , axis=0)-1
          
          prob_sum = prob_dl + prob_yolo + prob_fastscnn
          label['sum'] = np.argmax( prob_sum[1:] , axis=0)
          
          prob_weighted_sum = weights[0] * prob_dl + weights[1] * prob_yolo + weights[2] * prob_fastscnn
          label['weighted_sum'] = np.argmax( prob_weighted_sum[1:] , axis=0)
          
          
          prob_weighted_sum_temperature = weights_temperature [0] * prob_dl + weights_temperature[1] * prob_yolo + weights_temperature[2] * prob_fastscnn
          label['weighted_sum_temperature'] = np.argmax( prob_weighted_sum_temperature [1:] , axis=0)

          print("time to get predictions", time.time()-st_ )
          
          
          # RINGBUFFER THE PREDICTIONS
          label_list.append( copy.deepcopy( label) )
          global_idx_list.append(int( global_idx[b] ))
          if len(label_list) > window_size:
              label_list = label_list[-window_size:]
              global_idx_list = global_idx_list[-window_size:]
              # GET THE FLOW BETWEEN THE FRAMES
              suc, flow_list = pllo.get_flow( global_idx = global_idx_list)
              
              # CHECK IF THE GLOBAL IDX LIST ALIGNS
              suc_2 = valid_sequential_indi( global_idx_list, dataset_test.image_pths )
              suc = suc and suc_2
                  
              # CREATE PSEUDO LABEL
              if suc:
                  st_ = time.time()
                  
                  for k in list( ["weighted_sum",'dl',"fastscnn"] ): 
                      if len(s[k][0].shape) != 3:
                        seg = [ s[k][None] for s in label_list ]
                      else:
                        seg = [ s[k][None] for s in label_list ]
                       
                      seg.reverse()
                      _, pseudo_label, _ = plg.calculate_label(
                          index=None, 
                          seg= seg, 
                          flow= flow_list)
                      label[k+'_normal_flow'] = pseudo_label
                      
                      _, pseudo_2, _ = plg_ws_2.calculate_label(
                          index=None, 
                          seg= seg[:window_size_2], 
                          flow= flow_list[:window_size_2])
                      
                      label[k+'_w2_flow'] = pseudo_2
                      
                      _, pseudo_linear, _ = plg_ws_2.calculate_label(
                          index=None, 
                          seg= seg, 
                          flow= flow_list)
                      
                      label[k+'_pseudo_linear_flow'] = pseudo_linear
                      
                  print("time to create all pseudo labels", time.time()-st_ )   
                  pri =True
                  counts_flow += 1


          # EVALUATE ALL LABELS
          ret = get_max_acc(
                  label = list( label.values()) , 
                  gt=target[b].numpy(), 
                  names= list( label.keys()))

          for k in ret.keys():
              if k in acc_dict:
                  acc_dict[k] += ret[k]
              else:
                  acc_dict[k] = ret[k]
          counts += 1
      
      # LOGGING
      if j % 30 == 0 and j != 0:
          print_acc(acc_dict)
          mini =int((time.time()-st)/60)
          mini_left = int((time.time()-st)/j*(length-j)/60)
          print(f'{j}/{length} total time elapsed: {mini}min; time left {mini_left}min')
          plot = label
          plot['img'] = images[b]
          plot['gt'] = target[b].numpy()  
          plot_pseudo_labes( plot )
          
      if j > 20: 
          break