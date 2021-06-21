import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import pickle

import numpy as np
from PIL import Image


import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
from visu import Visualizer

#data
# img_pth = "/home/jonfrey/Datasets/scannet/scans/scene0000_00/color/738.jpg"
with open('/home/jonfrey/ASL/cfg/dataset/mappings/coco2017_nyu.pkl', 'rb') as handle:
    mappings = pickle.load(handle)
#visu
# visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=True, num_classes=41)

#model
# device = 'cpu'
# cfg = get_cfg()
# cfg['DEVICE'] = device
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# model = DefaultPredictor(cfg)
# model.model.to(device)



# img = np.array( Image.open(img_pth) )
# outputs = model( img )

# _,H,W = outputs['instances'][0].pred_masks.shape

# label = torch.zeros( (H,W) ,dtype=torch.long )
# for i in range(len( outputs['instances'])):
#     inst = outputs['instances'][i]
#     if inst.scores[0] > 0.5:
#         coco200 = int( inst.pred_classes ) 
#         label[ inst.pred_masks[0] ] = mappings['coco2017_id_nyu_id'][coco200] + 1
#         print( mappings["coco2017_id_name"][str(coco200)], mappings['coco2017_id_nyu_name'][coco200], inst.scores[0])
        
# visu.plot_detectron( img = img, label = label, tag='test', jupyter=True, store=False,alpha=0.3)

import imageio
import cv2 as cv
from scipy.interpolate import griddata
import scipy.ndimage as nd
from skimage.segmentation import slic

def txt_to_camera_info(cam_p, img_p):
    data = np.loadtxt(cam_p)
    img = imageio.imread(img_p)
    return data[:3, :3], (img.shape[0],img.shape[1])

class SuperpixelDepth():
    def __init__(self, scannet_scene_dir= "/home/jonfrey/Datasets/scannet/scans/scene0000_00" ):
        
        self.K_image, size_image = txt_to_camera_info(f"{scannet_scene_dir}/intrinsic/intrinsic_color.txt", 
                                                 f"{scannet_scene_dir}/color/0.jpg")
        self.K_depth, size_depth = txt_to_camera_info(f"{scannet_scene_dir}/intrinsic/intrinsic_depth.txt", 
                                                 f"{scannet_scene_dir}/depth/0.png")
        # maps from image to depth
        self.map1, self.map2 = cv.initUndistortRectifyMap(
            self.K_depth,
            np.array([0,0,0,0]),
            np.eye(3),
            self.K_image,
            size_image[::-1], # (W,H)
            cv.CV_32FC1)
        
        self.grid_x, self.grid_y = np.mgrid[0:size_image[0], 0:size_image[1]]
        
        points = np.stack([self.grid_x, self.grid_y],axis=2).reshape((-1,2))
        self.h_p = np.ones( (points.shape[0],3) )
        self.h_p[:,:2] = points 
        
    def get( self, depth_p, n_segments = 100, min_depth= 0.3, max_depth=3 , cap = 3,tag="", visu=None, plot=False, jupyter=False):
        
        depth = imageio.imread( depth_p )
        img = imageio.imread( depth_p.replace("depth","color")[:-4]+'.jpg' )

        depth_new = cv.remap( depth,
                     self.map1,
                     self.map2,
                     interpolation=cv.INTER_NEAREST,
                     borderMode=cv.BORDER_CONSTANT,
                     borderValue=0)

        values =depth_new.flatten()
        m = np.logical_and( values > min_depth * 1000, values < max_depth * 1000)
        depth_filled = griddata(self.h_p[:,:2] [m,:], values[m], (self.grid_x, self.grid_y), method='nearest')

        
        pcd = ((np.linalg.inv( self.K_image ) @ self.h_p.T) * ( depth_filled.reshape(-1) / 1000 )).T.reshape( (*depth_filled.shape,3))
        pdc_clamp = (np.clip( pcd, a_min= -cap, a_max = cap) + cap )/(2*cap)
        
        segments_slic = slic( pdc_clamp*256, n_segments=n_segments, compactness=10, sigma=4,start_label=0)

        
        if plot:
            img = visu.plot_depth(depth_filled/1000, vmin=min_depth, vmax= max_depth, tag=tag+"_depth_filled", jupyter=jupyter)
            segments_slic_plot = np.mod( segments_slic, np.full( segments_slic.shape,40 ))
            res = visu.plot_detectron( img = img, label = segments_slic_plot, tag=tag+"_seg_depth", jupyter=jupyter,alpha=0.1)
        
        return segments_slic
        
from skimage.segmentation import slic
from skimage.util import img_as_float
from utils_asl import LabelLoaderAuto

class Oracle():
    def __init__(self, jupyter=False, scene="scene0000_00"):
        self.jupyter = jupyter
        self.store = False
        
        with open('/home/jonfrey/ASL/cfg/dataset/mappings/coco2017_nyu.pkl', 'rb') as handle:
            self.mappings = pickle.load(handle)
        self.lla = LabelLoaderAuto(root_scannet="/home/jonfrey/Datasets/scannet", confidence=0.9)
        #model
        device ="cpu"# 'cuda:0'
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.DEVICE = device
        model = DefaultPredictor(cfg)
        print(cfg)
        # model.model.to(device)

        #visu
        # visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=True, num_classes=41)
        
        self.model = model
        self.visu = None #visu
        
        # depth
        scannet_scene_dir= f"/home/jonfrey/Datasets/scannet/scans/{scene}"
        self.spd = SuperpixelDepth( scannet_scene_dir )
        
    def inference(self, img_pth, tag="", plot=True):
        img = np.array( Image.open(img_pth) )
        outputs = self.model( img )

        _,H,W = outputs['instances'][0].pred_masks.shape

        label = torch.zeros( (H,W) ,dtype=torch.long )
        if len( outputs['instances']) == 0:
            return label
        for i in range(len( outputs['instances'])):
            inst = outputs['instances'][i]
            if inst.scores[0] > 0.5:
                coco200 = int( inst.pred_classes ) 
                label[ inst.pred_masks[0] ] = mappings['coco2017_id_nyu_id'][coco200] + 1
        if plot:
            self.visu.plot_detectron( img = img, label = label, tag=tag+"_mask_rcnn", jupyter=self.jupyter,alpha=0.3)
        return label
        
    def get_superpixel_image(self, img_pth, n_segments=40, tag="",  plot=True):
        img = np.array( Image.open( img_pth) )
        segments_slic = slic(img_as_float(img), n_segments=n_segments, compactness=20, sigma=5,
                             start_label=0)
        if plot:
            segments_slic_plot = np.mod( segments_slic, np.full(segments_slic.shape,40 ))
            self.visu.plot_detectron( img = img, label = segments_slic_plot, tag=tag+"_seg_image", jupyter=self.jupyter,alpha=0.3)
        
        return segments_slic
    
    def get_superpixel_depth(self, depth_pth, n_segments=40, tag="",  plot=True):
        return self.spd.get( depth_pth , n_segments=n_segments, tag=tag, plot=plot, visu=self.visu, jupyter=self.jupyter)   
        
        
    def combined(self, img_pth, depth_pth, n_segments=100, tag="",  plot=False ):
        d_seg = self.get_superpixel_depth(depth_pth, n_segments=n_segments, tag=tag,plot=plot)
        i_seg = self.get_superpixel_image(img_pth, n_segments=n_segments, tag=tag,plot=plot)
        
        shift_n = int(n_segments).bit_length()
        shift_mult = 2 ** shift_n
        
        com_seg = d_seg + (i_seg * shift_mult )
        
        res, index, inverse = np.unique( com_seg, return_index=True, return_inverse=True)
        if plot:
            img = np.array( Image.open( img_pth) )
            plot_inverse = np.mod( inverse, np.full( inverse.shape,40 ))
            self.visu.plot_detectron( img = img, label = plot_inverse.reshape( d_seg.shape ), tag=tag+"_seg_combined", jupyter=self.jupyter, store=False,alpha=0.3)
        return inverse.reshape( d_seg.shape )
    
    def oracle( self, img_pth, depth_pth, pred_pth, n_segments= 100,  tag="", plot=False):
        seg = self.combined(img_pth, depth_pth, n_segments=100, tag=tag, plot=plot )
        
        
        pred, _ = self.lla.get(pred_pth) #imageio.imread( pred_pth )
        mask_rcnn = self.inference( img_pth, tag=tag, plot=plot)
        
        out = np.zeros( pred.shape )
        for s in range(seg.max()):
            m = seg == s
            val, counts = np.unique( pred[m], return_counts=True)
            if val[np.argmax( counts )] != 0:
                out[m] = val[np.argmax( counts )]
            else:
                val, counts = np.unique( mask_rcnn[m], return_counts=True)
                out[m] = val[np.argmax( counts )]
                
        if plot:
            img = np.array( Image.open( img_pth) )
            
            print("PRED", pred.shape, pred.dtype, pred.max())
            self.visu.plot_detectron( img=img, label=pred, tag=tag+ '_network', jupyter=self.jupyter, alpha=0.6)
            
            print("OUT", out.shape, out.dtype, out.max())
            self.visu.plot_detectron( img = img, label =out, tag=tag+ '_final', jupyter=self.jupyter, alpha=0.6)
        return out


def process_frames( ls ):
    print("INIT ORACLE ", ls[0])
    oracle = Oracle()
    print("INIT ORACLE DONE", ls[0])
    
    print("OK")
    idfs = "labels_pretrain25k_correct_mapping_reprojected"
    idfs_out = "labels_pcmr_oracle"
    for j, p in enumerate(ls) :
        print(f"{j}/{len(ls)} ", p)
        scene = p.split('/')[-3]
        idx = int ( p.split('/')[-1][:-4])

        img_p = p
        depth_p = p.replace("color","depth").replace(".jpg",".png")
        pred_p = f"/home/jonfrey/Datasets/labels_generated/{idfs}/scans/{scene}/{idfs}/{idx}.png"
        out_p = f"/home/jonfrey/Datasets/labels_generated/{idfs_out}/scans/{scene}/{idfs_out}/{idx}.png"
        out = oracle.oracle( img_p, depth_p, pred_p, plot=False)
        print("Finished output ", out_p)
        Path(out_p).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_p, np.uint8( out ) )

from pathlib import Path 
import numpy as np 
  

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--number")
  parser.add_argument("--number_total")
  parser.add_argument("--scene")
  args = parser.parse_args()
  
  scene = args.scene 
  ls = [str(s) for s in Path( f"/home/jonfrey/Datasets/scannet/scans/{scene}/color/" ).rglob("*.jpg") if int((str(s).split('/')[-1][:-4]))% 10 == 0   ]##
  ls.sort(key= lambda x: int(x.split('/')[-1][:-4]))
  oracle = Oracle( scene = scene)
  ls [:4]
  
  tasks = [t.tolist() for t in np.array_split(np.array(ls), int( args.number_total) ) ]
  
  process_frames( tasks[int( args.number ) ] )