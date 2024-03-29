import cv2
import numpy as np
import imageio
from glob import glob
from scipy import ndimage
from torchvision.transforms.functional import resize
from torchvision import transforms as tf
import torch
from PIL import Image
import os
__all__ = ['readFlowKITTI', 'readDepth', 'readSegmentation','readImage',
	'getPathsDepth', 'getPathsFlow', 'getPathsSegmentation']


H,W = 640, 1280
def readImage(filename, H=640, W=1280, scale=True):
	_crop_center = tf.CenterCrop((H,W))
	img = Image.open(filename)    
	if scale:
			img = _crop_center( img )
	return np.array( img )

def readFlowKITTI(filename,H=960 ,W=1280):
	flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
	flow = flow[:,:,::-1].astype(np.float32)
	flow, valid = flow[:, :, :2], flow[:, :, 2]
	flow = (flow - 2**15) / 64.0
	
	flow = torch.from_numpy( flow ).permute(2,0,1)
	valid = torch.from_numpy( valid[:,:,None] ).permute(2,0,1)
	# h_scale = H/flow.shape[1]
	# w_scale = W/flow.shape[2]
	
	# flow = resize( flow, (H,W), interpolation= Image.BILINEAR ) 
	# flow[0,:,:] *= h_scale 
	# flow[1,:,:] *= w_scale

	# valid = resize( valid, (H,W), interpolation= Image.NEAREST ) 
	flow = flow.permute(1,2,0).numpy()
	valid = valid.permute(1,2,0).numpy()[:,:,0]

	return flow, valid

def readDepth(filename, H=960 ,W=1280): 
	im = imageio.imread(filename)
	im = im.astype(np.float32)
	im = im / 1000
	im = ndimage.zoom(im, (H/im.shape[0],W/im.shape[1]) , order=1)
	return im

def readSegmentation(filename):
	im = imageio.imread(filename)
	pred = im[:,:,0].astype(np.int32)
	target = im[:,:,1].astype(np.int32)
	valid = im[:,:,2].astype(bool)
	pred -= 1
	target -= 1
	return pred, target, valid

def getPathsDepth( base = "/home/jonfrey/datasets/scannet/scans/scene0000_00/depth_estimate", sub = 10):
	depth_pths = [str(p) for p in glob( base+'/**/*.png', recursive=True ) if str(p).find('depth_estimate') != -1 and str(p).find('preview') == -1 ]
	fun = (lambda x:
		x.split('/')[-3][-7:] + '_'+ 
		str( "0"*(6-len( x.split('/')[-1][:-4]))) + 
		x.split('/')[-1][:-4])
	depth_pths.sort(key=fun)
	depth_pths = [d for d in depth_pths if int( d.split('/')[-1][:-4]) % sub == 0]
	return depth_pths

def getPathsFlow( key= 'flow_sub_1', base = "/home/jonfrey/results/scannet_eval/run_24h_train_1gpu/scannet"):
	flow_pths = [str(p) for p in glob( base+'/**/*.png', recursive=True ) if str(p).find(key) != -1]
	fun = (lambda x:
		x.split('/')[-3]+ 
		'0'*(8-len((x.split('/')[-1]).split('_')[-1]))+
		(x.split('/')[-1]).split('_')[-1])
	flow_pths.sort(key=fun)
	return flow_pths

def getPathsSegmentation(base="/home/jonfrey/results/scannet_eval/run_24h_train_1gpu", key='segmentation_estimate'):
	segmentation_pths = [str(p) for p in glob( base+'/**/*.png', recursive=True ) if str(p).find(key) != -1]
	fun = lambda x : x.split('/')[-3][-7:] + '_'+ str( "0"*(6-len( x.split('/')[-1][:-4]))) + x.split('/')[-1][:-4]  
	segmentation_pths.sort(key=fun)
	return segmentation_pths