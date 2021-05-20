# Some basic setup:
# Setup detectron2 logger
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import os
import sys
import pickle
import sys
sys.path.append(os.path.join(os.getcwd(),"src"))
from visu import Visualizer


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)

with open('cfg/dataset/mappings/coco200_nyu.pkl', 'rb') as handle:
    mappings = pickle.load(handle)
stuff_ids_coco_200_ids = list(_get_builtin_metadata(  "coco_panoptic_separated" )['stuff_dataset_id_to_contiguous_id'].keys())


def map_function( outputs ):
	seg_pan = outputs['panoptic_seg'][0]
	sem_seg = torch.argmax( outputs['sem_seg'] , dim = 0) 
	seg_seg_nyu = sem_seg.clone()
	seg_seg_nyu[:,:] = -1

	for i in range( 53):
		if i == 0:
			m1 = sem_seg == 0
			for instance in outputs['panoptic_seg'][1] :
				ids = instance['id']
				category_id = instance['category_id']
				coco200 = COCO_CATEGORIES[ instance['category_id'] ]['id']
				nyuid = mappings['coco_id_nyu_id'][ coco200-1 ]
				mappings['coco_id_nyu_name'][ coco200-1  ]
				# print( "Mapped COCO", COCO_CATEGORIES[ instance['category_id'] ]['name'], " to NYU ", mappings['coco_id_nyu_name'][ coco200-1  ])
				m2 = seg_pan == ids
				seg_seg_nyu[ m2 * m1 ] = nyuid
		else:
			coco200 = stuff_ids_coco_200_ids[i-1]
			nyuid = mappings['coco_id_nyu_id'][ coco200-1 ]
			seg_seg_nyu[ sem_seg == i ] = nyuid
			# print( "Mapped COCO", _get_builtin_metadata(  "coco_panoptic_separated" )['stuff_classes'][i], " to NYU ", mappings['coco_id_nyu_name'][ coco200-1  ])
	return seg_seg_nyu


im = cv2.imread("/media/scratch2/jonfrey/labdata/2/undistorted_frame001000.png")
outputs = predictor(im)
seg_seg_nyu = map_function( outputs )
visu = Visualizer( p_visu=os.getcwd(), logger=None, epoch=0, store=True, num_classes=41)
visu.plot_segmentation( seg=seg_seg_nyu.cpu().numpy()+1,tag="DONE")

from pathlib import Path
inputs = [str(i) for i in Path("/home/jonfrey/datasets/scannet/scans/").rglob("*jpg") if str(i).find('color') != -1  and int(str(i).split('/')[-1][:-4])% 10 == 0]
outputs = [i.replace('color', 'label_detectron2').replace('.jpg', '.png') for i in inputs]

import imageio

for i, (in_p, out_p) in enumerate( zip( inputs, outputs) ):
	print( i ,"/", len(inputs))
	img = cv2.imread(in_p)
	outputs = predictor(img)
	seg_seg_nyu = map_function( outputs )
	Path( out_p ).parent.mkdir(parents=True, exist_ok=True)
	imageio.imwrite( out_p, (seg_seg_nyu+1).cpu().numpy().astype(np.uint16) )
	print(in_p, out_p)

# /home/jonfrey/datasets/scannet/scans/scene0000_00/label-filt/0.png
# imageio.imread( "/home/jonfrey/datasets/scannet/scans/scene0000_00/label-filt/0.png" )
# label = torch.from_numpy(imageio.imread( "/home/jonfrey/ASL/test.png" ).astype(np.int32)).type(torch.float32)



