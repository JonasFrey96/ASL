import torch
import cv2
import numpy as np
from PIL import Image

class PseudoLabelGenerator():
	def __init__(self,flow_format='sequential', optimization='simple', confidence='linear', seq_length=5):
		"""				
		flow_format :  
			'centering':  flow is calculated form frame i to N
			'sequential': flow is calculated from frame i -> i+1

		confidence:
			'equal': perfect optical flow -> all project labels are equally good
			'linear': linear rate -> per frame
			'exponential': exponential rate -> per frame 
		"""
		self._confidence_values = self._get_confidence_values( seq_length, confidence) 
		self._flow_format = flow_format

	


	def input(self, segmentations, flows, depths):
		"""
		segmentations: list( Tensors CxHxW ) of len N  (either one hot encoded or softmax output)
		flows: list( Tensors CxHxW ) of len N-1 
		depths: list( Tensors CxHxW ) of len N

		"""		

		new_segmentations = self.project_segmentation_to_reference( segmentations, flows)



	def project_segmentation_to_reference(self, segmentations, flows):
		if self._flow_format == 'centering':
			results_seg = []
			for s,f in zip(segmentations, flows):
				undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
				display(Image.fromarray(undistorted_img))
				results_seg.append(res)

		else:
			raise Exception
		return new_segmentation 


	def _get_confidence_values(self, seq_length, confidence ):
		"""
		returns list of length seq_lengt with values
		"""
		if confidence == 'linear':
			ret = []
			lin_rate = 0.1
			s = 0
			for i in range(seq_length):
				res = 1 - lin_rate* (seq_length-i)
				if res < 0: 
					res = 0
				s += ret
				ret.append(res)
			return [r/s for r in ret]

		elif confidence == 'equal':
			return [1/float(seq_length)]*int(seq_length])		
		
		elif confidence == 'exponential':
			ret = []
			exp_rate = 0.8
			s = 0
			for i in range(seq_length):
				res = exp_rate**(seq_length-i)
				if res < 0: 
					res = 0
				s += ret
				ret.append(res)
			return [r/s for r in ret]



idx = "00"
img_1 =f"/home/jonfrey/datasets/kitti/training/image_2/0000{idx}_10.png"
img_2 =f"/home/jonfrey/datasets/kitti/training/image_2/0000{idx}_11.png"
