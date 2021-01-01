from pytorch_lightning.metrics import Metric
import torch
import numpy as np
from pytorch_lightning.metrics.functional.classification import stat_scores_multiple_classes

__all__ = ['IoUTorch', 'PixAccTorch']

EPS = torch.finfo(torch.float).eps
class IoUTorch(Metric):
	"""Simple IoU implemented according to SCNN:
	ignores all targets labeled with -1
	IoU = TP / (TP + FP + FN)
	all pixels are simply summed up over the full batch

	Parameters
	----------
	Metric : [type]
			[description]
	"""
	def __init__(self, num_classes, dist_sync_on_step=False):
		super().__init__(dist_sync_on_step=dist_sync_on_step)
		self._num_classes = num_classes
		self.add_state("total_inter", default=torch.tensor(0), dist_reduce_fx="sum")
		self.add_state("total_union", default=torch.tensor(0), dist_reduce_fx="sum")

	def update(self, pred: torch.Tensor, target: torch.Tensor):
		# executed individual non-blocking on each thread/GPU
		assert pred.shape == target.shape

		# add 1 so the index ranges from 0 to NUM_CLASSES
		pred = pred.type(torch.int) + 1
		target = target.type(torch.int) + 1
		# NOW class=0 should not induce a loss
		
		# Set pixels that are predicted but no label is available to 0. These pixels dont enduce a loss. 
		# Neither does the IoU  of class 0 nor do these pixels count to the UNION for the other classes if predicted wrong. 
		pred = pred * (target > 0).type(pred.dtype) 
		# we have to do this calculation for each image. 
		TPS, FPS, TNS, FNS, _ = stat_scores_multiple_classes(pred, target, self._num_classes+1)

		self.total_inter += (TPS[1:]).sum().type(torch.int)
		self.total_union += (TPS[1:] + FPS[1:] + FNS[1:]).sum().type(torch.int)
	
	def compute(self):
		"""Gets the current evaluation result.
		Returns
		-------
		metrics : tuple of float
				pixAcc and mIoU
		"""
		# synchronizes accross threads

		IoU = 1.0 * self.total_inter / (EPS + self.total_union)
		mIoU = IoU.mean()
		return mIoU


class PixAccTorch(Metric):
	def __init__(self, dist_sync_on_step=False):
		super().__init__(dist_sync_on_step=dist_sync_on_step)
		self.add_state("total_correct", default=torch.tensor(0), dist_reduce_fx="sum")
		self.add_state("total_label", default=torch.tensor(0), dist_reduce_fx="sum")

	def update(self, pred: torch.Tensor, target: torch.Tensor):
		# executed individual non-blocking on each thread/GPU

		assert pred.shape == target.shape # 
		correct, labeled = batch_pix_accuracy(pred, target)

		self.total_correct += correct
		self.total_label += labeled

	def compute(self):
		"""Gets the current evaluation result.
		Returns
		-------
		metrics : tuple of float
				pixAcc and mIoU
		"""
		# synchronizes accross threads
		
		pixAcc = 1.0 * self.total_correct / (EPS + self.total_label)
		return pixAcc

def batch_pix_accuracy(predict, target):
	"""PixAcc"""
	# inputs are numpy array, output 4D, target 3D
	assert predict.shape == target.shape
	predict = predict.type(torch.int) + 1
	target = target.type(torch.int) + 1

	pixel_labeled = torch.sum(target > 0)
	pixel_correct = torch.sum((predict == target) * (target > 0))
	assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
	return pixel_correct, pixel_labeled
