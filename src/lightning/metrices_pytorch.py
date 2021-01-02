from pytorch_lightning.metrics import Metric
import torch
import numpy as np
from pytorch_lightning.metrics.functional.classification import stat_scores_multiple_classes
import warnings

__all__ = ['IoUTorch', 'PixAccTorch', 'meanIoUTorch', 'meanIoUTorchCorrect']

EPS = torch.finfo(torch.float).eps
class IoUTorch(Metric):
  """Simple IoU implemented according to SCNN:
  ignores all targets labeled with -1
  IoU = TP / (TP + FP + FN)
  all pixels are simply summed up over the full batch
  THIS ONLY WORKS FOR BATCHSIZE OF 1 correctly in my opinion !!!

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
    with warnings.catch_warnings():
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


class meanIoUTorch(Metric):
  """[summary]
  To assess performance, we rely on the standard Jaccard Index, commonly known as the 
  PASCAL VOC intersection-over-union metric IoU = TP ⁄ (TP+FP+FN) [1], 
  where TP, FP, and FN are the numbers of true positive, false positive, and false negative pixels, respectively, 
  determined over the whole test set. 
  Owing to the two semantic granularities, i.e. classes and categories, we report two separate mean performance scores: 
  IoUcategory and IoUclass. 
  In either case, pixels labeled as void do not contribute to the score.

  Parameters
  ----------
  pred : [torch.tensor]
      BSxD1xD2xD3 , predict class for each pixel. No need to predict the -1 class! element of 0-(num_classes-1)
  target : [torch.tensor]
      BSxD1xD2xD3	, -1 for the invalid pixels that should not induce an error! element of -1-(num_classes-1)
  num_classes : [int]
      invalid class does not count as a class. So lets say targets takes values -1 - 19 then you have 20 classes
  """
  def __init__(self, num_classes, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self._num_classes = num_classes
    self.add_state("iou", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
    self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, pred: torch.Tensor, target: torch.Tensor):
    # executed individual non-blocking on each thread/GPU
    BS = pred.shape[0]
    # add 1 so the index ranges from 0 to NUM_CLASSES
    pred = pred.type(torch.int) + 1
    target = target.type(torch.int) + 1
    # NOW class=0 should not induce a loss

    # Set pixels that are predicted but no label is available to 0. These pixels dont enduce a loss. 
    # Neither does the IoU  of class 0 nor do these pixels count to the UNION for the other classes if predicted wrong. 
    # pred = pred * (target > 0).type(pred.dtype) 
    iou = 0.0

    # we have to do this calculation for each image. 
    for b in range( BS ):
      with warnings.catch_warnings():

        TPS, FPS, TNS, FNS, _ = stat_scores_multiple_classes(pred[b], target[b], self._num_classes+1)
      IoU = TPS[1:] / (TPS[1:] + FPS[1:] + FNS[1:])
      mIoU = (IoU[torch.isnan(IoU)==False]).mean()
      iou += mIoU
      
    self.iou += iou
    self.total_batches += int(BS)
  
  def compute(self):
    """Gets the current evaluation result.
    Returns
    -------
    metrics : tuple of float
        pixAcc and mIoU
    """
    # synchronizes accross threads

    mIoU = self.iou / self.total_batches
    return mIoU

class meanIoUTorchCorrect(Metric):
  """
  This metric reproduces the reported results on the GitHub page.

  IoU = TP / (TP + FP + FN)

  Parameters
  ----------
  Metric : [type]
      [description]
  """
  def __init__(self, num_classes, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self._num_classes = num_classes
    self.add_state("iou_sum", default=torch.tensor( 0.0 ), dist_reduce_fx="sum")
    self.add_state("batches", default=torch.tensor( 0.0 ), dist_reduce_fx="sum")

  def update(self, pred: torch.Tensor, target: torch.Tensor):
    # executed individual non-blocking on each thread/GPU
    assert pred.shape == target.shape
    BS = pred.shape[0]

    p = pred.clone().type(torch.int) + 1
    t = target.clone().type(torch.int) + 1
    
    p[t == 0] = 0
    for b in range(BS):
      ious = torch.zeros( (self._num_classes) ) -1
      for c in range(0, self._num_classes) :
          # c+1 to ignore the first index
          t_c = t[b] == c+1 
          if torch.sum(t_c) == 0:
              continue
          p_c = p[b] == c+1
          intersection = torch.logical_and(p_c, t_c).sum()
          union = torch.logical_or(p_c, t_c).sum()
          if union != 0:
              ious[c] = intersection / union
      self.iou_sum += (ious[ious!=-1]).sum()/((ious!=-1).sum())
      self.batches += 1

  def compute(self):
    """Gets the current evaluation result.
    Returns
    -------
    metrics : tuple of float
        pixAcc and mIoU
    """
    # synchronizes accross threads
    return self.iou_sum / (self.batches+ EPS)

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
