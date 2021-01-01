###########################################################################
# Created by: Tramac
# Date: 2019-03-25
# Copyright (c) 2017
# https://github.com/Tramac/Fast-SCNN-pytorch 
# modified by Jonas Frey
# Licensed under Apache 2.0
###########################################################################

from pytorch_lightning.metrics import Metric
import torch
import numpy as np

__all__ = ['IoU', 'PixAcc']
class IoU(Metric):
  def __init__(self, num_classes, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self._num_classes = num_classes
    self.add_state("total_inter", default=torch.tensor(0), dist_reduce_fx="sum")
    self.add_state("total_union", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    # executed individual non-blocking on each thread/GPU

    assert preds.shape == target.shape # 

    inter, union = batch_intersection_union(preds, target, self._num_classes)

    self.total_inter += inter.sum()
    self.total_union += union.sum()

  def compute(self):
    """Gets the current evaluation result.
    Returns
    -------
    metrics : tuple of float
        pixAcc and mIoU
    """
    # synchronizes accross threads

    IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
    mIoU = IoU.mean()
    return mIoU

class PixAcc(Metric):
  def __init__(self, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.add_state("total_correct", default=torch.tensor(0), dist_reduce_fx="sum")
    self.add_state("total_label", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    # executed individual non-blocking on each thread/GPU

    assert preds.shape == target.shape # 
    correct, labeled = batch_pix_accuracy(preds, target)

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
    
    pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
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

    
def batch_intersection_union(predict, target, nclass):
  """mIoU"""
  # inputs are numpy array, output 4D, target 3D
  assert predict.shape == target.shape
  mini = 1
  maxi = nclass
  nbins = nclass
  predict = predict.type(torch.int) + 1
  target = target.type(torch.int) + 1

  predict = predict * (target > 0).type(predict.dtype)
  intersection = predict * (predict == target)
  # areas of intersection and union
  # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
  area_inter, _ = np.histogram(intersection.cpu().numpy(), bins=nbins, range=(mini, maxi))
  area_pred, _ = np.histogram(predict.cpu().numpy(), bins=nbins, range=(mini, maxi))
  area_lab, _ = np.histogram(target.cpu().numpy(), bins=nbins, range=(mini, maxi))
  area_union = area_pred + area_lab - area_inter
  
  assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
  return torch.tensor( area_inter, device=target.device) , torch.tensor( area_union, device=target.device)


def pixelAccuracy(imPred, imLab):
  """
  This function takes the prediction and label of a single image, returns pixel-wise accuracy
  To compute over many images do:
  for i = range(Nimages):
        (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
          pixelAccuracy(imPred[i], imLab[i])
  mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
  """
  # Remove classes from unlabeled pixels in gt image.
  # We should not penalize detections in unlabeled portions of the image.
  pixel_labeled = torch.sum(imLab >= 0)
  pixel_correct = torch.sum((imPred == imLab) * (imLab >= 0))
  pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
  return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
  """
  This function takes the prediction and label of a single image,
  returns intersection and union areas for each class
  To compute over many images do:
  for i in range(Nimages):
      (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
  IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
  """
  # Remove classes from unlabeled pixels in gt image.
  # We should not penalize detections in unlabeled portions of the image.
  imPred = imPred * (imLab >= 0)

  # Compute area intersection:
  intersection = imPred * (imPred == imLab)
  (area_intersection, _) = np.histogram(intersection.cpu().numpy(), bins=numClass, range=(1, numClass))

  # Compute area union:
  (area_pred, _) = np.histogram(imPred.cpu().numpy(), bins=numClass, range=(1, numClass))
  (area_lab, _) = np.histogram(imLab.cpu().numpy(), bins=numClass, range=(1, numClass))
  area_union = area_pred + area_lab - area_intersection
  return (torch.tensor(area_intersection, device=imPred.device) , torch.tensor(area_union, device=imPred.device))


def hist_info(pred, label, num_cls):
  assert pred.shape == label.shape
  k = (label >= 0) & (label < num_cls)
  labeled = torch.sum(k)
  correct = torch.sum((pred[k] == label[k]))
  inp = num_cls * label[k].type(torch.int) + pred[k]
  inp = inp.cpu().numpy()
  return np.bincount(inp, minlength=num_cls ** 2).reshape(num_cls,num_cls), labeled, correct


def compute_score(hist, correct, labeled):
  iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
  # print('right')
  # print(iu)
  mean_IU = np.nanmean(iu)
  mean_IU_no_back = np.nanmean(iu[1:])
  freq = hist.sum(1) / hist.sum()
  freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
  mean_pixel_acc = correct / labeled

  return iu, mean_IU, mean_IU_no_back, mean_pixel_acc