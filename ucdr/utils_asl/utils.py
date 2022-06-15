import torch
from pytorch_lightning.metrics.functional.classification import (
  stat_scores_multiple_classes,
)

__all__ = ["iIoU_class", "IoU_SCNN", "IoU_class"]


def iIoU_class(pred, target, num_classes, verbose=False):
  """[summary]
  It is well-known that the global IoU measure is biased toward object instances that cover a large image area.
  In street scenes with their strong scale variation this can be problematic.
  Specifically for traffic participants, which are the key classes in our scenario,
  we aim to evaluate how well the individual instances in the scene are represented in the labeling.
  To address this, we additionally evaluate the semantic labeling using an

  instance-level intersection-over-union metric iIoU = iTP ⁄ (iTP+FP+iFN).
  Again iTP, FP, and iFN denote the numbers of true positive, false positive, and false negative pixels, respectively.
  However, in contrast to the standard IoU measure, iTP and iFN are computed by weighting the contribution of each pixel by the ratio of the class’
  average instance size to the size of the respective ground truth instance.
  It is important to note here that unlike the instance-level task below,
  we assume that the methods only yield a standard per-pixel semantic class labeling as output.
  Therefore, the false positive pixels are not associated with any instance and thus do not require normalization.
  The final scores, iIoUcategory and iIoUclass, are obtained as the means for the two semantic granularities.


  Parameters
  ----------
  pred : [torch.tensor]
                  BSxD1xD2xD3 , predict class for each pixel. No need to predict the -1 class! element of 0-(num_classes-1)
  target : [torch.tensor]
                  BSxD1xD2xD3	, -1 for the VOID pixels that should not induce an error! element of -1-(num_classes-1)
  num_classes : [int]
                  invalid class does not count as a class. So lets say targets takes values -1 - 19 then you have 20 classes
  """

  BS = pred.shape[0]
  # add 1 so the index ranges from 0 to NUM_CLASSES
  pred = pred.type(torch.int) + 1
  target = target.type(torch.int) + 1
  # NOW class=0 should not induce a loss

  # Set pixels that are predicted but no label is available to 0. These pixels dont enduce a loss.
  # Neither does the IoU  of class 0 nor do these pixels count to the UNION for the other classes if predicted wrong.
  pred = pred * (target > 0).type(pred.dtype)
  iou_per_image = torch.zeros((BS), device=pred.device)

  # we have to do this calculation for each image.
  for b in range(BS):
    weight = torch.bincount(target[b].flatten())[1:]
    weight = weight / weight.sum()
    w = torch.zeros((num_classes), device=target.device, dtype=weight.dtype)
    w[: weight.shape[0]] = weight
    TPS, FPS, TNS, FNS, _ = stat_scores_multiple_classes(
      pred[b], target[b], num_classes + 1
    )
    if verbose:
      print(f"TPS:{TPS}, \nFPS:{FPS}, \nFNS:{FNS}, \nTNS:{TNS}")
      print(f"Inter: {TPS},\nUnion: {TPS+FPS+FNS}")
    IoU = (TPS[1:] * w) / ((TPS[1:] * w) + FPS[1:] + (FNS[1:] * w))

    mIoU = (IoU[not torch.isnan(IoU)]).mean()
    iou_per_image[b] = mIoU

  return torch.mean(iou_per_image)  # returns mean over batch


def IoU_class(pred, target, num_classes, verbose=False):
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

  BS = pred.shape[0]
  # add 1 so the index ranges from 0 to NUM_CLASSES
  pred = pred.type(torch.int) + 1
  target = target.type(torch.int) + 1
  # NOW class=0 should not induce a loss

  # Set pixels that are predicted but no label is available to 0. These pixels dont enduce a loss.
  # Neither does the IoU  of class 0 nor do these pixels count to the UNION for the other classes if predicted wrong.
  pred = pred * (target > 0).type(pred.dtype)
  iou_per_image = torch.zeros((BS), device=pred.device)

  # we have to do this calculation for each image.
  for b in range(BS):
    TPS, FPS, TNS, FNS, _ = stat_scores_multiple_classes(
      pred[b], target[b], num_classes + 1
    )
    if verbose:
      print(f"TPS:{TPS}, \nFPS:{FPS}, \nFNS:{FNS}, \nTNS:{TNS}")
      print(f"Inter: {TPS},\nUnion: {TPS+FPS+FNS}")
    IoU = TPS[1:] / (TPS[1:] + FPS[1:] + FNS[1:])
    mIoU = (IoU[torch.isnan(IoU) == False]).mean()
    iou_per_image[b] = mIoU

  return torch.mean(iou_per_image)  # returns mean over batch


def IoU_SCNN(pred, target, num_classes, verbose=False):
  """[summary]
  IoU = TP ⁄ (TP+FP+FN) simply summed up over the full batch.
  This is the fastest metric but also the worst one.

  Parameters
  ----------
  pred : [torch.tensor]
                  BSxD1xD2xD3 , predict class for each pixel. No need to predict the -1 class! element of 0-(num_classes-1)
  target : [torch.tensor]
                  BSxD1xD2xD3	, -1 for the invalid pixels that should not induce an error! element of -1-(num_classes-1)
  num_classes : [int]
                  invalid class does not count as a class. So lets say targets takes values -1 - 19 then you have 20 classes
  """

  BS = pred.shape[0]
  # add 1 so the index ranges from 0 to NUM_CLASSES
  pred = pred.type(torch.int) + 1
  target = target.type(torch.int) + 1
  # NOW class=0 should not induce a loss

  # Set pixels that are predicted but no label is available to 0. These pixels dont enduce a loss.
  # Neither does the IoU  of class 0 nor do these pixels count to the UNION for the other classes if predicted wrong.
  pred = pred * (target > 0).type(pred.dtype)
  iou_per_image = torch.zeros((BS), device=pred.device)

  # we have to do this calculation for each image.
  TPS, FPS, TNS, FNS, _ = stat_scores_multiple_classes(pred, target, num_classes + 1)
  IoU = (TPS[1:]).sum() / (TPS[1:] + FPS[1:] + FNS[1:]).sum()
  if verbose:
    print(f"TPS:{TPS}, \nFPS:{FPS}, \nFNS:{FNS}, \nTNS:{TNS}")
    print(f"Inter: {TPS},\nUnion: {TPS+FPS+FNS}")

  return IoU


if __name__ == "__main__":
  import time
  import sys
  import os

  os.chdir("/home/jonfrey/ASL")
  sys.path.append(os.path.join(os.getcwd(), "src"))
  from lightning import IoU

  BS, H, W, NC = 3, 100, 100, 4
  # metric by scnn
  metric_scnn = IoU(NC)

  # Prepare toy example
  target = torch.zeros((BS, H, W))
  target[:, :10, :10] = -1  # 100 pixels invalid
  target[:, 20:, 20:] = 2  # 640 pixel correct
  pred = torch.zeros((BS, H, W))
  pred[:, :, :] = 2

  # Evaluate
  print("TOY Example 1 Results:")
  st = time.time()
  res1 = IoU_class(pred, target, NC, verbose=True)
  print("   IoU class", time.time() - st, res1)
  st = time.time()
  res2 = iIoU_class(pred, target, NC, verbose=True)
  print("   iIoU class", time.time() - st, res2)
  st = time.time()
  res3 = metric_scnn(pred, target)
  print("   SCNN IoU orginal", time.time() - st, res3)
  st = time.time()
  res4 = IoU_SCNN(pred, target, NC, verbose=True)
  print("   SCNN IoU fast", time.time() - st, res4)
  assert res3 == res4

  # Random example
  pred = torch.randint(0, NC, (BS, H, W))
  target = torch.randint(0, NC, (BS, H, W))

  # Evaluate
  print("TOY Example 1 Results:")
  st = time.time()
  res1 = IoU_class(pred, target, NC, verbose=True)
  print("   IoU class", time.time() - st, res1)
  st = time.time()
  res2 = iIoU_class(pred, target, NC, verbose=True)
  print("   iIoU class", time.time() - st, res2)
  st = time.time()
  res3 = metric_scnn(pred, target)
  print("   SCNN IoU orginal", time.time() - st, res3)
  st = time.time()
  res4 = IoU_SCNN(pred, target, NC, verbose=True)
  print("   SCNN IoU fast", time.time() - st, res4)
  assert res3 == res4
