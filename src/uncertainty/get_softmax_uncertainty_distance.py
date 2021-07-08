import torch

__all__ = ["get_softmax_uncertainty_distance"]


def get_softmax_uncertainty_distance(pred, mask=None):
  """
  pred: BS,C,H,W before softmax is applied!

  distance between sec and first element
  # 1 if fully uncertain -> 2th best pixel estimate = 1th best pixel estimate, for all pixels
  # 0 if absolutly confident for all pixels. One class probability 1, 2th best class probability 0, for all pixels
  """
  BS, C, H, W = pred.shape
  if mask is None:
    mask = torch.ones((BS, H, W), device=pred.device, dtype=torch.bool)

  soft = torch.nn.functional.softmax(pred, dim=1)
  best_values = torch.topk(soft, 2, dim=1, largest=True, sorted=True)
  dif = best_values.values[:, 0, :, :] - best_values.values[:, 1, :, :]
  res = torch.zeros((BS), device=pred.device, dtype=pred.dtype)
  for b in range(BS):
    res[b] = dif[b][mask[b]].mean()
  return 1 - res


def test():
  # pytest -q -s src/uncertainty/get_softmax_uncertainty_distance.py
  import time

  BS, C, H, W = 16, 40, 300, 320
  mask = torch.rand((BS, H, W)) > 0.5

  pred = torch.rand((BS, C, H, W))
  res = get_softmax_uncertainty_distance(pred)

  print(res, "should be very low")

  pred = torch.rand((BS, C, H, W)) / 1000
  pred[:, 0, :, :] = 10
  res = get_softmax_uncertainty_distance(pred, mask)
  print(res, "should be nearly 0")

  pred[:, 1, :, :] = 8
  pred = pred.cuda()
  mask = mask.cuda()
  st = time.time()
  res = get_softmax_uncertainty_distance(pred, mask)
  print(res, "should be betweem 0-1", time.time() - st)
