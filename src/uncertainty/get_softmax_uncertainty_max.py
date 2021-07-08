import torch

__all__ = ["get_softmax_uncertainty_max"]


def get_softmax_uncertainty_max(pred, mask=None):
  """
  pred: BS,C,H,W before softmax is applied!

  (1 - max( softmax(pred))) mean over batch size

  # 1 if fully uncertain
  # 0 if absolutly confident for all pixels
  """
  BS, C, H, W = pred.shape
  if mask is None:
    mask = torch.ones((BS, H, W), device=pred.device, dtype=torch.bool)

  soft = torch.nn.functional.softmax(pred, dim=1)
  best_values = torch.topk(soft, 1, dim=1, largest=True, sorted=True)
  dif = best_values.values[:, 0, :, :]
  res = torch.zeros((BS), device=pred.device, dtype=pred.dtype)
  for b in range(BS):
    res[b] = dif[b][mask[b]].mean()
  return 1 - res


def test():
  # pytest -q -s src/uncertainty/get_softmax_uncertainty_max.py

  BS, C, H, W = 16, 40, 300, 320
  pred = torch.rand((BS, C, H, W))
  res = get_softmax_uncertainty_max(pred)
  print(res, "should be very low")

  pred = torch.rand((BS, C, H, W)) / 1000
  pred[:, 0, :, :] = 10
  res = get_softmax_uncertainty_max(pred)
  print(res, "should be nearly 0")

  pred[:, 1, :, :] = 8
  res = get_softmax_uncertainty_max(pred)
  print(res, "should be betweem 0-1")
