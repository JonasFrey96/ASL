import torch

__all__ = ['get_softmax_uncertainty_entropy']

def get_softmax_uncertainty_entropy(pred, mask=None):
  """
  pred: BS,C,H,W before softmax is applied!
  """
  BS,C,H,W = pred.shape
  if mask is None:
    mask = torch.ones( (BS,H,W), device=pred.device, dtype=torch.bool)
  
  p = torch.nn.functional.softmax(pred, dim=1)                                
  entropy = -torch.sum( p*torch.log(p), dim=1)
  res = []
  for b in range(BS):
    res.append( (entropy[b][mask[b]]).mean() )
  
  return torch.tensor(res, dtype=pred.dtype, device=pred.device)

def test():
  # pytest -q -s src/uncertainty/get_softmax_uncertainty_entropy.py
  BS,C,H,W = 16,5,3,3

  pred = torch.rand( ( BS,C,H,W) )
  res = get_softmax_uncertainty_entropy(pred)
  print(res, "should be very low")
  
  pred = torch.rand( ( BS,C,H,W) ) /1000
  pred[:,0,:,:] = 10
  res = get_softmax_uncertainty_entropy(pred)
  print(res, 'should be nearly 0')
  
  pred[:,1,:,:] = 8
  res = get_softmax_uncertainty_entropy(pred)
  print(res, 'should be betweem 0-1')
  