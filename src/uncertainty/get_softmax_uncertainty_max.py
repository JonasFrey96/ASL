import torch

__all__ = ['get_softmax_uncertainty_max']

def get_softmax_uncertainty_max(pred):
  """
  pred: BS,C,H,W before softmax is applied! 
  
  (1 - max( softmax(pred))) mean over batch size
  
  # 1 if fully uncertain
  # 0 if absolutly confident for all pixels
  """
  BS,C,H,W = pred.shape
  
  argm1 = torch.argmax(pred, 1)
  soft1 = torch.nn.functional.softmax(pred, dim=1)
  onehot_argm1 = torch.nn.functional.one_hot(argm1, num_classes=C).permute(0,3,1,2).type(torch.bool)
  
  soft1 = soft1.permute(0,2,3,1)
  onehot_argm1 = onehot_argm1.permute(0,2,3,1)
  
  res = []
  for b in range(BS):
    res_ = soft1[b][onehot_argm1[b]]
    res.append( torch.mean(res_) )
  
  return torch.tensor(res, dtype=pred.dtype, device=pred.device)

def test():
  BS,C,H,W = 16,40,300,320
  pred = torch.rand( ( BS,C,H,W) )
  res = get_softmax_uncertainty_max(pred)
  print(res, "should be very low")
  
  pred = torch.rand( ( BS,C,H,W) ) /1000
  pred[:,0,:,:] = 10
  res = get_softmax_uncertainty_max(pred)
  print(res, 'should be nearly 0')
  
  pred[:,1,:,:] = 8
  res = get_softmax_uncertainty_max(pred)
  print(res, 'should be betweem 0-1')
  