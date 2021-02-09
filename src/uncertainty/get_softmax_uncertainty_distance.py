import torch

__all__ = ['get_softmax_uncertainty_distance']

def get_softmax_uncertainty_distance(pred,mask = None):
  """
  pred: BS,C,H,W before softmax is applied! 
  
  distance between sec and first element
  # 1 if fully uncertain -> 2th best pixel estimate = 1th best pixel estimate, for all pixels
  # 0 if absolutly confident for all pixels. One class probability 1, 2th best class probability 0, for all pixels
  """
  BS,C,H,W = pred.shape
  if mask is None:
    mask = torch.ones( (BS,H,W), device=pred.device, dtype=torch.bool)
  
  argm1 = torch.argmax(pred, 1)
  soft1 = torch.nn.functional.softmax(pred, dim=1)

  onehot_argm1 = torch.nn.functional.one_hot(argm1, num_classes=C).permute(0,3,1,2).type(torch.bool)
  ten2 = pred.clone()
  ten2[ onehot_argm1 ] = 0

  argm2 = torch.argmax(ten2, 1)
  onehot_argm2 = torch.nn.functional.one_hot(argm2, num_classes=C).permute(0,3,1,2).type(torch.bool)
  res = [] 

  soft1 = soft1.permute(0,2,3,1)
  onehot_argm1 = onehot_argm1.permute(0,2,3,1)
  onehot_argm2 = onehot_argm2.permute(0,2,3,1)
  
  for b in range(BS):
    res_ = soft1[b][mask[b]][onehot_argm1[b][mask[b]]] - soft1[b][mask[b]][onehot_argm2[b][mask[b]]]
    res.append( res_.mean() )

  return torch.tensor(res, dtype=pred.dtype, device=pred.device)


def test():
  BS,C,H,W = 16,40,300,320
  mask = torch.rand( ( BS,H,W) ) > 0.5
    
  pred = torch.rand( ( BS,C,H,W) )
  res = get_softmax_uncertainty_distance(pred)
  print(res, "should be very low")
  
  pred = torch.rand( ( BS,C,H,W) ) /1000
  pred[:,0,:,:] = 10
  res = get_softmax_uncertainty_distance(pred, mask)
  print(res, 'should be nearly 0')
  
  pred[:,1,:,:] = 8
  res = get_softmax_uncertainty_distance(pred, mask)
  print(res, 'should be betweem 0-1')