import torch
import random
from torch.nn import functional as F
import time

__all__ = ["hierarchical_dissimilarity"]


def get_fast_distance(x):
  NR, C = x.shape
  x1 = x[:, None, :].repeat(1, NR, 1)
  # BS x BS distance matrix
  res = torch.cdist(x1, x1, p=2.0)
  return res.mean()


def get_knn_distance(x):
  return knn_search_nearest(x, x, k=2)


def knn_search_nearest(ref, query, k=2, norm=None):
  """return mean distance to the 5 losest datapoints over all vectors
  Args:
      ref ([type]): NR * C
      query ([type]): NR * C
  """
  mp2 = ref.unsqueeze(0).repeat(query.shape[0], 1, 1)
  tp2 = query.unsqueeze(1).repeat(1, ref.shape[0], 1)
  dist = torch.norm(mp2 - tp2, dim=2, p=norm)
  knn = dist.topk(k, largest=False)
  res = knn.values.mean()
  return res


def angle(x, k=2):
  NR, C = x.shape
  ref = x
  query = x
  mp2 = ref.unsqueeze(0).repeat(query.shape[0], 1, 1)
  tp2 = query.unsqueeze(1).repeat(1, ref.shape[0], 1)
  dist = F.cosine_similarity(
    mp2.view((-1, C)), tp2.view((-1, C))
  )  # if 1 same dir , if -1 other direction
  dist = dist.reshape(NR, NR)
  dist = -dist
  return dist.mean()


def gradient_dissimilarity_fast(X, K=50, iterations=1000, early_stopping=0.00001):
  """
  Parameters
  ----------
  X: torch.tensor NRxC -> move X on GPU before passing to the function
  """
  st = time.time()

  if X.shape[0] <= K:
    ret = torch.arange(0, X.shape[0])
    ret = ret.type(torch.int64)
    return ret

  init_selection = torch.randperm(X.shape[0])[:K]
  buffer_features_list = X[init_selection]

  prev_score = 0
  for i in range(iterations):

    swap_candidate = torch.randint(0, K, (1,))
    swap_candidate.to(swap_candidate.device)
    while True:
      replacement_candidate = torch.randint(0, X.shape[0], (1,))[0]
      if replacement_candidate not in init_selection:
        break

    stored_feature = buffer_features_list[swap_candidate].clone()
    buffer_features_list[swap_candidate] = X[replacement_candidate]

    new_score = angle(buffer_features_list)
    ra = random.random()
    if new_score > prev_score or ra > 0.95:
      # apply change
      init_selection[swap_candidate] = int(replacement_candidate)
      prev_score = new_score
    else:
      buffer_features_list[swap_candidate] = stored_feature
  print(f"Time {i} takes {time.time()-st}")
  init_selection = init_selection.type(torch.int64)
  return init_selection  # , prev_score


def hierarchical_dissimilarity(X, K=50, maxSize=100, device="cuda:0"):
  print("INPUT HIERARCHICAL", X.shape)
  if X.shape[0] < maxSize:
    # LEAF perform
    print("Leaf", X.shape, device)
    input_X = X.clone()
    input_X = input_X.to(device)
    c = gradient_dissimilarity_fast(input_X, K=K)
    del input_X
    print("Leaf returns ", c.shape[0])
    return c
  else:
    # BRANCHE
    s = X.shape[0]
    split = int(s / 2)
    indices = torch.arange(0, s)
    indices = torch.randperm(s)
    indices = indices.type(torch.int64)
    r = s - split
    right_indices = indices[split:]
    left_indices = indices[:split]
    print(f"Branch into l{split}, r{r}")

    print(right_indices.max(), left_indices.max(), X.shape)
    r = hierarchical_dissimilarity(
      X[right_indices], K=K, maxSize=maxSize, device=device
    )
    leng = hierarchical_dissimilarity(
      X[left_indices], K=K, maxSize=maxSize, device=device
    )
    sel_indices = torch.cat((indices[left_indices][leng], indices[right_indices][r]))
    in_data = (X[sel_indices]).clone()
    in_data = in_data.to(device)
    c = gradient_dissimilarity_fast(in_data, K=K)
    del in_data

    print(f"Joined -> ", c.shape)
    print("select", c.max(), c.min(), sel_indices.shape)
    sel_indices = sel_indices[c]
    return sel_indices
