from sklearn.cluster import KMeans
import torch

__all__ = ["get_kMeans_indices"]


def knn(ref, query):
  """return indices of ref for each query point. L2 norm

  Args:
      ref ([type]): points * 3
      query ([type]): tar_points * 3

  Returns:
      [knn]: distance = query * 1 , indices = query * 1
  """
  mp2 = ref.unsqueeze(0).repeat(query.shape[0], 1, 1)
  tp2 = query.unsqueeze(1).repeat(1, ref.shape[0], 1)
  dist = torch.norm(mp2 - tp2, dim=2, p=None)
  knn = dist.topk(1, largest=False)
  return knn


def get_kMeans_indices(feat, K_return=50, flag=False, max_matches=2):

  classes = feat.shape[1]
  sum_feat = feat.sum(2)
  sum_feat_mask = sum_feat != 0
  while True:
    class_candidates = []
    for c in range(classes):
      ret_val = min(max_matches, int(sum_feat_mask[:, c].sum()))

      if ret_val > 0:
        valid_data = feat[:, c, :][sum_feat_mask[:, c], :]
        kmeans = KMeans(n_clusters=ret_val, random_state=0).fit(
          valid_data.cpu().numpy()
        )
        res = knn(
          query=torch.from_numpy(kmeans.cluster_centers_), ref=feat[:, c, :].cpu()
        )
        class_candidates.append(
          res.indices.tolist()
        )  # contains index of closest cluster center
    candidates = torch.tensor(class_candidates, device=feat.device).flatten()
    res = torch.unique(candidates)
    if res.shape[0] > K_return or max_matches > 50:
      break
    else:
      max_matches += 1
  if res.shape[0] < K_return:
    raise Exception()
  if flag:
    return res
  sel = torch.randperm(res.shape[0])[:K_return]
  return res[sel]


# TODO write a test for this method
