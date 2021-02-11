import torch 
from torch.nn import functional as F
import random
__all__ = ['get_image_indices']

def get_image_indices(feat, gloable_indices, dis_metric= 'cos',
                     K_aggregate=50, K_return=50, most_dissimilar= True,
                     pick_mode='class_balanced'):
    N, NC, C = feat.shape   
    T = int(N*NC)
    y = torch.arange(0,NC, device=feat.device)[None].repeat(N,1).flatten()
    
    gloable_indices_all = gloable_indices[:,None].repeat(1,NC).flatten()
    feat = feat.view( (T,C) )
    
    # only mark a features as valid if the vector is not 0!
    y[ feat.sum(dim=1) == 0 ] = 999
    
    # create centroids over valid features
    feat_centroids = torch.zeros( (NC,C), device=feat.device, dtype=feat.dtype)
    for i in range(NC):
        feat_centroids[i] = feat[y==i].mean(dim=0)
        
    # create expanded centroid tensor that aligns with feat
    expanded_centroids = feat.clone()
    for i in range(NC):
        m = y == i
        expanded_centroids[ m,: ] = feat_centroids[i][None,:].repeat(T,1)[m, :]
    
    # compute similarity metric
    if dis_metric == 'cos':
        metric = F.cosine_similarity(
            feat.type(torch.float32), 
            expanded_centroids.type(torch.float32),
            dim=1, eps=1e-6) 
    elif dis_metric == 'pairwise':
        metric = F.pairwise_distance(
            feat.type(torch.float32), 
            expanded_centroids.type(torch.float32),
            dim=1, eps=1e-6)
    else:
        raise Exception(f'In get_image_indices dis_metric {dis_metric} not implemented')
    
    if pick_mode== 'most_hits':

        # select K best for each class
        candidates = []
        for i in range(NC):
            m = y==i
            _K = min(m.sum(), K_aggregate)
            values, indices = torch.topk( metric[m] ,_K, 
                largest = not most_dissimilar)
            candidates +=  gloable_indices_all[m][indices].tolist()
        candidates = torch.tensor( candidates, device=feat.device )

        # select the globale_image_indicies that are selected most
        indi, counts = torch.unique(candidates, return_counts=True, sorted=False)

        values, indices_of_indices = torch.topk( counts ,K_return) 
        ret_globale_indices = gloable_indices[indices_of_indices]
    
    elif pick_mode == 'class_balanced':
        candidates = []
        nr_features_valid = (y!=999).sum()
        
        for i in range(NC):
            m = y==i
            nr_features_valid_i = (y==i).sum()
            
            factor = float( nr_features_valid_i/nr_features_valid)
            if i == NC-1:
                _K = int(K_return-len(candidates))
            else:
                _K = round( K_return*factor )
                
            if _K > 0:
                max_ele = metric[m].shape[0]
                top_K = min(max_ele, _K+ len(candidates)+1)
                
                values, indices = torch.topk( metric[m] ,_K, 
                    largest = not most_dissimilar)
                added = 0
                for ele in gloable_indices_all[m][indices].tolist():
                    if not ele in candidates and added < _K:
                        added += 1
                        candidates = candidates + [int(ele)]
        
        random.shuffle(candidates)
        if len(candidates) > K_return:
            candidates = candidates[:K_return]
        if len(candidates) < K_return:
            base = gloable_indices.type(torch.int64).tolist()
            for c in candidates:
                base.remove( c )
            while len(candidates) < K_return:
                ca = random.choice(base)
                candidates.append(ca)
                base.remove(ca)
                print('Randomly added Image')   
                
        # select the globale_image_indicies that are selected most
        ret_globale_indices = torch.tensor(candidates, device=feat.device)
        
    return ret_globale_indices
  
def test():
  # pytest -q -s src/uncertainty/get_image_indices.py
  
  p = "/media/scratch1/jonfrey/models/master_thesis/dev/uncertainty_integration/latent_feature_tensor_0.pt"
  data = torch.load( p ).to('cuda:0')
  print("START")
  ret_gloable_indices = get_image_indices(data, torch.arange(0,data.shape[0],device=data.device), pick_mode='class_balanced')
  ret_gloable_indices = get_image_indices(data, torch.arange(0,data.shape[0],device=data.device), pick_mode='most_hits')
  print(data.device)
  print(ret_gloable_indices)
  # TODO write some more sophisticated tests