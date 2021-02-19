import torch 
import random
from random import randint
from torch.nn import functional as F

__all__ = ['distribution_matching']

def compute_metric(selected, features):
    target_dist = features.sum(0).type(torch.float32)
    target_dist = target_dist / target_dist.sum() #normalize

    selected_dist = features[selected,:].sum(0).type(torch.float32)
    selected_dist = selected_dist/ selected_dist.sum() #normalize

    number = F.mse_loss( selected_dist, target_dist)
    return number

def distribution_matching(features, K_return=50, iterations= 1000, early_stopping = 0.00001):
    """ returns k indices from globale_indices such that the returned indexed match the total feature distribution
				fully sampling based !
    
    Parameters
    ----------
    features : torch.tensor NRxC
    K_return : int, optional
            number of returned indices, by default 50
    """
    indices = torch.arange( 0, features.shape[0] )
    start_indices = torch.randperm( indices.shape[0] )[:K_return]
    selected = torch.zeros( (indices.shape[0]), dtype=torch.bool)
    for i in range(start_indices.shape[0]):
        selected[start_indices[i]] = True
        
    
    old_metric = compute_metric( selected, features)
    for i in range(iterations):
        if old_metric < early_stopping:
            break
        
        current_selection = torch.where(selected)[0]
        candidate = current_selection[randint(0,K_return-1)]

        # get a new candidate sample
        while True:
            sample = randint(0,selected.shape[0]-1)
            if sample not in current_selection:
                break

        selected[ candidate ] = False
        selected[ sample ] = True
        new_metric = compute_metric( selected, features)

        if new_metric < old_metric:
            # keep candidate change
            old_metric = new_metric
        else:
            if random.random() < 0.05:
                old_metric = new_metric
            else:
                selected[ candidate ] = True
                selected[ sample ] = False
        
    return selected, old_metric
 