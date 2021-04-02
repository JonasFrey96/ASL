import torch
import random
from random import randint
from torch.nn import functional as F

__all__ = ['gradient_dissimilarity']

def met_distance(candidate, buffer , l2=False, lowest= True):
    """
    candidate_feature: 128 dimensional
    buffer_features: Nx128 all features from the targeted classes
    """
    if l2:
        inp = candidate[None,:].repeat(buffer.shape[0],1).type(torch.float32) - buffer[:,:].type(torch.float32)
        metric = torch.norm(inp, p=2, dim=1)
    else:
        metric = F.cosine_similarity(
            candidate[None,:,None].repeat(buffer.shape[0],1,1).type(torch.float32), 
            buffer[:,:,None].type(torch.float32),
            dim=1, eps=1e-6) 
    if lowest:
        return metric.min()
    else:
        return metric.mean()


def compute_interclass_similarity_score( buffer_features ):
    score = 0
    for i in range( buffer_features.shape[0] ):
        valid = torch.ones( ( buffer_features.shape[0] ), dtype=bool)
        valid[i] = False
        res = met_distance(candidate= buffer_features[i], buffer = buffer_features[valid], l2=False, lowest=True )
        score += res
    return score


def gradient_dissimilarity(gradient_list, K_return=50, iterations= 1000, early_stopping = 0.00001):
    """
    Parameters
    ----------
    gradient_list: torch.tensor NRx128
    """
    
    init_selection = torch.randperm( gradient_list.shape[0] )[:K_return]
    
    buffer_features_list = gradient_list[init_selection]
    
    prev_score = 0
    for i in range(iterations):
        if i % 100 == 0:
          print( "gradient_dissimilarity iteration ", i)
        swap_candidate = torch.randint(0, K_return, (1,))
        swap_candidate.to( swap_candidate.device )
        while True:
            replacement_candidate = torch.randint(0,gradient_list.shape[0],(1,))[0]
            if replacement_candidate not in init_selection:
                break
        
        stored_feature = buffer_features_list[swap_candidate].clone()
        buffer_features_list[swap_candidate] = gradient_list[replacement_candidate]
        
        new_score = compute_interclass_similarity_score( 
            buffer_features_list)
        ra = random.random() 
        if new_score > prev_score or  ra > 0.95:
            # apply change
            init_selection[swap_candidate] = int(replacement_candidate)
            prev_score = new_score
        else:
            buffer_features_list[swap_candidate] = stored_feature
                
    return init_selection, prev_score


def test():
    import time
        
    st = time.time()
    print("Start")
    gradient = torch.randint( 0, 1000, (200,100000) )
    res = gradient_dissimilarity(gradient, iterations = 100)
    print("Total time", time.time()-st )
    
if __name__ == "__main__":  
	test()  