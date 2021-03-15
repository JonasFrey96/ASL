import torch.utils.data as data
import numpy as np
import torch
import random
import multiprocessing

__all__ = ['ReplayDataset', 'StaticReplayDataset']

class ReplayDataset(data.Dataset):
    def __init__(self, bins, elements, add_p=0.5, replay_p=0.5, current_bin=0, replay=False):
        if replay == False:
            return
           
        self._bins = [
            multiprocessing.Array(
                'I', (elements)) for i in range(bins)]
        self._valids = [
            multiprocessing.Array(
                'I', (elements)) for i in range(bins)]
        self._current_bin = multiprocessing.Value('I', 0)
         
        self.b = np.zeros(elements)
        self.v = np.zeros(elements)
        self._elements = elements
       
        self.replay_p = replay_p
        self._add_p = add_p

    def idx(self, index):
        bin = -1
        if random.random() < self.replay_p and self._current_bin.value != 0:
            index, bin = self.get_element(index)

        elif random.random() < self._add_p:
            self.add_element(index)
          
        return index, bin

    def get_replay_state(self):
        v_el = []
        for i in self._valids:
            with i.get_lock(): 
                self.v = np.frombuffer(i.get_obj(),dtype=np.uint32) # no data copying
                v_el.append( int( self.v.sum() ) )
        v = self._current_bin.value
        string = "ReplayDataset contains elements " + str( v_el) 
        string += f"\nCurrently selected bin: {v}"
        return string

    def set_current_bin(self, bin):       
        # might not be necessarry since we create the dataset with the correct
        # bin set
        if bin < len( self._bins ):
            self._current_bin.value = bin
        else:
            raise Exception(
                "Invalid bin selected. Bin must be element 0-" +
                len( self._bins))
            
    def get_full_state(self):
        bins = np.zeros( ( len(self._bins),self._elements))
        valids = np.zeros( ( len(self._valids),self._elements))
        
        for b in range( len(self._bins) ):
            bins[b,:] = self._bins[b][:]
            valids[b,:] = self._valids[b][:]
        
        return bins, valids
    
    def set_full_state(self, bins,valids, bin):
        if self.replay == False:
            return
    
        assert bins.shape[0] == valids.shape[0]
        assert valids.shape[0] == len(self._bins)
        assert bins.shape[1] == valids.shape[1]
        assert valids.shape[1] == self._elements
        assert bin >= 0 and bin < bins.shape[0]
        
        for b in range( len(self._bins) ):
            with self._bins[b].get_lock(): 
                self.b = np.frombuffer(self._bins[b].get_obj(),dtype=np.uint32) # no data copying
                self.b[:] = bins[b,:].astype(np.uint32) 
            with self._valids[b].get_lock(): 
                self.v = np.frombuffer(self._valids[b].get_obj(),dtype=np.uint32) # no data copying
                self.v[:] = valids[b,:].astype(np.uint32)
                
        self._current_bin.value = bin
         

    def get_element(self, index):

        v = self._current_bin.value
        if v > 0:
            if v > 1:
                b = int(np.random.randint(0, v - 1, (1,)))
            else:
                b = 0
            # we assume that all previous bins are filled. 
            # Therefore contain more values 
            with self._valids[b].get_lock(): # synchronize access
                with self._bins[b].get_lock(): 
                    
                    self.v = np.frombuffer(self._valids[b].get_obj(),dtype=np.uint32) # no data copying
                    self.b = np.frombuffer(self._bins[b].get_obj(),dtype=np.uint32) # no data copying
                    
                    indi = np.nonzero(self.v)[0]
                    if indi.shape[0] == 0:
                        sel_ele = 0
                    else:
                        sel_ele = np.random.randint(0, indi.shape[0], (1,))
                    
                    return int( self.b[ int(indi[sel_ele]) ] ), int(b)
                
        return -1, -1

    def add_element(self, index):
        v = self._current_bin.value
        
        with self._valids[v].get_lock(): # synchronize access
            with self._bins[v].get_lock(): 
                
                self.v = np.frombuffer(self._valids[v].get_obj(),dtype=np.uint32) # no data copying
                self.b = np.frombuffer(self._bins[v].get_obj(),dtype=np.uint32) # no data copying
                    
                if (index == self.b).sum() == 0:
                    # not in buffer
                    if self.v.sum() != self.v.shape[0]:
                        # free space simply add
                        indi = np.nonzero(self.v == 0)[0]
                        sel_ele = np.random.randint(0, indi.shape[0], (1,))
                        sel_ele = int(indi[sel_ele])

                        self.b[sel_ele] = index
                        self.v[sel_ele] = True

                    else:
                        # replace
                        sel_ele = np.random.randint(0, self.b.shape[0], (1,))
                        self.b[sel_ele] = index
                    
                    return True

        return False

    # Deleting (Calling destructor) 
    # def __del__(self): 
    #     for i in self._bins:
    #         del i
    #     for j in self._valids:
    #         del j
    #     del self._bins
    #     del self._valids


class StaticReplayDataset(data.Dataset):
    def __init__(self, bins, elements, add_p=0.5, replay_p=0.5, current_bin=0, replay=False):
        if replay == False:
            return
           
        self._bins = np.zeros( (bins,elements) ).astype(np.int32) 
        self._valids = np.zeros( (bins,elements) ).astype(np.int32) 
        self._current_bin = 0 
        
        self._elements = elements
       
        self.replay_p = replay_p

    def idx(self, index):
        bin = -1
        if random.random() < self.replay_p and self._current_bin != 0:
            index, bin = self.get_element(index)
        return index, bin

    def get_replay_state(self):
        string = 'get_replay_state not implemented yet'
        return string

    def set_current_bin(self, bin):       
        if bin <  self._bins.shape[0]:
            self._current_bin= bin
        else:
            raise Exception(
                "Invalid bin selected. Bin must be element 0-" +
                self._bins.shape[0])
            
    def get_full_state(self):
        return self._bins, self._valids
    
    def set_full_state(self, bins,valids, bin):
        if self.replay == False:
            return
        assert bins.shape[0] == valids.shape[0]
        assert valids.shape[0] == self._bins.shape[0]
        assert bins.shape[1] == valids.shape[1]
        assert valids.shape[1] == self._elements
        assert bin >= 0 and bin < bins.shape[0]
        
        self._bins = bins.astype(np.int32) 
        self._valids = valids.astype(np.int32)
        self._current_bin = bin
         

    def get_element(self, index):
        v = self._current_bin
        if v > 0:
            if v > 1:
                b = int(np.random.randint(0, v - 1, (1,)))
            else:
                b = 0
            
            indi = np.nonzero(self._valids[b])[0]
            if indi.shape[0] == 0:
                sel_ele = 0
            else:
                sel_ele = np.random.randint(0, indi.shape[0], (1,))
            
            # print(self._bins.shape, b, int(indi[sel_ele]))
            try:
                return int( self._bins[b, int(indi[sel_ele]) ] ), int(b)
            except:
                return -1,-1
        return -1, -1
