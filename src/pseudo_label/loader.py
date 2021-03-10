try:
	from helper_functions import *
except:
	from .helper_functions import *
import numpy as np 
import os

__all__ = ['PseudoLabelLoader']

class PseudoLabelLoader():
	def __init__(self, window_size, h=960, w=1280, sub=10, ignore_depth=False):
		self.depth_paths = getPathsDepth()
		self.segmentation_paths = getPathsSegmentation()
		self.flow_paths = getPathsFlow(key=f"flow_sub_{sub}")
		self.window_size = window_size 
		self.sub = sub
		self.H=h
		self.W=w
		self.ignore_depth = ignore_depth
		
		self.lists_to_ids()
		self.global_to_local_idx = self.get_global_idx_list()
		self.length = len( self.global_to_local_idx )

	def lists_to_ids(self):
		def list_to_ids( ls ):
			scene = [int( s.split('/')[-3][-7:-3])*100 + int( s.split('/')[-3][-2:]) for s in ls] 
			ids = [int( s.split('/')[-1][:-4]) for s in ls]
			return np.array( [scene, ids] ).T

		self.depth_ids = list_to_ids( self.depth_paths ) 
		self.flow_ids = list_to_ids( self.flow_paths ) 
		self.seg_ids = list_to_ids( self.segmentation_paths ) 

	def get_global_idx_list(self):
		global_to_local_idx = []

		for i in range( self.seg_ids.shape[0] ):
			scene = self.seg_ids[i,0]
			ids = self.seg_ids[i,1]
			brake = False

			__d_ids = []
			__f_ids = []
			__s_ids = []

			for k in range(0, self.window_size):
				# check if seg flow and depth are availalbe
				_ids = int( ids-k*self.sub )
				
				if not self.ignore_depth:
					# DEPTH
					scene_depth_filter = self.depth_ids[:,0] == scene
					if scene_depth_filter.sum() == 0:
						brake = True
						continue
					if (self.depth_ids[scene_depth_filter][:,1] == _ids).sum() == 0:
						brake = True
						continue
					idx_depth = np.where( self.depth_ids[scene_depth_filter][:,1] == _ids )
					idx_depth = int( idx_depth[0])
					__d_ids.append( idx_depth )

				# FLOW
				scene_flow_filter = self.flow_ids[:,0] == scene
				if scene_flow_filter.sum() == 0:
					brake = True
					continue	
				if (self.flow_ids[scene_flow_filter][:,1] == _ids).sum() == 0:
					brake = True
					continue
				
				idx_flow = np.where( self.flow_ids[scene_flow_filter][:,1] == _ids )
				idx_flow = int( idx_flow[0])
				__f_ids.append( idx_flow )

				scene_seg_filter = self.seg_ids[:,0] == scene
				if (self.seg_ids[scene_seg_filter][:,1] == _ids).sum() == 0:
					brake = True
					continue
				
				idx_seg = np.where( self.seg_ids[scene_seg_filter][:,1] == _ids )
				idx_seg = int( idx_seg[0])
				__s_ids.append(idx_seg)
			
			if not brake:
				global_to_local_idx.append( {'seg_ids': __s_ids, 'flow_ids': __f_ids, 'depth_ids': __d_ids } )
		return global_to_local_idx

	def __getitem__(self, index):
		di = self.global_to_local_idx[index]
		seg = []
		flow = []
		depth = []
		paths = []
		for i in range(len( di['seg_ids'] )):
			flow.append( readFlowKITTI( self.flow_paths[di['flow_ids'][i]], H=self.H ,W=self.W))
			seg.append( readSegmentation( self.segmentation_paths[di['seg_ids'][i]]))
			if not self.ignore_depth:
				depth.append( readDepth( self.depth_paths[di['depth_ids'][i]]))
			
			l = self.flow_paths[di['flow_ids'][i]].split('/')
			paths.append( os.path.join( 'scans',  l[-3], 'color', l[-1][:-4]+'.jpg' ) )
		return seg, depth, flow, paths

def test():
	pll = PseudoLabelLoader(window_size=5, sub=1, ignore_depth=True)
	print( pll[0] )


if __name__ == "__main__":
	print("Start Test")
	test()
