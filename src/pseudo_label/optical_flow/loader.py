import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/src/pseudo_label'))

import imageio

from optical_flow import readFlowKITTI
from pathlib import Path
from utils_asl import LabelLoaderAuto

class OpticalFlowLoader():
  def __init__(self, base_flow= "/home/jonfrey/Datasets/optical_flow_scannet",
               root_scannet= "/home/jonfrey/Datasets/scannet",
              lookahead = 5 ):
    scene_dirs = sorted( next(os.walk(base_flow))[1] )
    paths = [ ]
    total = 0
    self.images = []
    
    for s in scene_dirs:
      flow_paths = [str(s) for s in Path(os.path.join(base_flow,s)).rglob("*.png")]
      flow_paths.sort(key= lambda x:  int(x.split('/')[-1][:-4]))
      paths.append( flow_paths )
        
      images_paths = [os.path.join( root_scannet, 'scans', s, 'color' ,p.split('/')[-1][:-4] ) +'.jpg' for p in flow_paths]
      self.images.append( images_paths )
      
      total += len(flow_paths)
    
    
    self.paths = paths
    self.total = total
    self.first = [p[0] for p in paths]
    self.stop = paths[-1][-1]
    self.max_frame_ids = [len(p) for p in paths]
    self.max_scene_id = len(paths)
    self.current_scene_id = 0
    self.current_frame_id = 0
    self.scene_dirs = scene_dirs
    self.lookahead = lookahead
    self.label_paths = {}
    self.lla = LabelLoaderAuto(root_scannet)
    
  def register_predictions(self, name, path):
    # providde path to the scans folder
    scenes = sorted( next(os.walk(path))[1] )
    for s in self.scene_dirs:
        assert s in scenes, f"Did not find {s} in registered path dir"
    tmp = []
    for j,s in enumerate( self.paths ):
        tmp.append([])
        for p in s:
            loc = p.find(self.scene_dirs[j])
            p_new = os.path.join( path, p[loc:] )
            
            # FOR COSTUME PREDICTIONS
            if not os.path.exists( p_new ):
                el =  p[loc:].split('/')
                scene = el[0]
                img = el[-1]
                idtf = path.split('/')[-2]
                p_new = os.path.join( path, scene, idtf, img)
            
            # FOR GT DIR
            if not os.path.exists( p_new ):
                el =  p[loc:].split('/')
                scene = el[0]
                img = el[-1]
                idtf = "label-filt"
                p_new = os.path.join( path, scene, idtf, img)
            
            assert os.path.exists( p_new ), f"{p_new}"
            tmp[-1].append(p_new)
    
    self.label_paths[name] = tmp
    
    print("Done")
   
  def __iter__(self):
      return self

  def __next__(self):
    """
    Returns flow list:
        [ frame1, frame2, fram3, frame_lookahead]
    Returns label lost:
        [ {"detectron": frame_1,
            "scannet": frame_1 },
            
            ...
            ...
            
            {"detectron": frame_lookahead,
            "scannet": frame_lookahead } ]
    """ 
    if self.current_scene_id > self.max_scene_id:
        raise StopIteration
    
    p = self.paths[self.current_scene_id][self.current_frame_id]
    
    ret_images = []
    ret_flow = []
    ret_labels = []
    for i in range(self.current_frame_id, self.current_frame_id+self.lookahead):
            if i < self.max_frame_ids[self.current_scene_id]:
                p = self.paths[self.current_scene_id][i]
                ret_flow.append( readFlowKITTI(p) )
                p_img = self.images[self.current_scene_id][i]
                ret_images.append( imageio.imread(p_img))
                labels =  {}
                for k,v in self.label_paths.items():
                    labels[ k ], m = self.lla.get(v[self.current_scene_id][i]) 
                ret_labels.append( labels)
                          
    self.current_frame_id += 1
    if self.current_frame_id > self.max_frame_ids[self.current_scene_id]:
        self.current_frame_id = 0
        self.current_scene_id += 1
    
    return ret_flow, ret_labels, ret_images

if __name__ == "__main__":
  ofl = OpticalFlowLoader()
  ofl.register_predictions("pretrain25k", "/home/jonfrey/Datasets/labels_generated/labels_pretrain25k/scans")
  ofl.register_predictions("gt", "/home/jonfrey/Datasets/scannet/scans")

  for j,val in enumerate( ofl):
      flows, labels, images = val
      if j == 3: break
