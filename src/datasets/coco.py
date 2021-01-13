import os
import random
from pathlib import Path

import torch.utils.data as data
import torch
from torchvision import transforms as tf
from torchvision.transforms import functional as F

import numpy as np
import scipy
from  PIL import Image
from pycocotools.coco import COCO as COCOtool
try:
  from .helper import Augmentation
except Exception: #ImportError
  from helper import Augmentation

__all__ = ['COCo']

# some indexe in the coco dataset are simply skipped. This makes the objects go from 1-80 and 0 is the background class. 
coco_skipped_ids = [12,26,29,30,45,66,68,69,71,83]
coco_id_without_skip= {}
for i in range(91):
  minus = 0
  if i in coco_skipped_ids:
    continue
  for x in coco_skipped_ids:
    if i > x:
      minus+= 1
  coco_id_without_skip[i] = i-minus


class COCo(data.Dataset):
  def __init__(self, root='/media/scratch2/jonfrey/datasets/COCO/', 
              mode='train', scenes=[], output_trafo = None, 
              output_size=400, degrees = 10, flip_p = 0.5, jitter_bcsh=[0.3, 0.3, 0.3, 0.05], squeeze_80_labels_to_40 = True):
    """
    TODO Filtering is not implemented. Here additional to do when creating the mask take care!
    
    Parameters
    ----------
    root : str, path to the COCO folder
    mode : str, option ['train','val]
    """
    self._output_size = output_size
    self._mode = mode
    self._load(root, mode)
    self._augmenter = Augmentation(output_size,
                                    degrees,
                                    flip_p,
                                    jitter_bcsh)
    
    self._squeeze_80_labels_to_40 = squeeze_80_labels_to_40
    self._output_trafo = output_trafo
    
    self._resize_img = tf.Resize( size=self._output_size, interpolation=Image.BILINEAR )
    self._resize_label = tf.Resize( size=self._output_size, interpolation=Image.NEAREST)
    
   
  @staticmethod
  def get_classes(mode):
    # TODO 
    return {}
          
  def __getitem__(self, index):
    # Get image
    n = self._coco.loadImgs(self._img_ids[index])[0]['file_name']
    img_path = os.path.join(self._root_img, n)
    img = np.array( Image.open( img_path ).convert('RGB') ) # H W C
    img = (torch.from_numpy( img )/255).type(torch.float32).permute(2,0,1) # C H W
    
    # Get label
    _,H,W= img.shape
    ann_ids = self._coco.getAnnIds(imgIds=self._img_ids[index])
    anns = self._coco.loadAnns(ann_ids)
    label = np.zeros((H,W))
    
    for i in range(len(anns)):
      pixel_value = anns[i]['category_id']
      
      obj_label = self._coco.annToMask(anns[i])
      if self._squeeze_80_labels_to_40:
        obj_label[obj_label== 1] = int((coco_id_without_skip[pixel_value]+1)/2 )
        
      else:
        
        obj_label[obj_label == 1] = coco_id_without_skip[pixel_value]
      label = np.maximum(obj_label, label)
    label = label.astype(np.int32)
    
    if label.sum() == 0:
      print(len(anns))
      raise Exception
    label = torch.from_numpy( label ).type(torch.float32).permute(0,1)[None,:,:] # C H W
    
    # The resizing is only necessary for coco!
    img, label = self._resize(img,label)
    
    if self._mode == 'train':
        img, label = self._augmenter.apply(img, label)
    elif self._mode == 'val' or self._mode == 'test':
        img, label = self._augmenter.apply(img, label, only_crop=True)
    else:
        raise Exception('Invalid Dataset Mode')
           
    img_ori = img.clone()   
    img_ori.requires_grad = False
    if self._output_trafo is not None:
        img = self._output_trafo(img)
    
    return img, label.type(torch.int64)[0,:,:] - 1, img_ori

  def _resize(self, img, label):
    if (img.shape[1] < self._output_size or 
      img.shape[2] < self._output_size):
      
      return self._resize_img(img),  self._resize_label(label)
    else:
      return img, label
    
  def __len__(self):
    return self._length

  def _load(self, root, mode):
    print(mode)
    annFile = os.path.join( root, f'annotations/instances_{mode}2014.json')
    self._coco = COCOtool(annFile)
    self._img_ids = list(sorted(self._coco.imgs.keys()))
    
    if mode == 'train':
      self._img_ids = [self._img_ids[i] for i in range(len(self._img_ids)) if i not in NOT_ANNOTATED_FRAMES_TRAIN]
      # self._img_ids.pop(NOT_ANNOTATED_FRAMES_TRAIN)
    elif mode == 'val':
      self._img_ids = [self._img_ids[i] for i in range(len(self._img_ids)) if i not in NOT_ANNOTATED_FRAMES_VAL]
      # self._img_ids.pop(NOT_ANNOTATED_FRAMES_VAL)
    elif mode == 'test':
      # TODO
      pass
    
    
    self._root_img = os.path.join( root, f'{mode}2014')
    catIDs = self._coco.getCatIds()
    self._cats = self._coco.loadCats(catIDs)
    self._length = len( self._img_ids)
    
        
  def _filter_scene(self, scenes):
    pass


def test():
  # pytest -q -s src/datasets/coco.py 
  
  ds = COCo( root='/media/scratch2/jonfrey/datasets/COCO/', 
              mode='train', scenes=[], output_trafo = None, 
              output_size=400, degrees = 10, flip_p = 0.5, 
              jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
              squeeze_80_labels_to_40 = True)
  print(ds[0])
  emp = []
  for index in range ( 0, len( ds._img_ids )):
    anns = ds._coco.getAnnIds(imgIds=ds._img_ids[index])
    if len(anns) == 0:
      print(index)
      emp.append( index)
      
NOT_ANNOTATED_FRAMES_TRAIN = [31, 61, 149, 201, 281, 481, 486, 514, 553, 579, 585, 733, 864, 1009, 1018, 1033, 1053, 1131, 1331, 1334, 1382, 1425, 1457, 1512, 1723, 1741, 1802, 2140, 2157, 2212, 2310, 2880, 2969, 3105, 3162, 3177, 3341, 3421, 3654, 3663, 3695, 3723, 4023, 4097, 4101, 4114, 4219, 4307, 4490, 4544, 4754, 4979, 5041, 5051, 5095, 5191, 5334, 5384, 5456, 5464, 5495, 5614, 5668, 5754, 5820, 6021, 6453, 6506, 6521, 6739, 7068, 7070, 7325, 7333, 7622, 7750, 7879, 7880, 7922, 8055, 8223, 8304, 8407, 8418, 8454, 8465, 8523, 8671, 8750, 8751, 9109, 9160, 9315, 9390, 9572, 9742, 9895, 9917, 10184, 10234, 10399, 10729, 10766, 10862, 10947, 10995, 11064, 11335, 11535, 11545, 11731, 11777, 12069, 12096, 12378, 12382, 12384, 12526, 12643, 12736, 12813, 12833, 12882, 13010, 13040, 13367, 13395, 13484, 13790, 13932, 13933, 13985, 13991, 14061, 14113, 14166, 14400, 14409, 14609, 14659, 14685, 14746, 14834, 15041, 15227, 15382, 15436, 15438, 15489, 15549, 15676, 15739, 15934, 16007, 16094, 16359, 16383, 16454, 16491, 16506, 16525, 16795, 17049, 17224, 17275, 17429, 17604, 17709, 17723, 17750, 17778, 17788, 17825, 18002, 18154, 18413, 18524, 18638, 18701, 18739, 18964, 19146, 19189, 19417, 19545, 19574, 19790, 19839, 19905, 19911, 19976, 20109, 20156, 20185, 20208, 20443, 20493, 20550, 20581, 20654, 21159, 21223, 21233, 21245, 21303, 21319, 21547, 21815, 22009, 22091, 22099, 22601, 22667, 22728, 22806, 22817, 22846, 23093, 23196, 23231, 23499, 23728, 23806, 23912, 24092, 24208, 24384, 24619, 24806, 24913, 24926, 25187, 25199, 25761, 25979, 26134, 26325, 26391, 26524, 26738, 26744, 26851, 26870, 27229, 27271, 27423, 27460, 27478, 27521, 27544, 27562, 27671, 27724, 27761, 27762, 27808, 27874, 27965, 28143, 28205, 28272, 28484, 28613, 28767, 28822, 28827, 28889, 29408, 29802, 30016, 30382, 30433, 30573, 30800, 30874, 30975, 31265, 31318, 31372, 31414, 31424, 31501, 31533, 31670, 31779, 32033, 32263, 32291, 32393, 32437, 32712, 32758, 32788, 32885, 33211, 33274, 33321, 33323, 33394, 33429, 33577, 33733, 33778, 33986, 34058, 34167, 34272, 34339, 34408, 34437, 34461, 34473, 34547, 34615, 34726, 34783, 34806, 34838, 35057, 35097, 35218, 35411, 35458, 35580, 35588, 35726, 35738, 35764, 35834, 35846, 35914, 36030, 36085, 36394, 36453, 36604, 36657, 36808, 36878, 36955, 36958, 37173, 37233, 37358, 37526, 37551, 37599, 37746, 37779, 37875, 37891, 37933, 38111, 38193, 38445, 38617, 38983, 39051, 39134, 39138, 39185, 39510, 39562, 39574, 39748, 39782, 39793, 39819, 39955, 40058, 40291, 40324, 40528, 40780, 40839, 40891, 40892, 41032, 41293, 41306, 41378, 41424, 41501, 41698, 42171, 42259, 42425, 42426, 42486, 42490, 42498, 42701, 42965, 43007, 43087, 43571, 43879, 44050, 44572, 44735, 44865, 45022, 45130, 45373, 45674, 45991, 46053, 46158, 46385, 46411, 46559, 46782, 47149, 47247, 47440, 47561, 47772, 48275, 48395, 48449, 48768, 48817, 48822, 49034, 49047, 49117, 49146, 49241, 49302, 49327, 49374, 49648, 49683, 49684, 49723, 50184, 50493, 50769, 50793, 51124, 51142, 51217, 51430, 51505, 51556, 51571, 51588, 51668, 51840, 52049, 52082, 52254, 52348, 52418, 52511, 52650, 52735, 52754, 52803, 52888, 52910, 52911, 52971, 53413, 53474, 53631, 53683, 53698, 53905, 53938, 53967, 53981, 54192, 54384, 54493, 54837, 54978, 55152, 55344, 55686, 55744, 56073, 56113, 56404, 56549, 56856, 57006, 57172, 57222, 57343, 57400, 57558, 57682, 57783, 57787, 57816, 57843, 57864, 57964, 57967, 58015, 58388, 58436, 58444, 58544, 58553, 58623, 58816, 58872, 58888, 58889, 59002, 59036, 59080, 59212, 59331, 59498, 59837, 60268, 60468, 60523, 60576, 60600, 60630, 60937, 60982, 61034, 61052, 61189, 61210, 61261, 61352, 61481, 61579, 61695, 61762, 61768, 61917, 61992, 62001, 62103, 62701, 62733, 62770, 62929, 63096, 63174, 63193, 63205, 63324, 63464, 63482, 63543, 63570, 63721, 63909, 63949, 64389, 64399, 64417, 64457, 64480, 64488, 64531, 64630, 64696, 64731, 64865, 65063, 65258, 66073, 66079, 66186, 66213, 66215, 66395, 66478, 66611, 66990, 67402, 67501, 67739, 67778, 68173, 68209, 68216, 68695, 68766, 68841, 69330, 69415, 69444, 69646, 69703, 69773, 69932, 70485, 70546, 70855, 70969, 70980, 71003, 71053, 71126, 71136, 71161, 71353, 71510, 71533, 71757, 71830, 71871, 71893, 71956, 71967, 72000, 72037, 72122, 72176, 72187, 72260, 72387, 72397, 72426, 72524, 72528, 72999, 73128, 73195, 73321, 73464, 73503, 73528, 73594, 73762, 73819, 74010, 74072, 74128, 74274, 74351, 74498, 74564, 74688, 74702, 74841, 74858, 74940, 75299, 75463, 75477, 75571, 75584, 75682, 75792, 76034, 76108, 76293, 76409, 76447, 76478, 76631, 76743, 76839, 76924, 77143, 77156, 77229, 77364, 77517, 77753, 77896, 77961, 78281, 78285, 78460, 78554, 78566, 78797, 78875, 78955, 79215, 79236, 79360, 79489, 79768, 79959, 80070, 80486, 80601, 80677, 80974, 81073, 81350, 81358, 81446, 81510, 81708, 81958, 82184, 82351, 82379]
NOT_ANNOTATED_FRAMES_VAL = [265, 693, 733, 743, 954, 971, 1182, 1199, 1519, 1649, 1740, 1801, 1818, 1914, 2010, 2077, 2120, 2385, 2400, 2405, 2446, 2803, 2959, 3058, 3133, 3222, 3328, 3460, 3499, 3510, 3549, 3614, 3767, 4137, 4170, 4271, 4433, 4437, 4550, 4692, 4846, 4854, 4917, 4931, 5111, 5253, 5277, 5281, 5508, 5529, 5532, 5573, 5801, 5848, 6150, 6274, 6348, 6448, 6451, 6679, 6746, 6840, 6857, 7027, 7052, 7057, 7275, 7279, 7508, 7626, 7848, 8233, 8338, 8407, 8550, 8654, 8667, 8675, 8738, 8806, 8844, 8997, 9002, 9047, 9260, 9365, 9437, 9801, 10486, 10499, 10632, 10881, 10905, 11341, 11586, 11605, 12061, 12102, 12183, 12260, 12261, 12263, 12296, 12299, 12471, 12603, 12851, 13223, 13362, 13400, 13459, 13523, 13801, 13875, 13964, 14252, 14550, 14605, 14620, 14705, 14761, 14772, 14835, 15147, 15244, 15364, 15404, 15478, 15512, 15542, 15693, 15706, 15784, 15786, 15825, 15953, 15979, 16047, 16058, 16111, 16612, 16632, 16778, 16800, 16829, 17029, 17062, 17186, 17233, 17270, 17573, 17669, 17678, 18155, 18240, 18268, 18279, 18549, 18581, 18663, 18705, 18712, 18721, 18737, 18830, 19006, 19083, 19139, 19249, 19289, 19331, 19410, 19513, 19571, 19601, 19719, 19849, 20177, 20460, 20512, 20517, 20752, 20772, 20843, 20911, 21045, 21221, 21305, 21349, 21396, 21532, 21569, 21594, 21624, 21702, 21768, 21821, 22046, 22144, 22145, 22170, 22243, 22361, 22423, 22520, 22691, 22692, 22716, 22883, 22885, 23061, 23063, 23134, 23240, 23347, 23419, 23493, 23498, 23544, 23552, 23579, 23667, 23734, 23840, 23992, 23999, 24025, 24074, 24335, 24398, 24656, 24944, 25001, 25036, 25047, 25060, 25246, 25306, 25322, 25723, 25873, 26131, 26171, 26445, 26609, 26625, 26631, 26642, 26676, 26734, 26958, 27188, 27356, 27403, 27522, 27526, 27650, 27663, 27665, 27750, 27957, 27981, 28011, 28072, 28090, 28099, 28173, 28185, 28214, 28233, 28290, 28294, 28320, 28431, 28798, 28960, 29200, 29367, 29389, 29662, 29736, 29825, 29878, 29981, 30000, 30093, 30117, 30185, 30200, 30749, 30757, 30927, 31157, 31282, 31324, 31343, 31428, 31933, 31971, 31982, 32109, 32329, 32498, 32502, 32653, 33020, 33167, 33218, 33499, 33850, 33927, 34069, 34190, 34366, 34435, 34704, 34794, 34848, 34985, 35009, 35035, 35062, 35337, 35447, 35700, 35796, 35933, 36118, 36238, 36316, 36794, 36837, 36957, 37152, 37229, 37326, 37516, 37590, 37707, 37886, 37909, 37925, 38050, 38099, 38110, 38175, 38307, 38344, 38481, 38700, 38915, 38967, 39141, 39265, 39377, 39384, 39567, 39642, 39721, 39813, 39855, 40089, 40107, 40231, 40309, 40340, 40448]