import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors
__all__ = ['SCANNET_COLOR_MAP', 'RG_PASTEL', "RG_PASTEL_r"]
# SCANNET_COLOR_MAP = {
#     (0, (0, 0, 0)),
#     ('wall', (174, 199, 232)),
#     ('floor', (152, 223, 138)),
#     ('cabinet', (31, 119, 180)),
#     ('bed', (255, 187, 120)),
#     ('chair', (188, 189, 34)),
#     ('sofa', (140, 86, 75)),
#     ('table', (255, 152, 150)),
#     ('door', (214, 39, 40)),
#     ('window', (197, 176, 213)),
#     ('bookshelf', (148, 103, 189)),
#     ('picture', (196, 156, 148)),
#     ('counter', (23, 190, 207)),
#     ('blinds', (178, 76, 76)),
#     ('desk', (247, 182, 210)),
#     ('shelves', (66, 188, 102)),
#     ('curtain', (219, 219, 141)),
#     ('dresser', (140, 57, 197)),
#     ('pillow', (202, 185, 52)),
#     ('mirror', (51, 176, 203)),
#     ('floormat', (200, 54, 131)),
#     ('clothes', (92, 193, 61)),
#     ('ceiling', (78, 71, 183)),
#     ('books', (172, 114, 82)),
#     ('refrigerator', (255, 127, 14)),
#     ('television', (91, 163, 138)),
#     ('paper', (153, 98, 156)),
#     ('towel', (140, 153, 101)),
#     ('showercurtain', (158, 218, 229)),
#     ('box', (100, 125, 154)),
#     ('whiteboard', (178, 127, 135)),
#     ('person', (120, 185, 128)),
#     ('nightstand', (146, 111, 194)),
#     ('toilet', (44, 160, 44)),
#     ('sink', (112, 128, 144)),
#     ('lamp', (96, 207, 209)),
#     ('bathtub', (227, 119, 194)),
#     ('bag', (213, 92, 176)),
#     ('otherstructure', (94, 106, 211)),
#     ('otherfurniture', (82, 84, 163)),
#     ('otherprop', (100, 85, 144)),
# }

col = { "red":[255,89,94],
 "yellow":[255,202,58],
 "green":[138,201,38],
 "blue":[25,130,196],
 "purple":[106,76,147] }

li = [ [*(v),255] for v in col.values()]
li = (np.array(li)/255).tolist()
COL_MAP = cm.colors.ListedColormap(li)


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
RG_PASTEL = make_colormap(
    [ COL_MAP(0)[:3] , COL_MAP(2)[:3]] )
RG_PASTEL_r = make_colormap(
    [ COL_MAP(0)[:3] , COL_MAP(2)[:3]] )

from collections import OrderedDict
od = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('wall', (174, 199, 232)),
    ('floor', (152, 223, 138)),
    ('cabinet', (31, 119, 180)),
    ('bed', (255, 187, 120)),
    ('chair', (188, 189, 34)),
    ('sofa', (140, 86, 75)),
    ('table', (255, 152, 150)),
    ('door', (214, 39, 40)),
    ('window', (197, 176, 213)),
    ('bookshelf', (148, 103, 189)),
    ('picture', (196, 156, 148)),
    ('counter', (23, 190, 207)),
    ('blinds', (178, 76, 76)),
    ('desk', (247, 182, 210)),
    ('shelves', (66, 188, 102)),
    ('curtain', (219, 219, 141)),
    ('dresser', (140, 57, 197)),
    ('pillow', (202, 185, 52)),
    ('mirror', (51, 176, 203)),
    ('floormat', (200, 54, 131)),
    ('clothes', (92, 193, 61)),
    ('ceiling', (78, 71, 183)),
    ('books', (172, 114, 82)),
    ('refrigerator', (255, 127, 14)),
    ('television', (91, 163, 138)),
    ('paper', (153, 98, 156)),
    ('towel', (140, 153, 101)),
    ('showercurtain', (158, 218, 229)),
    ('box', (100, 125, 154)),
    ('whiteboard', (178, 127, 135)),
    ('person', (120, 185, 128)),
    ('nightstand', (146, 111, 194)),
    ('toilet', (44, 160, 44)),
    ('sink', (112, 128, 144)),
    ('lamp', (96, 207, 209)),
    ('bathtub', (227, 119, 194)),
    ('bag', (213, 92, 176)),
    ('otherstructure', (94, 106, 211)),
    ('otherfurniture', (82, 84, 163)),
    ('otherprop', (100, 85, 144)),
])

SCANNET_COLOR_MAP = {i:v for i,v in enumerate(od.values())}