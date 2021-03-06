import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors

SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

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

# from collections import OrderedDict
# od = OrderedDict([
#     ('unlabeled', (0, 0, 0)),
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
# ])