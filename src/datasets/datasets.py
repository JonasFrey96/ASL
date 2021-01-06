from .cityscapes import Cityscapes
from .nyu_v2 import NYUv2
datasets = {
    'cityscapes': Cityscapes,
    'nyuv2': NYUv2,
}

def get_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
