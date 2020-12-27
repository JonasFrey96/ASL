from .cityscapes import Cityscapes

datasets = {
    'cityscapes': Cityscapes,
}

def get_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
