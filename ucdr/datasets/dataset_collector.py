from ucdr.datasets.scannet import ScanNet

datasets = {
    "scannet": ScanNet,
}

__all__ = ["get_dataset"]


def get_dataset(name, env, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](root=env[name], labels_generic=env["labels_generic"], **kwargs)
