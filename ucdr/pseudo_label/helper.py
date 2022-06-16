import numpy as np
from torchvision import transforms as tf
from PIL import Image

__all__ = ["readImage"]


def readImage(filename, H=640, W=1280, scale=True):
    _crop_center = tf.CenterCrop((H, W))
    img = Image.open(filename)
    if scale:
        img = _crop_center(img)
    return np.array(img)
