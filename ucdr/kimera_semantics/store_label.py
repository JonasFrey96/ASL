import imageio
import numpy as np
import torch


def store_hard_label(label, path):
    # H,W np.int32 0 == invalid_class 40 == class number 39
    label = np.uint8(label)
    imageio.imwrite(path, label)


def store_soft_label(label, path, max_classes=40):
    # H,W, num_classes with normalized probabilities
    assert len(label.shape) == 3
    assert label.shape[2] == max_classes
    H, W, _ = label.shape
    idxs = torch.zeros((3, H, W), dtype=torch.uint8, device=label.device)
    values = torch.zeros((3, H, W), device=label.device)
    label_c = label.clone()
    max_val_10bit = 1023
    for i in range(3):
        idx = torch.argmax(label_c, dim=2)
        idxs[i] = idx.type(torch.uint8)

        m = torch.eye(max_classes)[idx] == 1
        values[i] = ((label_c[m] * max_val_10bit).reshape(H, W)).type(torch.int32)
        values[i][values[i] > max_val_10bit] = max_val_10bit
        label_c[m] = 0

    values = values.type(torch.int32)
    idxs = idxs.type(torch.int32)

    png = torch.zeros((H, W, 4), dtype=torch.int32, device=values.device)
    for i in range(3):
        png[:, :, i] = values[i]
        png[:, :, i] = torch.bitwise_or(png[:, :, i], idxs[i] << 10)

    png = png.cpu().numpy().astype(np.uint16)
    imageio.imwrite(path, png, format="PNG-FI", compression=9)
