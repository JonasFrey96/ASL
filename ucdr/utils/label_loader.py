import pandas
import numpy as np
import imageio

import torch
import os

__all__ = ["LabelLoaderAuto"]


class LabelLoaderAuto:
    def __init__(self, root_scannet="no_path", confidence=0, H=968, W=1296):
        self._get_mapping(root_scannet)
        self._confidence = confidence
        # return label between 0-40

        self.max_classes = 40
        # H, W = imageio.imread(
        #   os.path.join(root_scannet, "scans/scene0000_00/label-filt/0.png")
        # ).shape
        self.label = np.zeros((H, W, self.max_classes))
        iu16 = np.iinfo(np.uint16)
        mask = np.full((H, W), iu16.max, dtype=np.uint16)
        self.mask_low = np.right_shift(mask, 6, dtype=np.uint16)

    def get(self, path):
        img = imageio.imread(path, format="PNG-FI")
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                H, W, _ = img.shape
                self.label = np.zeros((H, W, self.max_classes))
                for i in range(3):
                    prob = np.bitwise_and(img[:, :, i], self.mask_low) / 1023
                    cls = np.right_shift(img[:, :, i], 10, dtype=np.uint16)
                    m = np.eye(self.max_classes)[cls] == 1
                    self.label[m] = prob.reshape(-1)
                m = np.max(self.label, axis=2) < self._confidence
                self.label = np.argmax(self.label, axis=2).astype(np.int32) + 1
                self.label[m] = 0
                method = "RGBA"
            else:
                raise Exception("Type not know")
        elif len(img.shape) == 2 and img.dtype == np.uint8:
            self.label = img.astype(np.int32)
            method = "FAST"
        elif len(img.shape) == 2 and img.dtype == np.uint16:
            if not self.scannet:
                raise Exception("ScanNet Mapping was not loaded given wrong Scannet Path")
            self.label = torch.from_numpy(img.astype(np.int32)).type(torch.float32)[None, :, :]
            sa = self.label.shape
            self.label = self.label.flatten()
            self.label = self.mapping[self.label.type(torch.int64)]
            self.label = self.label.reshape(sa).numpy().astype(np.int32)[0]
            method = "MAPPED"
        else:
            raise Exception("Type not know")
        return self.label, method

    def get_probs(self, path):
        img = imageio.imread(path, format="PNG-FI")
        assert len(img.shape) == 3
        assert img.shape[2] == 4
        H, W, _ = img.shape
        probs = np.zeros((H, W, self.max_classes))
        for i in range(3):
            prob = np.bitwise_and(img[:, :, i], self.mask_low) / 1023
            cls = np.right_shift(img[:, :, i], 10, dtype=np.uint16)
            m = np.eye(self.max_classes)[cls] == 1
            probs[m] = prob.reshape(-1)

        return probs

    def _get_mapping(self, root):
        try:
            tsv = os.path.join(root, "scannetv2-labels.combined.tsv")
            df = pandas.read_csv(tsv, sep="\t")
            mapping_source = np.array(df["id"])
            mapping_target = np.array(df["nyu40id"])

            self.mapping = torch.zeros((int(mapping_source.max() + 1)), dtype=torch.int64)
            for so, ta in zip(mapping_source, mapping_target):
                self.mapping[so] = ta
            self.scannet = True
        except:
            print("LabelLoaderAuto failed to load ScanNet labels")
            self.scannet = False


if __name__ == "__main__":
    lla = LabelLoaderAuto("/home/jonfrey/Datasets/scannet")
    import time

    st = time.time()
    for i in range(40):
        a, m = lla.get("/home/jonfrey/Datasets/scannet/scans/scene0000_00/label-filt/0.png")
        # b, m = lla.get( "/home/jonfrey/Datasets/labels_generated/new_format_scene0-10/scans/scene0000_00/new_format_scene0-10/0.png")
        # c, m = lla.get( "/home/jonfrey/Datasets/scannet/scannet_frames_25k/scene0010_01/label/001300.png")
    print(time.time() - st, m)
    # print(a.shape, a.max(), a.min())
    # print(b.shape, b.max(), b.min())
    # print(c.shape, c.max(), c.min())
