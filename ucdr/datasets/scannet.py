from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms as tf
import random
import os
from pathlib import Path
import copy
import imageio
import pandas
import pickle

from ucdr.datasets import AugmentationList
from ucdr.utils import LabelLoaderAuto
from ucdr import UCDR_ROOT_DIR

__all__ = ["ScanNet"]


class ScanNet(Dataset):
    def __init__(
        self,
        root="/media/scratch2/jonfrey/datasets/scannet/",
        mode="train",
        scenes=[],
        output_trafo=None,
        output_size=(480, 640),
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
        sub=10,
        data_augmentation=True,
        label_setting="default",
        confidence_aux=0,
        labels_generic="",
    ):

        """
        Dataset dosent know if it contains replayed or normal samples !

        Some images are stored in 640x480 other ins 1296x968
        Warning scene0088_03 has wrong resolution -> Ignored
        Parameters
        ----------
        root : str, path to the ML-Hypersim folder
        mode : str, option ['train','val]
        """

        super(ScanNet, self).__init__()

        if mode.find("val") != -1:
            mode = mode.replace("val", "test")

        self._labels_generic = labels_generic
        self._sub = sub
        self._mode = mode

        self._confidence_aux = confidence_aux

        self._label_setting = label_setting

        if mode.find("_25k") == -1:
            self._load(root, mode, label_setting=label_setting)
            self._filter_scene(scenes)
        else:
            self._load_25k(root, mode)

        self._augmenter = AugmentationList(output_size, degrees, flip_p, jitter_bcsh)

        self._output_trafo = output_trafo
        self._data_augmentation = data_augmentation

        self._scenes_loaded = scenes
        self.unique = False

        self._label_loader = LabelLoaderAuto(root_scannet=root, confidence=self._confidence_aux)

        if self.aux_labels:
            self._preprocessing_hack()

    def make_replay(self, percentage=0.1):
        torch.manual_seed(0)
        s = self.global_to_local_idx.shape[0]
        sel_indices = torch.randperm(s)[: int(s * percentage)]
        self.global_to_local_idx = self.global_to_local_idx[sel_indices.numpy()]
        self.length = self.global_to_local_idx.shape[0]

    def __getitem__(self, index):
        """
        Option 5 (AUX_LABEL + VISU):
        img
        label
        aux_label
        aux_valid
        img_orginal
        """

        global_idx = self.global_to_local_idx[index]

        # Read Image and Label
        label, _ = self._label_loader.get(self.label_pths[global_idx])
        label = torch.from_numpy(label).type(torch.float32)[None, :, :]  # C H W -> contains 0-40
        label = [label]

        # Fetch auxilary label
        if self.aux_labels:
            _p = self.aux_label_pths[global_idx]
            if os.path.isfile(_p):
                aux_label, _ = self._label_loader.get(_p)
                aux_label = torch.from_numpy(aux_label).type(torch.float32)[None, :, :]
                label.append(aux_label)
            else:
                # TODO: Remove when this offline preprocessing failed
                if _p.find("_.png") != -1:
                    print(_p)
                    print("Processed not found")
                    _p = _p.replace("_.png", ".png")
                    aux_label, _ = self._label_loader.get(_p)
                    aux_label = torch.from_numpy(aux_label).type(torch.float32)[None, :, :]
                    label.append(aux_label)

        img = imageio.imread(self.image_pths[global_idx])
        img = torch.from_numpy(img).type(torch.float32).permute(2, 0, 1) / 255  # C H W range 0-1

        # Apply data augmentations
        if self._mode.find("train") != -1 and self._data_augmentation:
            img, label = self._augmenter.apply(img, label)
        else:
            img, label = self._augmenter.apply(img, label, only_crop=True)

        img_ori = img.clone()
        if self._output_trafo is not None:
            img = self._output_trafo(img)

        for k in range(len(label)):
            label[k] = label[k] - 1  # 0 == chairs 39 other prop  -1 invalid

        # REJECT LABEL
        if (label[0] != -1).sum() < 10:
            assert False
            idx = random.randint(0, len(self) - 1)
            if not self.unique:
                return self[idx]
            else:
                return False

        ret = (img, label[0].type(torch.int64)[0, :, :])

        if not self.aux_labels:
            ret += (label[0].type(torch.int64)[0, :, :], torch.tensor(False))
        else:
            ret += (label[1].type(torch.int64)[0, :, :], torch.tensor(True))

        ret += (img_ori,)
        return ret

    def __len__(self):
        return self.length

    def __str__(self):
        string = "=" * 90
        string += "\nScannet Dataset: \n"
        length = len(self)
        string += f"    Total Samples: {length}"
        string += f"  »  Mode: {self._mode} \n"
        string += f"    Replay: {self.replay}"
        string += f"  »  DataAug: {self._data_augmentation}"
        string += f"  »  DataAug Replay: {self._data_augmentation_for_replay}\n"
        string += "=" * 90
        return string

    def _get_mapping(self, root):
        tsv = os.path.join(root, "scannetv2-labels.combined.tsv")
        df = pandas.read_csv(tsv, sep="\t")
        self.df = df
        mapping_source = np.array(df["id"])
        mapping_target = np.array(df["nyu40id"])

        self.mapping = torch.zeros((int(mapping_source.max() + 1)), dtype=torch.int64)
        for so, ta in zip(mapping_source, mapping_target):
            self.mapping[so] = ta

    def _load(self, root, mode, train_val_split=0.2, label_setting="default"):
        self._get_mapping(root)

        self.train_test, self.scenes, self.image_pths, self.label_pths = self._load_cfg(root, train_val_split)

        self.image_pths = [os.path.join(root, i) for i in self.image_pths]
        self.label_pths = [os.path.join(root, i) for i in self.label_pths]

        if label_setting != "default":
            self.aux_label_pths = [i.replace("label-filt", label_setting) for i in self.label_pths]
            
            if not os.path.isfile(self.aux_label_pths[0]):
                print("LABEL FILE DOSENT EXIST -> MAYBE ON Workstation")
                self.aux_label_pths = [
                    i.replace(
                        root,
                        os.path.join(self._labels_generic, label_setting),
                    )
                    for i in self.aux_label_pths
                ]
                self.aux_labels = True
        else:
            self.aux_labels = False

        if mode.find("_strict") != -1:
            # TODO This is broken
            r = os.path.join(root, "scans")
            colors = [str(p) for p in Path(r).rglob("*color/*.jpg") if str(p).find("scene0088_03") == -1]
            fun = (
                lambda x: 100000000 * int(x.split("/")[-3].split("_")[0][5:])
                + int(1000000 * int(x.split("/")[-3].split("_")[1]))
                + int(x.split("/")[-1][:-4])
            )
            colors.sort(key=fun)
            self.train_test = []
            for c in colors:
                if int(c.split("/")[-3].split("_")[0][5:]) < 100 * (1 - train_val_split):
                    self.train_test.append("train_strict")
                else:
                    self.train_test.append("test_strict")

        self.valid_mode = np.array(self.train_test) == mode

        idis = np.array([int(p.split("/")[-1][:-4]) for p in self.image_pths])

        for k in range(idis.shape[0]):
            if idis[k] % self._sub != 0:
                self.valid_mode[k] = False

        self.global_to_local_idx = np.arange(self.valid_mode.shape[0])
        self.global_to_local_idx = (self.global_to_local_idx[self.valid_mode]).tolist()
        self.length = len(self.global_to_local_idx)

    @staticmethod
    def get_classes(train_val_split=0.2):
        data = pickle.load(open(f"cfg/dataset/scannet/scannet_trainval_{train_val_split}.pkl", "rb"))
        names, counts = np.unique(data["scenes"], return_counts=True)

        return names.tolist(), counts.tolist()

    def _load_cfg(self, root="not_def", train_val_split=0.2):
        # if pkl file already created no root is needed. used in get_classes
        try:
            data = pickle.load(open(f"cfg/dataset/scannet/scannet_trainval_{train_val_split}.pkl", "rb"))
            return (
                data["train_test"],
                data["scenes"],
                data["image_pths"],
                data["label_pths"],
            )
        except:
            pass
        return self._create_cfg(root, train_val_split)

    def _create_cfg(self, root, train_val_split=0.2):
        """Creates a pickle file containing all releveant information.
        For each train_val split a new pkl file is created
        """
        r = os.path.join(root, "scans")
        ls = [os.path.join(r, s[:9]) for s in os.listdir(r)]
        all_s = [os.path.join(r, s) for s in os.listdir(r)]

        scenes = np.unique(np.array(ls)).tolist()
        scenes = [s for s in scenes if int(s[-4:]) <= 100]  # limit to 100 scenes

        sub_scene = {s: [a for a in all_s if a.find(s) != -1] for s in scenes}
        for s in sub_scene.keys():
            sub_scene[s].sort()

        image_pths = []
        label_pths = []
        train_test = []
        scenes_out = []
        for s in scenes:
            for sub in sub_scene[s]:
                print(s)
                colors = [str(p) for p in Path(sub).rglob("*color/*.jpg")]
                labels = [str(p) for p in Path(sub).rglob("*label-filt/*.png")]
                fun = lambda x: int(x.split("/")[-1][:-4])
                colors.sort(key=fun)
                labels.sort(key=fun)

                for i, j in zip(colors, labels):
                    assert int(i.split("/")[-1][:-4]) == int(j.split("/")[-1][:-4])

                if len(colors) > 0:
                    assert len(colors) == len(labels)

                    nr_train = int(len(colors) * (1 - train_val_split))
                    nr_test = int(len(colors) - nr_train)
                    train_test += ["train"] * nr_train
                    train_test += ["test"] * nr_test
                    scenes_out += [s.split("/")[-1]] * len(colors)
                    image_pths += colors
                    label_pths += labels
                else:
                    print(sub, "Color not found")

        if root[-1] != "/":
            root += "/"
        image_pths = [i.replace(root, "") for i in image_pths]
        label_pths = [i.replace(root, "") for i in label_pths]
        data = {
            "train_test": train_test,
            "scenes": scenes_out,
            "image_pths": image_pths,
            "label_pths": label_pths,
        }

        pickle.dump(
            data,
            open(f"cfg/dataset/scannet/scannet_trainval_{train_val_split}.pkl", "wb"),
        )
        return train_test, scenes_out, image_pths, label_pths

    def _filter_scene(self, scenes):
        self.valid_scene = copy.deepcopy(self.valid_mode)

        if len(scenes) != 0:
            vs = np.zeros(len(self.valid_mode), dtype=bool)
            for sce in scenes:
                tmp = np.array(self.scenes) == sce
                vs[tmp] = True
            # self.valid_scene = np.logical_and(tmp, self.valid_scene )
        else:
            vs = np.ones(len(self.valid_mode), dtype=bool)
        self.valid_scene = vs * self.valid_scene

        self.global_to_local_idx = np.arange(self.valid_mode.shape[0])
        self.global_to_local_idx = (self.global_to_local_idx[self.valid_scene]).tolist()
        self.length = len(self.global_to_local_idx)

        # verify paths found:
        for global_idx in self.global_to_local_idx:
            if not os.path.exists(self.label_pths[global_idx]):
                print("Label not found ", self.label_pths[global_idx])
            if self.aux_labels:
                if not os.path.exists(self.aux_label_pths[global_idx]):
                    print("AuxLa not found ", self.aux_label_pths[global_idx])
            if not os.path.exists(self.image_pths[global_idx]):
                print("Image not found ", self.image_pths[global_idx])

    def _preprocessing_hack(self, force=False):
        """
        If training with aux_labels ->
        generates label for fast loading with a fixed certainty.
        """

        # check if this has already been performed
        aux_label, method = self._label_loader.get(self.aux_label_pths[self.global_to_local_idx[0]])
        if method == "RGBA":
            # This should always evaluate to true
            if self.aux_label_pths[self.global_to_local_idx[0]].find("_.png") == -1:
                print(
                    "self.aux_label_pths[self.global_to_local_idx[0]]",
                    self.aux_label_pths[self.global_to_local_idx[0]],
                    self.global_to_local_idx[0],
                )
                if (
                    os.path.isfile(self.aux_label_pths[self.global_to_local_idx[0]].replace(".png", "_.png"))
                    and os.path.isfile(self.aux_label_pths[self.global_to_local_idx[-1]].replace(".png", "_.png"))
                    and not force
                ):
                    # only perform simple renaming
                    print("Only do renanming")
                    self.aux_label_pths = [a.replace(".png", "_.png") for a in self.aux_label_pths]
                else:
                    print("Start multithread preprocessing of images")

                    def parallel(gtli, aux_label_pths, label_loader):
                        print("Start take care of: ", gtli[0], " - ", gtli[-1])
                        for i in gtli:
                            aux_label, method = label_loader.get(aux_label_pths[i])
                            imageio.imwrite(
                                aux_label_pths[i].replace(".png", "_.png"),
                                np.uint8(aux_label),
                            )

                    def parallel2(aux_pths, label_loader):
                        for a in aux_pths:
                            aux_label, method = label_loader.get(a)
                            imageio.imwrite(
                                a.replace(".png", "_.png"),
                                np.uint8(aux_label),
                            )

                    cores = 16
                    tasks = [t.tolist() for t in np.array_split(np.array(self.global_to_local_idx), cores)]

                    from multiprocessing import Process

                    for i in range(cores):
                        p = Process(
                            target=parallel2,
                            args=(
                                np.array(self.aux_label_pths)[np.array(tasks[i])].tolist(),
                                self._label_loader,
                            ),
                        )
                        p.start()
                    p.join()
                    print("Done multithread preprocessing of images")
                    self.aux_label_pths = [a.replace(".png", "_.png") for a in self.aux_label_pths]

    def _load_25k(self, root, mode, ratio=0.8):
        self.aux_labels = False
        self._get_mapping(root)
        pa = Path(os.path.join(root, "scannet_frames_25k"))
        paths = [str(s) for s in pa.rglob("*.jpg") if str(s).find("color") != -1]
        fun = (
            lambda x: int(x.split("/")[-3][5:9]) * 100000
            + int(x.split("/")[-3][10:]) * 10000
            + int(x.split("/")[-1][:-4])
        )
        paths.sort(key=fun)
        self.image_pths = paths
        self.label_pths = [p.replace("color", "label").replace("jpg", "png") for p in paths]
        idx_train = int(ratio * len(self.image_pths))
        if mode.find("train") != -1:
            self.global_to_local_idx = np.arange(idx_train)
        else:
            self.global_to_local_idx = np.arange(idx_train, len(self.label_pths))
        self.length = len(self.global_to_local_idx)


def test():
    # pytest -q -s src/datasets/ml_hypersim.py

    # Testing
    output_transform = tf.Compose(
        [
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # create new dataset
    dataset = ScanNet(
        root="/home/jonfrey/datasets/scannet",
        mode="val",
        scenes=[],
        output_trafo=output_transform,
        output_size=(480, 640),
        degrees=10,
        flip_p=0.5,
        jitter_bcsh=[0.3, 0.3, 0.3, 0.05],
        replay=True,
    )
    dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=0, pin_memory=False, batch_size=4)
    print(dataset)
    import time

    st = time.time()
    print("Start")
    for j, data in enumerate(dataloader):
        label = data[1]
        assert type(label) == torch.Tensor
        assert (label != -1).sum() > 10
        assert (label < -1).sum() == 0
        assert label.dtype == torch.int64
        if j % 10 == 0 and j != 0:

            print(j, "/", len(dataloader))
        if j == 100:
            break

    print("Total time", time.time() - st)
    # print(j)
    # img, label, _img_ori= dataset[i]    # C, H, W

    # label = np.uint8( label.numpy() * (255/float(label.max())))[:,:]
    # img = np.uint8( img.permute(1,2,0).numpy()*255 ) # H W C
    # imageio.imwrite(f'/home/jonfrey/tmp/{i}img.png', img)
    # imageio.imwrite(f'/home/jonfrey/tmp/{i}label.png', label)


if __name__ == "__main__":
    test()
