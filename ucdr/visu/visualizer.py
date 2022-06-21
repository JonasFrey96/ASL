# STD
import os
import copy

# MISC
import numpy as np
import torch
import imageio
import cv2
from PIL import Image

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches
import pickle
from PIL import Image, ImageDraw
import skimage
from skimage import measure
from skimage.segmentation import mark_boundaries, find_boundaries
from ucdr.visu.colors import *
from ucdr.visu.flow_viz import *

__all__ = ["Visualizer", "MainVisualizer"]
from PIL import ImageFont, ImageDraw, Image


def find_font_size(text, font, image, target_width_ratio):
    tested_font_size = 100
    tested_font = ImageFont.truetype(font, tested_font_size)
    observed_width, observed_height = get_text_size(text, image, tested_font)
    estimated_font_size = tested_font_size / (observed_width / image.width) * target_width_ratio
    return round(estimated_font_size)


def get_text_size(text, image, font):
    im = Image.new("RGB", (image.width, image.height))
    draw = ImageDraw.Draw(im)
    return draw.textsize(text, font)


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    buf = np.asarray(buf)
    buf = Image.fromarray(buf)
    buf = buf.convert("RGB")
    return buf


def image_functionality(func):
    def wrap(*args, **kwargs):
        log = False
        if kwargs.get("method", "def") == "def":
            img = func(*args, **kwargs)
            log = True
        elif kwargs.get("method", "def") == "left":
            kwargs_clone = copy.deepcopy(kwargs)
            kwargs_clone["store"] = False
            kwargs_clone["jupyter"] = False
            res = func(*args, **kwargs_clone)
            args[0]._storage_left = res
        elif kwargs.get("method", "def") == "right":
            kwargs_clone = copy.deepcopy(kwargs)
            kwargs_clone["store"] = False
            kwargs_clone["jupyter"] = False
            res = func(*args, **kwargs_clone)
            args[0]._storage_right = res

        if args[0]._storage_right is not None and args[0]._storage_left is not None:
            img = np.concatenate([args[0]._storage_left, args[0]._storage_right], axis=1)
            args[0]._storage_left = None
            args[0]._storage_right = None
            log = True
        log *= not kwargs.get("not_log", False)
        if log:
            if args[0].pl_module is not None:
                log_exp = args[0].pl_module.logger is not None
            else:
                log_exp = False
            tag = kwargs.get("tag", "TagNotDefined")
            jupyter = kwargs.get("jupyter", False)
            # Each logging call is able to override the setting that is stored in the visualizer
            if kwargs.get("store", None) is not None:
                store = kwargs["store"]
            else:
                store = args[0]._store

            if kwargs.get("epoch", None) is not None:
                epoch = kwargs["epoch"]
            else:
                epoch = args[0]._epoch

            # Store & Log & Display in Jupyter
            if store:
                p = os.path.join(args[0].p_visu, f"{epoch}_{tag}.png")
                imageio.imwrite(p, img)

            if log_exp:
                H, W, C = img.shape
                ds = cv2.resize(img, dsize=(int(W / 2), int(H / 2)), interpolation=cv2.INTER_CUBIC)

                try:
                    from neptune.new.types import File

                    # logger == neptuneai
                    args[0].pl_module.logger.experiment[tag].log(File.as_image(np.float32(ds) / 255), step=epoch)
                except:
                    try:
                        # logger == tensorboard
                        args[0].pl_module.logger.experiment.add_image(
                            tag=tag, img_tensor=ds, global_step=epoch, dataformats="HWC"
                        )
                    except:
                        print("Tensorboard Logging and Neptune Logging failed !!!")
                        pass

            if jupyter:
                display(Image.fromarray(img))

        return func(*args, **kwargs)

    return wrap


class MainVisualizer:
    def __init__(self, p_visu, pl_module=None, epoch=0, store=True, num_classes=22):
        self.p_visu = p_visu
        self.pl_module = pl_module

        if not os.path.exists(self.p_visu):
            os.makedirs(self.p_visu)

        self._epoch = epoch
        self._store = store
        self._storage_left = None
        self._storage_right = None

        jet = cm.get_cmap("jet")
        self.SEG_COLORS = (np.stack([jet(v) for v in np.linspace(0, 1, num_classes)]) * 255).astype(np.uint8)
        self.SEG_COLORS_BINARY = (np.stack([jet(v) for v in np.linspace(0, 1, 2)]) * 255).astype(np.uint8)

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, store):
        self._store = store

    @image_functionality
    def plot_segmentation(self, seg, **kwargs):
        try:
            seg = seg.clone().cpu().numpy()
        except:
            pass

        if seg.dtype == np.bool:
            col_map = self.SEG_COLORS_BINARY
        else:
            col_map = self.SEG_COLORS
            seg = seg.round()

        H, W = seg.shape[:2]
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for i, color in enumerate(col_map):
            img[seg == i] = color[:3]

        return img

    @image_functionality
    def plot_image(self, img, **kwargs):
        """
        ----------
        img : CHW HWC accepts torch.tensor or numpy.array
              Range 0-1 or 0-255
        """
        try:
            img = img.clone().cpu().numpy()
        except:
            pass

        if img.shape[2] == 3:
            pass
        elif img.shape[0] == 3:
            img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
        else:
            raise Exception("Invalid Shape")
        if img.max() <= 1:
            img = img * 255

        img = np.uint8(img)
        return img

    @image_functionality
    def plot_matrix(
        self,
        data_matrix,
        higher_is_better=True,
        title="TitleNotDefined",
        max_tasks=None,
        max_tests=None,
        label_x=None,
        label_y=None,
        color_map="custom",
        col_map=None,
        **kwargs,
    ):

        if max_tasks is None and max_tests is None:
            max_tasks = data_matrix.shape[0]
            max_tests = data_matrix.shape[1]
        else:
            d1 = data_matrix.shape[0]
            d2 = data_matrix.shape[1]
            assert d2 <= max_tests
            data = np.zeros((max_tasks, max_tests))
            if max_tasks > d1:

                data[:d1, :d2] = data_matrix
            else:
                data[:max_tasks, :d2] = data_matrix[:max_tasks, :d2]

            data_matrix = data

        if label_y is None:
            label_y = ["Task  " + str(i) for i in range(max_tasks)]
        if label_x is None:
            label_x = ["Test " + str(i) for i in range(max_tests)]

        fig, ax = plt.subplots()
        if col_map != None:
            im = ax.imshow(data_matrix, cmap=col_map)
        else:
            if higher_is_better:
                if color_map == "custom":
                    im = ax.imshow(data_matrix, cmap=RG_PASTEL)
                else:
                    im = ax.imshow(data_matrix, cmap=cm.get_cmap("PiYG"))
            else:
                if color_map == "custom":
                    im = ax.imshow(data_matrix, cmap=RG_PASTEL_r)
                else:
                    im = ax.imshow(data_matrix, cmap=cm.get_cmap("PiYG_r"))

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(label_x)))
        ax.set_yticks(np.arange(len(label_y)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(label_x)
        ax.set_yticklabels(label_y)

        # Rotate the tick labels and set their alignment.

        # ax.invert_xaxis()
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(label_x)):
            for j in range(len(label_y)):
                text = ax.text(
                    i,
                    j,
                    data_matrix[j, i],
                    ha="center",
                    va="center",
                    color="w",
                    fontdict={"backgroundcolor": (0, 0, 0, 0.2)},
                )

        ax.set_title(title)
        # fig.tight_layout()
        arr = get_img_from_fig(fig, dpi=600)
        plt.close()
        return np.uint8(arr)

    @image_functionality
    def plot_lines_with_background(
        self,
        x,
        y,
        count=None,
        x_label="x",
        y_label="y",
        title="Title",
        task_names=None,
        **kwargs,
    ):
        # y = list of K  np.arrays with len N  . first tasks goes first
        # x : np.array N
        # both x and y might be just an array
        # optional x might be given for each y as a list
        # task_names: list of K: str
        # count: list of K points when next task started

        fig, ax = plt.subplots()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)

        if type(y) is not list:
            y = [y]

        keys = list(COL_DICT.keys())
        for j, y_ in enumerate(y):
            if type(x) is list:
                if len(x) == len(y):
                    x_ = x[j]
                else:
                    x_ = x[0]
            else:
                x_ = x
            if x_.shape[0] == 1:
                x_ = x_.repeat(2, 0)
                y_ = y_.repeat(2, 0)

            ax.plot(x_, y_, color=np.array(COL_DICT[keys[j]]) / 255)
        if task_names is not None:
            plt.legend(task_names)

        length = x.max() - x.min()

        nr_tasks = len(y)
        if count is None:
            for i in range(0, nr_tasks):
                print("Plotting")
                print((i) * length / nr_tasks)
                plt.axvspan(
                    (i) * length / nr_tasks,
                    (i + 1) * length / nr_tasks,
                    facecolor=np.array(COL_DICT[keys[i]]) / 255,
                    alpha=0.2,
                )
        else:
            start = x.min()
            for i in range(0, len(count)):

                stop = count[i]
                plt.axvspan(
                    max(start, x.min()),
                    min(stop, x.max()),
                    facecolor=np.array(COL_DICT[keys[i]]) / 255,
                    alpha=0.2,
                )
                start = stop

        arr = get_img_from_fig(fig, dpi=300)
        plt.close()
        return np.uint8(arr)

    @image_functionality
    def plot_cont_validation_eval(self, task_data, **kwargs):
        """
        res1 =  np.linspace(0., 0.5, 6)
        res2 =  np.linspace(0., 0.5, 6)*0.5
        res3 =  np.linspace(0., 0.5, 6)**2
        T1 = {'name': 'TrainTask1' ,'val_task_results': [(np.arange(0,6), res1), (np.arange(0,6), res2), (np.arange(0,6), res3) ] }
        T2 = {'name': 'TrainTask2' ,'val_task_results': [(np.arange(5,11), res1), (np.arange(5,11),res2), (np.arange(5,11),res3) ] }
        T3 = {'name': 'TrainTask3' ,'val_task_results': [(np.arange(10,16),res1), (np.arange(10,16),res2), (np.arange(10,16),res3) ] }
        task_data = [T1, T2]
        """

        line_styles = ["-", "--", "-.", ":"]
        steps_min = 999
        steps_max = 0
        for t in task_data:
            for v in t["val_task_results"]:
                if np.min(v[0]) < steps_min:
                    steps_min = np.min(v[0])
                if np.max(v[0]) > steps_max:
                    steps_max = np.max(v[0])
        span = steps_max - steps_min

        fig, axs = plt.subplots(len(task_data), sharex=True, sharey=True, figsize=(10, len(task_data) * 2))
        if len(task_data) == 1:
            axs = [axs]
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=1, wspace=0.2, hspace=0.8)
        for nr, task in enumerate(task_data):
            name = task["name"]
            axs[nr].set_title(name)
            axs[nr].set_xlabel("Step")
            axs[nr].set_ylabel("Acc")
            axs[nr].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            axs[nr].grid(True, linestyle="-", linewidth=1)
            for j, i in enumerate(task["val_task_results"]):
                k = list(COL_DICT.keys())
                val = COL_DICT[k[j]]
                val = [v / 255 for v in val]
                axs[nr].plot(i[0], i[1], color=val, linestyle=line_styles[j], label=task["eval_names"][j])
            plt.legend(loc="upper left")
        arr = get_img_from_fig(fig, dpi=600)
        plt.close()
        return np.uint8(arr)

    @image_functionality
    def plot_bar(
        self,
        data,
        x_label="Sample",
        y_label="Value",
        title="Bar Plot",
        sort=True,
        reverse=True,
        **kwargs,
    ):
        def check_shape(data):
            if len(data.shape) > 1:
                if data.shape[0] == 0:
                    data = data[0, :]
                elif data.shape[1] == 0:
                    data = data[:, 0]
                else:
                    raise Exception("plot_hist: Invalid Data Shape")
            return data

        if type(data) == list:
            pass
        elif type(data) == torch.Tensor:
            data = check_shape(data)
            data = list(data.clone().cpu())
        elif type(data) == np.ndarray:
            data = check_shape(data)
            data = list(data)
        else:
            raise Exception("plot_hist: Unknown Input Type" + str(type(data)))

        if sort:
            data.sort(reverse=reverse)

        fig = plt.figure()
        plt.bar(list(range(len(data))), data, facecolor=COL_MAP(2))

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        arr = get_img_from_fig(fig, dpi=300)
        plt.close()
        return np.uint8(arr)


def colorize(value, vmin=0.1, vmax=10, cmap="plasma"):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


class Visualizer:
    def __init__(self, p_visu, pl_module=None, epoch=0, store=True, num_classes=22):
        self.p_visu = p_visu
        self.pl_module = pl_module

        if not os.path.exists(self.p_visu):
            os.makedirs(self.p_visu)

        self._epoch = epoch
        self._store = store
        self._storage_left = None
        self._storage_right = None

        jet = cm.get_cmap("jet")
        self.SEG_COLORS = (np.stack([jet(v) for v in np.linspace(0, 1, num_classes)]) * 255).astype(np.uint8)

        if num_classes == 41:
            self.SEG_COLORS = np.array([(*SCANNET_COLOR_MAP[k], 255) for k in SCANNET_COLOR_MAP.keys()], dtype=np.uint8)

        self.SEG_COLORS_BINARY = (np.stack([jet(v) for v in np.linspace(0, 1, 2)]) * 255).astype(np.uint8)

        class DotDict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        self._meta_data = {
            "stuff_classes": list(ORDERED_DICT.keys()),
            "stuff_colors": list(SCANNET_COLOR_MAP.values()),
        }

        self._meta_data = DotDict(self._meta_data)

    @image_functionality
    def plot_detectron(
        self,
        img,
        label,
        text_off=False,
        alpha=0.5,
        draw_bound=True,
        scale=1,
        shift=2.5,
        font_size=12,
        **kwargs,
    ):
        img = self.plot_image(img, not_log=True)
        try:
            label = label.clone().cpu().numpy()
        except:
            pass
        label = label.astype(np.long)

        H, W, C = img.shape
        uni = np.unique(label)
        overlay = np.zeros_like(img)

        centers = []
        for u in uni:
            m = label == u
            col = self._meta_data["stuff_colors"][u]
            overlay[m] = col
            labels_mask = measure.label(m)
            regions = measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area, reverse=True)
            cen = np.mean(regions[0].coords, axis=0).astype(np.uint32)[::-1]

            centers.append((self._meta_data["stuff_classes"][u], cen))

        back = np.zeros((H, W, 4))
        back[:, :, :3] = img
        back[:, :, 3] = 255
        fore = np.zeros((H, W, 4))
        fore[:, :, :3] = overlay
        fore[:, :, 3] = alpha * 255
        img_new = Image.alpha_composite(Image.fromarray(np.uint8(back)), Image.fromarray(np.uint8(fore)))
        draw = ImageDraw.Draw(img_new)

        if not text_off:

            for i in centers:
                pose = i[1]
                pose[0] -= len(str(i[0])) * shift
                pose[1] -= font_size / 2

                # font = ImageFont.truetype("cfg/arial.ttf", font_size)
                # font = ImageFont.truetype("/usr/share/fonts/truetype/arial.ttf", font_size)

                draw.text(tuple(pose), str(i[0]), fill=(255, 255, 255, 128))

        img_new = img_new.convert("RGB")
        mask = mark_boundaries(np.array(img_new), label, color=(255, 255, 255))
        mask = mask.sum(axis=2)
        m = mask == mask.max()
        img_new = np.array(img_new)
        if draw_bound:
            img_new[m] = (255, 255, 255)
        return np.uint8(img_new)

    @image_functionality
    def plot_detectron_true_false(
        self,
        img,
        pred,
        gt,
        text_off=False,
        alpha=0.7,
        draw_bound=True,
        scale=1,
        shift=2.5,
        font_size=12,
        **kwargs,
    ):
        img = self.plot_image(img, not_log=True)
        try:
            pred = pred.clone().cpu().numpy()
        except:
            pass

        try:
            gt = gt.clone().cpu().numpy()
        except:
            pass

        label = (gt == pred).astype(np.long)

        H, W, C = img.shape
        uni = np.unique(label)
        overlay = np.zeros_like(img)

        centers = []
        for u in uni:
            m = label == u
            if u == 0:
                col = np.array(COL_DICT["red"])
            else:
                col = np.array(COL_DICT["green"])
            overlay[m] = col
            labels_mask = measure.label(m)
            regions = measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area, reverse=True)
            cen = np.mean(regions[0].coords, axis=0).astype(np.uint32)[::-1]

            centers.append((self._meta_data["stuff_classes"][u], cen))

        back = np.zeros((H, W, 4))
        back[:, :, :3] = img
        back[:, :, 3] = 255
        fore = np.zeros((H, W, 4))
        fore[:, :, :3] = overlay
        fore[:, :, 3] = alpha * 255
        img_new = Image.alpha_composite(Image.fromarray(np.uint8(back)), Image.fromarray(np.uint8(fore)))
        draw = ImageDraw.Draw(img_new)

        if not text_off:

            for i in centers:
                pose = i[1]
                pose[0] -= len(str(i[0])) * shift
                pose[1] -= font_size / 2

                font = ImageFont.truetype("cfg/arial.ttf", font_size)

                # font = ImageFont.truetype("/usr/share/fonts/truetype/arial.ttf", font_size)

                draw.text(tuple(pose), str(i[0]), fill=(255, 255, 255, 128), font=font)

        img_new = img_new.convert("RGB")
        mask = mark_boundaries(img_new, label, color=(255, 255, 255))
        mask = mask.sum(axis=2)
        m = mask == mask.max()
        img_new = np.array(img_new)
        if draw_bound:
            img_new[m] = (255, 255, 255)
        return np.uint8(img_new)

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, store):
        self._store = store

    def plot_flow(self, flow, **kwargs):
        # flow input HxWx2 or 2xHxW tensor or array, dtype float
        if type(flow) == torch.Tensor:
            if flow.device != "cpu":
                flow = flow.cpu()
            flow = flow.numpy()

        if flow.shape[0] == 2:
            flow = np.moveaxis(flow, [0, 1, 2], [2, 1, 0])
        flow = flow.astype(np.float32)

        img = flow_to_image(flow)
        return self.plot_image(img=img, **kwargs)

    def plot_depth(self, depth, vmin=0.1, vmax=10, **kwargs):
        img = colorize(depth, vmin=vmin, vmax=vmax)
        return self.plot_image(img=img, **kwargs)

    @image_functionality
    def plot_nyu_confusion_matrix(self, conf, title="NYU40 Confusion Matrix", **kwargs):
        with open("cfg/dataset/mappings/coco_nyu.pkl", "rb") as handle:
            mappings = pickle.load(handle)

        label_y = [str(i) for i in range(conf.shape[0] + 1)]
        label_x = [str(i) for i in range(conf.shape[1] + 1)]
        fig, ax = plt.subplots(figsize=(10, 10))
        conf_expa = np.zeros((conf.shape[0] + 1, conf.shape[1] + 1))
        conf_expa[1:, 1:] = conf

        diago = np.eye(conf_expa.shape[0], dtype=bool)
        conf_expa[diago == False] *= -1
        fac = -conf_expa.max() / conf_expa.min()
        conf_expa[diago == False] *= fac
        im = ax.imshow(conf_expa, cmap=cm.get_cmap("PiYG"))

        for i in range(len(label_y)):
            v = self.SEG_COLORS[i]
            vals = (v[0] / 255, v[1] / 255, v[2] / 255, 1)
            rect = patches.Rectangle((i - 0.5, -0.5), 1, 1, linewidth=1, edgecolor=None, facecolor=vals)
            rect2 = patches.Rectangle((0 - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor=None, facecolor=vals)
            ax.add_patch(rect)
            ax.add_patch(rect2)

        ax.set_xticks(np.arange(len(label_x)))
        ax.set_yticks(np.arange(len(label_y)))
        ax.set_xticklabels(list(mappings["nyu_name_id"].keys())[:])
        ax.set_yticklabels(list(mappings["nyu_name_id"].keys())[:])
        ax.xaxis.tick_top()

        plt.xlabel("predicted_class")
        plt.ylabel("targets")
        ax.yaxis.set_label_position("right")

        ax.set_axisbelow(True)
        ax.yaxis.grid(color="gray", linestyle="dashed")
        ax.xaxis.grid(color="gray", linestyle="dashed")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        ax.set_title(title)

        arr = get_img_from_fig(fig, dpi=600)
        plt.close()
        return np.uint8(arr)

    @image_functionality
    def plot_segmentation(self, seg, **kwargs):
        """
        Input segmentation NYU: 0 unlabeled, 1 wall, 40 highest label
        """
        if not type(seg).__module__ == np.__name__:
            try:
                seg = seg.clone().cpu().numpy()
            except:
                try:
                    seg = seg.numpy()
                except:
                    pass

        if seg.dtype == np.bool:
            col_map = self.SEG_COLORS_BINARY
        else:
            col_map = self.SEG_COLORS
            # seg = seg.astype(np.float32)
            # seg = seg.round()

        H, W = seg.shape[:2]
        img = np.zeros((H, W, 3), dtype=np.uint8)

        for i, color in enumerate(col_map):
            img[seg == i] = color[:3]
        return img

    @image_functionality
    def plot_bar(
        self,
        data,
        x_label="Sample",
        y_label="Value",
        title="Bar Plot",
        sort=True,
        reverse=True,
        **kwargs,
    ):
        def check_shape(data):
            if len(data.shape) > 1:
                if data.shape[0] == 0:
                    data = data[0, :]
                elif data.shape[1] == 0:
                    data = data[:, 0]
                else:
                    raise Exception("plot_hist: Invalid Data Shape")
            return data

        if type(data) == list:
            pass
        elif type(data) == torch.Tensor:
            data = check_shape(data)
            data = list(data.clone().cpu())
        elif type(data) == np.ndarray:
            data = check_shape(data)
            data = list(data)
        else:
            raise Exception("plot_hist: Unknown Input Type" + str(type(data)))

        if sort:
            data.sort(reverse=reverse)

        fig = plt.figure()
        plt.bar(list(range(len(data))), data, facecolor=COL_MAP(2))

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        arr = get_img_from_fig(fig, dpi=300)
        plt.close()
        return np.uint8(arr)

    @image_functionality
    def plot_image(self, img, **kwargs):
        try:
            img = img.clone().cpu().numpy()
        except:
            pass
        if img.shape[2] == 3:
            pass
        elif img.shape[0] == 3:
            img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
        else:
            raise Exception("Invalid Shape")
        if img.max() <= 1:
            img = img * 255
        img = np.uint8(img)
        return img


def test():
    # pytest -q -s src/visu/visualizer.py
    visu = Visualizer(os.getenv("HOME") + "/tmp", pl_module=None, epoch=0, store=False, num_classes=41)
    # vis = MainVisualizer( p_visu='/home/jonfrey/tmp', logger=None, epoch=0, store=True, num_classes=41)
    # x = np.arange(100)
    # y = [ np.random.normal(0, 1, 100), np.random.normal(0.5, 0.2, 100), np.random.normal(-0.5, 0.1, 100), np.random.normal(0.2, 1, 100)]
    # vis.plot_lines_with_bachground(x,y, count=[5,55,60,100], task_names=['a', 'b', 'c', 'd'])


if __name__ == "__main__":
    test()
