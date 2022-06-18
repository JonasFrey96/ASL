import torch
import numpy as np
import os
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics import ConfusionMatrix

def np_nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()
    
def nanmean(data, **args):
    """Finding the mean along dim"""
    mask = ~torch.isnan(data)
    masked = data[mask]  # Apply the mask using an element-wise multiply
    return masked.sum() / mask.sum()  # Find the average!


class TorchSemanticsMeter(torch.nn.Module):
    def __init__(self, number_classes):
        super().__init__()
        self.conf_mat = None
        self.conf_mat_np = None
        self.number_classes = number_classes
        self.mask = torch.zeros((number_classes,), requires_grad=False, dtype=torch.bool)

        self.mask_np = np.zeros((number_classes,), dtype=np.bool)

        self.cm = ConfusionMatrix(num_classes=number_classes)

    def clear(self):
        self.conf_mat = None
        self.conf_mat_np = None

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    @torch.no_grad()
    def update(self, preds, truths):
        truths = truths.detach()
        preds = preds.detach()
        un = torch.unique(truths)
        self.mask[un[un != -1].type(torch.long)] = True
        preds = preds.flatten()
        truths = truths.flatten()
        valid_pix_ids = truths != -1
        preds = preds[valid_pix_ids]
        truths = truths[valid_pix_ids]
        conf_mat_current = self.cm(truths, preds).T

        if self.conf_mat is None:
            self.conf_mat = conf_mat_current
        else:
            self.conf_mat += conf_mat_current

    @torch.no_grad()
    def measure(self):
        conf_mat = self.conf_mat
        conf_mat_np = self.conf_mat_np
        norm_conf_mat = (conf_mat.T / conf_mat.type(torch.float32).sum(axis=1)).T
        missing_class_mask = torch.isnan(norm_conf_mat.sum(1))
        existing_class_mask = ~missing_class_mask
        class_average_accuracy = nanmean(torch.diagonal(norm_conf_mat))
        total_accuracy = torch.sum(torch.diagonal(conf_mat)) / torch.sum(conf_mat)
        ious = torch.zeros(self.number_classes, device=self.conf_mat.device)
        for class_id in range(self.number_classes):
            ious[class_id] = conf_mat[class_id, class_id] / (
                torch.sum(conf_mat[class_id, :]) + torch.sum(conf_mat[:, class_id]) - conf_mat[class_id, class_id]
            )

        miou_valid_class = torch.mean(ious[existing_class_mask])
        # miou_valid_class_np = np.mean(ious_np[existing_class_mask_np])

        return miou_valid_class, total_accuracy, class_average_accuracy


class SemanticsMeter:
    def __init__(self, number_classes):
        self.conf_mat = None
        self.number_classes = number_classes
        self.mask = np.zeros((number_classes,), dtype=np.bool)

    def clear(self):
        self.conf_mat = None

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        un = np.unique(truths)
        self.mask[un[un != -1].astype(np.uint32)] = True
        preds = preds.flatten()
        truths = truths.flatten()
        valid_pix_ids = truths != -1
        preds = preds[valid_pix_ids]
        truths = truths[valid_pix_ids]
        conf_mat_current = confusion_matrix(truths, preds, labels=list(range(self.number_classes)))
        if self.conf_mat is None:
            self.conf_mat = conf_mat_current
        else:
            self.conf_mat += conf_mat_current

    def measure(self):
        conf_mat = self.conf_mat
        norm_conf_mat = np.transpose(np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

        missing_class_mask = np.isnan(norm_conf_mat.sum(1))  # missing class will have NaN at corresponding class
        exsiting_class_mask = ~missing_class_mask

        class_average_accuracy = np_nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)
        ious = np.zeros(self.number_classes)
        for class_id in range(self.number_classes):
            ious[class_id] = conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) - conf_mat[class_id, class_id]
            )
        miou_valid_class = np.mean(ious[exsiting_class_mask])
        return miou_valid_class, total_accuracy, class_average_accuracy
