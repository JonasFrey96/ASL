# STD
import copy
import sys
import os
import time
import shutil
import argparse
import logging
import signal
import pickle
import math
from pathlib import Path
import random 
from math import pi
from math import ceil

# MISC 
import numpy as np
import pandas as pd

# DL-framework
import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from torchvision import transforms
# MODULES
from models import FastSCNN
from datasets import get_dataset
from loss import  MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from .metrices import IoU, PixAcc
from visu import Visualizer
__all__ = ['Network']

class Network(LightningModule):
  def __init__(self, exp, env):
    super().__init__()
    self._exp = exp
    self._env = env
    self.hparams['lr'] = self._exp['lr']

    if self._exp['model']['name'] == 'Fast-SCNN': 
      self.model = FastSCNN(**self._exp['model']['cfg'])
    else:
      raise Exception('Model name not implemented')
    self.criterion = MixSoftmaxCrossEntropyLoss(aux=False, aux_weight=0.001,
      ignore_index=-1)
    p_visu = os.path.join( self._exp['name'], 'visu')
    
    self.visualizer = Visualizer(
      p_visu=p_visu,
      writer=None,
      num_classes=self._exp['model']['cfg']['num_classes'])

    self.metric_val_IoU = IoU(num_classes=self._exp['model']['cfg']['num_classes'])
    self.metric_val_PixAcc = PixAcc()
    
    self.logged_images_train = 0
    self.logged_images_val = 0
    self.logged_images_test = 0

  def forward(self, batch):
    outputs = self.model(batch)
    return outputs
    
  def on_train_start(self):
    self.visualizer.writer = self.logger.experiment

  def on_epoch_start(self):
    self.visualizer.epoch = self.current_epoch
    self.logged_images_train = 0
    self.logged_images_val = 0
    self.logged_images_test = 0

  def training_step(self, batch, batch_idx):
    images = batch[0]
    targets = batch[1]
    st = time.time()
    outputs = self(images)
    loss = self.criterion(outputs, targets)

    if self._exp['visu'].get('train_images',0) > self.logged_images_train:
      pred = torch.argmax(outputs[0], 1)
      self.logged_images_train += 1
      self.visualizer.plot_segmentation(tag='pred_train', seg=pred[0])
      self.visualizer.plot_segmentation(tag='gt_train', seg=targets[0])
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    images = batch[0]
    targets = batch[1]
    outputs = self(images)
    loss = self.criterion(outputs, targets)

    pred = torch.argmax(outputs[0], 1)
    self.metric_val_IoU(pred, targets)
    self.metric_val_PixAcc(pred, targets)
    self.log('metric_val_IoU', self.metric_val_IoU, on_step=True, on_epoch=True)
    self.log('metric_val_PixAcc', self.metric_val_PixAcc, on_step=True, on_epoch=True)

    if self._exp['visu'].get('val_images',0) > self.logged_images_val:
      self.logged_images_val += 1
      self.visualizer.plot_segmentation(tag='pred_val', seg=pred[0])

    return {'val_loss': loss}

  #def validation_epoch_end(self, outputs):

  #def training_epoch_end(self, outputs):

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(
        [{'params': self.model.parameters()}], lr=self.hparams['lr'])
    if self._exp.get('lr_scheduler',{}).get('active', False):
      #polynomial lr-scheduler
      init_lr = self.hparams['lr']
      max_epochs = self._exp['lr_scheduler']['cfg']['max_epochs'] 
      target_lr = self._exp['lr_scheduler']['cfg']['target_lr'] 
      power = self._exp['lr_scheduler']['cfg']['power'] 
      lambda_lr= lambda epoch: (((max_epochs-epoch)/max_epochs)**(power) ) + (1-(((max_epochs -epoch)/max_epochs)**(power)))*target_lr/init_lr
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1, verbose=True)
      ret = [optimizer], [scheduler]
    else:
      ret = [optimizer]
    return ret
  
  def train_dataloader(self):
    input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    dataset_train = get_dataset(
      **self._exp['d_train'],
      root = self._env['cityscapes'],
      transform = input_transform,
    )
                                      
    # initalize train and validation indices
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
      shuffle = self._exp['loader']['shuffle'],
      num_workers = ceil(self._exp['loader']['num_workers']/torch.cuda.device_count()),
      pin_memory = self._exp['loader']['pin_memory'],
      batch_size = self._exp['loader']['batch_size'], 
      drop_last = True)
    return dataloader_train
    
  def val_dataloader(self):
    input_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    dataset_val = get_dataset(
      **self._exp['d_val'],
      root = self._env['cityscapes'],
      transform = input_transform
    )

    dataloader_val = torch.utils.data.DataLoader(dataset_val,
      shuffle = False,
      num_workers = ceil(self._exp['loader']['num_workers']/torch.cuda.device_count()),
      pin_memory = self._exp['loader']['pin_memory'],
      batch_size = self._exp['loader']['batch_size'])
    return dataloader_val