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
from pytorch_lightning import metrics as pl_metrics
from torch.nn import functional as F
# MODULES
from models import FastSCNN
from datasets import get_dataset
from loss import  MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
#from .metrices import IoU, PixAcc
from visu import Visualizer
from lightning import IoUTorch 

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
    self.criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=False, aux_weight=0.4,
      ignore_index=-1)
    p_visu = os.path.join( self._exp['name'], 'visu')
    
    self.visualizer = Visualizer(
      p_visu=p_visu,
      writer=None,
      num_classes=self._exp['model']['cfg']['num_classes']+1)

    self.train_acc = pl_metrics.classification.Accuracy()
    self.val_acc = pl_metrics.classification.Accuracy()

    self.train_iou = IoUTorch(self._exp['model']['cfg']['num_classes'])
    self.val_iou = IoUTorch(self._exp['model']['cfg']['num_classes'])

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
    target = batch[1]
    outputs = self(images)
    #_loss = self.criterion(outputs, target)
    loss =  F.cross_entropy(outputs[0], target, ignore_index=-1 ) 
    #print(_loss, loss)
    if self._exp['visu'].get('train_images',0) > self.logged_images_train:
      pred = torch.argmax(outputs[0], 1)
      self.logged_images_train += 1
      self.visualizer.plot_segmentation(tag='pred_train', seg=pred[0])
      self.visualizer.plot_segmentation(tag='gt_train', seg=target[0])

    if False: 
      pred = torch.argmax(outputs[0], 1)
      train_iou = self.train_iou(pred,target)
      # calculates acc only for valid labels
      m  =  target > -1
      train_acc = self.train_acc(pred[m], target[m])
      
      self.log('train_iou', self.train_iou, on_step=True, on_epoch=True)
      self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

    self.log('train_loss', loss, on_step=True, on_epoch=True)
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    images = batch[0]
    target = batch[1]    #[-1,n] labeled with -1 should not induce a loss
    outputs = self(images)
    loss = self.criterion(outputs, target)

    pred = torch.argmax(outputs[0], 1)
    self.val_iou(pred,target)
    # calculates acc only for valid labels
    m  =  target > -1
    self.val_acc(pred[m], target[m])
    
    self.log('val_iou', self.val_iou, on_step=True, on_epoch=True)
    self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)

    if self._exp['visu'].get('val_images',0) > self.logged_images_val:
      self.logged_images_val += 1
      self.visualizer.plot_segmentation(tag='pred_val', seg=pred[0])

    self.log('val_loss', loss, on_step=True, on_epoch=True)

    return {'val_loss': loss}

  # def validation_epoch_end(self, outputs):
  #   print("END")

  def configure_optimizers(self):
    if self._exp['optimizer']['name'] == 'ADAM':
      optimizer = torch.optim.Adam(
          [{'params': self.model.parameters()}], lr=self.hparams['lr'])
    elif self._exp['optimizer']['name'] == 'SGD':
      optimizer = torch.optim.SGD(
          [{'params': self.model.parameters()}], lr=self.hparams['lr'],
          **self._exp['optimizer']['sgd_cfg'] )
    else:
      raise Exception

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


import pytest
class TestLightning:
  def  test_iout(self):
    input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    di = {
      'name': 'cityscapes',
      'base_size': 1024,
      'crop_size': 768,
      'split': 'val',
      'mode': 'val',
      'overfit': 5 }
    root = '/media/scratch1/jonfrey/datasets/Cityscapes'
    self.dataset_train = get_dataset(
      **di,
      root = root,
      transform = input_transform,
    )
    BS, H, W = 1,100,100
    NC = 4
    target = torch.zeros( (BS,H,W) )
    
    target[:,:10,:10] = -1 # 100 pixels invalid
    target[:,20:,20:] = 2 # 640 pixel correct
    print((target==0).sum())
    pred = torch.zeros( (BS,H,W) )
    pred[:,:,:] = 2
    


    #self.val_iou = pl_metrics.functional.classification.iou()
    self.val_acc = pl_metrics.classification.Accuracy()


    pred = pred.type(torch.int) + 1
    target = target.type(torch.int) + 1
    m  =  target > 0
    acc1 = self.val_acc(pred[m], target[m]) # only call acc for labeld pixels !    
    
    pred = pred * (target > 0).type(pred.dtype) 
    iou = pl_metrics.functional.classification.iou(pred, target, num_classes=NC, reduction='none')[1:]
    m = torch.zeros( iou.shape, device=iou.device, dtype=torch.bool)
    for i in range(NC-1):
      if ((target-1) == i).any():
        m[i] = True
    masked_iou = iou[m]
    mIOU = masked_iou.mean()
    assert acc_or ==  acc1