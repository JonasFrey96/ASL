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
import logging

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
from pytorch_lightning.utilities import rank_zero_info

from torch.nn import functional as F
# MODULES
from models import FastSCNN
from datasets import get_dataset
from loss import  MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
#from .metrices import IoU, PixAcc
from visu import Visualizer
from lightning import meanIoUTorchCorrect
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

    self.test_acc = pl_metrics.classification.Accuracy()
    self.train_acc = pl_metrics.classification.Accuracy()
    self.val_acc = pl_metrics.classification.Accuracy()

    self.test_mIoU = meanIoUTorchCorrect(self._exp['model']['cfg']['num_classes'])
    self.train_mIoU = meanIoUTorchCorrect(self._exp['model']['cfg']['num_classes'])
    self.val_mIoU = meanIoUTorchCorrect(self._exp['model']['cfg']['num_classes'])

    self.logged_images_train = 0
    self.logged_images_val = 0
    self.logged_images_test = 0
    
    self._task_name = 'NotDefined' # is used for model checkpoint nameing

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
    loss = F.cross_entropy(outputs[0], target, ignore_index=-1 ) 

    self.log('train_loss', loss, on_step=True, on_epoch=True)
    return {'loss': loss, 'pred': outputs[0], 'target': target}
  
  def training_step_end(self, outputs):
    # Logging + Visu
    if ( self._exp['visu'].get('train_images',0) > self.logged_images_train and 
         self.current_epoch % self._exp['visu'].get('every_n_epochs',1) == 0):
      pred = torch.argmax(outputs['pred'], 1)
      self.logged_images_train += 1
      self.visualizer.plot_segmentation(tag=f'', seg=pred[0], method='right')
      self.visualizer.plot_segmentation(tag=f'train_gt_left_pred_right_{self.logged_images_train}', seg=outputs['target'][0], method='left')

    if False: 
      pred = torch.argmax(outputs['pred'], 1)
      train_mIoU = self.train_mIoU(pred,outputs['target'])
      # calculates acc only for valid labels
      m  =  target > -1
      train_acc = self.train_acc(pred[m], outputs['target'][m])
      
      self.log('train_mIoU', self.train_iou, on_step=True, on_epoch=True)
      self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
      return {'loss': outputs['loss'],
              'progress_bar': { 'mIoU': train_mIoU, 
                                'Acc': train_acc} }
    
    return {'loss': outputs['loss']}
  
  def validation_step(self, batch, batch_idx):
    images = batch[0]
    target = batch[1]    #[-1,n] labeled with -1 should not induce a loss
    outputs = self(images)
    loss = F.cross_entropy(outputs[0], target, ignore_index=-1 ) 

    pred = torch.argmax(outputs[0], 1)

    self.log('val_loss', loss, on_step=True, on_epoch=True)

    return {'pred': pred, 'target': target}

  def validation_step_end( self, outputs):
    # Logging + Visu
    pred, target = outputs['pred'],outputs['target']
    if ( self._exp['visu'].get('val_images',0) > self.logged_images_val and 
         self.current_epoch % self._exp['visu'].get('every_n_epochs',1)== 0) :
      self.logged_images_val += 1
      self.visualizer.plot_segmentation(tag=f'', seg=pred[0], method='right')
      self.visualizer.plot_segmentation(tag=f'val_gt_left_pred_right_{self.logged_images_val}', seg=target[0], method='left')
      
    self.val_mIoU(pred,target)
    # calculates acc only for valid labels

    m  =  target > -1
    self.val_acc(pred[m], target[m])
    self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
    self.log('val_mIoU', self.val_mIoU, on_step=True, on_epoch=True)
    # self.log('task_name', self._task_name, on_epoch=True)
    
  def validation_epoch_end(self, outputs):
    metrics = self.trainer.logger_connector.callback_metrics
    me =  copy.deepcopy ( metrics ) 
    for k in me.keys():
      try:
        me[k] = "{:10.4f}".format( me[k])
      except:
        pass
    
    if 'train_loss_epoch' in me.keys():
      t_l = me[ 'train_loss_epoch']  
    else:
      t_l = me.get('train_loss', 'Not Defined')
    
    v_acc = me.get('val_acc_epoch', 'Not Defined')
    v_mIoU = me.get('val_mIoU_epoch', 'Not Defined')  
    epoch = self.current_epoch
    rank_zero_info(
      f"Epoch: {epoch} >> Train-Loss: {t_l } <<    >> Val-Acc: {v_acc}, Val-mIoU: {v_mIoU} <<"
    )


  def test_step(self, batch, batch_idx):
    images = batch[0]
    target = batch[1]    #[-1,n] labeled with -1 should not induce a loss
    outputs = self(images) 
    pred = torch.argmax(outputs[0], 1)
   
    return {'pred': pred, 'target': target}

  def test_step_end( self, outputs):
    # Logging + Visu
    pred, target = outputs['pred'],outputs['target']
    self.test_mIoU(pred,target)
    # calculates acc only for valid labels

    m  =  target > -1
    self.test_acc(pred[m], target[m])
    self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)
    self.log('test_mIoU', self.test_mIoU, on_step=True, on_epoch=True)
    
    if ( self._exp['visu'].get('test_images',0) > self.logged_images_test):
      self.logged_images_test += 1
      self.visualizer.plot_segmentation(tag=f'', seg=pred[0], method='right')
      self.visualizer.plot_segmentation(tag=f'test_gt_left_pred_right_{self.logged_images_test}', seg=target[0], method='left') 
      

  def test_epoch_end( self, outputs):
    metrics = self.trainer.logger_connector.callback_metrics
    me =  copy.deepcopy ( metrics ) 
    new_dict = {}
    for k in me.keys():
      if k.find('test') != -1 and k.find('epoch') != -1:  
        try:
          new_dict[k] = "{:10.4f}".format( me[k])
        except:
          pass
    rank_zero_info(
      f"Test Epoch Results: "+ str(new_dict)
    )

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
      lambda_lr= lambda epoch: (((max_epochs-min(max_epochs,epoch) )/max_epochs)**(power) ) + (1-(((max_epochs -min(max_epochs,epoch))/max_epochs)**(power)))*target_lr/init_lr
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1, verbose=True)
      ret = [optimizer], [scheduler]
    else:
      ret = [optimizer]
    return ret
  
  def train_dataloader(self):
    output_transform = transforms.Compose([
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    dataset_train = get_dataset(
      **self._exp['d_train'],
      env = self._env,
      output_trafo = output_transform,
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
    output_transform = transforms.Compose([
      transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    dataset_val = get_dataset(
      **self._exp['d_val'],
      env = self._env,
      output_trafo = output_transform
    )

    dataloader_val = torch.utils.data.DataLoader(dataset_val,
      shuffle = False,
      num_workers = ceil(self._exp['loader']['num_workers']/torch.cuda.device_count()),
      pin_memory = self._exp['loader']['pin_memory'],
      batch_size = self._exp['loader']['batch_size'])
    return dataloader_val

  def test_dataloader(self):
    output_transform = transforms.Compose([
      transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    dataset_test = get_dataset(
      **self._exp['d_test'],
      env = self._env,
      output_trafo = output_transform
    )

    dataloader_test = torch.utils.data.DataLoader(dataset_test,
      shuffle = False,
      num_workers = ceil(self._exp['loader']['num_workers']/torch.cuda.device_count()),
      pin_memory = self._exp['loader']['pin_memory'],
      batch_size =  self._exp['loader']['batch_size_test'])
    return dataloader_test
  
  def set_train_dataset_cfg(self, dataset_train_cfg, dataset_val_cfg, task_name):
    self._exp['d_train'] = dataset_train_cfg
    self._exp['d_val'] = dataset_val_cfg
    self._task_name = task_name
    
    # TODO this creation of the dataset to check its length should be avoided
    output_transform = transforms.Compose([
      transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    d2 = get_dataset(
      **self._exp['d_val'],
      env = self._env,
      output_trafo = output_transform
    )
    d1 = get_dataset(
      **self._exp['d_train'],
      env = self._env,
      output_trafo = output_transform
    )
        
    return bool(len(d1) > self._exp['loader']['batch_size'] and 
      len(d2) > self._exp['loader']['batch_size'])
    
  def set_test_dataset_cfg(self, dataset_test_cfg):
    self._exp['d_test'] = dataset_test_cfg
    
    # TODO this creation of the dataset to check its length should be avoided
    output_transform = transforms.Compose([
      transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    d1 = get_dataset(
      **self._exp['d_test'],
      env = self._env,
      output_trafo = output_transform
    )
    
    return bool( len(d1) > self._exp['loader']['batch_size_test'])
      
    

import pytest
class TestLightning:
  def  test_iout(self):
    output_transform = transforms.Compose([
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
      transform = output_transform,
    )
    BS, H, W = 1,100,100
    NC = 4