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
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning import metrics as pl_metrics
from pytorch_lightning.utilities import rank_zero_info

from torch.nn import functional as F
# MODULES
from models import FastSCNN, Teacher, ReplayStateSyncBack
from datasets import get_dataset
from loss import cross_entropy_soft

#from .metrices import IoU, PixAcc
from visu import Visualizer
from lightning import meanIoUTorchCorrect
from latent_replay import LatentReplayBuffer
import datetime

__all__ = ['Network']
def wrap(s,length, hard=False):
  if len(s) < length:
    return s + ' '*(length - len(s))
  if len(s) > length and hard:
    return s[:length]
  return s

class Network(LightningModule):
  def __init__(self, exp, env):
    super().__init__()
    
    self._epoch_start_time = time.time()
    self._exp = exp
    self._env = env
    self.hparams['lr'] = self._exp['lr']
    
    if self._exp['model']['name'] == 'Fast-SCNN': 
      self.model = FastSCNN(**self._exp['model']['cfg'])
    else:
      raise Exception('Model name not implemented')
    
    p_visu = os.path.join( self._exp['name'], 'visu')
    
    self.visualizer = Visualizer(
      p_visu=p_visu,
      writer=None,
      num_classes=self._exp['model']['cfg']['num_classes']+1)

    max_tests = 4
    self.test_acc = pl_metrics.classification.Accuracy()
    self.train_acc = pl_metrics.classification.Accuracy()
    self.val_acc = torch.nn.ModuleList( [pl_metrics.classification.Accuracy() for i in range(max_tests)] )

    self.test_mIoU = meanIoUTorchCorrect(self._exp['model']['cfg']['num_classes'])
    self.train_mIoU = meanIoUTorchCorrect(self._exp['model']['cfg']['num_classes'])
    
    self.val_mIoU = torch.nn.ModuleList( [meanIoUTorchCorrect(self._exp['model']['cfg']['num_classes']) for i in range(max_tests)] )
    # self.val_mIoU = 

    self.logged_images_train = 0
    self.logged_images_val = 0
    self.logged_images_test = 0
    self._dataloader_index_store = 0
    
    self._task_name = 'NotDefined' # is used for model checkpoint nameing
    self._task_count = 0 # so this here might be a bad idea. Decide if we know the task or not
    self._type = torch.float16 if exp['trainer'].get('precision',32) == 16 else torch.float32
    self._train_start_time = time.time()
    
    if self._exp.get('latent_replay_buffer',{}).get('active',False):
      
      if self._exp['latent_replay_buffer']['extraction_level'] == 'compressed':
        extraction_size = (128,12,12)
      elif self._exp['latent_replay_buffer']['extraction_level'] == 'input':
        s = self._exp['model']['input_size']
        extraction_size = (3,s,s)
      else: 
        raise Exception()
      
      self._lrb = LatentReplayBuffer(
        size = extraction_size,
        size_label = (exp['model']['input_size'],exp['model']['input_size']),
        **self._exp['latent_replay_buffer']['cfg'],
        dtype = self._type,
        device=self.device)

      
    # set correct teaching mode.
    self._teaching = self._exp.get('teaching',{}).get('active',False)
    if self._teaching: 
      if self._exp['task_generator']['total_tasks'] == 1:
        self._teaching = False
      if not (self._exp['task_generator']).get('replay',False):
        self._teaching = False
    
    if self._teaching: 
      self.teacher = Teacher(
        num_classes= self._exp['model']['cfg']['num_classes'],
        n_teacher= self._exp['task_generator']['total_tasks']-1,
        soft_labels= self._exp['teaching']['soft_labels'] )
    
    self._replayed_samples = 0
    self._real_samples = 0
    
    if self._exp['replay_state_sync_back']['active']:
      self._rssb_active = True
      if self._exp['replay_state_sync_back']['get_size_from_task_generator']:
        bins = self._exp['task_generator']['cfg_replay']['bins']
        elements = self._exp['task_generator']['cfg_replay']['elements']
      else:
        raise Exception('Not Implemented something else')
      self._rssb = ReplayStateSyncBack(bins=bins, elements=elements)
    else:
      self._rssb_active = False
  def forward(self, batch, **kwargs):
    if kwargs.get('injection', None) is not None:
      outputs = self.model.injection_forward(
        x = batch, 
        injection = kwargs['injection'], 
        injection_mask = kwargs['injection_mask'])
    else:
      outputs = self.model(batch)
    
    return outputs
      
  def on_train_start(self):
    self.visualizer.writer = self.logger.experiment
    if self._rssb_active: 
      bins, valids = self._rssb.get()
      self.trainer.train_dataloader.dataset.set_full_state(
        bins=bins, 
        valids=valids,
        bin= self._task_count)
  def on_save_checkpoint(self, params):
    if self._rssb_active:
      bins, valids = self.trainer.train_dataloader.dataset.get_full_state()
      self._rssb.absorbe(bins,valids)
    
  def on_train_end(self):
    if self._teaching:
      rank_zero_info( "Store current model as new teacher")
      self.teacher.absorbe_model( self.model, self._task_count)

    if self._rssb_active:
      bins, valids = self.trainer.train_dataloader.dataset.get_full_state()
      self._rssb.absorbe(bins,valids)
      
  def on_epoch_start(self):
    self.visualizer.epoch = self.current_epoch
    
    if self._exp['model'].get('freeze',{}).get('active', False):
      mask = self._exp['model']['freeze']['mask']
      self.model.freeze_module(mask)
      rank_zero_info(f'Model Freeze Following Layers {mask}') 
       

    self.logged_images_train = 0
    self.logged_images_val = 0
    self.logged_images_test = 0


      
  def compute_loss(self, pred, target, images, replayed ):
    nr_replayed = (replayed != -1).sum()
    BS = replayed.shape[0]
    self._replayed_samples += int( nr_replayed )
    self._real_samples += int( BS - nr_replayed )
    
    if self._teaching and (replayed != -1).sum() != 0:
            
      if self._exp['teaching']['soft_labels']:
        # TODO Results during training in NANs ???? 
        
        t_aux = target.clone()
        mask = (target > -1).type(target.dtype)
        t_aux = torch.clamp(t_aux,0) # inplace operator
        target_onehot = torch.nn.functional.one_hot(t_aux, 
          num_classes= self._exp['model']['cfg']['num_classes']).permute(0,3,1,2).type(images.dtype)
      else:
        target_accum = target.clone()
        
      for teacher in range(self._exp['task_generator']['total_tasks']-1):
        if (replayed == teacher).sum() != 0:
          m = (replayed == teacher)[:,0]
          with torch.no_grad():
            tar = self.teacher( images.detach(), teacher) 
          tar.requires_grad = True
          
          if self._exp['teaching']['soft_labels']:
            
            # if we are not doing the softmax this would we prefect and results in loss of 0 for the next task
            tar = torch.nn.functional.softmax(tar, dim=1) # to get a valid prob distribution 
            
            rep_mask = m[:,None,None,None].repeat(1,*tar.shape[1:]).type(images.dtype)
            ori_mask = (rep_mask == 0).type(images.dtype)
            target_onehot = target_onehot * ori_mask + tar * rep_mask
          else:
            tar = torch.argmax(tar, dim=3)
            rep_mask = m[:,None,None].repeat(1,*target_accum.shape[1:]).type(torch.int64)
            ori_mask = (rep_mask == 0).type(torch.int64)
            target_accum = (target_accum * ori_mask + tar * rep_mask).type(torch.int64)
            
          
      if self._exp['teaching']['soft_labels']:
        # MSE for soft labels. CategoricalCrossEntropy for gt-labels
        loss_soft = F.mse_loss(pred,target_onehot,reduction='none').mean(dim=[1,2,3])
        loss = F.cross_entropy(pred, target, ignore_index=-1,reduction='none').mean(dim=[1,2])
        loss = loss * (replayed[:,0]== -1) + loss_soft * (replayed[:,0]!= -1) * self._exp['teaching']['soft_labels_weight']
        loss = loss.mean()
      else:
        loss = F.cross_entropy(pred, target_accum, ignore_index=-1)
      
    else:
      
      loss = F.cross_entropy(pred, target, ignore_index=-1)
    return loss
  
  def training_step(self, batch, batch_idx):
    images = batch[0]
    target = batch[1]
    BS = images.shape[0]
    
    ori_img = batch[2]
    if self._exp.get('latent_replay_buffer',{}).get('active',False):
      injection, injection_labels, mask_injection = self._lrb(BS, device=images.device)
      outputs = self(batch = images, 
        injection = injection, 
        injection_mask = mask_injection)
      
      # self._replayed_samples += int( mask_injection.sum() )
      # self._real_samples += int( BS - mask_injection.sum() )
      if mask_injection.sum() != 0:
        # set the labels to the stored labels
        target[mask_injection] = injection_labels[mask_injection]
        
        # set the masked image to green
        img = torch.zeros(ori_img.shape, device= self.device)
        img[:,1,:,:] = 1
        ori_img[mask_injection] = img[mask_injection]
    else:
      outputs = self(batch = images)

    loss = self.compute_loss(  
              pred = outputs[0], 
              target = target, 
              images= images, 
              replayed = batch[3] )
    # write to latent replay buffer
    if self._exp.get('latent_replay_buffer',{}).get('active',False) and not self.trainer.running_sanity_check:
      valid_elements = torch.nonzero(mask_injection==0, as_tuple=False)
      if valid_elements.shape[0] != 0:
        # check for the case only latent replay performed
        if random.random() < self._exp['latent_replay_buffer']['fill_sample_from_batch_ratio']: 
          ele = torch.randint(0, valid_elements.shape[0], (1,))
          
          if self._exp['latent_replay_buffer']['extraction_level'] == 'compressed':
            self._lrb.add(x=outputs[1][ele].detach(),y=target[ele].detach())
          elif self._exp['latent_replay_buffer']['extraction_level'] == 'input':
            self._lrb.add(x=images[ele].detach(),y=target[ele].detach())
        
    self.log('train_loss', loss, on_step=False, on_epoch=True)
    return {'loss': loss, 'pred': outputs[0], 'target': target, 'ori_img': ori_img }
  
  def training_step_end(self, outputs):
    # Log replay buffer stats
    if self._exp.get('latent_replay_buffer',{}).get('active',False):
      if self.global_step % (self._exp['visu']).get('log_every_y_global_steps',10) == 0:
        dic = {}
        for i in range( len( self._lrb.bins) ):
          filled =  self._lrb._bin_counts[i]
          dic[f'lrb_bin_{i}'] = filled.clone().detach()    
        self.logger.experiment.add_scalars('lr_bins', 
          dic, 
          global_step=self.global_step)
        
    self.logger.experiment.add_scalars('samples', 
      {'real': torch.tensor(self._real_samples),
      'replayed': torch.tensor(self._replayed_samples)}, 
      global_step=self.global_step)
  
    # Logging + Visu
    if self.current_epoch % self._exp['visu'].get('log_training_metric_every_n_epoch',9999) == 0 : 
      pred = torch.argmax(outputs['pred'], 1)
      target = outputs['target']
      train_mIoU = self.train_mIoU(pred,target)
      # calculates acc only for valid labels
      m  =  target > -1
      train_acc = self.train_acc(pred[m], target[m])
      self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True, prog_bar = True)
      self.log('train_mIoU_epoch', self.train_mIoU, on_step=False, on_epoch=True, prog_bar = True)
    
    if ( self._exp['visu'].get('train_images',0) > self.logged_images_train and 
         self.current_epoch % self._exp['visu'].get('every_n_epochs',1) == 0):
      pred = torch.argmax(outputs['pred'], 1).clone().detach()
      target = outputs['target'].clone().detach()
      
      pred[0][ target[0] == -1 ] = -1
      pred[0] = pred[0]+1
      target[0] = target[0] +1
      self.logged_images_train += 1
      self.visualizer.plot_segmentation(tag=f'', seg=pred[0], method='right')
      self.visualizer.plot_segmentation(tag=f'train_gt_left_pred_right_{self._task_name}_{self.logged_images_train}', seg=target[0], method='left')  
      self.visualizer.plot_segmentation(tag=f'', seg=pred[0], method='right')
      self.visualizer.plot_image(tag=f'train_img_ori_left_pred_right_{self._task_name}_{self.logged_images_train}', img=outputs['ori_img'][0], method='left')
    
    return {'loss': outputs['loss']}
  
  def on_train_epoch_end(self, outputs):
    if self.current_epoch % self.trainer.check_val_every_n_epoch != 0:
      val_acc_epoch = torch.tensor( 999, device=self.device)
      val_mIoU_epoch = torch.tensor( 999, device=self.device)
      val_loss = torch.tensor( 999, device=self.device)
      self.log('val_acc_epoch',val_acc_epoch)
      self.log('val_mIoU_epoch',val_mIoU_epoch,)
      self.log('val_loss',val_loss)
    
    
    
      
  def validation_step(self, batch, batch_idx, dataloader_idx):
    if self._dataloader_index_store != dataloader_idx:
      self._dataloader_index_store = dataloader_idx
      self.logged_images_val = 0
      
    images = batch[0]
    target = batch[1]    #[-1,n] labeled with -1 should not induce a loss
    outputs = self(images)
    
    loss = F.cross_entropy(outputs[0], target, ignore_index=-1 ) 

    pred = torch.argmax(outputs[0], 1)

    self.log('val_loss', loss, on_epoch=True)

    return {'pred': pred, 'target': target, 'ori_img': batch[2], 'dataloader_idx': dataloader_idx}

  def validation_step_end( self, outputs ):
    # Logging + Visu
    dataloader_idx = outputs['dataloader_idx']
    pred, target = outputs['pred'],outputs['target']
    if ( self._exp['visu'].get('val_images',0) > self.logged_images_val and 
         self.current_epoch % self._exp['visu'].get('every_n_epochs',1)== 0) :
      self.logged_images_val += 1
      pred_c = pred.clone().detach()
      target_c = target.clone().detach()
      pred_c[0][ target_c[0] == -1 ] = -1
      pred_c[0] = pred_c[0]+1
      target_c[0] = target_c[0] +1
      self.visualizer.plot_segmentation(tag=f'', seg=pred_c[0], method='right')
      self.visualizer.plot_segmentation(tag=f'val_gt_left_pred_right__Name_{self._task_name}__Val_Task_{dataloader_idx}__Sample_{self.logged_images_val}', seg=target_c[0], method='left')
      self.visualizer.plot_segmentation(tag=f'', seg=pred_c[0], method='right')
      self.visualizer.plot_image(tag=f'val_img_ori_left_pred_right__Name_{self._task_name}_Val_Task_{dataloader_idx}__Sample_{self.logged_images_train}', img=outputs['ori_img'][0], method='left')
      
      
    self.val_mIoU[dataloader_idx] (pred,target)
    # calculates acc only for valid labels

    m  =  target > -1
    self.val_acc[dataloader_idx] (pred[m], target[m])
    self.log(f'val_acc', self.val_acc[dataloader_idx] , on_epoch=True, prog_bar=False)
    self.log(f'val_mIoU', self.val_mIoU[dataloader_idx] , on_epoch=True, prog_bar=False)
    self.log(f'task_count', self._task_count, on_epoch=True, prog_bar=False)
    # self.log('task_name', self._task_name, on_epoch=True)
    
  def validation_epoch_end(self, outputs):

    
    metrics = self.trainer.logger_connector.callback_metrics
    me =  copy.deepcopy ( metrics ) 
    for k in me.keys():
      try:
        me[k] = "{:10.4f}".format( me[k])
      except:
        pass
    
    # if 'train_loss_epoch' in me.keys():
    #   t_l = me[ 'train_loss_epoch']  
    # else:
    
    t_l = me.get('train_loss', 'NotDef')
    v_acc = me.get('val_acc', 'NotDef')
    v_mIoU = me.get('val_mIoU', 'NotDef')  
    epoch = str(self.current_epoch)
    
    t = time.time()- self._epoch_start_time
    t = str(datetime.timedelta(seconds=round(t)))
    t2 = time.time()- self._train_start_time
    t2 = str(datetime.timedelta(seconds=round(t2))) 
    if not self.trainer.running_sanity_check:
      rank_zero_info('Time for a complete epoch: '+ t)
      n = self._task_name
      n = wrap(n,20)
      t = wrap(t,10,True)
      epoch =  wrap(epoch,3)
      t_l = wrap(t_l,6)
      v_acc = wrap(v_acc,6)
      v_mIoU = wrap(v_mIoU,6)
      
      rank_zero_info(
        f"Exp: {n} | Epoch: {epoch} | TimeEpoch: {t} | TimeStart: {t2} |  >>> Train-Loss: {t_l } <<<   >>> Val-Acc: {v_acc}, Val-mIoU: {v_mIoU} <<<"
      )
    self._epoch_start_time = time.time()
    

  
  def test_step(self, batch, batch_idx):
    images = batch[0]
    target = batch[1]    #[-1,n] labeled with -1 should not induce a loss
    outputs = self(images) 
    pred = torch.argmax(outputs[0], 1)
    return {'pred': pred, 'target': target, 'ori_img': batch[2]}

  def test_step_end( self, outputs):
    # Logging + Visu
    pred, target = outputs['pred'],outputs['target']
    self.test_mIoU(pred,target)
    # calculates acc only for valid labels

    m  =  target > -1
    self.test_acc(pred[m], target[m])
    self.log('test_acc', self.test_acc, on_epoch=True, prog_bar=True)
    self.log('test_mIoU', self.test_mIoU, on_epoch=True, prog_bar=True)
    
    if ( self._exp['visu'].get('test_images',0) > self.logged_images_test):
      self.logged_images_test += 1
      pred_c = pred.clone().detach()
      target_c = target.clone().detach()
      pred_c[0][ target_c[0] == -1 ] = -1
      pred_c[0] = pred_c[0]+1
      target_c[0] = target_c[0] +1
      self.visualizer.plot_segmentation(tag=f'', seg=pred_c[0], method='right')
      self.visualizer.plot_segmentation(tag=f'test_gt_left_pred_right_{self._task_name}_{self.logged_images_test}', seg=target_c[0], method='left')

  def test_epoch_end( self, outputs):
    metrics = self.trainer.logger_connector.callback_metrics
    me = copy.deepcopy( metrics ) 
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
    self.logged_images_test = 0

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
  
  # def train_dataloader(self):
  #   output_transform = transforms.Compose([
  #           transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
  #   ])
  #   # dataset and dataloader
  #   dataset_train = get_dataset(
  #     **self._exp['d_train'],
  #     env = self._env,
  #     output_trafo = output_transform,
  #   )
  #   print(dataset_train)
  #   # self.trainer.train_dataloader                              
  #   # initalize train and validation indices
  #   if self.trainer.train_dataloader is not None:
  #     bins, valids = self.trainer.train_dataloader.dataset.get_full_state()
  #     dataset_train.set_full_state(bins,valids, self._task_count_task_name)
  #   dataloader_train = torch.utils.data.DataLoader(dataset_train,
  #     shuffle = self._exp['loader']['shuffle'],
  #     num_workers = ceil(self._exp['loader']['num_workers']/torch.cuda.device_count()),
  #     pin_memory = self._exp['loader']['pin_memory'],
  #     batch_size = self._exp['loader']['batch_size'], 
  #     drop_last = True)
    
  #   return dataloader_train
    
  # def val_dataloader(self):
  #   output_transform = transforms.Compose([
  #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
  #   ])
  #   dataset_val = get_dataset(
  #     **self._exp['d_val'],
  #     env = self._env,
  #     output_trafo = output_transform
  #   )

  #   dataloader_val = torch.utils.data.DataLoader(dataset_val,
  #     shuffle = False,
  #     num_workers = ceil(self._exp['loader']['num_workers']/torch.cuda.device_count()),
  #     pin_memory = self._exp['loader']['pin_memory'],
  #     batch_size = self._exp['loader']['batch_size'])
  #   return dataloader_val

  # def test_dataloader(self):
  #   output_transform = transforms.Compose([
  #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
  #   ])
  #   dataset_test = get_dataset(
  #     **self._exp['d_test'],
  #     env = self._env,
  #     output_trafo = output_transform
  #   )

  #   dataloader_test = torch.utils.data.DataLoader(dataset_test,
  #     shuffle = False,
  #     num_workers = ceil(self._exp['loader']['num_workers']/torch.cuda.device_count()),
  #     pin_memory = self._exp['loader']['pin_memory'],
  #     batch_size =  self._exp['loader']['batch_size_test'])
  #   return dataloader_test
  
  # def set_train_dataset_cfg(self, dataset_train_cfg, dataset_val_cfg, task_name):
  #   self._exp['d_train'] = dataset_train_cfg
  #   self._exp['d_val'] = dataset_val_cfg
  #   self._task_name = task_name
  #   self._task_count += 1
    
  #   if self._exp.get('latent_replay_buffer',{}).get('active',False):
  #     self._lrb.set_bin( self._task_count)
      
    
    
  #   # TODO this creation of the dataset to check its length should be avoided
  #   output_transform = transforms.Compose([
  #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
  #   ])
  #   d2 = get_dataset(
  #     **self._exp['d_val'],
  #     env = self._env,
  #     output_trafo = output_transform
  #   )
  #   d1 = get_dataset(
  #     **self._exp['d_train'],
  #     env = self._env,
  #     output_trafo = output_transform
  #   )
        
  #   return bool(len(d1) > (self._exp['loader']['batch_size']*self._exp['trainer']['gpus']) and 
  #     len(d2) > (self._exp['loader']['batch_size']*self._exp['trainer']['gpus']))
    
  # def set_test_dataset_cfg(self, dataset_test_cfg, task_name):
  #   self._exp['d_test'] = dataset_test_cfg
  #   self._task_name = task_name
  #   # TODO this creation of the dataset to check its length should be avoided
  #   output_transform = transforms.Compose([
  #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
  #   ])
  #   d1 = get_dataset(
  #     **self._exp['d_test'],
  #     env = self._env,
  #     output_trafo = output_transform
  #   )
    
  #   return bool( len(d1) > self._exp['loader']['batch_size_test']*self._exp['trainer']['gpus'])
      
    

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
      'mode': 'val'}
    root = '/media/scratch1/jonfrey/datasets/Cityscapes'
    self.dataset_train = get_dataset(
      **di,
      root = root,
      transform = output_transform,
    )
    BS, H, W = 1,100,100
    NC = 4