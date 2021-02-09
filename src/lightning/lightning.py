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
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from torchvision.utils import make_grid
from torch.nn import functional as F
# MODULES
from models import FastSCNN, Teacher, ReplayStateSyncBack
from datasets import get_dataset
from loss import cross_entropy_soft

#from .metrices import IoU, PixAcc
from visu import Visualizer
from lightning import meanIoUTorchCorrect
import datetime
from math import ceil

# Uncertainty
from uncertainty import get_softmax_uncertainty_max, get_softmax_uncertainty_distance


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
      logger=None,
      num_classes=self._exp['model']['cfg']['num_classes']+1)
    self._mode = 'train'
    max_tests = 4
    self.test_acc = pl_metrics.classification.Accuracy()
    self.train_acc = pl_metrics.classification.Accuracy()
    self.val_acc = torch.nn.ModuleList( 
      [pl_metrics.classification.Accuracy() for i in range(max_tests)] )

    self.test_mIoU = meanIoUTorchCorrect(self._exp['model']['cfg']['num_classes'])
    self.train_mIoU = meanIoUTorchCorrect(self._exp['model']['cfg']['num_classes'])
    
    self.val_mIoU = torch.nn.ModuleList( 
      [meanIoUTorchCorrect(self._exp['model']['cfg']['num_classes']) for i in range(max_tests)] )

    self.logged_images_train = 0
    self.logged_images_val = 0
    self.logged_images_test = 0
    self._dataloader_index_store = 0
    
    self._task_name = 'NotDefined' # is used for model checkpoint nameing
    self._task_count = 0 # so this here might be a bad idea. Decide if we know the task or not
    self._type = torch.float16 if exp['trainer'].get('precision',32) == 16 else torch.float32
    self._train_start_time = time.time()
    
      
    # set correct teaching mode.
    self._teaching = (self._exp.get('teaching',{}).get('active',False) or 
                      self._exp.get('latent_replay',{}).get('active',False))
    if self._teaching: 
      if self._exp['task_generator']['total_tasks'] == 1:
        self._teaching = False
      if not (self._exp['task_generator']).get('replay',False):
        self._teaching = False
    
    if self._teaching: 
      self.teacher = Teacher(
        num_classes= self._exp['model']['cfg']['num_classes'],
        n_teacher= self._exp['task_generator']['total_tasks']-1,
        soft_labels= self._exp['teaching']['soft_labels'],
        fast_params = self._exp['model']['cfg'])
    
    self._replayed_samples = 0
    self._real_samples = 0
    self._val_results = {} 
    # resetted on_train_end. filled in validation_epoch_end
    
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
    if kwargs.get('replayed', None) is not None:
      injection_mask = kwargs['replayed'] != -1
      outputs = self.model.injection_forward(
        x = batch, 
        injection_features = kwargs['injection_features'], 
        injection_mask = injection_mask)
    else:
      outputs = self.model(batch)
    return outputs
  
  def on_train_epoch_start(self):
    self._mode = 'train'
     
  def on_train_start(self):
    self.visualizer.logger= self.logger
      
    if (self._task_count != 0 and
        self.model.extract):
      self.model.freeze_module(layer=self.model.extract_layer)
      rank_zero_warn( 'Freezed the model given that we start to extract data from teacher')
      rank_zero_warn( 'Therefore not training upper layers')
    
    if self._rssb_active: 
      bins, valids = self._rssb.get()
      rank_zero_info( 'Reload saved buffer state')
      for i in range(self._task_count):
        s = 'Restored for task {i}: \n' + str(bins[i])
        rank_zero_info( s )
      self.trainer.train_dataloader.dataset.set_full_state(
        bins=bins, 
        valids=valids,
        bin= self._task_count)
      
  def on_save_checkpoint(self, params):
    if self._rssb_active:
      rank_zero_info( 'When saving checkpoint also save buffer state')
      bins, valids = self.trainer.train_dataloader.dataset.get_full_state()
      self._rssb.absorbe(bins,valids)
    
  def on_train_end(self):
    if self._rssb_active:
      bins, valids = self.trainer.train_dataloader.dataset.get_full_state()
      self._rssb.absorbe(bins,valids)
    
    if self._exp.get('buffer',{}).get('fill_after_fit', False):
      self.fill_buffer()
      
    self._val_results = {} # reset the buffer for the next task
    
  def on_epoch_start(self):
    self.visualizer.epoch = self.current_epoch
    
    if self._exp['model'].get('freeze',{}).get('active', False):
      mask = self._exp['model']['freeze']['mask']
      self.model.freeze_module(mask)
      rank_zero_info(f'Model Freeze Following Layers {mask}') 
       
    self.logged_images_train = 0
    self.logged_images_val = 0
    self.logged_images_test = 0

    if self._teaching:
      self.teacher.print_weight_summary()
    
    string = ''
    sum = 0
    for i in self.model.parameters():
      sum += i[0].sum()
    string += f'   CurrentModel: WeightSum == {sum}\n'
    rank_zero_info(string)
      
  def compute_loss(self, pred, target, teacher_targets, images, replayed ):
    nr_replayed = (replayed != -1).sum()
    BS = replayed.shape[0]
    self._replayed_samples += int( nr_replayed )
    self._real_samples += int( BS - nr_replayed )
    
    if self._teaching and (replayed != -1).sum() != 0:
      
      if self._exp['teaching']['soft_labels']:
        # MSE for soft labels. CategoricalCrossEntropy for gt-labels
        
        if self._exp['teaching'].get('loss_function', 'MSE') == 'MSE':
          loss_soft = F.mse_loss( torch.nn.functional.softmax(pred, dim=1),teacher_targets,reduction='none').mean(dim=[1,2,3])
        else: 
          raise Exception('Invalid Input for loss_function')
        loss = F.cross_entropy(pred, target, ignore_index=-1,reduction='none').mean(dim=[1,2])
        loss = loss * (replayed[:,0]== -1) + loss_soft * (replayed[:,0]!= -1) * self._exp['teaching']['soft_labels_weight']
        loss = loss.mean()
        
      else: 
        target = target * (replayed[:,0]== -1)[:,None,None].repeat(1,target.shape[1],target.shape[2]) + teacher_targets * (replayed[:,0] != -1)[:,None,None].repeat(1,target.shape[1],target.shape[2])
        loss = F.cross_entropy(pred, target, ignore_index=-1)
    else:
      # use gt labels
      loss = F.cross_entropy(pred, target, ignore_index=-1)
    return loss
  
  def training_step(self, batch, batch_idx):
    images = batch[0]
    target = batch[1]
    ori_img = batch[2]
    replayed = batch[3]
    BS = images.shape[0]
    
    teacher_targets = None
    if ( (replayed != -1).sum() != 0 and
      ( self._exp.get('latent_replay',{}).get('active',False) or self._teaching)):
        teacher_targets, injection_features = self.teacher.get_latent_replay(images, replayed)
        
        if self._exp.get('latent_replay',{}).get('active',False):
          outputs = self(batch = images, 
                        injection_features = injection_features, 
                        replayed = replayed)
        else:
          outputs = self(batch = images) 
    else:
      outputs = self(batch = images)

    loss = self.compute_loss(  
              pred = outputs[0], 
              target = target,
              teacher_targets = teacher_targets,
              images= images, 
              replayed = replayed)
        
    self.log('train_loss', loss, on_step=False, on_epoch=True)
    return {'loss': loss, 'pred': outputs[0], 'target': target, 'ori_img': ori_img }
  
  def training_step_end(self, outputs):
    # Log replay buffer stats
    self.logger.log_metrics( 
      metrics = { 'real': torch.tensor(self._real_samples),
                  'replayed': torch.tensor(self._replayed_samples)},
      step = self.global_step)
    
  
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
  
  def on_validation_epoch_start(self):
    self._mode = 'val'
  
  def validation_epoch_end(self, outputs):
    metrics = self.trainer.logger_connector.callback_metrics
    me =  copy.deepcopy ( metrics ) 
    for k in me.keys():
      try:
        me[k] = "{:10.4f}".format( me[k])
      except:
        pass
    
    t_l = me.get('train_loss', 'NotDef')
    v_acc = me.get('val_acc', 'NotDef')
    v_mIoU = me.get('val_mIoU', 'NotDef')  
    
    if len( self._val_results ) == 0:
      for i in range(self._exp['max_tasks']):
        self._val_results[f'val_acc/dataloader_idx_{i}'] = float(metrics[f'val_acc/dataloader_idx_{i}'])
    else:
      val_results = {}
      for i in range(self._exp['max_tasks']):
        val_results[f'val_acc/dataloader_idx_{i}'] = float(metrics[f'val_acc/dataloader_idx_{i}'])
        res = (self._val_results[f'val_acc/dataloader_idx_{i}'] -
          val_results[f'val_acc/dataloader_idx_{i}'])
        self.log(f'forgetting/acc_idx_{i}', res, on_epoch=True, prog_bar=False)
      
      if self._task_count > 0:
        res = 0
        for i in range(self._task_count):
          res += (self._val_results[f'val_acc/dataloader_idx_{i}'] -
            val_results[f'val_acc/dataloader_idx_{i}'])
        
        res /= self._task_count
        self.log(f'forgetting/acc_avg_pervious', res, on_epoch=True, prog_bar=False)
      
      res = ( val_results[f'val_acc/dataloader_idx_{self._task_count}']-
        self._val_results[f'val_acc/dataloader_idx_{self._task_count}'] )
      self.log(f'learning/acc_current', res, on_epoch=True, prog_bar=False)
      
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
  
  def fill_buffer(self):
    # how can we desig this
    # this function needs to be called the number of tasks -> its not so important that it is effiecent
    # load the sample with the datloader in a blocking way -> This might take 5 Minutes per task
    rank_zero_info('Fill buffer')
    
    BS = self._exp['loader']['batch_size']
    auxillary_dataloader = torch.utils.data.DataLoader(
      self.trainer.train_dataloader.dataset,
      shuffle = False,
      num_workers = ceil(self._exp['loader']['num_workers']/torch.cuda.device_count()),
      pin_memory = self._exp['loader']['pin_memory'],
      batch_size = BS, 
      drop_last = True)
    
    ret = torch.zeros( (int(BS*len(auxillary_dataloader)),3), device=self.device )
    # ret should be filles with globale index and measurment
    
    s = 0
    for batch_idx, batch in enumerate(auxillary_dataloader):
      for b in range(len(batch)):
        batch[b] = batch[b].to(self.device)
      
      res, global_index = self.fill_step(batch, batch_idx)
      
      for ba in range( res.shape[0] ):  
        ret[s,0] = res[ba]
        ret[s,1] = global_index[ba]
        s += 1
    
    _, indi = torch.topk( ret[:,0] , self._rssb.bins.shape[1] )
    self._rssb.bins[self._task_count,:] = ret[:,1][indi]
    
    # use the top 16 images to create an buffer overview
    _, indi = torch.topk( ret[:,0] , 16 )
    images = []
    labels = []
    for ind in list( indi ):
      batch = self.trainer.train_dataloader.dataset[ind]
      images.append( batch[2]) # 3,H,W
      labels.append( batch[1][None].repeat(3,1,1)) # 3,H,W    

    grid_images = make_grid(images,nrow = 4,padding = 2,
            scale_each = False, pad_value = 0)
    grid_labels = make_grid(labels,nrow = 4,padding = 2,
            scale_each = False, pad_value = -1)

    self.visualizer.plot_image(grid_images, tag=f'{self._task_count}_Buffer_Sample_Images')
    self.visualizer.plot_segmentation( seg = grid_labels[0], tag=f'{self._task_count}_Buffer_Sample_Labels')
    m = self._exp.get('buffer',{}).get('mode', 'softmax_max')
    
    self.visualizer.plot_bar(ret[:,0], x_label='Sample', y_label='Value '+m ,
                             title='Bar Plot', sort=True, reverse=True, tag=f'{self._task_count}_Buffer_Eval_Metric')
    rank_zero_info('Set bin selected the following values: \n'+ str( self._rssb.bins[self._task_count,:]) )
      
  def fill_step(self, batch, batch_idx):
    BS = batch[0].shape[0]
    global_index =  batch[4]    
    outputs = self(batch = batch[0]) 
    pred = outputs[0]
    
    m = self._exp.get('buffer',{}).get('mode', 'softmax_max')
    if m == 'softmax_max':
      res = get_softmax_uncertainty_max(pred) # confident 0 , uncertain 1
    elif m == 'softmax_distance':
      res = get_softmax_uncertainty_distance(pred) # confident 0 , uncertain 1
    elif m == 'loss':
	    res = F.cross_entropy(pred, batch[1], ignore_index=-1,reduction='none').mean(dim=[1,2]) # correct 0 , incorrect high
    else:
      raise Exception('Mode to fill buffer is not defined!')
    return res.detach(), global_index.detach()
  
   
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