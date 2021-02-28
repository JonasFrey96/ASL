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

from uncertainty import get_softmax_uncertainty_max, get_softmax_uncertainty_distance
from uncertainty import get_image_indices
from uncertainty import distribution_matching
from uncertainty import get_kMeans_indices

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
    
    self._buffer_elements = 0 
    if self._exp['replay_state_sync_back']['active']:
      self._rssb_active = True
      self._rssb_last_epoch = -1
      if self._exp['replay_state_sync_back']['get_size_from_task_generator']:
        bins = self._exp['task_generator']['cfg_replay']['bins']
        self._buffer_elements = self._exp['task_generator']['cfg_replay']['elements']
      else:
        raise Exception('Not Implemented something else')
      self._rssb = ReplayStateSyncBack(bins=bins, elements=self._buffer_elements)
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
    print('Start')
    self.visualizer.logger= self.logger
      
    if (self._task_count != 0 and
        self.model.extract):
      self.model.freeze_module(layer=self.model.extract_layer)
      rank_zero_warn( 'ON_TRAIN_START: Freezed the model given that we start to extract data from teacher')
      rank_zero_warn( 'ON_TRAIN_START: Therefore not training upper layers')
    
    if self._rssb_active:
      self.rssb_to_dataset()
      bins, valids = self.trainer.train_dataloader.dataset.datasets.get_full_state()
  
  def rssb_to_dataset(self):
    if self._rssb_active:
      bins, valids = self._rssb.get()
      print( 'RSSB_TO_DATASET: Reload saved buffer state to dataset')
      for i in range(self._task_count):
        s = f'Restored for task {i}: \n' + str(bins[i])
        print( 'RSSB_TO_DATASET: ' + s )
        
      self.trainer.train_dataloader.dataset.datasets.set_full_state( 
        bins=bins, 
        valids=valids,
        bin= self._task_count)
      
  def on_save_checkpoint(self, params):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0: 
      if self._rssb_active:
        if self.current_epoch != self._rssb_last_epoch and self.trainer.train_dataloader.dataset.datasets.replay:
          self._rssb_last_epoch = self.current_epoch
          print( 'ON_SAVE_CHECKPOINT: Before saving checkpoint sync-back buffer state')
          bins, valids = self.trainer.train_dataloader.dataset.datasets.get_full_state()
          if self._mode != 'test':
            pass
            # self._rssb.absorbe(bins,valids)
      
  def teardown(self, stage):
    print('TEARDOWN: Called')
  
  def on_fit_end(self):
    if self._mode != 'test':
      print('ON_FIT_END: Called')
      if self._rssb_active and self.trainer.train_dataloader.dataset.datasets.replay:
        self._rssb_last_epoch = self.current_epoch
        print( 'ON_FIT_END: Before finishing training sync-back buffer state')
        bins, valids = self.trainer.train_dataloader.dataset.datasets.get_full_state()
        # self._rssb.absorbe(bins,valids)
        
      print('ON_FIT_END: Training end reset val_results.')
      self._val_results = {} # reset the buffer for the next task
    
  def on_train_end(self):
    bins, valids = self.trainer.train_dataloader.dataset.datasets.get_full_state()
    
  def on_epoch_start(self):
    self.visualizer.epoch = self.current_epoch
    
    if self._exp['model'].get('freeze',{}).get('active', False):
      mask = self._exp['model']['freeze']['mask']
      self.model.freeze_module(mask)
      print(f'ON_EPOCH_START: Model Freeze Following Layers {mask}') 
       
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
    print('ON_EPOCH_START:\n' + string)
      
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
  
    
  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    if self._dataloader_index_store != dataloader_idx:
      self._dataloader_index_store = dataloader_idx
      self.logged_images_val = 0
      
    images = batch[0]
    target = batch[1]    #[-1,n] labeled with -1 should not induce a loss
    outputs = self(images)
    
    loss = F.cross_entropy(outputs[0], target, ignore_index=-1 ) 
    pred = torch.argmax(outputs[0], 1)

    

    return {'pred': pred, 'target': target, 'ori_img': batch[2], 'dataloader_idx': dataloader_idx, 'loss_ret': loss }

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
    self.log('val_loss', outputs['loss_ret'], on_epoch=True)
    # self.log(f'val_acc/dataloader_idx_{dataloader_idx}', self.val_acc[dataloader_idx] , on_epoch=True, prog_bar=False)
    # self.log(f'val_mIoU/dataloader_idx_{dataloader_idx}', self.val_mIoU[dataloader_idx] , on_epoch=True, prog_bar=False)
    # self.log(f'task_count/dataloader_idx_{dataloader_idx}', self._task_count, on_epoch=True, prog_bar=False)
    # self.log(f'val_loss/dataloader_idx_{dataloader_idx}', outputs['loss_ret'], on_epoch=True)
    
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
    
    try:
      # only works when multiple val-dataloader are set!
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
    except:
      pass
      
    epoch = str(self.current_epoch)
    
    t = time.time()- self._epoch_start_time
    t = str(datetime.timedelta(seconds=round(t)))
    t2 = time.time()- self._train_start_time
    t2 = str(datetime.timedelta(seconds=round(t2))) 
    if not self.trainer.running_sanity_check:
      print('VALIDATION_EPOCH_END: Time for a complete epoch: '+ t)
      n = self._task_name
      n = wrap(n,20)
      t = wrap(t,10,True)
      epoch =  wrap(epoch,3)
      t_l = wrap(t_l,6)
      v_acc = wrap(v_acc,6)
      v_mIoU = wrap(v_mIoU,6)
      
      print('VALIDATION_EPOCH_END: '+ 
        f"Exp: {n} | Epoch: {epoch} | TimeEpoch: {t} | TimeStart: {t2} |  >>> Train-Loss: {t_l } <<<   >>> Val-Acc: {v_acc}, Val-mIoU: {v_mIoU} <<<"
      )
    self._epoch_start_time = time.time()
    
    # if self.current_epoch == self.trainer.max_epochs:
      
    # if ( self.trainer.should_stop and
    #  
    print( "SELF TRAINER SHOULD STOP", self.trainer.should_stop, self.device )

  def on_test_epoch_start(self):
    self._mode = 'test'
    # modify extraction settings to get uncertainty
    self._extract_store = self.model.extract
    self._extract_layer_store = self.model.extract_layer
    self.model.extract = True
    self.model.extract_layer = 'fusion' # if we use fusion we get a 128 feature
    
    # how can we desig this
    # this function needs to be called the number of tasks -> its not so important that it is effiecent
    # load the sample with the datloader in a blocking way -> This might take 5 Minutes per task
    print('ON_TEST_EPOCH_START: Called')
    BS = self.trainer.test_dataloaders[0].batch_sampler.batch_size
    
    # compute all metrics
    self._t_ret_metrices = 3
    self._t_ret_name = ['softmax_max', 'softmax_distance', 'loss']

    self._t_l = len(self.trainer.test_dataloaders[0])
    
    self._t_ret = torch.zeros( (int(BS*self._t_l),1+self._t_ret_metrices), 
                              device=self.device )
    self._t_latent_feature_all = torch.zeros( (int(BS*self._t_l),40,128),
                                             device=self.device, dtype= torch.float16 )
    self._t_feature_labels = torch.zeros( (int(BS*self._t_l),40),
                                             device=self.device, dtype= torch.int64)
    
    
    self._t_label_sum = torch.zeros( (self._exp['model']['cfg']['num_classes']),
                                    device=self.device, dtype = torch.int64)
    self._t_s = 0
    self._t_st = time.time()
    
    
  def test_step(self, batch, batch_idx):
    self._t_s
    
    if batch_idx % int(self.trainer.log_every_n_steps/5) == 0:
      info = f'TEST_STEP: Analyzed Dataset {batch_idx}/{self._t_l} Time: {time.time()-self._t_st }'
      print(info)
    
    res, global_index, latent_feature = self.fill_step( batch, batch_idx)
    
    for ba in range( res.shape[0] ):  
      if batch[4][ba] != -999:
        
        label_indi, label_counts = torch.unique( batch[1][ba] , return_counts = True) 
        for i,cou in zip( list(label_indi), list(label_counts) ):
          if i != -1:
            self._t_label_sum[int( i )] += int( cou )
            self._t_feature_labels[self._t_s,int(i)] += int( cou )
            
        self._t_ret[self._t_s,:self._t_ret_metrices] = res[ba]
        self._t_ret[self._t_s,self._t_ret_metrices] = global_index[ba]
        self._t_latent_feature_all[self._t_s] = latent_feature[ba]
        self._t_s += 1
    
  def on_test_epoch_end(self):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
      s = self._t_s
      self._t_ret = self._t_ret[:s]
      self._t_latent_feature_all = self._t_latent_feature_all[:s]
      self._t_feature_labels = self._t_feature_labels[:s]
      
      
      torch.save(self._t_feature_labels.cpu(),self._exp['name'] + 
                f'/labels_tensor_{self._task_count}.pt')
      torch.save(self._t_ret[:,self._t_ret_metrices].cpu(),self._exp['name'] + 
                f'/indices_tensor_{self._task_count}.pt')
      torch.save(self._t_latent_feature_all.cpu(),self._exp['name'] + 
                f'/latent_feature_tensor_{self._task_count}.pt')
      self._t_label_sum = self._t_label_sum/self._t_label_sum.sum()
      
      # Method according to which the buffer will be filled
      m = self._exp['buffer']['mode']
      # calculated ret_globale_indices
      if m == 'latent_feat':
        # more complex evaluate according to latent feature space
        ret_globale_indices = self.use_latent_features( self._t_latent_feature_all, 
                                                      self._t_ret[:,self._t_ret_metrices] )
        
      elif m == 'softmax_max' or m == 'softmax_distance' or m == 'loss':
        if m == 'softmax_max':
          metric_idx = 0
        elif m == 'softmax_distance': 
          metric_idx = 1
        elif m == 'loss':
          metric_idx = 2
        # simple method use top-K  of the computed metric in fill_step
        _, indi = torch.topk( input = self._t_ret[:,metric_idx],
                              k =self._rssb.bins.shape[1],
                              largest =self._exp['buffer'].get('metric_cfg',{}).get('use_highest_uncertainty',True))
        ret_globale_indices = self._t_ret[:,self._t_ret_metrices][indi]
        
      elif m == 'distribution_matching':
        selected, metric = distribution_matching(self._t_feature_labels, 
                              K_return=self._rssb.bins.shape[1], 
                              **self._exp['buffer']['distribution_matching_cfg'])
        ret_globale_indices = self._t_ret[:,self._t_ret_metrices][selected]
      elif m == 'random':
        selected = torch.randperm( self._t_ret.shape[0], device=self.device)[:self._rssb.bins.shape[1]]
        ret_globale_indices = self._t_ret[:,self._t_ret_metrices][selected]
      elif m == 'kmeans':
        flag = m = self._exp['buffer']['kmeans']['perform_distribution_matching_based_on_subset']
        if flag:
          start_can = 4
        else:
          start_can = 2 
        selected = get_kMeans_indices( self._t_latent_feature_all, self._rssb.bins.shape[1], flag, start_can )
        
        if flag: 
          # perfrom dist matching
          selected, metric = distribution_matching(self._t_feature_labels, 
                              K_return=self._rssb.bins.shape[1], 
                              **self._exp['buffer']['distribution_matching_cfg'],
                              sub_selection = selected)
          
        ret_globale_indices = self._t_ret[:,self._t_ret_metrices][selected]        
      else:
        raise Exception('Undefined mode on_test_epoch_end')
      
      
      # Writes to RSSB
      if self._exp['buffer'].get('sync_back', True):
          if self._rssb_active:
            self._rssb.bins[self._task_count,:] = ret_globale_indices
            self._rssb.valid[self._task_count,:] = True
        
      
      ### THIS IS ONLY FOR VISUALIZATION
      dataset_indices = ret_globale_indices.clone() # create clone that will be filled with correct indices
      
      # converte global to locale indices of task dataloader
      gtli = torch.tensor( self.trainer.test_dataloaders[0].dataset.global_to_local_idx, 
                          device=self.device)
      for i in range( ret_globale_indices.shape[0] ):
        dataset_indices[i] = torch.where( gtli == ret_globale_indices[i])[0]
        
      # extract data from buffer samples:
      #              - pixelwise labels
      #              - image, label picture
      nr_images = 16  
      images = []
      labels = []
      label_sum_buffer = torch.zeros( (self._exp['model']['cfg']['num_classes']),
                                    device=self.device )
      for images_added, ind in enumerate( list( dataset_indices )):
        batch = self.trainer.test_dataloaders[0].dataset[int( ind )]
        
        indi, counts = torch.unique( batch[1] , return_counts = True)
        for i,cou in zip( list(indi), list(counts) ):
          if i != -1:
            label_sum_buffer[int( i )] += int( cou )
        if images_added < nr_images:
          images.append( batch[2]) # 3,H,W
          labels.append( batch[1][None].repeat(3,1,1)) # 3,H,W 
            
      label_sum_buffer = label_sum_buffer / label_sum_buffer.sum()
      
      # Plot Pixelwise
      self.visualizer.plot_bar(self._t_label_sum, x_label='Label', y_label='Count',
                                title=f'Task-{self._task_count} Pixelwise Class Count',
                                sort=False, reverse=True, 
                                tag=f'Pixelwise_Class_Count_Task', method='left')
      self.visualizer.plot_bar(label_sum_buffer, x_label='Label', y_label='Count',
                            title=f'Buffer-{self._task_count} Pixelwise Class Count',
                            sort=False, reverse=True, 
                            tag=f'Buffer_Pixelwise_Class_Count', method='right')
      
      # Plot Images
      grid_images = make_grid(images,nrow = 4,padding = 2,
              scale_each = False, pad_value = 0)
      grid_labels = make_grid(labels,nrow = 4,padding = 2,
              scale_each = False, pad_value = -1)
      self.visualizer.plot_image( img = grid_images, 
                                  tag = f'{self._task_count}_Buffer_Sample_Images', 
                                  method = 'left')
      self.visualizer.plot_segmentation( seg = grid_labels[0], 
                                          tag = f'Buffer_Sample_Images_Labels_Task-{self._task_count}',
                                          method = 'right')

      # Plot return metric statistics
      for i in range(self._t_ret_metrices):
        m = self._t_ret_name[i]
        self.visualizer.plot_bar(self._t_ret[:,i], x_label='Sample', y_label=m+'-Value' ,
                                title=f'Task-{self._task_count}: Top-K direct selection metric {m}', 
                                sort=True, 
                                reverse=True, 
                                tag=f'Buffer_Eval_Metric_{m}')
      if self._rssb_active:
        print('ON_TEST_EPOCH_END: Set bin selected the following values: \n'+ 
                      str( self._rssb.bins[self._task_count,:]) )
      
      # restore the extraction settings
      self.model.extract = self._extract_store
      self.model.extract_layer = self._extract_layer_store
      # restore replay state
    
  def use_latent_features(self, feat,global_indices,plot=True):
    cfg = self._exp['buffer']['latent_feat']
  
    ret_globale_indices = get_image_indices(feat, global_indices,
      **cfg.get('get_image_cfg',{}) , K_return= self._buffer_elements )
    if plot:
      # common sense checking
      val, counts = torch.unique( global_indices[global_indices!=0], return_counts=True )
      if counts.max() > 1:
        raise Exception('USE_LATENT_FEATURES: Something is wrong the global_indices are repeated! ')
      
      
      classes = torch.zeros( (feat.shape[1]), device=self.device )
      for i in range(ret_globale_indices.shape[0]):
          idx = int( torch.where( global_indices == ret_globale_indices[i] )[0])
          for n in range( feat.shape[1] ):
              if feat[idx,n,:].sum() != 0:
                  classes[n] += 1
      
      pm = cfg.get('get_image_cfg',{}).get('pick_mode', 'class_balanced')
      self.visualizer.plot_bar(classes, sort=False, title=f'Buffer-{self._task_count}: Class occurrences latent features {pm}',
                                tag = f'Labels_class_balanced_cos_Buffer-{self._task_count}',y_label='Counts',x_label='Classes', method='left' )
      
      classes = torch.zeros( (feat.shape[1]), device=self.device)
      for i in range(feat.shape[0]):
          idx = int(i)
          for n in range( feat.shape[1] ):
              if feat[idx,n,:].sum() != 0:
                  classes[n] += 1
      self.visualizer.plot_bar(classes, sort=False, title=f'Task-{self._task_count}: Class occurrences in in full Task',
                                tag = f'Buffer_Class Occurrences per Image (Latent Features)',y_label='Counts',x_label='Classes', method='right' )
      return ret_globale_indices

  def fill_step(self, batch, batch_idx):
    """
    Extract metric + latent features
    """
    BS = batch[0].shape[0]
    global_index =  batch[4]    
    
    outputs = self(batch = batch[0]) 
    pred = outputs[0]
    features = outputs[1]
    label = batch[1]
    _BS,_C,_H, _W = features.shape
    label_features = F.interpolate(label[:,None].type(features.dtype), (_H,_W), mode='nearest')[:,0].type(label.dtype)
    
    NC = self._exp['model']['cfg']['num_classes']
    
    latent_feature = torch.zeros( (_BS,NC,_C), device=self.device ) #10kB per Image if 16 bit
    for b in range(BS): 
      for n in range(NC):
        m = label_features[b]==n
        if m.sum() != 0:
          latent_feature[b,n] = features[b][:,m].mean(dim=1)
    
    res1 = get_softmax_uncertainty_max(pred) # confident 0 , uncertain 1
    res2 = get_softmax_uncertainty_distance(pred) # confident 0 , uncertain 1
    res3 = F.cross_entropy(pred, batch[1], ignore_index=-1,reduction='none').mean(dim=[1,2]) # correct 0 , incorrect high
    res = torch.stack([res1,res2,res3],dim=1)
    return res.detach(), global_index.detach(), latent_feature.detach()
  
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