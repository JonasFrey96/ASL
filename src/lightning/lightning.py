# STD
import copy
import time

# MISC 
import numpy as np
import pandas as pd

# DL-framework
import torch
from pytorch_lightning.core.lightning import LightningModule
from torchvision import transforms
from pytorch_lightning import metrics as pl_metrics
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from torch.nn import functional as F
import datetime

# MODULES
from models_asl import FastSCNN, ReplayStateSyncBack



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
    
    self._rssb = ReplayStateSyncBack( **exp['replay']['cfg_rssb'] )

    self._mode = 'train'
    
    self.train_acc = pl_metrics.classification.Accuracy()
    self.train_aux_acc = pl_metrics.classification.Accuracy()
    self.train_aux_vs_gt_acc = pl_metrics.classification.Accuracy()

    
    self.val_acc = torch.nn.ModuleList( 
      [pl_metrics.classification.Accuracy() for i in range(exp['replay']['cfg_rssb']['bins'])] )
    self.val_aux_acc = torch.nn.ModuleList( 
      [pl_metrics.classification.Accuracy() for i in range(exp['replay']['cfg_rssb']['bins'])] ) 
    self.val_aux_vs_gt_acc = torch.nn.ModuleList( 
      [pl_metrics.classification.Accuracy() for i in range(exp['replay']['cfg_rssb']['bins'])] ) 


    self._task_name = 'NotDefined' # is used for model checkpoint nameing
    self._task_count = 0 # so this here might be a bad idea. Decide if we know the task or not
    self._type = torch.float16 if exp['trainer'].get('precision',32) == 16 else torch.float32
    self._train_start_time = time.time()
    

    self._replayed_samples = 0
    self._real_samples = 0
    self._val_results = {} 

    self._visu_callback = None
    self._ltmene = self._exp['visu'].get('log_training_metric_every_n_epoch',9999)


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
        
  def compute_loss(self, pred, label, aux_valid, replayed, aux_label=None, **kwargs):
    """
    Args:
        pred (torch.tensor): BSxCxHxW.
        label (torch.tensor]): BSxHxW.
        aux_label (torch.tensor): BSxHxW or BSxCxHxW.
        replayed (torch.long): BS wheater a sampled is replayed or not.
        use_aux (torch.bool): Wheater to use aux_label or label to calculate the loss.
        not_reduce (bool, optional): reduce the loss or return for each element in batch.
    Returns:
        [type]: [description]
    """
    use_soft = len(label.shape) == 4
    
    nr_replayed = (replayed != -1).sum()
    BS = replayed.shape[0]
    self._replayed_samples += int( nr_replayed )
    self._real_samples += int( BS - nr_replayed )
    
    if aux_valid.sum() != 0:        
      # compute auxillary loss
      if use_soft: 
        aux_loss = F.mse_loss( torch.nn.functional.softmax(pred, dim=1), aux_label,reduction='none').mean(dim=[1,2,3])
      else:
        aux_loss = F.cross_entropy(pred, aux_label, ignore_index=-1, reduction='none').mean(dim=[1,2])
    else:
      aux_loss = torch.zeros( (BS), device= pred.device)


    if aux_valid.sum() != BS:
      # compute normal loss on labels
      non_aux_loss = F.cross_entropy(pred, label, ignore_index=-1, reduction='none').mean(dim=[1,2])
    else:
      non_aux_loss = torch.zeros( (BS), device= pred.device)
    
    # return the reduce mean
    return ( (aux_loss * aux_valid).sum() + (non_aux_loss * ~aux_valid).sum() ) / BS 
  
  def parse_batch(self, batch):
    ba = {}
    if len( batch ) == 1:
      raise Exception("Dataloader is set to unique and not implemented")
    ba['images'] = batch[0]
    ba['label'] = batch[1]
    ba['replayed'] = batch[2]
    if len(batch) == 4:
      ba['ori_img'] = batch[3]
    if len(batch) == 5:
      ba['aux_label'] = batch[3]
      ba['aux_valid'] = batch[4]
    if len(batch) == 6:
      ba['aux_label'] = batch[3]
      ba['aux_valid'] = batch[4]
      ba['ori_img'] = batch[5]
    return ba

  ##################
  #### TRAINING ####
  ##################
  
  def on_fit_start(self):
    print(" ================ START FITTING ==================")
    print(f" TASK NAME: {self._task_name } ")
    print(f" TASK COUNT: {self._task_count } ")
    print(f" CURRENT EPOCH: {self.current_epoch } ")
    print(f" CURRENT EPOCH: {self.global_step } ")
    print(f" RSSB STATE: ", self._rssb.valid.sum(dim=1) )
    print(" ================ START FITTING ==================")

  def on_train_epoch_start(self):
    self._mode = 'train'
    
  def training_step(self, batch, batch_idx):
    ba = self.parse_batch( batch )
    outputs = self(batch = ba['images'])

    if not ('aux_valid' in ba.keys()):
      ba['aux_valid'] = torch.zeros( (ba['images'].shape[0]), 
                                      device=ba['images'].device, 
                                      dtype=torch.bool)
    loss = self.compute_loss(  
                pred = outputs[0], 
                **ba)
    self.log('train_loss', loss, on_step=False, on_epoch=True)
    ret = {'loss': loss, 'pred': outputs[0], 'label': ba['label'], 'ori_img': ba['ori_img']  }
    
    if 'aux_label' in ba.keys():
      ret["aux_label"] = ba['aux_label']
      ret["aux_vaild"] = ba['aux_valid']
    
    return ret
  
  def training_step_end(self, outputs):
    with torch.no_grad():
      self._visu_callback.training_step_end(self.trainer, self, outputs)

    # LOG REPLAY / REAL
    self.logger.log_metrics( 
      metrics = { 'real': torch.tensor(self._real_samples),
                  'replayed': torch.tensor(self._replayed_samples)},
      step = self.global_step)

    if self.current_epoch % self._ltmene == 0 :
      # LOG ACCURRACY
      self._acc_cal( outputs, 
                  self.train_acc ,
                  self.train_aux_acc ,
                  self.train_aux_vs_gt_acc )

    return {'loss': outputs['loss']}
  

  ####################
  #### VALIDATION ####
  ####################

  def on_validation_epoch_start(self):
    self._mode = 'val'
    
  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    images, label = batch[:2]

    outputs = self(images)
   
    loss = F.cross_entropy(outputs[0], label , ignore_index=-1 ) 

    ret = {'pred': outputs[0], 'label': label,
            'dataloader_idx': dataloader_idx, 'loss_ret': loss }

    if len(batch) == 3:
      ret['ori_img'] = batch[2]
    if len(batch) > 3 :
      ret['aux_label'] = batch[2]
      ret['aux_valid'] = batch[3]
      ret['ori_img'] = batch[4]
    
    return ret

  def validation_step_end( self, outputs ):
    with torch.no_grad():
      self._visu_callback.validation_step_end(self.trainer, self, outputs)
    dataloader_idx = outputs['dataloader_idx']
    self._acc_cal( outputs, 
                  self.val_acc[dataloader_idx],
                  self.val_aux_acc[dataloader_idx],
                  self.val_aux_vs_gt_acc[dataloader_idx] )

  @torch.no_grad()
  def _acc_cal(self, outputs, acc, aux_acc, aux_vs_gt_acc ):
    pred = torch.argmax(outputs['pred'], 1)

    m = outputs['label'] > -1
    acc( pred[m], outputs['label'][m])
    self.log(f'{self._mode}_acc', acc, on_step=False, on_epoch=True)

    if 'aux_valid' in outputs.keys():
      aux_m = outputs['aux_label'] > -1
      aux_acc( pred[aux_m], outputs['aux_label'][aux_m])
      self.log(f'{self._mode}_aux_acc', aux_acc, on_step=False, on_epoch=True)

      aux_m2 = aux_m * m
      aux_vs_gt_acc( outputs['label'][aux_m2], outputs['aux_label'][aux_m2])
      self.log(f'{self._mode}_aux_vs_gt_acc', aux_vs_gt_acc, on_step=False, on_epoch=True)


  def validation_epoch_end(self, outputs):
    self.log(f'task_count', self._task_count, on_step=False, on_epoch=True, prog_bar=False)


    metrics = self.trainer.logger_connector.callback_metrics
    me =  copy.deepcopy ( metrics ) 
    for k in me.keys():
      try:
        me[k] = "{:10.4f}".format( me[k])
      except:
        pass
    
    t_l = me.get('train_loss', 'NotDef')
    v_acc = me.get('val_acc', 'NotDef')
    
    try:
      # only works when multiple val-dataloader are set!
      if len( self._val_results ) == 0:
        for i in range(self._exp['replay']['cfg_rssb']['bins']):
          self._val_results[f'val_acc/dataloader_idx_{i}'] = float(metrics[f'val_acc/dataloader_idx_{i}'])
      else:
        val_results = {}
        for i in range(self._exp['replay']['cfg_rssb']['bins']):
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
      
      print('VALIDATION_EPOCH_END: '+ 
        f"Exp: {n} | Epoch: {epoch} | TimeEpoch: {t} | TimeStart: {t2} |  >>> Train-Loss: {t_l } <<<   >>> Val-Acc: {v_acc} <<<"
      )
    self._epoch_start_time = time.time()
    print( "SELF TRAINER SHOULD STOP", self.trainer.should_stop, self.device )

  def on_save_checkpoint(self, params):
    pass

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