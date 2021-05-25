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
    self._visu_callback.validation_step_end(self.trainer, self, outputs)
    dataloader_idx = outputs['dataloader_idx']
    self._acc_cal( outputs, 
                  self.val_acc[dataloader_idx],
                  self.val_aux_acc[dataloader_idx],
                  self.val_aux_vs_gt_acc[dataloader_idx] )

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

  """
  from uncertainty import get_softmax_uncertainty_max, get_softmax_uncertainty_distance
  from uncertainty import get_image_indices
  from uncertainty import distribution_matching
  from uncertainty import get_kMeans_indices
  from uncertainty import interclass_dissimilarity
  from uncertainty import gradient_dissimilarity
  from uncertainty import hierarchical_dissimilarity
  from gradient_helper import * # get_weights, get_grad, set_grad, gem_project, sum_project, random_project


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
    self._t_gradient_list = []
    
  def test_step(self, batch, batch_idx):
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
  
  @torch.no_grad()
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
      elif m == 'gradient_selection':
        self._t_gradient_list = [t[None] for t in self._t_gradient_list]
        # selected, metric = gradient_dissimilarity( 
        #   torch.cat( self._t_gradient_list ) , 
        #   K_return= self._rssb.bins.shape[1], 
        #   **self._exp['buffer']['gradient_selection_cfg'])
        # hierarchical_dissimilarity(X, K=50, maxSize=100)
        sub = 50
        grad = torch.cat( self._t_gradient_list )
        print("orignal_size", grad.shape)
        new_grad = grad[:,::sub]
        selected = hierarchical_dissimilarity( new_grad , K=self._rssb.bins.shape[1], maxSize=100, device= self._t_ret.device)
        
        
        selected.to( self._t_ret.device )
        
        ret_globale_indices = self._t_ret[:,self._t_ret_metrices][selected]
      elif m == 'distribution_matching':
        selected, metric = distribution_matching(self._t_feature_labels, 
                              K_return=self._rssb.bins.shape[1], 
                              **self._exp['buffer']['distribution_matching_cfg'])

        
        ret_globale_indices = self._t_ret[:,self._t_ret_metrices][selected]
      elif m == 'random':
        selected = torch.randperm( self._t_ret.shape[0], device=self.device)[:self._rssb.bins.shape[1]]
        ret_globale_indices = self._t_ret[:,self._t_ret_metrices][selected]
      elif m == "interclass_dissimilarity":
        selected, metric = interclass_dissimilarity(self._t_latent_feature_all, 
                                 self._t_feature_labels, 
                                 K_return=self._rssb.bins.shape[1], iterations= 5000)
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
    with torch.set_grad_enabled(True):
        
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
      
      self.model.zero_grad()
      l = res3.mean()
      l.backward()
      res = get_grad( self.model.named_parameters() ) #4 MB per sample 
      self._t_gradient_list.append( res.detach().cpu() )
      
    res = torch.stack([res1,res2,res3],dim=1)
    return res.detach(), global_index.detach(), latent_feature.detach()
    
  """

