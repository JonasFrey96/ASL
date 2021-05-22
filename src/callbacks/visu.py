# TODO: Jonas Frey write test for this
# TODO:Decide of I should i remove the logging from thsi callback ? log() 

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_warn

import torch
from torchvision.utils import make_grid
import os

from visu import Visualizer


__all__ = ['VisuCallback']

class VisuCallback(Callback):
  def __init__(self, exp):
    self.visualizer = Visualizer(
      p_visu=os.path.join( exp['name'], 'visu'),
      logger=None,
      num_classes= exp['model']['cfg']['num_classes']+1)

    self.logged_images = {
      'train': 0,
      'val': 0,
      'test': 0
    }

  def on_epoch_start(self, trainer, pl_module):
    self.visualizer.epoch = pl_module.current_epoch
    # reset logged images counter
    self.logged_images = dict.fromkeys(self.logged_images, 0)

  def on_train_start(self, trainer, pl_module):
    # Set the Logger given that on init not initalized yet-
    self.visualizer.logger= pl_module.logger

  def training_step_end(self, trainer, pl_module, outputs):
    # Logging + Visu
    if pl_module.current_epoch % pl_module._exp['visu'].get('log_training_metric_every_n_epoch',9999) == 0 : 
      pred = torch.argmax(outputs['pred'], 1)
      target = outputs['target']
      train_mIoU = pl_module.train_mIoU(pred,target)
      # calculates acc only for valid labels
      m  =  target > -1
      train_acc = pl_module.train_acc(pred[m], target[m])
      
      pl_module.log('train_acc_epoch', pl_module.train_acc, on_step=False, on_epoch=True, prog_bar = True)
      pl_module.log('train_mIoU_epoch', pl_module.train_mIoU, on_step=False, on_epoch=True, prog_bar = True)
      
      if "aux_target" in outputs:
        m2 = m* (outputs['aux_target']>-1)
        pl_module.train_aux_acc(pred[m2], outputs['aux_target'][m2])
        pl_module.train_aux_vs_gt_acc(target[m2], outputs['aux_target'][m2])
        pl_module.log('train_PRED_VS_AUX_acc_epoch', pl_module.train_aux_acc )
        pl_module.log('train_GT_VS_AUX_acc_epoch', pl_module.train_aux_vs_gt_acc )

    if ( pl_module._exp['visu'].get('train_images',0) > pl_module.logged_images_train and 
         pl_module.current_epoch % pl_module._exp['visu'].get('every_n_epochs',1) == 0):
      pred = torch.argmax(outputs['pred'], 1).clone().detach()
      target = outputs['target'].clone().detach()
      
      pred[ target == -1 ] = -1
      pred += 1
      target += 1
      
      self.logged_images['train'] += 1
      
      BS = pred.shape[0]
      rows = int( BS**0.5 )
      grid_pred = make_grid(pred[:,None].repeat(1,3,1,1),nrow = rows,padding = 2,
              scale_each = False, pad_value = 0)
      grid_target = make_grid(target[:,None].repeat(1,3,1,1),nrow = rows,padding = 2,
              scale_each = False, pad_value = 0)
      grid_image = make_grid(outputs['ori_img'],nrow = rows,padding = 2,
              scale_each = False, pad_value = 0)

      if "aux_target" in outputs:
        aux_target = outputs['aux_target'].clone().detach()
        grid_aux_target = make_grid(aux_target[:,None].repeat(1,3,1,1),nrow = rows,padding = 2,
              scale_each = False, pad_value = 0)
        self.visualizer.plot_segmentation(tag=f'', seg=grid_pred[0], method='right')
        self.visualizer.plot_segmentation(tag=f'{self._mode}_AUX_left_pred_right_{self._task_name}_{self.logged_images_train}', seg=grid_aux_target[0], method='left')  

        self.visualizer.plot_segmentation(tag=f'', seg=grid_target[0], method='right')
        self.visualizer.plot_segmentation(tag=f'{self._mode}_AUX_left_GT_right_{self._task_name}_{self.logged_images_train}', seg=grid_aux_target[0], method='left')  
      

      print(grid_pred.shape, grid_target.shape, grid_image.shape)
      self.visualizer.plot_segmentation(tag=f'', seg=grid_pred[0], method='right')
      self.visualizer.plot_segmentation(tag=f'{self._mode}_gt_left_pred_right_{self._task_name}_{self.logged_images_train}', seg=grid_target[0], method='left')  
      self.visualizer.plot_segmentation(tag=f'', seg=grid_pred[0], method='right')
      self.visualizer.plot_image(tag=f'{self._mode}_img_ori_left_pred_right_{self._task_name}_{self.logged_images_train}', img=grid_image, method='left')
    
  def validation_step(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
    if pl_module._dataloader_index_store != dataloader_idx:
      pl_module._dataloader_index_store = dataloader_idx
      pl_module.logged_images['val'] = 0

  def validation_step_end( self, trainer, pl_module, outputs ):
    dataloader_idx = outputs['dataloader_idx']
    pred, target = outputs['pred'],outputs['target']
    if ( pl_module._exp['visu'].get('val_images',0) > self.logged_images['val'] and 
         pl_module.current_epoch % pl_module._exp['visu'].get('every_n_epochs',1)== 0) :
      self.logged_images['val'] += 1
      pred_c = pred.clone().detach()
      target_c = target.clone().detach()
      
      pred_c[ target_c == -1 ] = -1
      pred_c += 1
      target_c += 1
      BS = pred_c.shape[0]
      rows = int( BS**0.5 )
      grid_pred = make_grid(pred_c[:,None].repeat(1,3,1,1),nrow = rows,padding = 2,
              scale_each = False, pad_value = 0)
      grid_target = make_grid(target_c[:,None].repeat(1,3,1,1),nrow = rows,padding = 2,
              scale_each = False, pad_value = 0)
      grid_image = make_grid(outputs['ori_img'],nrow = rows,padding = 2,
              scale_each = False, pad_value = 0)
      self.visualizer.plot_segmentation(tag=f'', seg=grid_pred[0], method='right')
      self.visualizer.plot_segmentation(tag=f'val_gt_left_pred_right__Name_{pl_module._task_name}__Val_Task_{dataloader_idx}__Sample_{self.logged_images_val}', seg=grid_target[0], method='left')
      self.visualizer.plot_segmentation(tag=f'', seg=grid_pred[0], method='right')
      self.visualizer.plot_image(tag=f'val_img_ori_left_pred_right__Name_{pl_module._task_name}_Val_Task_{dataloader_idx}__Sample_{self.logged_images_train}', img=grid_image, method='left')
      