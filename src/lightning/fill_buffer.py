import os
import time

import torch
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from torch.nn import functional as F
from torchvision.utils import make_grid
# Uncertainty
from uncertainty import get_softmax_uncertainty_max, get_softmax_uncertainty_distance
from uncertainty import get_image_indices


__all__ = ['fill_buffer']

def fill_buffer(model, root_gpu, dataloader_buffer):
	model.cuda(root_gpu)
			
	# modify extraction settings to get uncertainty
	extract_store = model.model.extract
	extract_layer_store = model.model.extract_layer
	model.model.extract = True
	model.model.extract_layer = 'fusion' # if we use fusion we get a 128 feature
	store_replay_state = model.trainer.train_dataloader.dataset.replay
	model.trainer.train_dataloader.dataset.replay = False
	model.trainer.train_dataloader.dataset.unique = True
	
	# how can we desig this
	# this function needs to be called the number of tasks -> its not so important that it is effiecent
	# load the sample with the datloader in a blocking way -> This might take 5 Minutes per task
	rank_zero_info('FILL_BUFFER: Called')
	BS = model._exp['loader']['batch_size']
	
	if model._exp.get('buffer',{}).get('mode', 'softmax_max')  == 'all':
		ret_metrices = 3
		ret_name = ['softmax_max', 'softmax_distance', 'loss']
	else:
		ret_metrices = 1
		ret_name = [model._exp.get('buffer',{}).get('mode', 'softmax_max')]
		
	ret = torch.zeros( (int(BS*len(dataloader_buffer)),1+ret_metrices), device=model.device )
	latent_feature_all = torch.zeros( (int(BS*len(dataloader_buffer)),40,128), device=model.device, dtype=torch.float16 )
	label_sum = torch.zeros( (model._exp['model']['cfg']['num_classes']),device=model.device )
	s = 0
	
	
	# calculation of: 
	#                 - pixelwise labels full dataset
	#                 - return metric full dataset
	#                 - latent features
	st = time.time()
	l = len(dataloader_buffer)
	
	for batch_idx, batch in enumerate(dataloader_buffer):
		if batch_idx % int(model.trainer.log_every_n_steps/5) == 0:
			info = f'FILL_BUFFER: Analyzed Dataset {batch_idx}/{l} Time: {time.time()-st}'
			rank_zero_info(info)
		# if batch_idx > 20:
		# 	break # TODO undo
 
		for b in range(len(batch)):
			batch[b] = batch[b].to(model.device)
		
		indi, counts = torch.unique( batch[1] , return_counts = True)
		for i,cou in zip( list(indi), list(counts) ):
			if i != -1:
				label_sum[int( i )] += int( cou )
			
		res, global_index, latent_feature = fill_step(model, batch, batch_idx)
		
		for ba in range( res.shape[0] ):  
			if batch[4][ba] != -999:
				ret[s,:ret_metrices] = res[ba]
				ret[s,ret_metrices] = global_index[ba]
				latent_feature_all[s] = latent_feature[ba]
				s += 1
	ret = ret[:s]
	latent_feature_all = latent_feature_all[:s]
	torch.save(latent_feature_all.cpu(), model._exp['name'] + f'/latent_feature_tensor_{model._task_count}.pt')
	label_sum = label_sum/label_sum.sum()
	
	# write results to the rssb
	if model._exp['buffer'].get('latent_feat',{}).get('active',False):
		# more complex evaluate according to latent feature space
		ret_globale_indices = use_latent_features(model, latent_feature_all, ret[:,ret_metrices] )
		if model._exp['buffer'].get('sync_back', True):
			model._rssb.bins[model._task_count,:] =  ret_globale_indices
			model.rssb_to_dataset()
		else:
			rank_zero_warn('Fill Buffer: Did not sync back to RSSB')
	else:
		# simple method use top-K  of the computed metric in fill_step
		_, indi = torch.topk( input = ret[:,0],
													k = model._rssb.bins.shape[1],
													largest = model._exp['buffer'].get('use_highest_uncertainty',True))
		if model._exp['buffer'].get('sync_back', True):
			model._rssb.bins[model._task_count,:] = ret[:,ret_metrices][indi]
			model.rssb_to_dataset()
		else:
			rank_zero_warn('Fill Buffer: Did not sync back to RSSB')
	
	
	
	# calculate indices of dataloader to get pixelwise classes for images in buffer
	if not model._exp['buffer'].get('latent_feat',{}).get('active',False):
		_, indi = torch.topk( ret[:,0] , model._buffer_elements )
		globale_indices_selected = ret[:,ret_metrices][indi]
	else:
		globale_indices_selected = ret_globale_indices
	dataset_indices = globale_indices_selected.clone()
	
	# converte global to locale indices of task dataloader
	gtli = torch.tensor( model.trainer.train_dataloader.dataset.global_to_local_idx , device=model.device)
	for i in range( globale_indices_selected.shape[0] ):
		dataset_indices[i] = torch.where( gtli == globale_indices_selected[i])[0]
		
		
	# extract data from buffer samples:
	#              - pixelwise labels
	#              - image, label picture
	nr_images = 16  
	images = []
	labels = []
	label_sum_buffer = torch.zeros( (model._exp['model']['cfg']['num_classes']),device=model.device )
	for images_added, ind in enumerate( list( dataset_indices )):
		batch = model.trainer.train_dataloader.dataset[int( ind )]
		
		indi, counts = torch.unique( batch[1] , return_counts = True)
		for i,cou in zip( list(indi), list(counts) ):
			if i != -1:
				label_sum_buffer[int( i )] += int( cou )
		if images_added < nr_images:
			images.append( batch[2]) # 3,H,W
			labels.append( batch[1][None].repeat(3,1,1)) # 3,H,W 
				
	label_sum_buffer = label_sum_buffer/label_sum_buffer.sum()
	
	
	# Plot Pixelwise
	model.visualizer.plot_bar(label_sum, x_label='Label', y_label='Count',
														title=f'Task-{model._task_count} Pixelwise Class Count', sort=False, reverse=True, 
														tag=f'Pixelwise_Class_Count_Task', method='left')
	model.visualizer.plot_bar(label_sum_buffer, x_label='Label', y_label='Count',
												title=f'Buffer-{model._task_count} Pixelwise Class Count', sort=False, reverse=True, 
												tag=f'Buffer_Pixelwise_Class_Count', method='right')
	
	# Plot Images
	grid_images = make_grid(images,nrow = 4,padding = 2,
					scale_each = False, pad_value = 0)
	grid_labels = make_grid(labels,nrow = 4,padding = 2,
					scale_each = False, pad_value = -1)
	model.visualizer.plot_image( img = grid_images, 
															tag = f'{model._task_count}_Buffer_Sample_Images', 
															method = 'left')
	model.visualizer.plot_segmentation( seg = grid_labels[0], 
																			tag = f'Buffer_Sample_Images_Labels_Task-{model._task_count}',
																			method = 'right')

	# Plot return metric statistics
	for i in range(ret_metrices):
		m = ret_name[i]
		model.visualizer.plot_bar(ret[:,i], x_label='Sample', y_label=m+'-Value' ,
														title=f'Task-{model._task_count}: Top-K direct selection metric {m}', 
														sort=True, 
														reverse=True, 
														tag=f'Buffer_Eval_Metric_{m}')
	
	rank_zero_info('FILL_BUFFER: Set bin selected the following values: \n'+ str( model._rssb.bins[model._task_count,:]) )
	
	# restore the extraction settings
	model.model.extract = extract_store
	model.model.extract_layer = extract_layer_store
	# restore replay state
	model.trainer.train_dataloader.dataset.replay = store_replay_state
	model.trainer.train_dataloader.dataset.unique = False
	
	model.cpu()
	torch.cuda.empty_cache()
	print('Finished')
	
def fill_step(model, batch, batch_idx):
	BS = batch[0].shape[0]
	global_index =  batch[4]    
	
	outputs = model(batch = batch[0]) 
	pred = outputs[0]
	features = outputs[1]
	label = batch[1]
	_BS,_C,_H, _W = features.shape
	label_features = F.interpolate(label[:,None].type(features.dtype), (_H,_W), mode='nearest')[:,0].type(label.dtype)
	
	NC = model._exp['model']['cfg']['num_classes']
	
	latent_feature = torch.zeros( (_BS,NC,_C), device=model.device ) #10kB per Image if 16 bit
	for b in range(BS): 
		for n in range(NC):
			m = label_features[b]==n
			if m.sum() != 0:
				latent_feature[b,n] = features[b][:,m].mean(dim=1)
	
	m = model._exp.get('buffer',{}).get('mode', 'softmax_max') 
	if m == 'softmax_max':
		res = get_softmax_uncertainty_max(pred) # confident 0 , uncertain 1
	elif m == 'softmax_distance':
		res = get_softmax_uncertainty_distance(pred) # confident 0 , uncertain 1
	elif m == 'loss':
		res = F.cross_entropy(pred, batch[1], ignore_index=-1,reduction='none').mean(dim=[1,2]) # correct 0 , incorrect high
	elif m == 'all':
		res1 = get_softmax_uncertainty_max(pred) # confident 0 , uncertain 1
		res2 = get_softmax_uncertainty_distance(pred) # confident 0 , uncertain 1
		res3 = F.cross_entropy(pred, batch[1], ignore_index=-1,reduction='none').mean(dim=[1,2]) # correct 0 , incorrect high
		res = torch.stack([res1,res2,res3],dim=1)
	else:
		raise Exception('Mode to fill buffer is not defined!')
	return res.detach(), global_index.detach(), latent_feature.detach()

def use_latent_features(model, feat,global_indices,plot=True):
	cfg = model._exp['buffer']['latent_feat']
	
	ret_globale_indices = get_image_indices(feat, global_indices,
		**cfg.get('get_image_cfg',{}) , K_return=model._buffer_elements )
	if plot:
		# model.trainer.train_dataloader.dataset[0][-1] 
		
		# common sense checking
		val, counts = torch.unique( global_indices, return_counts=True )
		if counts.max() > 1:
			raise Exception('USE_LATENT_FEATURES: Something is wrong the global_indices are repeated! ')
		
		
		classes = torch.zeros( (feat.shape[1]), device=model.device )
		for i in range(ret_globale_indices.shape[0]):
				idx = int( torch.where( global_indices == ret_globale_indices[i] )[0])
				for n in range( feat.shape[1] ):
						if feat[idx,n,:].sum() != 0:
								classes[n] += 1
		
		pm = cfg.get('get_image_cfg',{}).get('pick_mode', 'class_balanced')
		model.visualizer.plot_bar(classes, sort=False, title=f'Buffer-{model._task_count}: Class occurrences latent features {pm}',
															tag = f'Labels_class_balanced_cos_Buffer-{model._task_count}',y_label='Counts',x_label='Classes', method='left' )
		
		classes = torch.zeros( (feat.shape[1]), device=model.device)
		for i in range(feat.shape[0]):
				idx = int(i)
				for n in range( feat.shape[1] ):
						if feat[idx,n,:].sum() != 0:
								classes[n] += 1
		model.visualizer.plot_bar(classes, sort=False, title=f'Task-{model._task_count}: Class occurrences in in full Task',
															tag = f'Buffer_Class Occurrences per Image (Latent Features)',y_label='Counts',x_label='Classes', method='right' )
	return ret_globale_indices