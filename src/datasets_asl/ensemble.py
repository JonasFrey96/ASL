# Idee pass in mutiple datasets
# Pass in the ratios according to which they should be sampled
# Allows internally masking a dataset such that only a subset of elements is accessible via filtering
# Fully replace the memory thing


from torch.utils.data import Dataset
import numpy as np

eps = 0.0001

class Ensemble(Dataset):
	def __init__(self, main_dataset, replay_datasets, probs):
		"""Initalizes the datasets

		Args:
				datasets (list of datasets): must return same sized elements
		"""
		self.replay_datasets = replay_datasets
		self.main_dataset = main_dataset
		self.probs = np.array( probs )
		
		# Quick Systematic Problem Explained:
		# Define a new length based on the probs !
		assert np.abs(np.array(probs).sum() -1 ) < eps

		self._length = len(main_dataset) *(1+ (1/(1-probs)))
		
		self.replayed = [False]*len(main_dataset) # either_get_from_main_dataset
		self.replayed += [True]*( len(main_dataset)*1/(1-probs))  # either_get_from_main_dataset

		def __getitem__(self,index):
			if not self.replayed[index]:
				# Return value from main dataset
				return self.main_dataset[index]
			else:
				# Return value from replayed datasets
				return self.get_random_replay_item()

		def get_random_replay_item(self):
			dataset_idx = np.argmax( np.random.random( len(self.replay_datasets) ) * self.probs)
			dataset_ele_idx = np.random.randint( 0, len(self.replay_datasets[dataset_idx]) )
			return self.replay_datasets[dataset_idx][dataset_ele_idx]

		def __len__(self):
			return self._length