import torch
"""
Hardcode everything in the task and make it more flexible when needed
A task is not responsible how it is trained this is fully decided by the algortihm itself. 
It is simply responsible for providing data and the logging interface.
"""


class Task():
	def __init__(self,info):
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
		self.train_dataset = 
		self.test_dataset = 
		self.val_dataset =
		
  