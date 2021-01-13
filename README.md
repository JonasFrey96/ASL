# ASL Master Thesis 

## Challenge
Continual learning for semantic segmentation. 
We focus on adaptation to different environments.


### Similar fields in semantic segmentation: 
- **Unsupervised semantic segmentation:** 
> > >  In a lot of robotic application and especially semantic segmentation labeled data is expensive therefore employing unsupervised or semi-supervised methods is desirable to achieve the adaptation.
- **Domain adaptation:** 
> > > Semantic segmentation or optical flow training data is often generated using game engines or simulators (GTA 5 Dataset or Similar ones).
Here the challenge is how to deal with the domain shift between the target domain (real data) and training domain (synthetic data)- 

- **Class Incremental Continual Learning:**
> > > To study catastrophic forgetting and algorithms on how to update a Neural Network class incremental learning is a nice toy task. 
There should be a clear carry over between tasks. The skill of classifying CARS and ROADS help each other. Also this scenario is easily set up. 

- **Knowledge Distillation:**
> > > Extracting knowledge from a trained network to train an other one. Student Teacher models. 

- **Transfer Learning:**
> > > Learning one task and transferring the learned knowledge to achieve other tasks. This is commonly done with auxillary training losses/tasks or pre-training on an other task/dataset. 

- **Contrastive Learning:**
> > > Can be used to fully unsupervised learn meaningful embeddings.

## Use case
A simple use case can be a robot that is pre-trained on several indoor datasets but is deployed to a new indoor lab.

Instead of generalizing to all indoor-environments in the world we tackle the challenge by continually integrating gathered knowledge of the robots environment in an unsupervised fashion in the semantic segmentation network.

The problem of **GENERALIZATION**: NN are shown to work great when you are able to densely sample within the desired target distribution with provided training data. 
This is often not possible (lack of data). Generalization to a larger domain of inputs can be achieved by data augmentation or providing other suitable human designed priors. Non or the less these methods are intrinsically limited.  

Given this intrinsic limitation and the fact that data is nearly always captured in time-series and humans learn in the same fashion the continual learning paradigm arises.

## First Step Dataset
Given that there is now benchmark for continual learning for semantic segmentation we propose benchmark based on a existing semantic segmentation dataset. 

With additional metrics such as compute, inference time and memory consumption this allows for a comparison of different methods. 

Additionally we provide multiple baselines:
1. Training in an non continual setting
2. Training in a naive-way 
3. Our method

## Implementation
http://taskonomy.stanford.edu/

Synthetic 
### Dataset Format:

### Project:
The repository is organized in the following way:


**root/**
- **src/**
	- **datasets/**
	- **loss/**
	- **models/**
	- **utils/**
	- **visu/**
- **cfg/**
	- **dataset/**: config files for dataset
	- **env/**: cluster workstation environment
	- **exp/**: experiment configuration
	- **setup/**: conda environment files
- **notebooks**
- **tools**
	- **leonhard/**
	- **natrix/**
- main.py

### Tasks and Evaluation
We define a task T being offered by a dataloader. A dataloader can offer multiple tasks. 
Each task is trained once. As normal training a task might consists of multiple epochs to keep it general!
```
Task1:
	Name: Kitchens
	Epochs: 10
	DatasetConfig: 'something'
```
```
monitoring = []
 
for task in tasks:
	model.set_task( task.train_cfg )
	trainer.fit()
	model.set_tasks_to_eval( tasks.eval_cfg )
	res = model.eval()
	monitoring.append( res )
```

Logging: 
For each task a seperate tensorboard logfile is created.
Also a logfile for tracking the joint results is created at the beginning. 

```
model.set_task( 1 )
	 -> sets up dataloader correctly
	 -> maybe updates the optimizer
	 -> sets up  tensorboard logfile
```

## Scaling performance:
4 1080Ti GPUs 20 Cores BS 8 (effective BS 32)
Rougly running at 1.8 it/s 
-> 57.6 Images/s 

## Datasets:


|-----------------|---------------|----------------------------------------|
| Dataset         | Parameters    | Values                                 |
|-----------------|---------------|----------------------------------------|
| **NYU_v2**      | Images Train: | 1449 (total)                           |
|                 | Images Val:   |                                        |
|                 | Annotations:  | NYU-40                                 |
|                 | Optical Flow: | True                                   |
|                 | Depth:        | True                                   |
|                 | Total Size:   | 3.7GB                                  |
| **ML-Hypersim** | Images Train: | 74619 (total)                          |
|                 | Images Val:   |                                        |
|                 | Annotations:  | NYU-40                                 |
|                 | Optical Flow: | False                                  |
|                 | Depth:        | True                                   |
|                 | Total Size:   | 247GB                                  |
| **COCO 2014**   | Images Train: | 330K >200K labeled                     |
|                 | Images Val:   |                                        |
|                 | Annotations:  | Object cat 90 Classes (80 used)        |
|                 | Optical Flow: | False                                  |
|                 | Depth:        | False                                  |
|                 | Total Size:   | 20GB                                   |


		
# Acknowledgement
Special thanks to:

People at <http://continualai.org> for the inspiration and feedback.

The authors of Fast-SCNN: Fast Semantic Segmentation Network. ([arxiv](https://arxiv.org/pdf/1902.04502.pdf))

TRAMAC <https://github.com/Tramac> for implementing Fast-SCNN in PyTorch [Fast-SCNN-pytorch](https://github.com/Tramac/Fast-SCNN-pytorch).
