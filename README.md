# ASL Master Thesis 

## Challenge
Continous learning for semantic segmentation. 
We focus on adaptation to different enviroments.


Similar fields in semantic segmentation: 
-Unsupervised semantic segmentation: 
In a lot of robotic application and especially semantic segmentation labeled data is expensive therefore employing unsupervised or semi-supervised methods methods is desirable to achieve the adaptaion.
- Domain adaptation: 
Semantic segmentation or optical flow training data is often generated using game engines or simulators (GTA 5 Dataset or Similar ones).
Here the chellenge is how to deal with the domain shift between the target domain (real data) and training domain (synthetic data)- 
- Class Incremental Continual Learning: 
To study catastopic forgetting and aloghrithms on how to update a Neural Network class incremental learning is a nice toy task. 
There should be a clear carry over between tasks. The skill of classifying CARS and ROADS help each other. Also this scenario is easly set up. 
- Knoledge Destilation: 
Extracting knowlege from a trained network to train an other one. Student Teacher models. 
- Transfer Learning: 
Learning one task and transfering the learned knowledge to achieve other tasks. This is commonly done wiht auxilary training losses/tasks or pretraining on an other task/dataset. 
- Contrastive Learning: 
Can be used to fully unsupervised learn meaningful embeedings.

## Usecase
A simple usecase can be a robot that is pretrained on several indoor datasets but is deployed to a new indoor lab.
Instead of generalizing to all indoor-enviroments in the world we tackle the challenge by continously integrating gathered knowledge of the robots enviroment in an unsupervised fashion in the semantic segmentation network.

The problem of **GENERALIZATION**: NN are shown to work great when you are able to densly sample within the desired target distribution with provided training data. 
This is often not possible (lack of data). Generalization to a larger domain of inputs can be achieved by data augmentation or providing other suitable human designed priors. Non or the less these methods are inrinisically limited.  
Given this inrinsic limitation and the fact that data is nearly always captured in timeseries and humans learn in the same fashion the continual learning paradim arises.

## First Step Dataset
Given that there is now benchmark for continous learning for semantic segmentation we propose benchmark based on a existing semantic segmentation dataset. 
With addtional metrices such as compute, inference time and memory consumption this allows for a comparision of different methods. 

Additionally we provide mutiple baselines:
1. Training in an non continous setting
2. Training in a navie-way 
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
	- **setup/**: conda enviorment files
- **notebooks**
- **tools**
	- **leonhard/**
	- **natrix/**
- main.py



