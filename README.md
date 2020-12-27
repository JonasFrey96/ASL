# ASL Master Thesis 

## Challenge
Master Thesis researching continous learning for semantic segmentation. 
Here the challenge is to adapt to different enviroments. 
We face the same challenges as catastropic forgetting but also integrate into unsupervised semantic segmentation methods. 

## Usecase
A simple usecase can be a robot that is pretrained on several indoor datasets but is deployed to a new indoor lab.
Instead of generalizing to all indoor-enviroments in the world we tackle the challenge by continously integrating gathered knowledge of the robots enviroment in an unsupervised fashion in the semantic segmentation network.

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
	- **loaders/**
	- **models/**
	- **utils/**
	- **visu/**
- **cfg/**
	- **dataset/**: config files for dataset
	- **env/**: cluster workstation environment
	- **exp/**: experiment configuration
- main.py
