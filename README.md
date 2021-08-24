# 1. ASL Master Thesis

https://markdownlivepreview.com/
This is the repository accompanying my master's thesis.
 
Additionally, we created the accompanying repositories:  
 1.) Pose estimation using [ORBSLAM2 Fork](https://github.com/JonasFrey96/orb_slam_2_ros)  
 2.) Optical Flow estimation [RAFT Fork](https://github.com/JonasFrey96/RAFT)   
 3.) [Kimera Semantics Fork](https://github.com/JonasFrey96/Kimera-Semantics) (Allows fast protobuf export of semantics)  
 4.) [Kimera Interfacer](https://github.com/JonasFrey96/Kimera-Interfacer) (Generate maps and pseudo labels for ScanNet and Lab data)  
 
We only elaborate on how to run all the experiments and reproduce the results.  
For the technical background, I refer to my [Master's Thesis](https://github.com/JonasFrey96/ASL/tree/main/docs/Jonas_Frey_Master_Thesis.pdf). 

## Repository Overview:
All code related to continual learning experiments is contained in the **src** folder.  

* **src** Contains all code for training the neural network
  * **src/supervisor.py** Starts all the training tasks.  
  * **src/train_task.py** Can be called directly or is executed by the supervisor.  
Set's up logger, trainer, and model and executes a single task.  
    * **src/lightning/lightning.py** Contains the full neural network model and logging.  
It is the best starting point to get familiar with the implementation. 
    * **src/task/task_generator.py** Depending on the configuration returns a list of parameters to create multiple training and associated test datasets for each task. 

* **docker** Contains the docker files to generate the pseudo labels using 3D mapping  
  * **docker/orbslam2_ros** ORBSLAM2 Implementation  
  * **docker/semantics** Kimera Semantics + Kimera Interfacer  
  * **docker/vio** Not used in the final report (Kimera VIO)  
  * **docker/maplab** Not used in the final report (Maplab)  

* **notebooks** 
  * **notebooks/report** includes all the jupyter notebooks used to generate all figures in the report  
 
* **cfg** includes configuration files for:  
  * **cfg/env** Setting up your environment file
  * **cfg/exp** Experiment files to run all experiments
  * **cfg/conda** Python conda enviornment


## Continual Learning Experiments

### Getting started:
Clone the repository:  
```bash 
cd ~ && git clone https://github.com/JonasFrey96/ASL
```
 
Create and activate conda environment:
```bash 
cd ~/ASL && conda env create -f cfg/conda/cl.yml
conda activate cl 
```
Assumes CUDA 11.0 is installed and uses ```pytorch=1.7.1``` and ```pytorch-lightning=1.2.0```.  

You may have to adapt the conda environment depending on your system. 

#### Environment Configuration Explained
Setting up the enviornment file depending on your system.  
You can either create a new enviornment file or edit an existin one in the ```env``` folder.  
For example you can create ```ASL/cfg/env/your_env_name.yml```, with the following content:

```yaml
# Configuration
workstation: True                 
# If False uses Proxy and copies Dataset to $TMPDIR folder
# -> allows to train on Leonhard and Euler Cluster

base: path_to_your_logging_directory # Here all your experiments are logged
 
# Datasets
cityscapes: ../datasets/cityscapes
nyuv2: ../datasets/nyuv2
coco2014: ../datasets/coco
mlhypersim: ../datasets/mlhypersim
scannet: ../datasets/scannet
scannet_frames_25k: ../datasets/scannet_frames_25k
cocostuff164k: ../datasets/cocostuff164k
 
# Pseudo Labels Path
labels_generic: ../datasets/
```
 
The correct env file is chosen based on the global variable ```ENV_WORKSTATION_NAME```.
```bash
export ENV_WORKSTATION_NAME="""your_env_name"""
```
If available set your NeptuneAPI Token to log all the experiments to NeptuneAI:
```bash
export NEPTUNE_API_TOKEN="""YOUR_NEPTUNE_KEY"""
```
With this you have setup the configuration. 
How to download and install the datasets is explained in [Dataset Preperation](Dataset-Preperation).
 
### Running Experiments
 
Running a full CL-scenario:
```bash
cd ~/ASL && python supervisor.py --exp=cfg/exp/MA/scannet/25k_pretrain/pretrain.py
```
```exp``` arguments provides the correct experiment configuration path.
 
#### Experiment Configuration Explained
 
```yaml
name: test/run1 #Name of the experiment folders
neptne_project_name: YourName/YourProject #NeptuneAI project name  
offline_mode: False # False uses tensorboard, True uses NeptuneAI
TODO
```
#### Overview Provided Experiments
**Hypersim**
1. basic: Standard Memory Buffer
2. memory_size: Evaluate different Memory Sizes
3. sgd_setting: Training Setting Search
 
 

**ScanNet**
1. basic: Standard Memory Buffer
2. buffer_filling: Different Buffer Filling Strategies
3. buffer_size: Buffer Size
4. latent_replay: Latent Replay Experiment
5. no_replay: Catastrophic Forgetting
6. replay_ratios: Replay Ratios
7. single_task: Train on all datasets simultaneously
8. student_teacher: Experiments with soft and hard teacher for replay
 
## Supervision Signal Generation

How to run

### CRF & SLIC
TODO
 
### Optical Flow
To create the optical flow we used [RAFT](https://github.com/princeton-vl/RAFT)  
Download the pre-trained models.  
You can create the optical flow for the ScanNet dataset using the jupyter Notebook **notebook/optical_flow/**  
You need to make sure to add the correct paths to import the RAFT model and the dataset.  
The notebook **Report/pseudo_labels_flow.ipynb** creates all reported plots and figures.  

### 3D Mapping
Result:  
![](https://github.com/JonasFrey96/ASL/tree/main/docs/create_map_gt_rviz.gif)

## Dataset Preparation
### Hypersim
Download the Hypersim Dataset and extract the files following the [author's instructions](https://github.com/apple/ml-hypersim)  
We additionally provide a .tar file with the first 10 scenes.
 
Folder structure:
```
mlhypersim
  ai_XXX_XXX
    images
      scene
        scene_cam_00_final_hdf5
          frame.0000.color.hdf5
          ...
          frame.XXXX.color.hdf5
        scene_cam_00_geometry_hdf5
          frame.0000.semantic.hdf5
          ...
          frame.XXX.semantic.hdf5
  ...
  ...
```		
### ScanNet
Download the ScanNet dataset and follow the [author's instructions](http://www.scan-net.org/) to extract the .sens files into individual files.  
Also, make sure to download the 25k files if you want to perform the pre-training.  
Here it's important to delete samples from the first 10 scenes to avoid overlapping continual learning and pre-training tasks.  
We additionally provide a .tar file with the extracted first 10 scenes.  
 
Folder structure:
```
scannet
  scannet_frames_25k
    scene0010_00
      color
        000000.jpg
        ...
        XXXXXX.jpg
      depth
        000000.png
        ...
        XXXXXX.png
      label
        000000.png
        ...
        XXXXXX.png
      pose
        000000.txt
        ...
        XXXXXX.txt
      intriniscs_color.txt
      intrinsics_depth.txt
    ...
    ...
    scene0706_00
      ...
  scans
    scene0000_00
      color
        000000.jpg
        ...
        XXXXXX.jpg
      depth
        000000.png
        ...
        XXXXXX.png
      label-filt
        000000.png
        ...
        XXXXXX.png
      pose
        000000.txt
        ...
        XXXXXX.txt
      intrinsics
        intriniscs_color.txt
        intrinsics_depth.txt
    ...
    scene0009_02
    ...

```
### COCO2014
This dataset is only used for pre-training.
Download the MS COCO 2014 dataset https://cocodataset.org/#home.
 
Folder structure:
```
coco
  train2014
    COCO_train2014_000000000009.jpg
    ...
    COCO_train2014_000000581921.jpg
  val2014
    COCO_val2014_000000000042.jpg
    ...
    COCO_val2014_000000581929.jpg
  annotations
    instances_train2014.json
    instances_val2014.json
```
### COCO164k
This dataset is only used for pre-training.
Download the MS COCO 2017 dataset https://cocodataset.org/#home.
 
Folder structure:
```
cocostuff164k
  images
    train2017
      000000000009.jpg
      ...
      000000581929.jpg
    val2017
      000000000139.jpg
      ...
      000000581781.jpg
  annotations
    instances_train2014.json
    instances_val2014.json
```

# 2. Acknowledgment 

- The authors of [Fast-SCNN](https://arxiv.org/pdf/1902.04502.pdf)  
- TRAMAC implementing [Fast-SCNN in PyTorch](https://github.com/Tramac/Fast-SCNN-pytorch)  
- The authors of [RAFT](https://github.com/princeton-vl/RAFT)  
-  The authors of [ORBSLAM2](https://github.com/appliedAI-Initiative/orb_slam_2_ros)  
- People at <http://continualai.org> for the inspiration

