# Continual Learning of Semantic Segmentation **U**sing **C**omplementary 2D-3D **D**ata **R**epresentations

# Paper

Jonas Frey, Hermann Blum, Francesco Milano, Roland Siegwart, Cesar Cadena, **Continual Learning of Semantic Segmentation using Complementary 2D-3D Data Representations**”, in *IEEE Robotics and Automation Letters(RA-L)*, 2022.

```latex
@inproceedings{frey2022traversability,
  author={Jonas Frey and Hermann Blum and Francesco Milano and Roland Siegwart and Cesar Cadena},
  journal={under review: IEEE Robotics and Automation Letters(RA-L},
  title={Continual Learning of Semantic Segmentation using Complementary 2D-3D Data Representations},
  year={2022}
}
```

# Installation


# Preperation
# Create Conda Environment
```
conda env create -f cfg/conda/ucdr.yml
```


# Create Docker Container
```
cd cfg/docker && ./build.sh
mkdir exports && cd exports && SINGULARITY_NOHTTPS=1 singularity build --sandbox ucdr.sif docker-daemon://ucdr:latest
sudo tar -cvf ucdr.tar ucdr.sif
scp ucdr.tar username@euler:/cluster/work/rsl/username/ucdr/containers
```

# Dataset Tar Files
Move to the directory that contains the individual scenes (scene0000_00, ...)
Tar it without compression:
```
tar -cvf scannet.tar ./
```

Repeat this for the folder containing 
- scannet25k (without scene 0-10).
- scannet (scene 0-10 only with all images subsampled correctly).
- generated labels directory (scene 0-10). 


# Acknowledgment 
- The authors of [Fast-SCNN](https://arxiv.org/pdf/1902.04502.pdf)  
- TRAMAC implementing [Fast-SCNN in PyTorch](https://github.com/Tramac/Fast-SCNN-pytorch)   
-  The authors of [ORBSLAM2](https://github.com/appliedAI-Initiative/orb_slam_2_ros)  
- People at <http://continualai.org> for the inspiration 