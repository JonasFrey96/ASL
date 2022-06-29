# Continual Adaptation of Semantic Segmentation **U**sing **C**omplementary 2D-3D **D**ata **R**epresentations
---
Code, Model Checkpoints, Generated Data, Documentation, Installation Instructions.
<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#citation">Citation</a> •
  <a href="#setup">Setup</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#credits">Credits</a>
</p>


<img align="left" height="300" src="https://github.com/JonasFrey96/continual_adaptation_ucdr/blob/main/docs/scene_0000_00_seg.png" alt="MultiView"> 
<img align="right"  height="300" src="https://github.com/JonasFrey96/continual_adaptation_ucdr/blob/main/docs/scene_0000_00_multiview.png" alt="MultiView"> 

# Citation

Jonas Frey, Hermann Blum, Francesco Milano, Roland Siegwart, Cesar Cadena, **Continual Learning of Semantic Segmentation using Complementary 2D-3D Data Representations**”, in *IEEE Robotics and Automation Letters(RA-L)*, 2022.

```latex
@inproceedings{frey2022traversability,
  author={Jonas Frey and Hermann Blum and Francesco Milano and Roland Siegwart and Cesar Cadena},
  journal={under review: IEEE Robotics and Automation Letters(RA-L},
  title={Continual Learning of Semantic Segmentation using Complementary 2D-3D Data Representations},
  year={2022}
}
```

# Setup

## Create Conda Environment
```
conda env create -f cfg/conda/ucdr.yaml
```

## Create Docker Container
```
cd cfg/docker && ./build.sh
mkdir exports && cd exports && SINGULARITY_NOHTTPS=1 singularity build --sandbox ucdr.sif docker-daemon://ucdr:latest
sudo tar -cvf ucdr.tar ucdr.sif
scp ucdr.tar username@euler:/cluster/work/rsl/username/ucdr/containers
```

## Dataset Tar Files
Move to the directory that contains the individual scenes (scene0000_00, ...)
Tar it without compression:
```
tar -cvf scannet.tar ./
```

Repeat this for the folder containing 
- scannet25k (without scene 0-10).
- scannet (scene 0-10 only with all images subsampled correctly).
- generated labels directory (scene 0-10). 







# Experiments Reproduce Results
Generate Score for 1-Pseudo Adap:
```
python scripts/eval_pseudo_labels.py --pseudo_label_idtf=labels_individual_scenes_map_2 --mode=val --scene=scene0000,scene0001,scene0002,scene0003,scene0004
```

Generate Score for 1-Pseudo-GT Adap:
```
python scripts/eval_pseudo_labels.py --pseudo_label_idtf=labels_gt_reprojected_3cm --mode=val --scene=scene0000,scene0001,scene0002,scene0003,scene0004
```

Generate Score for Continual Learning and Finetuning:
```
python scripts/eval_model.py --eval=eval_pred_1.yaml
python scripts/eval_model.py --eval=eval_pred_2_00.yaml
python scripts/eval_model.py --eval=eval_pred_2_02.yaml
python scripts/eval_model.py --eval=eval_pred_2_05.yaml
```


# Credits
- The authors of [Fast-SCNN](https://arxiv.org/pdf/1902.04502.pdf)  
- TRAMAC implementing [Fast-SCNN in PyTorch](https://github.com/Tramac/Fast-SCNN-pytorch)   
-  The authors of [ORBSLAM2](https://github.com/appliedAI-Initiative/orb_slam_2_ros)  
- People at <http://continualai.org> for the inspiration 