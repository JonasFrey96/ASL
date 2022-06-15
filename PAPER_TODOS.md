1. We can reproduce generalization measurments

# Generalization IoU
Pre-Trained Model: scannet25k_24h_lr_decay_from_scratch/2021-06-05T14:36:26_scannet25k_24h_lr_decay_from_scratch/task0-epoch=64--step=158340.ckpt

# ONLY fine 2 Fine-tune and 2 Pred model


1-Pred (Check) 
1-Pseudo (No Clue where) 
/media/jonfrey/Fieldtrip1/Jonas_Frey_Master_Thesis/labels_generated/labels_individual_scenes_map_2/scans

2-Finetune (should be good) 
2-Pred (should be good CL)


First iterations models:
(Test here which timestep is used the paper or paper2 models!)

Fine-Tuning:
/home/jonfrey/Results/paper/paper/xxx
/home/jonfrey/Results/paper2/paper2/xxx

Paper-Result:
/home/jonfrey/Results/paper/paper/2021-09-13T20:35:09_s2_iteration2_r02
/home/jonfrey/Results/paper2/paper2/2021-09-14T09:20:24_s2_iteration2_r02

It may be also this one here: 
/home/jonfrey/Results/individual_scenes/2021-09-07T08:12:20_labels_iter2_5cm_replay02_e20_s2
/home/jonfrey/Results/individual_scenes/2021-09-06T22:58:11_labels_iter2_5cm_replay02_s2

1-Pred (Pretrained) 1-Pseudo (labels_individual_scenes_map_2 ?)  2-Pred (One of the above)  2-Pseudo 3-Pred 3-Pseudo

(2-Pseudo):
/media/jonfrey/Fieldtrip1/Jonas_Frey_Master_Thesis/labels_generated/labels_iteration_2_individual_replay02_reprojected

(3-Pred):
/media/jonfrey/Fieldtrip1/Jonas_Frey_Master_Thesis/labels_generated/labels_iteration_3_individual_replay02

(3-Pseudo):
/media/jonfrey/Fieldtrip1/Jonas_Frey_Master_Thesis/labels_generated/labels_iteration_3_individual_replay02_reprojected