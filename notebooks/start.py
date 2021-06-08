import os

total = 10
d = "/home/jonfrey/Datasets/labels_generated/labels_pretrain25k_correct_mapping_reprojected/scans"
scenes = [o for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]  

print(scenes)
for s in scenes:
  if s != "scene0000_00":
    for i in range(total):
      if i == total-1 :
	      os.system( f"cd /home/jonfrey/ASL/notebooks/ && /home/jonfrey/miniconda3/envs/track4/bin/python mask_rcnn_demo_scenes.py --number={i} --number_total={total} --scene={s} ")
      else: 
        os.system( f"cd /home/jonfrey/ASL/notebooks/ && /home/jonfrey/miniconda3/envs/track4/bin/python mask_rcnn_demo_scenes.py --number={i} --number_total={total} --scene={s} &")