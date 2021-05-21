import os 
from pathlib import Path

p = "/home/jonfrey/datasets/scannet/scans"
scenes = [x for x in next(os.walk(p))[1]]
scenes.sort()
scenes = ["scene0000_00","scene0000_01","scene0000_02","scene0001_00", "scene0001_01"]
for j, s in enumerate( scenes[:5] ):
	# s = s.replace("scene", "scan")
	print(j,s)
	if j % 5 == 0 and j != 0:
		#os.system( f"""python2.7 /home/jonfrey/ASL/cfg/dataset/scannet/download.py --out_dir="/media/scratch1/jonfrey" --id={s} --type=_2d-instance-filt.zip """ )
		os.system( f"""python2.7 /home/jonfrey/ScanNet/SensReader/python/reader.py --filename=/media/scratch1/jonfrey/scans/{s}/{s}.sens --output_path=/media/scratch1/jonfrey/dataset/scannet/scans/{s} --export_color_images --export_depth_images --export_poses --export_intrinsics """)
	else:
		#os.system( f"""python2.7 /home/jonfrey/ASL/cfg/dataset/scannet/download.py --out_dir="/media/scratch1/jonfrey" --id={s} --type=_2d-instance-filt.zip & """ )
		os.system( f"""python2.7 /home/jonfrey/ScanNet/SensReader/python/reader.py --filename=/media/scratch1/jonfrey/scans/{s}/{s}.sens --output_path=/media/scratch1/jonfrey/scannet/scans/{s} --export_color_images --export_depth_images --export_poses --export_intrinsics &""")
	

# python2.7 /home/jonfrey/ScanNet/SensReader/python/reader.py --filename=/media/scratch1/jonfrey/old/scans/scene0000_01/scene0000_01.sens --output_path=/media/scratch1/jonfrey/dataset/scannet/scans/scene0000_01 --export_depth_images 
# 
# --export_color_images --export_depth_images --export_poses --export_intrinsics


python2.7 /home/jonfrey/ScanNet/SensReader/python/reader.py --filename=/media/scratch1/jonfrey/old/scans/scene0000_02/scene0000_02.sens --output_path=/media/scratch1/jonfrey/dataset/scannet/scans/scene0000_02 --export_depth_images 

python2.7 /home/jonfrey/ScanNet/SensReader/python/reader.py --filename=/media/scratch1/jonfrey/old/scans/scene0001_00/scene0001_00.sens --output_path=/media/scratch1/jonfrey/dataset/scannet/scans/scene0001_00 --export_depth_images 


scene0001_01 label-filt and detectron2 
scene0002_00 label-filt and detectron2  color
scene0002_01 depth

python2.7 /home/jonfrey/ScanNet/SensReader/python/reader.py --filename=/media/scratch1/jonfrey/old/scans/scene0001_01/scene0000_01.sens --output_path=/media/scratch1/jonfrey/dataset/scannet/scans/scene0001_01 --export_color_images --export_poses --export_intrinsics