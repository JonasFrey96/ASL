from pathlib import Path
scans = "/home/jonfrey/datasets/scannet/scans/"
res = Path( scans ).rglob("*label_detectron2*")
res = [str(i).replace("/home/jonfrey/datasets/", "") for i in res ]
import os
os.chdir("/home/jonfrey/datasets/")

cmd = "tar -cvf /home/jonfrey/datasets/scannet/label_v1_detectron.tar "
for i in range(len(res)):
	cmd += res[i] + " " 
print( cmd )
os.system(cmd)