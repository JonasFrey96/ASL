import os
from pathlib import Path


import os
d = '/home/jonfrey/datasets/scannet/scans'
subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
subdirs.sort()

folders = [s[-12:] for s in subdirs]
print(folders)

depth_base = "/home/"

j = 0
for s in folders:
	
	path = os.path.join('/home/jonfrey/datasets/scannet/scans', s, "depth/0.png")
	
	if not os.path.exists(path):
		print(path, " FAILED")
	else:
		print("worked")
	#target = os.path.join(d,s)

	# os.system(f"rm -r -f {path}")
	#os.system( f" mv {path} {target}")
	# j += 1
	# if s.find("83") != -1:
	# 	break