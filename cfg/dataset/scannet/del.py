import os
from pathlib import Path

# out = [str(s) for s in Path("/home/jonfrey/datasets/scannet/scans").rglob("**/depth/*")  if str(s).find(".png") != -1]

# print(out)
# for o in out:
# 	os.remove(o)


import os

d = "/home/jonfrey/datasets/scannet/scans"
subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
subdirs.sort()

folders = [s[-12:] for s in subdirs]
print(folders)


depth_base = "/media/scratch1/jonfrey/old/scannet/scans"


dl_paths = [str(s) for s in Path(d).rglob("*.jpg") if str(s).find("label_detectron2") != -1]
print(dl_paths, len(dl_paths))

for p in dl_paths:
    os.system(f"rm {p}")
# for s in folders:
# 	path = os.path.join(depth_base, s, "depth/0.png")
# 	p2 = "/media/scratch1/jonfrey/old/sensefiles/" + s + f"/{s}.sens"

# 	if not os.path.exists(path):
# 		print(s , " does not exist")
# 		os.system( f"""python2.7 /home/jonfrey/ScanNet/SensReader/python/reader.py --filename={p2} --output_path=/media/scratch1/jonfrey/old/scannet/scans/{s} --export_depth_images""")


# 	if not os.path.exists(  p2 ):
# 		print("Download", s)
# 		os.system( f"""python2.7 /home/jonfrey/ASL/cfg/dataset/scannet/download.py --out_dir="/media/scratch1/jonfrey/old/sensefiles" --id={s} --type=.sens""" )

# 		print (s, " redownload")
