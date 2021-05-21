import os
from pathlib import Path

p = '/media/scratch2/jonfrey/datasets/scannet/scans'
labels = [str(p) for p in Path(p).rglob('*.zip') if int(str(p).split('/')[-1][5:9]) < 101 ]
labels.sort()
 
print(labels)

for j, l in enumerate( labels):
	print(j)
	pa = str( Path(l).parent) 
	os.system(f'cd {pa} && unzip -o {l}')