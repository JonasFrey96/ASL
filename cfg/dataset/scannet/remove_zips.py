import os
from pathlib import Path

p = '/media/scratch2/jonfrey/datasets/scannet/scans'
labels = [str(p) for p in Path(p).rglob('*.sens') ]
labels.sort()
 
print(labels)

for j, l in enumerate( labels):
	print(j) #os.system
	pa = str( Path(l).parent) 
	os.system(f'rm {l}')