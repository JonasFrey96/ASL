# python ScanNet/SensReader/python/reader.py --filename=scannet_test/scans/scene0000_00/scene0000_00.sens --output_path=scannet_test/uncompressed/

import os

base = '/media/scratch2/jonfrey/datasets/scannet/scans'

scenes = [os.path.join(base,s) for s in os.listdir(base)]
scenes.sort()
scenes = scenes[:221]


scenes = [ 
"/media/scratch2/jonfrey/datasets/scannet/scans/scene0000_00",
"/media/scratch2/jonfrey/datasets/scannet/scans/scene0000_02",
"/media/scratch2/jonfrey/datasets/scannet/scans/scene0002_01",
"/media/scratch2/jonfrey/datasets/scannet/scans/scene0012_00",
"/media/scratch2/jonfrey/datasets/scannet/scans/scene0024_00",
"/media/scratch2/jonfrey/datasets/scannet/scans/scene0030_00",
"/media/scratch2/jonfrey/datasets/scannet/scans/scene0031_00"]
# extracting scene 0 - 100
j = 0
for b in scenes:
  fn = b+'/'+b.split('/')[-1]+'.sens'
  op = base+'/'+b.split('/')[-1]
  print(fn) 
  # if j % 15 == 0 and j != 0:
  os.system(f'/usr/bin/python /media/scratch2/jonfrey/datasets/scannet/ScanNet/SensReader/python/reader.py --filename=/media/scratch1/jonfrey/old/sensefiles/scene0002_01/scene0002_01.sens --output_path=/media/scratch1/jonfrey/old/sensefiles/scene0002_01/{op}')
  # else:
    # os.system(f'/usr/bin/python /media/scratch2/jonfrey/datasets/scannet/ScanNet/SensReader/python/reader.py --filename={fn} --output_path={op} &')
  # j += 1