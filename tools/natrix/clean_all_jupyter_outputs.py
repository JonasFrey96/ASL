import os
from pathlib import Path

notebooks = [str(p) for p in Path("/home/jonfrey/ASL").rglob('*.ipynb') ] 
print(notebooks)

for n in notebooks:
  cmd = "jupyter nbconvert --clear-output --inplace " + n
  res = os.system(cmd)
  print(res)