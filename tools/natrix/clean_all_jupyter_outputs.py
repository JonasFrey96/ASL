import os
from pathlib import Path
import time

notebooks = [str(p) for p in Path("/home/jonfrey/ASL").rglob('*.ipynb') ] 

notebooks.sort( key = lambda x: Path(x).stat().st_mtime )
print(notebooks)

for n in notebooks:
  cmd = "jupyter nbconvert --clear-output --inplace " + n
  res = os.system(cmd)
  time.sleep(0.05)
  print(res)