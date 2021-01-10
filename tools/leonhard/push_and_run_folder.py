from pathlib import Path
import os
import argparse
import yaml
from os.path import expanduser
"""
For execution run:
python tools/leonhard/push_and_run_folder.py --exp=1 --time=00:30 --gpus=1 --mem=10240 --workers=20 --ram=60
python tools/leonhard/push_and_run_folder.py --exp=1 --time=00:30 --gpus=1 --mem=10240 --workers=20 --ram=60 --fake=True

python tools/leonhard/push_and_run_folder.py --exp=1 --time=4 --gpus=4 --workers=20 --ram=60
"""
parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='exp',  required=True,
                    help='Folder containing experiment yaml file.')
parser.add_argument('--time', default=24, required=True,
                    help='Runtime.')
parser.add_argument('--mem', default=10240, help='Min GPU Memory')
parser.add_argument('--gpus', default=1)
parser.add_argument('--workers', default=20)
parser.add_argument('--ram', default=60)
parser.add_argument('--env', default='cfg/env/leonhard.yml')
parser.add_argument('--scratch', default=0, help="Total Scratch space in GB")
parser.add_argument('--fake', default=False, help="Not schedule")

args = parser.parse_args()
w = int(args.workers)
gpus = int(args.gpus)
ram = int(int(args.ram)*1000/w)
env = args.env
print(f'\nAll jobs will be run for {args.time}h')
mem = args.mem
if args.time == '120':
  s1 = '119:59'
elif args.time == '24':
  s1 = '23:59'
elif args.time == '4':
  s1 = '3:59'
elif  isinstance( args.time, str):
  s1 = args.time
else:
  raise Exception
scratch = int( int(args.scratch) / w)
fake = args.fake
 
# Get all model_paths
home = expanduser("~")
p = f'{home}/ASL/cfg/exp/{args.exp}/'
exps = [str(p) for p in Path(p).rglob('*.yml')]
model_paths = []
print('\nFound Config Files in directory:')
for j,e in enumerate(exps):
  print('   ' + e)
  with open(e) as f:
    doc = yaml.load(f, Loader=yaml.FullLoader) 
  
  # Validate if config trainer settings fits with job.
  if gpus > 1 and doc['trainer']['accelerator'] != 'ddp':
    print('Error in exp {e}: Mutiple GPUs but not using ddp')
    exps.remove(e)
  elif doc['trainer']['gpus'] != gpus and doc['trainer']['gpus'] != -1:
    print('Error in exp {e}: Nr GPUS does not match job')
    exps.remove(e)
  else:
    model_paths.append( doc['name'] ) 
    
print('\n')
  
export_cmd = """export LSF_ENVDIR=/cluster/apps/lsf/conf; export LSF_SERVERDIR=/cluster/apps/lsf/10.1/linux3.10-glibc2.17-x86_64/etc;"""
bsub_cmd = """/cluster/apps/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bsub"""
  
with open(os.path.join( home, 'ASL', env )) as f:
  doc = yaml.load(f, Loader=yaml.FullLoader) 
  base = doc['base']
model_paths = [os.path.join(base,i) for i in model_paths] 

# Push to cluster 
cmd = f"""rsync -a --delete {home}/ASL/* jonfrey@login.leonhard.ethz.ch:/cluster/home/jonfrey/ASL"""
os.system(cmd)

# Executue commands on cluster
import paramiko
try:

  host = "login.leonhard.ethz.ch"
  port = 22
  username = "jonfrey"
  ssh = paramiko.SSHClient()
  ssh.load_system_host_keys()
  #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  ssh.connect(host, port, username)
  
  N = len(exps)
  print(f'Using bsub to schedule {N}-jobs:')
  for j, e in enumerate(exps):
    e = e.replace('/home/jonfrey/ASL/','')
    print('MP', model_paths)
    p = model_paths[j].split('/')
    p = '/'.join(p[:-1])

    #Remote make Path
    cmd = f'mkdir -p {p}'
    stdin, stdout, stderr = ssh.exec_command(cmd)

    name = model_paths[j].split('/')[-1] + str(j) + '.out'
    o = f""" -oo {p}/{name} """
    cmd = f"""{export_cmd} cd $HOME/ASL && {bsub_cmd}{o}-n {w} -W {s1} -R "rusage[mem={ram},ngpus_excl_p={gpus}]" -R "select[gpu_mtotal0>={mem}]" """ 
    if scratch > 0:
      cmd += f"""-R "rusage[scratch={scratch}]" """
    cmd += f"""./tools/leonhard/submit.sh --env={env} --exp={e}"""  
    
    cmd = cmd.replace('\n', '')
    print(f'   {j}-Command: {cmd}')
    
    if not fake:
      stdin, stdout, stderr = ssh.exec_command(cmd)
      #a = stdin.readlines()
      b = stdout.readlines()[0]
      c = stderr.readlines()
      print(f'   {j}-Results: {b}')
    else:
      print('   Fake Flag is set')
    #Remote schedule jobs
finally:  
  if ssh is not None:
    ssh.close()
    try:
      del ssh, stdin, stdout, stderr
    except:
      pass
