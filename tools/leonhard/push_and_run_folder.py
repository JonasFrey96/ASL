from pathlib import Path
import os
import argparse
import yaml
from os.path import expanduser
import logging
import coloredlogs
coloredlogs.install()
logging.getLogger("paramiko").setLevel(logging.WARNING)
"""
For execution run:
python tools/leonhard/push_and_run_folder.py --exp=1 --time=00:30 --gpus=1 --mem=10240 --workers=20 --ram=60
python tools/leonhard/push_and_run_folder.py --exp=1 --time=00:30 --gpus=1 --mem=10240 --workers=20 --ram=60 --fake=True

python tools/leonhard/push_and_run_folder.py --exp=1 --time=4 --gpus=4 --workers=20 --ram=60

# FOR COCO:
python tools/leonhard/push_and_run_folder.py --exp=pretrain-coco --time=4 --gpus=4 --mem=10240 --workers=20 --ram=60 --scratch=100

# MLhypersim
python tools/leonhard/push_and_run_folder.py --exp=ml-hypersim --time=4 --gpus=4 --mem=10240 --workers=20 --ram=60 --scratch=300

# ScanNet
python tools/leonhard/push_and_run_folder.py --exp=scannet --time=4 --gpus=4 --mem=10240 --workers=16 --ram=60 --scratch=80

python tools/leonhard/push_and_run_folder.py --exp=scannet --time=4 --gpus=4 --mem=10240 --workers=20 --ram=60 --scratch=80
"""
parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='exp',  required=True,
                    help='Folder containing experiment yaml file.')
parser.add_argument('--time', default=24, required=True,
                    help='Runtime.')
parser.add_argument('--mem', default=10240, help='Min GPU Memory')
parser.add_argument('--gpus', default=1)
parser.add_argument('--workers', default=16)
parser.add_argument('--ram', default=60)
parser.add_argument('--env', default='cfg/env/leonhard.yml')
parser.add_argument('--scratch', default=0, help="Total Scratch space in GB")
parser.add_argument('--fake', default=False, help="Not schedule")
parser.add_argument('--ignore_workers', default=False, help="Ignore workers")
parser.add_argument('--script', default='supervisor', choices=['main', 'supervisor'], help="Select script to start")


args = parser.parse_args()
w = int(args.workers)
gpus = int(args.gpus)
ram = int(int(args.ram)*1000/w)
env = args.env
logging.info('#'*80)
logging.info(' '*25+f'All jobs will be run for {args.time}h')
logging.info('#'*80)
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
scratch = int( int(args.scratch)*1000 / w)
fake = args.fake
ign = args.ignore_workers 
# Get all model_paths
home = expanduser("~")
p = f'{home}/ASL/cfg/exp/{args.exp}/'
exps = [str(p) for p in Path(p).rglob('*.yml') if str(p).find('_tmp.yml') == -1]
model_paths = []
logging.info('')
logging.info('Found Config Files in directory:')
for j,e in enumerate(exps):
  logging.info('   ' + e)
  with open(e) as f:
    doc = yaml.load(f, Loader=yaml.FullLoader) 
  
  if not ign and doc['loader']['num_workers'] != w:
    logging.warning('   Error: Number of workers dosent align with requested cores!')
    logging.warning('   Error: Either set ignore_workers flag true or change config')
    exps.remove(e)
  # Validate if config trainer settings fits with job.
  elif gpus > 1 and doc['trainer']['accelerator'].find('ddp') == -1:
    logging.warning('   Error: Mutiple GPUs but not using ddp')
    exps.remove(e)
  elif doc['trainer']['gpus'] != gpus and doc['trainer']['gpus'] != -1:
    logging.warning(f'   Error: Nr GPUS does not match job')
    exps.remove(e)
  else:
    model_paths.append( doc['name'] ) 
    
logging.info('')

if len(model_paths) == 0:
  logging.info('Model Paths Empty!')
  
else:  
  export_cmd = """export LSF_ENVDIR=/cluster/apps/lsf/conf; export LSF_SERVERDIR=/cluster/apps/lsf/10.1/linux3.10-glibc2.17-x86_64/etc;"""
  bsub_cmd = """/cluster/apps/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bsub"""
    
  with open(os.path.join( home, 'ASL', env )) as f:
    doc = yaml.load(f, Loader=yaml.FullLoader) 
    base = doc['base']
  model_paths = [os.path.join(base,i) for i in model_paths] 

  # Push to cluster 
  cmd = f"""rsync -a --delete --exclude='.git/' --exclude='cfg/exp/tmp/*' {home}/ASL/* jonfrey@login.leonhard.ethz.ch:/cluster/home/jonfrey/ASL"""
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
    logging.info(f'Using bsub to schedule {N}-jobs:')
    for j, e in enumerate(exps):
      e = e.replace('/home/jonfrey/ASL/','')
      logging.info('   Model Path:'+ model_paths[j])
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
        
      if args.script == 'main':
        subscr = 'submit'
      elif args.script == 'supervisor':
        subscr = 'submit_supervisor'
          
      cmd += f"""./tools/leonhard/{subscr}.sh --env={env} --exp={e}"""  
      
      cmd = cmd.replace('\n', '')
      logging.info(f'   {j}-Command: {cmd}')
      
      if not fake:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        #a = stdin.readlines()
        b = stdout.readlines()[0]
        c = stderr.readlines()
        logging.info(f'   {j}-Results: {b}')
      else:
        logging.info('   Fake Flag is set')
      #Remote schedule jobs
  finally:  
    if ssh is not None:
      ssh.close()
      try:
        del ssh, stdin, stdout, stderr
      except:
        pass
