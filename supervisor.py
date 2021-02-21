import argparse
import os
import yaml

def load_yaml(path):
  with open(path) as file:  
    res = yaml.load(file, Loader=yaml.FullLoader) 
  return res

def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()    
  parser.add_argument('--exp', type=file_path, default='/home/jonfrey/ASL/cfg/exp/2/dist_match.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--env', type=file_path, default='cfg/env/env.yml',
                      help='The environment yaml file.')
  args = parser.parse_args()
  exp = load_yaml(args.exp)
  env = load_yaml(args.env)
  if exp['max_tasks'] > exp['task_generator']['total_tasks']:
    print('Max Tasks largen then total tasks -> Setting max_tasks to total_tasks')
    exp['max_tasks'] = exp['task_generator']['total_tasks']
  
  for i in range( exp['max_tasks'] ):
    init = int(bool(i==0))
    close = int(bool(i==exp['task_generator']['total_tasks']))
    cmd = 'cd $HOME/ASL && $HOME/miniconda3/envs/track3/bin/python main.py' 
    cmd += f' --exp={args.exp} --env={args.env} --init={init} --task_nr={i} --close={close}'
    print(cmd)
    os.system(cmd)
  