import os
import sys
import yaml
from pathlib import Path

ASL = os.path.join( str(Path.home()), "ASL" )
src = os.path.join( str(Path.home()), "ASL", "src" )
sys.path.append( ASL )
sys.path.append( src )

from lightning import Network
from utils_asl import load_yaml
name = os.getenv('ENV_WORKSTATION_NAME')
env_cfg_path =os.path.join( ASL, f"cfg/env/{name}.yml")  
exp_cfg_path =os.path.join( ASL, "cfg/exp/debug.yml")

env = load_yaml(env_cfg_path)
exp = load_yaml(exp_cfg_path)

model = Network(exp=exp, env=env)
