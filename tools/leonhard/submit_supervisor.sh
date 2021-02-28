module load python_gpu/3.7.4 gcc/6.3.0 eth_proxy cuda/11.0.3 cudnn/8.0.5 StdEnv jpeg/9b libpng/1.6.27 nccl/2.4.8-1

source ~/.bashrc
cd $HOME/ASL
module load eth_proxy
/cluster/home/jonfrey/miniconda3/envs/track4/bin/python supervisor.py $@
