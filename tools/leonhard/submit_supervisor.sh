module load python_gpu/3.7.4 gcc/6.3.0 eth_proxy
source ~/.bashrc
cd $HOME/ASL
/cluster/home/jonfrey/miniconda3/envs/track4/bin/python supervisor.py $@
