module list &> /dev/null || source /cluster/apps/modules/init/bash
module purge
module load legacy new gcc/6.3.0 hdf5 eth_proxy

source ~/.bashrc
cd $HOME/ASL
/cluster/home/jonfrey/miniconda3/envs/track4/bin/python supervisor.py $@
