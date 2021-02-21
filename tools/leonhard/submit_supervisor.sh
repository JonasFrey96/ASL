module load python_gpu/3.7.4 gcc/6.3.0 eth_proxy
source ~/.bashrc
conda activate track3
cd $HOME/ASL
module load eth_proxy
$HOME/miniconda3/envs/track3/bin/python supervisor.py $@
