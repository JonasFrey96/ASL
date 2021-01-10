module load python_gpu/3.7.4 gcc/6.3.0
source ~/.bashrc
conda activate track3
cd $HOME/ASL
$HOME/miniconda3/envs/track3/bin/python main.py $@
