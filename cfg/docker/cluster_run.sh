#!/bin/bash

echo "Execute cluster_run.sh"
echo $@
module load gcc/6.3.0 cuda/11.4.2
tar -xf /cluster/work/rsl/jonfrey/ucdr/containers/ucdr.tar -C $TMPDIR

singularity exec -B $WORK/ucdr:/home/work -B $HOME/ucdr:/home/git --nv --writable --containall $TMPDIR/ucdr.sif /home/git/ASL/cfg/docker/container_run.sh $@
echo "Execute cluster_run.sh done"
bkill $LSB_JOBID
exit 0