#!/bin/bash
echo "Execute cluster_bsub.sh"
echo "Args: $@"
h=`echo $@`

if [[ $ENV_WORKSTATION_NAME = "ge76" ]]
then
    rsync -r -v --exclude=.git/* --exclude=results/* --exclude=.neptune/* /home/jonfrey/git/ASL/ jonfrey@euler:/cluster/home/jonfrey/ASL
    ssh jonfrey@euler " source /cluster/home/jonfrey/.bashrc; bsub -n 24 -R singularity -R \"rusage[mem=2596,ngpus_excl_p=1]\" -W $TIME -o $OUTFILE_NAME -R \"select[gpu_mtotal0>=10000]\" -R \"rusage[scratch=1500]\" -R \"select[gpu_driver>=470]\" /cluster/home/jonfrey/ASL/cfg/docker/cluster_run.sh $h"
else
    TIME=24:00
    bsub -n 24 -R singularity -R "rusage[mem=1596,ngpus_excl_p=1]" -W $TIME -R "select[gpu_mtotal0>=10000]" -R "rusage[scratch=1500]" -R "select[gpu_driver>=470]" /cluster/home/jonfrey/ASL/cfg/docker/cluster_run.sh $h
fi