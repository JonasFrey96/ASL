#!/bin/bash
echo "Execute cluster_bsub.sh"
echo "Args: $@"
h=`echo $@`

if ! [[ $ENV_WORKSTATION_NAME = "euler" ]]
then
    ssh $USER@euler " source /cluster/home/$USER/.bashrc; bsub -n 18 -R singularity -R \"rusage[mem=2596,ngpus_excl_p=1]\" -W $TIME -o $OUTFILE_NAME -R \"select[gpu_mtotal0>=10000]\" -R \"rusage[scratch=12000]\" -R \"select[gpu_driver>=470]\" /cluster/home/$USER/ASL/cfg/docker/cluster_run.sh $h"
else
    TIME=4:00
    bsub -n 18 -R singularity -R "rusage[mem=1596,ngpus_excl_p=1]" -W $TIME -R "select[gpu_mtotal0>=10000]" -R "rusage[scratch=12000]" -R "select[gpu_driver>=470]" /cluster/home/$USER/ASL/cfg/docker/cluster_run.sh $h
fi