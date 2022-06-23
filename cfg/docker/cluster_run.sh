#!/bin/bash

echo "Execute cluster_run.sh"
echo $@
module load gcc/6.3.0 cuda/11.3.1

mkdir -p $TMPDIR/scannet/scans
mkdir -p $TMPDIR/scannet/scannet_frames_25k
cp /cluster/work/rsl/$USER/ucdr/datasets/scannetv2-labels.combined.tsv $TMPDIR/scannet

echo "Start copying datasets"
tar -xf /cluster/work/rsl/$USER/ucdr/datasets/scannet.tar -C $TMPDIR/scannet/scans
tar -xf /cluster/work/rsl/$USER/ucdr/datasets/scannet_25k.tar -C $TMPDIR/scannet/scannet_frames_25k
tar -xf /cluster/work/rsl/$USER/ucdr/datasets/labels_individual_scenes_map_2.tar -C $TMPDIR/scannet/scans
tar -xf /cluster/work/rsl/$USER/ucdr/containers/ucdr.tar -C $TMPDIR
echo "Finished copying datasets"

# singularity shell --env NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN -B $TMPDIR:/home/tmpdir -B $WORK/ucdr:/home/work -B $HOME:/home/git --nv --writable --containall $TMPDIR/ucdr.sif bash
singularity exec --env NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN -B $TMPDIR:/home/tmpdir -B $WORK/ucdr:/home/work -B $HOME:/home/git --nv --writable --containall $TMPDIR/ucdr.sif /home/git/ASL/cfg/docker/container_run.sh $@
echo "Execute cluster_run.sh done"
bkill $LSB_JOBID
exit 0