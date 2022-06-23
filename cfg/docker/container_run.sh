#!/bin/bash

echo "Execute cluster_run_container.sh"
h=`echo $@`
echo $h
export ENV_WORKSTATION_NAME=euler

cd /home/git/ASL
python -m pip install --root ./  ./
pip install torchmetrics==0.9.1 
export PYTHONPATH=$PYTHONPATH:/home/git/ASL

exec bash -c "python3 -u scripts/train.py $h"