docker build ~/ASL/docker/vio -t vio
docker build ~/ASL/docker/rviz -t rviz

docker run --rm -it --network host -v /home/jonfrey/Datasets/:/Datasets --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  vio:latest bash


#!/bin/bash

# Allow X server connection
xhost +local:root
sudo docker run -it --rm \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    kimera_vio
# Disallow X server connection
#   --network="host"\
xhost -local:root

docker run --rm -it --network host -v /home/jonfrey/Datasets/:/Datasets --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.  --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 X11-unix:rw" vio:latest bash

docker run -it --rm --privileged --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 rviz /bin/bash


docker run -it --rm --privileged --net=host -v /home/jonfrey/Datasets/:/Datasets -env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 vio:latest tmux


docker run -it --rm --privileged --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix --gpus 2 rviz /bin/bash



docker run -it --rm \
--privileged \
--net=host \
--gpus=all \
--runtime=nvidia \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e DISPLAY \
-e QT_X11_NO_MITSHM=1 \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw rviz /bin/bash