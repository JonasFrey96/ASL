XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run -it \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="/home/jonfrey/ASL/docker/vio/launch:/home/catkin_ws/src/Kimera-VIO-ROS/launch" \
    --volume="/home/jonfrey/ASL/docker/vio/params:/home/catkin_ws/src/Kimera-VIO/params" \
    --volume="/home/jonfrey/ASL/docker/vio/scripts:/home/catkin_ws/src/Kimera-VIO-ROS/scripts" \
    --volume="$XAUTH:$XAUTH" \
    --runtime=nvidia \
    -v /home/jonfrey/Datasets/:/Datasets  \
    --privileged \
    --net=host \
    --name="vio_tmp" \
    vio_new \
    tmux