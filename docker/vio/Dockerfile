# This is an auto generated Dockerfile for ros:perception
# generated from docker_images/create_ros_image.Dockerfile.em
# FROM ros:melodic-ros-base-bionic
FROM osrf/ros:melodic-desktop-full

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

MAINTAINER Jonas Frey "jonfrey@ethz.ch"

SHELL ["/bin/bash", "--login", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
# To avoid tzdata asking for geographic location...
ENV DEBIAN_frontend noninteractive

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-perception=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*


RUN echo 'source /opt/ros/melodic/setup.bash' >> ~/.bashrc
RUN mkdir /Datasets
RUN apt install python-rosdep
RUN apt-get update
RUN apt install wget
RUN apt-get install -y wget 
RUN apt-get -y install curl
RUN rosdep update
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'

RUN wget http://packages.ros.org/ros.key -O - | apt-key add - 
RUN apt-get update 
RUN apt-get install -y python-catkin-tools
RUN mkdir -p /home/catkin_ws/src


RUN source /opt/ros/melodic/setup.bash && cd /home/catkin_ws/  && catkin init && catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DGTSAM_TANGENT_PREINTEGRATION=OFF && catkin config --merge-devel
RUN echo 'source /home/catkin_ws/devel/setup.bash' >> ~/.bashrc
RUN cd /home/catkin_ws/src && git clone https://github.com/JonasFrey96/Kimera-VIO-ROS.git && wstool init && wstool merge Kimera-VIO-ROS/install/kimera_vio_ros_https.rosinstall && wstool update && rosdep install --from-paths . --ignore-src -r -y && rm -r -f glog_catkin && git clone https://github.com/tu-darmstadt-ros-pkg/glog_catkin.git

RUN cd /home/catkin_ws/ && source /opt/ros/melodic/setup.bash && catkin build -j8

RUN apt install -y tmux
RUN sudo echo """# Make mouse useful in copy mode \
setw -g mode-mouse on \
# Allow mouse to select which pane to use \
set -g mouse-select-pane on \
# Allow xterm titles in terminal window, terminal scrolling with scrollbar, and setting overrides of C-Up, C-Down, C-Left, C-Right \
set -g terminal-overrides "xterm*:XT:smcup@:rmcup@:kUP5=\eOA:kDN5=\eOB:kLFT5=\eOD:kRIT5=\eOC" \
# Scroll Histor y\
set -g history-limit 30000 \
# Set ability to capture on start and restore on exit window data when running an application \
setw -g alternate-screen on \
# Lower escape timing from 500ms to 50ms for quicker response to scroll-buffer access. \
set -s escape-time 50""" >> /home/.tmux.conf 
