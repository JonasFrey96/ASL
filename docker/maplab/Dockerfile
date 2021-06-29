FROM osrf/ros:kinetic-desktop-full

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
    ros-kinetic-perception=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*
    
# install ros packages
RUN apt-get update

RUN echo 'source /opt/ros/kinetic/setup.bash' >> ~/.bashrc
RUN mkdir /Datasets


RUN echo 'export UBUNTU_VERSION=xenial' >> ~/.bashrc  #(Ubuntu 16.04: xenial, Ubuntu 14.04: trusty, Ubuntu 18.04: bionic)
RUN echo 'export ROS_VERSION=kinetic' >> ~/.bashrc  #(Ubuntu 16.04: kinetic, Ubuntu 14.04: indigo, Ubuntu 18.04: melodic)

RUN apt install autotools-dev ccache doxygen dh-autoreconf git liblapack-dev libblas-dev libgtest-dev libreadline-dev libssh2-1-dev pylint clang-format-3.8 python-autopep8 python-catkin-tools python-pip python-git python-setuptools python-termcolor python-wstool libatlas3-base --yes

RUN pip install requests


RUN apt install -y python-rosdep
RUN apt-get update
RUN apt install wget
RUN apt-get install -y wget 
RUN apt-get -qq -y install curl
RUN rosdep update
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'

RUN wget http://packages.ros.org/ros.key -O - | apt-key add - 
RUN apt-get update
RUN apt-get install -y python-catkin-tools
RUN mkdir -p /home/maplab_ws/src


RUN source /opt/ros/kinetic/setup.bash && cd /home/maplab_ws/  && catkin init &&  catkin config --merge-devel && catkin config --extend /opt/ros/kinetic && catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
RUN echo 'source /home/maplab_ws/devel/setup.bash' >> ~/.bashrc
RUN cd /home/maplab_ws/src && git clone https://github.com/ethz-asl/maplab.git --recursive
RUN cd /home/maplab_ws/src && git clone https://github.com/ethz-asl/maplab_dependencies --recursive
RUN cd /home/maplab_ws/src/maplab && ./tools/linter/init-git-hooks.py

RUN cd /home/maplab_ws && catkin build maplab

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
