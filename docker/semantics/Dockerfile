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

RUN echo 'source /opt/ros/melodic/setup.bash' >> ~/.bashrc
RUN mkdir /Datasets
RUN apt install python-rosdep
RUN apt-get update
RUN apt install wget
RUN apt-get install -y wget 
RUN apt-get -qq -y install curl
RUN rosdep update
RUN apt install -y tmux
RUN apt install -y wget
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'

RUN wget http://packages.ros.org/ros.key -O - | apt-key add - 
RUN apt-get update 
RUN apt-get install -y python-catkin-tools
RUN mkdir -p /home/catkin_ws/src

RUN source /opt/ros/melodic/setup.bash && cd /home/catkin_ws/  && catkin init && catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release && catkin config --merge-devel
RUN echo 'source /home/catkin_ws/devel/setup.bash' >> ~/.bashrc
RUN echo "2ts"
RUN cd /home/catkin_ws/src && git clone https://github.com/JonasFrey96/Kimera-Semantics.git && wstool init && wstool merge Kimera-Semantics/install/kimera_semantics_https.rosinstall && wstool update && rosdep install --from-paths . --ignore-src -r -y 
# && rm -r -f glog_catkin && git clone https://github.com/tu-darmstadt-ros-pkg/glog_catkin.git

# RUN cd /home/catkin_ws/ && source /opt/ros/melodic/setup.bash && catkin build -j8
# RUN cd /home/catkin_ws/src && chmod +x Kimera-Semantics/kimera_semantics_ros/include/proto/gen_proto.sh && ./Kimera-Semantics/kimera_semantics_ros/include/proto/gen_proto.sh

