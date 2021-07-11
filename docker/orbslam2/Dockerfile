FROM osrf/ros:melodic-desktop-full

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

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
RUN source /opt/ros/melodic/setup.bash && rosdep fix-permissions && rosdep update

RUN apt-get install -y python-catkin-tools

# TMUX and VIM
RUN apt install -y tmux vim
RUN echo """# Make mouse useful in copy mode \
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

# PANGOLIN
RUN apt install -y vim
RUN apt install -y libgl1-mesa-dev
RUN apt install -y libglew-dev
RUN apt install -y cmake
RUN apt install -y python3-pip
# RUN python -mpip install numpy pyopengl Pillow pybind11
RUN apt install -y pkg-config
RUN apt install -y libegl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols

RUN cd /home && git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build .

# ORB-SLAM2
RUN cd /home && git clone https://github.com/JonasFrey96/ORB_SLAM2.git ORB_SLAM2
RUN cd /home/ORB_SLAM2 && chmod +x build.sh && ./build.sh

# ORB-ROS
RUN echo 'export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/ORB_SLAM2/Examples/ROS' >> ~/.bashrc
RUN apt-get install -y ros-melodic-libg2o ros-melodic-cv-bridge
RUN source /opt/ros/melodic/setup.sh && export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/ORB_SLAM2/Examples/ROS && cd /home/ORB_SLAM2 && chmod +x /home/ORB_SLAM2/build_ros.sh && ./build_ros.sh

RUN mkdir -p /home/catkin_ws/src
RUN echo 'source /home/catkin_ws/devel/setup.bash' >> ~/.bashrc
#RUN cd /home/catkin_ws && catkin init && catkin config --extend /opt/ros/melodic && \
#	catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release && \
#	catkin config --merge-devel && \
#	cd /home/catkin_ws/src/ && git clone https://github.com/ethz-asl/image_undistort && wstool update &&  catkin build ./

RUN chmod +x /home/ORB_SLAM2/Examples/RGB-D/orbslam_rosbag_adapter.py 
RUN apt install -y python-pip
RUN apt-get install -y ros-melodic-ros-numpy
