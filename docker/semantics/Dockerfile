# This is an auto generated Dockerfile for ros:perception
# generated from docker_images/create_ros_image.Dockerfile.em
# FROM ros:melodic-ros-base-bionic
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

RUN echo 'source /opt/ros/melodic/setup.bash' >> ~/.bashrc
RUN mkdir /Datasets
RUN apt install python-rosdep
RUN apt-get update
RUN apt install wget
RUN apt-get install -y wget 
RUN apt-get -qq -y install curl
RUN rosdep update
RUN apt install -y tmux
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
RUN apt-get update 
RUN apt install -y wget
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-get install -y vim 

RUN wget http://packages.ros.org/ros.key -O - | apt-key add - 
RUN apt-get update 
RUN apt-get install -y python-catkin-tools
RUN mkdir -p /home/catkin_ws/src

RUN source /opt/ros/melodic/setup.bash && cd /home/catkin_ws/  && catkin init && catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release && catkin config --merge-devel
RUN echo 'source /home/catkin_ws/devel/setup.bash' >> ~/.bashrc
RUN echo "2ts"

# INSTALL LATEST VERSION OF PROTBUF
RUN apt-get install -y autoconf automake libtool curl make g++ unzip 

# https://github.com/protocolbuffers/protobuf/blob/master/src/README.md
RUN cd /home && git clone https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && ./autogen.sh && ./configure && make && make install && ldconfig

# RUN gem install google-protobuf -v 3.17.3
RUN apt install -y gcc-8 g++-8
RUN rm /usr/bin/gcc
RUN rm /usr/bin/g++
RUN ln -s /usr/bin/gcc-8 /usr/bin/gcc
RUN ln -s /usr/bin/g++-8 /usr/bin/g++

# RUN cd /home && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-linux-x86_64.zip && unzip protoc-3.17.3-linux-x86_64.zip && ls && mv ./bin/* /usr/local/bin/ && mv ./include/* /usr/local/include/
RUN cd /home/catkin_ws/src && git clone -b melodic https://github.com/JonasFrey96/Kimera-Semantics.git && wstool init && wstool merge Kimera-Semantics/install/kimera_semantics_https.rosinstall && wstool update && rosdep install --from-paths . --ignore-src -r -y 

RUN protoc -I=/home/catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/include/proto/ --cpp_out=/home/catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/include/proto/ /home/catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/include/proto/semantic_map.proto
RUN protoc -I=/home/catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/include/proto/ --python_out=/home/catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/include/proto/ /home/catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/include/proto/semantic_map.proto
RUN protoc -I=/home/catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/include/proto/ --python_out=/home/catkin_ws/src/Kimera-Interfacer/kimera_interfacer/scripts /home/catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/include/proto/semantic_map.proto

RUN cd /home/catkin_ws/ && source /opt/ros/melodic/setup.bash && catkin build -j20

# RUN wget https://github.com/protocolbuffers/protobuf/releases/latest
# RUN cd protobuf && git submodule update --init --recursive ./autogen.sh && ./configure && make && make install && ldconfig

# && rm -r -f gcd 
# RUN cd /home/catkin_ws/ && source /opt/ros/melodic/setup.bash && catkin build -j8
# RUN cd /home/catkin_ws/src && chmod +x Kimera-Semantics/kimera_semantics_ros/include/proto/gen_proto.sh && ./Kimera-Semantics/kimera_semantics_ros/include/proto/gen_proto.sh