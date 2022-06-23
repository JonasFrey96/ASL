FROM pytorchlightning/pytorch_lightning:latest-py3.7-torch1.10
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

LABEL maintainer="Jonas Frey" mail="jonfrey@ethz.ch"

SHELL ["/bin/bash", "--login", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_frontend noninteractive

RUN mkdir /home/git
RUN mkdir /home/results
RUN mkdir /home/code
RUN mkdir /home/dataset
RUN mkdir /home/work
RUN mkdir /home/tmpdir

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install imageio
RUN pip3 install neptune-client
RUN pip3 install wandb
RUN pip3 install opencv-python==4.2.0.34
RUN pip install scikit-image

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN pip install torch-geometric