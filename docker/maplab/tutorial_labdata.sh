#!/usr/bin/env bash
# Script to run ROVIOLI from a Euroc rosbag data source.
# Usage: tutorial_euroc <output save folder> <rosbag path> [<additional rovioli flags>]

LOCALIZATION_MAP_OUTPUT="/root/save_folder"

ROVIO_CONFIG_DIR="/home/maplab_ws/src/maplab/applications/rovioli/scripts/tutorials/labdata"
NCAMERA_CALIBRATION="$ROVIO_CONFIG_DIR/ncamera-euroc.yaml"
IMU_PARAMETERS_MAPLAB="$ROVIO_CONFIG_DIR/imu-adis16488.yaml"
IMU_PARAMETERS_ROVIO="$ROVIO_CONFIG_DIR/imu-sigmas-rovio.yaml"

rosrun rovioli rovioli \
  --alsologtostderr=1 \
  --v=2 \
  --ncamera_calibration=$NCAMERA_CALIBRATION  \
  --imu_parameters_maplab=$IMU_PARAMETERS_MAPLAB \
  --imu_parameters_rovio=$IMU_PARAMETERS_ROVIO \
  --datasource_type="rosbag" \
  --save_map_folder="$LOCALIZATION_MAP_OUTPUT" \
  --optimize_map_to_localization_map=false \
  --map_builder_save_image_as_resources=false \
  --datasource_rosbag="/Datasets/labdata/2021-06-17-17-56-33.bag"
