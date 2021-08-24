roslaunch orb_slam2_ros orb_slam2_rosbag.launch   
rosbag play -l -r 0.25 /Datasets/labdata/scenes_slow/_2021-07-07-10-56-59.bag

rviz -d ~/catkin_ws/src/Kimera-Interfacer/kimera_interfacer/rviz/labdata.rviz 

python3 ros/labdata_node.py
python3 ros/fastscnn_node.py
roslaunch kimera_interfacer labdata.launch
rviz -d ~/catkin_ws/src/Kimera-Interfacer/kimera_interfacer/rviz/labdata.rviz
