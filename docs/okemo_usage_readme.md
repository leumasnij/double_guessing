
### copy the repo into your ROS catkin workspace and catkin_make and source it 

### Start the Okemo hardware interface
  
        roslaunch ur_robotouch bringup_okemo.launch 

### Start External Control URCap in robot controller touchscreen

### Start the moveit planner for okemo

        roslaunch okemo_moveit_config move_group.launch 

### Start the rViz visualizer (optional)

        roslaunch robotouch_ur view_okemo.launch

### Add the table and wall
        
        roslaunch robotouch_ur table_wall.launch

### Start the gripper script

	      roslaunch wsg_50_driver wsg_50_tcp.launch
