# Line Trace DQN

Line trace robot simulation with reinforcement learning, Deep Q-Network.

The final report of "Intelligent Informatics", Creative Informatics, the University of Tokyo.


## Requirements

- Ubuntu 20.04
- ROS Melodic


## Installation

Clone packages.

```bash  
$ mkdir -p ~/catkin_ws/src  
$ cd ~/catkin_ws/src  
$ wstool init .  
$ wstool set --git line_trace_dqn git@github.com:ketaro-m/line_trace_dqn.git -y  
$ wstool update  
$ wstool merge -t . line_trace_dqn/.rosinstall  
$ wstool update  
$ rosdep install -y -r --from-paths . --ignore-src  
```

Build packages.

```bash  
$ cd ..  
$ source /opt/ros/${ROS_DISTRO}/setup.bash  
$ catkin build  
$ source ./devel/setup.bash
```

### Python virtualenv

Python 3.8.6

```bash  
$ roscd line_trace_dqn/scripts  
$ python3 --version  
Python 3.8.6  
$ python3 -m venv .venv  
$ source .venv/bin/activate  
(.venv) $ pip install --extra-index-url https://rospypi.github.io/simple rospy-all  
```

## Usage

### 1. Launch Gazebo simulator

```bash  
$ roslaunch line_trace_dqn turtlebot3_linetrace.launch  
```


### 2. Python sample scripts

```bash  
$ roscd line_trace_dqn/scripts  
$ source .venv/bin/activate  
(.venv) $ python test_rospy/wanderbot.py  
```


## References

### Gazebo setup
- [ROSの勉強　第24弾：ライントレース（OpenCV）](https://qiita.com/Yuya-Shimizu/items/5c408fb06878471ad486)
- [ROS講座37 gazebo worldを作成する](https://qiita.com/srs/items/9b23ad12bea9e3ec0480)
  - [How to include uri relatively?](https://answers.gazebosim.org//question/16159/how-to-include-uri-relatively/)
  - [Gazebo UR [Err] [REST.cc:205] Error in REST request](https://qiita.com/hoshianaaa/items/4ec14775ad11cefccff3)


### Python3
- [ROSでPython3を使う方法](https://qiita.com/tnjz3/items/4d64fc2d36b75e604ab1)
- [Python3のvirtualenvでrospyを使う](https://qiita.com/otamasan/items/7ac7732a5c3d47ec3028)

### Sample

- [ROSの勉強　第12弾：センシングと移動](https://qiita.com/Yuya-Shimizu/items/66dd6fa254957ca773e9)
- [PythonでTurtlebot3を動かしてみた(gazeboを使って)](https://zenn.dev/kmiura55/articles/ros-turtlesim3-wander)
