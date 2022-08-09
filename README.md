# Line Trace DQN

Line trace robot simulation with reinforcement learning, Deep Q-Network.

The final report of "Intelligent Informatics", Creative Informatics, the University of Tokyo.


<img width="550" src="https://user-images.githubusercontent.com/52503908/183637813-aee73986-b706-45ba-9455-006092f64958.gif">

## Requirements

- Ubuntu 18.04
- ROS Melodic
- Python 3.6.x (must be installed on the system like `/usr/bin/python3`)


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
```

Build packages.

```bash  
$ cd ..  
$ rosdep install -y -r --from-paths src --ignore-src  
$ source /opt/ros/${ROS_DISTRO}/setup.bash  
$ catkin build line_trace_dqn  
$ source ./devel/setup.bash
```

## Usage

### 1. Launch Gazebo simulator

```bash  
$ roslaunch line_trace_dqn turtlebot3_linetrace.launch  
```


### (ex. Python sample scripts)

```bash  
$ roslaunch line_trace_dqn test_rospy.launch script:=follower  
```


### 2. Train DQN

```bash  
$ roslaunch line_trace_dqn result_graph.launch # realtime score/action plot  
$ # roslaunch line_trace_dqn turtlebot3_dqn_train.launch --ros-args # see the parameter descriptions  
$ roslaunch line_trace_dqn turtlebot3_dqn_train.launch lr:=0.1  
```


### 3. Plot learning results

```bash  
$ rosrun line_trace_dqn log_plotter.py  
```


## References

### Gazebo setup
- [ROSの勉強　第24弾：ライントレース（OpenCV）](https://qiita.com/Yuya-Shimizu/items/5c408fb06878471ad486)
- [ROS講座37 gazebo worldを作成する](https://qiita.com/srs/items/9b23ad12bea9e3ec0480)
  - [How to include uri relatively?](https://answers.gazebosim.org//question/16159/how-to-include-uri-relatively/)
  - [Gazebo UR [Err] [REST.cc:205] Error in REST request](https://qiita.com/hoshianaaa/items/4ec14775ad11cefccff3)

### Gazebo Tips
- [How to reset the simulation ?](https://answers.gazebosim.org//question/8801/how-to-reset-the-simulation/)

### Python3
- ~~[ROSでPython3を使う方法](https://qiita.com/tnjz3/items/4d64fc2d36b75e604ab1)~~
- ~~[Python3のvirtualenvでrospyを使う](https://qiita.com/otamasan/items/7ac7732a5c3d47ec3028)~~
- [catkin_virtualenv](https://github.com/locusrobotics/catkin_virtualenv)
- [Installing Python scripts and modules](http://docs.ros.org/en/jade/api/catkin/html/howto/format2/installing_python.html)

### Sample

- [ROSの勉強　第12弾：センシングと移動](https://qiita.com/Yuya-Shimizu/items/66dd6fa254957ca773e9)
- [PythonでTurtlebot3を動かしてみた(gazeboを使って)](https://zenn.dev/kmiura55/articles/ros-turtlesim3-wander)


### OpenCV

- [ROSの勉強　第45弾：ROS (melodic)においてpython3系でcv_bridgeが使えないことへの解決](https://qiita.com/Yuya-Shimizu/items/ba73c9959067fa94a7c5)


### DQN
- [DQN（Deep Q Network）を理解したので、Gopherくんの図を使って説明](https://qiita.com/ishizakiiii/items/5eff79b59bce74fdca0d)
- [ゼロからDeepまで学ぶ強化学習](https://qiita.com/icoxfog417/items/242439ecd1a477ece312)
- [Pytorch, REINFORCEMENT LEARNING (DQN) TUTORIAL](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [ROBOTIS e-Manual, 9. Machine Learning](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/)
- [ライントレーサーをDeep Q Learningで教育する - Chainer](https://qiita.com/chachay/items/555638e3079fce9d59c9)
- [自前の環境で深層強化学習](https://www.scsk.jp/product/oss/tec_guide/chainer_rl/1_chainer_rl2_1.html)
