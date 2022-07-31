#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

import rospy
import rosparam
import os
import pyqtgraph as pg
import sys
import joblib
from std_msgs.msg import Float32MultiArray, Float32
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Result")
        self.setGeometry(50, 50, 600, 650)
        self.graph_sub = rospy.Subscriber('result', Float32MultiArray, self.data_callback)
        self.ep = []
        self.data = []
        self.rewards = []
        self.x = []
        self.count = 1
        self.size_ep = 0
        load_data = False

        if load_data:
            self.ep, self.data = self.load_data()
            self.size_ep = len(self.ep)
        self.plot()

    def __del__(self):
        self.save_data([self.ep, self.data])

    def data_callback(self, data):
        self.data.append(data.data[0])
        self.ep.append(self.size_ep + self.count)
        self.count += 1
        self.rewards.append(data.data[1])


    def plot(self):
        self.qValuePlt = pg.PlotWidget(self, title="Steps")
        self.qValuePlt.move(0, 320)
        self.qValuePlt.resize(600, 300)
        self.timer1 = pg.QtCore.QTimer()
        self.timer1.timeout.connect(self.update)
        self.timer1.start(200)

        self.rewardsPlt = pg.PlotWidget(self, title="Total reward")
        self.rewardsPlt.move(0, 10)
        self.rewardsPlt.resize(600, 300)

        self.timer2 = pg.QtCore.QTimer()
        self.timer2.timeout.connect(self.update)
        self.timer2.start(100)

        self.show()

    def update(self):
        self.rewardsPlt.showGrid(x=True, y=True)
        self.qValuePlt.showGrid(x=True, y=True)
        self.rewardsPlt.plot(self.ep, self.data, pen=(255, 0, 0))
        self.qValuePlt.plot(self.ep, self.rewards, pen=(0, 255, 0))

    def load_data(self):
        log_dir = rosparam.get_param("/turtlebot3_dqn_train/log_path")
        log_fname = os.path.join(log_dir, "result_graph.txt")
        try:
            x, y = joblib.load(log_fname)
        except:
            x, y = [], []
        return x, y

    def save_data(self, data):
        log_dir = rosparam.get_param("/turtlebot3_dqn_train/log_path")
        log_fname = os.path.join(log_dir, "result_graph.txt")
        joblib.dump(data, log_fname, compress=3)


def run():
        rospy.init_node('graph')
        app = QApplication(sys.argv)
        GUI = Window()
        sys.exit(app.exec_())

run()