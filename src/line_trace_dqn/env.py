#! /usr/bin/env python

import rospy
import ros_numpy
import os
import time
import numpy as np
import cv2
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

WAFFLE_MAX_LIN_VEL = 0.26 # can be 1.0?
WAFFLE_MAX_ANG_VEL = 1.82

STATE_SIZE = (64, 36) # image_raw (1920 : 1080)= (16 : 9)

class Env():

    def __init__(self, action_size=5, max_angular_vel=1.5):
        self.action_size = action_size
        self.max_angular_vel = max_angular_vel

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        # self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_image = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # for debug: only one of them can be True
        self.show_camera_image = False
        self.show_state_image = True
        self.show_center_point = False


    def reset(self):
        """reset gazebo simulator

        Returns:
            np.ndarray: state vector
        """
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        state = self.get_state()
        return np.asarray(state)


    def image_callback(self, img_msg):
        """callback function for image_raw

        Args:
            img_msg (sensor_msgs.msg.Image): ros msg
        """
        image = ros_numpy.numpify(img_msg) # instead of cv_bridge
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # for ros_numpy
        self.image_raw = image

        if self.show_camera_image:
            h, w = image.shape[:2]
            RESIZE = (w//3, h//3)
            display_image = cv2.resize(image, RESIZE)
            cv2.imshow('BGR Image', display_image)
            cv2.waitKey(1)


    def get_state(self):
        """gettter for state (state is image itself)

        Returns:
            cv2.image: subscribed cv image
        """
        state = self.image_raw
        return state


    def step(self, action):
        """environment step

        Args:
            action (int): action number (0 <= action < self.action_size)

        Returns:
            np.ndarray, float, bool: state-vector, reward-value, done(failure)
        """
        ang_vel = ((self.action_size - 1)/2 - action) * self.max_angular_vel * 0.5 # 0->action_size: max_angular_vel->(-max_angular_vel)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        state = self.get_state()
        reward, done = self.reward_function(state, action)
        state = cv2.resize(self.image_raw, STATE_SIZE) # resize for agent network

        # for debug
        if self.show_state_image:
            cv2.imshow('State Image', state)
            cv2.waitKey(1)

        return np.asarray(state), reward, done


    def get_distance_from_center(self, img):
        """calculate the absolute distance the center line (0: center, 1: edge)

        Args:
            img (cv2.image): camera image := state

        Returns:
            float, bool: absolute distance [0,1], no line detected
        """
        distance_fron_center = 0.0
        done = False

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([10, 10, 10])       # yellow threshold（lower）
        upper_yellow = np.array([255, 255, 250])    # yellow threshold（upper）
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  # binary mask image based on the threshold

        h, w = img.shape[:2]
        search_top = (h//4)*3
        search_bot = search_top + 20    # only focus on the 20 lines in front
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0

        M = cv2.moments(mask)    # center point of mask image
        if M['m00'] > 0:    # if center point exists
            cx = int(M['m10']/M['m00']) # x coordinate of the center point
            cy = int(M['m01']/M['m00']) # y coordinate of the center point
            cv2.circle(img, (cx, cy), 20, (0, 0, 255), -1) # red circle on image
            distance_fron_center = abs((cx / w - 0.5) / 0.5)
        else:
            distance_fron_center = float('Inf')
            done = True

        if self.show_center_point:
            RESIZE = (w//3, h//3)
            display_image = cv2.resize(img, RESIZE)
            cv2.imshow('Center point', display_image)
            cv2.waitKey(1)
        return distance_fron_center, done


    def reward_function(self, state, action):
        """reward function: r_t = R(s_t, a_t, s_{t+1})

        Args:
            state (cv2.image): state image
            action (int): action number

        Returns:
            float, bool: reward, done(failure)
        """

        distance_from_center, done = self.get_distance_from_center(state)
        # TODO: reward
        
        print(distance_from_center, done)

        return distance_from_center, done




if __name__ == "__main__":
    rospy.init_node('env')
    env = Env(action_size=5)
    time.sleep(0.5)
    try:
        while True:
            time.sleep(0.1)
            env.step(3)
    except KeyboardInterrupt:
        print('stop!')
        rospy.loginfo('Stop Turtlebot')
        env.cmd_vel_pub.publish(Twist())