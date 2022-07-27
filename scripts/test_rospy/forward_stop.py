#!/usr/bin/env python3

"""
PythonでTurtlebot3を動かしてみた(gazeboを使って)
https://zenn.dev/kmiura55/articles/ros-turtlesim3-wander
"""

import rospy
from rospy.exceptions import ROSInterruptException
from geometry_msgs.msg import Twist

def shutdown_turtlebot():
    rospy.loginfo('Stop Turtlebot')
    cmd_vel_pub.publish(Twist())
    rospy.sleep(1)

cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
rospy.init_node('red_light_green_light')
rospy.on_shutdown(shutdown_turtlebot)

red_light_twist = Twist()
green_light_twist = Twist()
green_light_twist.linear.x = 0.5

driving_forward = False
light_change_time = rospy.Time.now()
rate = rospy.Rate(1)

while not rospy.is_shutdown():
    try:
        if driving_forward:
            cmd_vel_pub.publish(green_light_twist)
        else:
            cmd_vel_pub.publish(red_light_twist)
        if light_change_time > rospy.Time.now():
            driving_forward = not driving_forward
        light_change_time = rospy.Time.now() + rospy.Duration(3)
        rate.sleep()
    except ROSInterruptException:
        rospy.loginfo('Finish')
