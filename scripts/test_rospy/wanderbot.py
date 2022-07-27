#! /usr/bin/env python3

"""
ROSの勉強　第12弾：センシングと移動
https://qiita.com/Yuya-Shimizu/items/66dd6fa254957ca773e9
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

def scan_callback(msg):
    """読み取ったセンサデータをグローバル変数に渡すことで，プログラム全体でその情報を扱えるようにする"""
    global g_range_ahead
    g_range_ahead = msg.ranges[len(msg.ranges)//2]  #正面の距離をグローバル変数に格納


g_range_ahead = 1   #初期値
scan_sub = rospy.Subscriber('scan', LaserScan, scan_callback)   #センサデータの購読
cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = 1) #移動トピックの配信準備

rospy.init_node('wanderbot')    #ノードの初期化

state_change_time = rospy.Time.now()    #現在時刻を記録
driving_forward = True  #直進可能と仮定
rate = rospy.Rate(10)   #10Hz

#ctrlキーが押されるまでループ
while not rospy.is_shutdown():

    #直進可
    if driving_forward:
        #条件1と2を満たすか
        if (g_range_ahead < 0.8 or rospy.Time.now() > state_change_time):
            driving_forward = False #直進不可
            state_change_time = rospy.Time.now() + rospy.Duration(5)    #rosの基準で5duration先の時刻を現在時刻とする　→　5durationの間回転を続けるということ

    #直進不可
    else:
        #方向転換が終了したか
        if rospy.Time.now() > state_change_time:
            driving_forward = True  #直進可能
            state_change_time = rospy.Time.now() + rospy.Duration(30)   #rosの基準で30duration先の時刻を現在時刻とする　→　30durationの間直進を続けるということ


    twist = Twist() #全体を0で初期化するために毎回Twistインスタンスを生成    これにより，以下では動かしたい要素のみの変更で済む

    #直進可能　→　直線方向の速度1
    if driving_forward:
        twist.linear.x = -1

    #直進不可　→　方向転換　→　回転の速度(z軸周り)1
    else:
        twist.angular.z = 1

    cmd_vel_pub.publish(twist)  #移動トピックを配信

    rate.sleep()    #10Hz(=0.1秒)待つ
