#! /usr/bin/env python3

"""
https://qiita.com/Yuya-Shimizu/items/5c408fb06878471ad486
"""

import rospy, ros_numpy
import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

class Follower:
    def __init__(self):
        rospy.init_node('follower')
        rospy.on_shutdown(self.cleanup)
        # self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)   #Image型で画像トピックを購読し，コールバック関数を呼ぶ
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        self.twist = Twist()

    def reset_sim(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

    def image_callback(self, msg):
        # image = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8') # cannot use in python3 virtualenv
        image = ros_numpy.numpify(msg) # instead of cv_bridge
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # for ros_numpy

        hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)  #色空間の変換(BGR→HSV)
        lower_yellow = np.array([90, 90, 90])       #黄色の閾値（下限）
        upper_yellow = np.array([255, 255, 250])    #黄色の閾値（上限）
        mask = cv.inRange(hsv, lower_yellow, upper_yellow)  #閾値によるHSV画像の2値化（マスク画像生成）
        masked = cv.bitwise_and(image, image, mask = mask)  #mask画像において，1である部分だけが残る（フィルタに通している）

        h, w = image.shape[:2]
        RESIZE = (w//3, h//3)
        search_top = (h//4)*3
        search_bot = search_top + 20    #目の前の線にだけに興味がある→20行分くらいに絞る
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0


        M = cv.moments(mask)    #maskにおける1の部分の重心
        if M['m00'] > 0:    #重心が存在する
            cx = int(M['m10']/M['m00']) #重心のx座標
            cy = int(M['m01']/M['m00']) #重心のy座標
            cv.circle(image, (cx, cy), 20, (0, 0, 255), -1) #赤丸を画像に描画

            ##P制御
            err = cx - w//2 #黄色の先の重心座標(x)と画像の中心(x)との差
            self.twist.linear.x = 0.2
            #self.twist.angular.z = -float(err)/100 #画像が大きいためか，-1/100では絶対値がまだ十分に大きく，ロボットが暴れてしまう
            self.twist.angular.z = -float(err)/1000 #誤差にあわせて回転速度を変化させる（-1/1000がP制御でいうところの比例ゲインにあたる）
            self.cmd_vel_pub.publish(self.twist)
        #大きすぎるため，サイズ調整
        display_mask = cv.resize(mask, RESIZE)
        display_masked = cv.resize(masked, RESIZE)
        display_image = cv.resize(image, RESIZE)

        #表示
        cv.imshow('BGR Image', display_image)   #'BGR Image'ウィンドウにimageを表示
        cv.imshow('MASK', display_mask)         #'MASK'ウィンドウにimageを表示
        cv.imshow('MASKED', display_masked)     #'MASKED'ウィンドウにimageを表示
        cv.waitKey(3)   #3ミリ秒待つ

    def cleanup(self):
        cv.destroyAllWindows()
        rospy.loginfo('Stop Turtlebot')
        self.cmd_vel_pub.publish(Twist())


follower = Follower()   #Followerクラスのインスタンスを作成（init関数が実行される）
follower.reset_sim()
rospy.spin()    #ループ