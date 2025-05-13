#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import yaml
from collections import deque
import numpy as np

import cv2

import rospy
from cv_bridge import CvBridge, CvBridgeError
import actionlib
from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal, SGRCtrlResult
from sensor_msgs.msg import Image




def object_detector(img, lower_hsv, upper_hsv):
    cv_image_cp = img.copy()
    cv_image_hsv = cv2.cvtColor(cv_image_cp, cv2.COLOR_BGR2HSV)
    cv_image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if lower_hsv[0] > upper_hsv[0]: # hue will jump between 360(180) and 0 in red area
        lower = np.array([0 / 2, lower_hsv[1], lower_hsv[2]])
        upper = np.array([upper_hsv[0], upper_hsv[1], upper_hsv[2]])

        lower2 = np.array([lower_hsv[0], lower_hsv[1], lower_hsv[2]])
        upper2 = np.array([180, upper_hsv[1], upper_hsv[2]])
        cv_image_gray = cv2.add(
            cv2.inRange(cv_image_hsv, lower, upper),
            cv2.inRange(cv_image_hsv, lower2, upper2)
        )
    else:
        cv_image_gray = cv2.inRange(cv_image_hsv, lower_hsv, upper_hsv)

    # use gray to predict
    # smooth a_picnd clean noise
    cv_image_gray = cv2.erode(cv_image_gray, None, iterations=2)
    cv_image_gray = cv2.dilate(cv_image_gray, None, iterations=2)
    cv_image_gray = cv2.GaussianBlur(cv_image_gray, (5, 5), 0)


    # find contours
    contours, hier = cv2.findContours(
        cv_image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # find the max one
        largest_contours = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contours)
        max_ball = (x, y, radius)
    else:
        max_ball = None
    return max_ball


def predict_image_trajectory(trajectory, steps=5, grab_time=5):
    """
    average a and v
    """
    if len(trajectory) < 2:
        raise 
    vx = 0
    vy = 0
    
    vx_0 = 0
    vy_0 = 0
    
    ax = 0
    ay = 0
    for i in range(len(trajectory) - 1, 0, -1):
        dt = trajectory[i][0] - trajectory[i-1][0]
        dx = trajectory[i][1] - trajectory[i-1][1]
        dy = trajectory[i][2] - trajectory[i-1][2]
        
        dt_0 = trajectory[i-1][0] - trajectory[i-2][0]
        dx_0 = trajectory[i-1][1] - trajectory[i-2][1]
        dy_0 = trajectory[i-1][2] - trajectory[i-2][2]
        
        vx += dx / dt / (len(trajectory) - 1)
        vy += dy / dt / (len(trajectory) - 1)

        vx_0 += dx_0 / dt_0 / (len(trajectory) - 1)
        vy_0 += dy_0 / dt_0 / (len(trajectory) - 1)

        ax += (vx - vx_0) / dt / (len(trajectory) - 1)    
        ay += (vy - vy_0) / dt / (len(trajectory) - 1)
    

    last_point = trajectory[-1]
    pred_points = []
    for i in range(1, steps+1):
        x = int(last_point[1] + vx * i * dt + 0.5 * ax * (i * dt)**2)
        y = int(last_point[2] + vy * i * dt + 0.5 * ay * (i * dt)**2)
        pred_points.append((x, y))
        
    
    # grab 5 seconds later
    grab_x = int(last_point[1] + vx * grab_time + 0.5 * ax * (grab_time)**2)
    grab_y = int(last_point[2] + vy * grab_time + 0.5 * ay * (grab_time)**2)
    grab_point = (grab_x, grab_y)
    return pred_points, grab_point



def draw_region(frame, ball):
    x, y, radius = map(int, ball)
    cv2.circle(frame, (x, y), radius ,(0, 255, 0), 2)
    cv2.circle(frame, (x, y), 3, (255, 0, 0), 3)
    return frame



    
def draw_trajectory(frame, trajectory, predicted_points):
    for i in range(1, len(trajectory)):
        cv2.line(frame, 
                (int(trajectory[i-1][1]), int(trajectory[i-1][2])),
                (int(trajectory[i][1]), int(trajectory[i][2])),
                (0, 255, 0), 2)  # green
        
    for i in range(len(predicted_points)):
        color = (0, 0, 255)
        cv2.circle(frame, 
                  (predicted_points[i][0], predicted_points[i][1]),
                  5, color, -1)
    return frame        


class GrabPointHandler:
    def __init__(self):
        self._intialzie()

        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.trajectory = deque(maxlen=20)
        self.pred_points = deque(maxlen=10)
        self.grab_point = None 


    def _intialzie(self, ):
        # 
        rospy.init_node('ball_catch_node', anonymous=False)

        # sagittarius control action
        self.client = actionlib.SimpleActionClient(
        rospy.get_param("~arm_name", "sgr532") + '/' + 'sgr_ctrl', SGRCtrlAction)
        self.client.wait_for_server()

        # initialize parameter
        filename = rospy.get_param("~vision_config")
        try:
            with open(filename, "r") as f:
                content = yaml.load(f.read())
        except:
            rospy.logerr("can't not open hsv file: ", filename)
            exit(1)

        self.k1 = content['LinearRegression']['k1']
        self.b1 = content['LinearRegression']['b1']
        self.k2 = content['LinearRegression']['k2']
        self.b2 = content['LinearRegression']['b2']

        self.lower_HSV = np.array([content['red']['hmin'] / 2, content['red']['smin'], content['red']['vmin']])
        self.upper_HSV = np.array([content['red']['hmax'] / 2, content['red']['smax'], content['red']['vmax']])

        # initialize action
        self.goal_search = SGRCtrlGoal()
        self.goal_search.action_type = self.goal_search.ACTION_TYPE_XYZ_RPY
        self.goal_search.grasp_type = self.goal_search.GRASP_OPEN
        self.goal_search.pos_x = 0.2
        self.goal_search.pos_z = 0.15
        self.goal_search.pos_pitch = 1.57

        self.goal_pick = SGRCtrlGoal()
        self.goal_pick.grasp_type = self.goal_pick.GRASP_OPEN
        self.goal_pick.action_type = self.goal_pick.ACTION_TYPE_PICK_XYZ
        self.goal_pick.pos_z = 0.02
        self.goal_pick.pos_pitch = 1.57


        # move to search pose
        rospy.loginfo('Move to Search Pose')
        self.client.send_goal_and_wait(self.goal_search, rospy.Duration.from_sec(30))


    def image_callback(self, frame):
        try:
            frame = CvBridge().imgmsg_to_cv2(frame, "bgr8")
        except CvBridgeError as e:
            print(e)        
        ball = object_detector(frame, self.lower_HSV, self.upper_HSV)

        if ball is not None:
            current_x, current_y, _ = ball
            
            current_time = rospy.Time.now().to_sec()
            self.trajectory.append(
                (current_time, int(current_x), int(current_y))
                )
            detect_frame = draw_region(frame, ball)
            detect_frame = draw_trajectory(detect_frame, self.trajectory, self.pred_points)

            if len(self.trajectory) >= 3:
                # predict trajectory and grab point
                # Todo: make this function in class
                predicted_points, self.grab_point = predict_image_trajectory(self.trajectory, steps=10, grab_time=2)
                self.pred_points.extend(predicted_points)
                if self.grab_point is not None:
                    cv2.circle(
                        detect_frame, 
                        self.grab_point,
                        5, (0, 255, 255), -1
                        )

            cv2.imshow("detected", detect_frame)
        else:
            self.trajectory.clear()
            self.pred_points.clear()
            cv2.imshow("detected", frame)
        cv2.waitKey(1)


    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.grab_point is not None:
                self.goal_pick.pos_x = self.k1 * self.grab_point[1] + self.b1
                self.goal_pick.pos_y = self.k2 * self.grab_point[0] + self.b2
                self.client.send_goal_and_wait(self.goal_pick, rospy.Duration.from_sec(30))

                ret = self.client.get_result()
                if ret.result == SGRCtrlResult.PLAN_NOT_FOUND:
                    rospy.logwarn("no plan return. pass")
                elif ret.result == SGRCtrlResult.GRASP_FAILD:
                    rospy.logwarn("grasp faild. pass")
                else:
                    rospy.loginfo("grasp success")
                    
                self.client.send_goal_and_wait(self.goal_search, rospy.Duration.from_sec(30))    
                self.grab_point = None               
            rate.sleep()


if __name__ == '__main__':
    handler = GrabPointHandler()
    handler.run()
