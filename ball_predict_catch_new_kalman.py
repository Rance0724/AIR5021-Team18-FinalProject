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

# import signal

class KalmanFilter:
    def __init__(
        self, 
        initial_time, 
        initial_pos, 
        initial_v,
        initial_a, 
        accel_noise=0.1, 
        obs_noise=5.0):
        """
        - initial_time: 
        - initial_pos: 
        - accel_noise: 
        - obs_noise:
        """
        # state: [x, y, vx, vy, ax, ay]
        self.state = np.array(
            [initial_pos[0], initial_pos[1], initial_v[0], initial_v[1], initial_a[0], initial_a[1]
             ], dtype=np.float32)
        # ajust this
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # obtain position
        self.H = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0]])
        # observation variance 
        self.R = np.diag([obs_noise, obs_noise])
        # accleration variance
        self.sigma_a = accel_noise

        self.last_time = initial_time
        self.dt = None

    def _build_F(self, dt):
        """Update F"""
        return np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

    def _build_Q(self, dt):
        """update Q"""
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        q11 = (dt4/4) * self.sigma_a**2
        q13 = (dt3/2) * self.sigma_a**2
        q22 = q11
        q24 = q13
        q33 = dt2 * self.sigma_a**2
        q44 = q33
        q55 = self.sigma_a**2
        q66 = q55
        return np.array([
            [q11, 0,   q13, 0,   0,    0],
            [0,   q22, 0,   q24, 0,    0],
            [q13, 0,   q33, 0,   0,    0],
            [0,   q24, 0,   q44, 0,    0],
            [0,   0,   0,   0,   q55,  0],
            [0,   0,   0,   0,   0,   q66]
        ], dtype=np.float32)

    def update(self, current_time, measurement):
        """
        update state according new observation
        - current_time: used to calculate dt 
        - measurement: (x, y)
        """
        self.dt = current_time - self.last_time
        self.last_time = current_time

        F = self._build_F(self.dt)
        Q = self._build_Q(self.dt)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def predict(self, t):
        """
        predict position aafter t second
        - return: predict point
        """
        F_pred = self._build_F(t)

        pred_state = F_pred @ self.state
        return pred_state[0], pred_state[1]




# class KalmanFilter:
#     def __init__(self, initial_time, initial_pos, intial_v, velocity_noise=0.1, obs_noise=5.0):
#         """
#         - initial_time: 
#         - initial_pos: 
#         - accel_noise: 
#         - obs_noise:
#         """
#         # state: [x, y, vx, vy,]
#         self.state = np.array([initial_pos[0], initial_pos[1], intial_v[0], intial_v[1],], dtype=np.float32)
#         # ajust this
#         self.P = np.diag([0.1, 0.1, 0.1, 0.1])
#         # obtain position
#         self.H = np.array([[1,0,0,0,], [0,1,0,0,]])
#         # observation variance 
#         self.R = np.diag([obs_noise, obs_noise])
#         # accleration variance
#         self.sigma_v = velocity_noise

#         self.last_time = initial_time
#         self.dt = None

#     def _build_F(self, dt):
#         """Update F"""
#         return np.array([
#             [1, 0, dt, 0, ],
#             [0, 1, 0, dt, ],
#             [0, 0, 1, 0, ],
#             [0, 0, 0, 1, ],
#         ], dtype=np.float32)

#     def _build_Q(self, dt):
#         """update Q"""
#         dt2 = dt ** 2
#         dt3 = dt ** 3
#         q11 = (dt3/3) * self.sigma_v**2
#         q13 = (dt2/2) * self.sigma_v**2
#         q22 = q11
#         q24 = q13
#         q33 = dt * self.sigma_v**2
#         q44 = q33
#         return np.array([
#             [q11, 0, q13, 0,],
#             [0, q22, 0, q24,],
#             [q13, 0, q33, 0,],
#             [0, q24, 0, q44,],
#         ], dtype=np.float32)

#     def update(self, current_time, measurement):
#         """
#         update state according new observation
#         - current_time: used to calculate dt 
#         - measurement: (x, y)
#         """
#         self.dt = current_time - self.last_time
#         self.last_time = current_time

#         F = self._build_F(self.dt)
#         Q = self._build_Q(self.dt)

#         self.state = F @ self.state
#         self.P = F @ self.P @ F.T + Q

#         y = measurement - self.H @ self.state
#         S = self.H @ self.P @ self.H.T + self.R
#         K = self.P @ self.H.T @ np.linalg.inv(S)

#         self.state = self.state + K @ y
#         self.P = (np.eye(4) - K @ self.H) @ self.P

#     def predict(self, t):
#         """
#         predict position aafter t second
#         - return: predict point
#         """
#         pred_x = self.state[0] + self.state[2] * t
#         pred_y = self.state[1] + self.state[3] * t
#         return pred_x, pred_y




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


def predict_image_trajectory(trajectory, steps=5):
    """
    匀速
    """
    if len(trajectory) < 2:
        raise 
    
    # 计算平均速度 in a trajectory
    vx, vy = 0, 0
    for i in range(1, len(trajectory)):
        dt = trajectory[i][0] - trajectory[i-1][0]
        dx = trajectory[i][1] - trajectory[i-1][1]
        dy = trajectory[i][2] - trajectory[i-1][2]
        
        vx += dx / dt
        vy += dy / dt
    # average
    vx /= (len(trajectory)-1)
    vy /= (len(trajectory)-1)
    
    # 生成预测点
    last_point = trajectory[-1]
    pred_points = []
    for i in range(1, steps+1):
        x = int(last_point[1] + vx * i * dt)
        y = int(last_point[2] + vy * i * dt)
        pred_points.append((x, y))
        
    
    # grab 5 seconds later
    grab_x = int(last_point[1] + vx * 5)
    grab_y = int(last_point[2] + vy * 5)
    grab_point = (grab_x, grab_y)
    return pred_points, grab_point



def draw_region(frame, ball):
    x, y, radius = map(int, ball)
    cv2.circle(frame, (x, y), radius ,(0, 255, 0), 2)
    cv2.circle(frame, (x, y), 3, (255, 0, 0), 3)
    return frame



    
def draw_trajectory(frame, trajectory, predicted_points):
    # 绘制历史轨迹
    for i in range(1, len(trajectory)):
        cv2.line(frame, 
                (int(trajectory[i-1][1]), int(trajectory[i-1][2])),
                (int(trajectory[i][1]), int(trajectory[i][2])),
                (0, 255, 0), 2)  # green
        
    for i in range(len(predicted_points)):
        color = (0, 0, 255)  # red
        cv2.circle(frame, 
                  (int(predicted_points[i][0]), int(predicted_points[i][1])),
                  5, color, -1)
    return frame        



class GrabPointHandler:
    def __init__(self, t_pred):
        self._intialzie()

        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.trajectory = deque(maxlen=20)
        self.pred_points = deque(maxlen=10)
        self.grab_point = None 

        self.kf = None
        self.t_pred = t_pred


        # video stuff
        # self.is_recording = True
        # self.video_writer = None
        # self.frame_width = 640
        # self.frame_height = 480
        # self.fps = 30

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.video_writer = cv2.VideoWriter(
        #     'predict_video.mp4',
        #     fourcc,
        #     self.fps,
        #     (self.frame_width, self.frame_height)
        # )
        # if not self.video_writer.isOpened():
        #     rospy.logerr("can not create video")
        #     sys.exit(1)

        # signal.signal(signal.SIGINT, self.signal_handler)

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


    def _init_kalman_filter(self, initial_time, initial_pos, init_v, init_a):
        self.kf = KalmanFilter(
            initial_time=initial_time,
            initial_pos=initial_pos,
            initial_v=init_v,
            initial_a=init_a,
            accel_noise=0.02,  # adjust this
            obs_noise=0.01  # adjust this
        )
        self.last_kf_update_time = initial_time

    # def signal_handler(self, sig, frame):
    #     print("saving video...")
    #     self.is_recording = False


    def image_callback(self, frame):
        try:
            frame = CvBridge().imgmsg_to_cv2(frame, "bgr8")
        except CvBridgeError as e:
            print(e)        
        ball = object_detector(frame, self.lower_HSV, self.upper_HSV)

        if ball is None:
            self.trajectory.clear()
            self.pred_points.clear()
            self.kf = None
            cv2.imshow("detected", frame)
            cv2.waitKey(1)
            # self.write_frame(cv_frame)
            return


        current_time = rospy.Time.now().to_sec()
        current_x, current_y, _ = ball
        
        self.trajectory.append(
            (current_time, current_x, current_y)
            )
        
        
        # initialze kalman
        # detect not well at edge, use later detection
        if len(self.trajectory) >= 5 and self.kf is None:
            dt = self.trajectory[4][0] - self.trajectory[3][0]
            dx = self.trajectory[4][1] - self.trajectory[3][1]
            dy = self.trajectory[4][2] - self.trajectory[3][2]
            
            dt_0 = self.trajectory[3][0] - self.trajectory[2][0]
            dx_0 = self.trajectory[3][1] - self.trajectory[2][1]
            dy_0 = self.trajectory[3][2] - self.trajectory[2][2]
            
            
            vx = dx / dt
            vy = dy / dt
            
            vx_0 = dx_0 / dt_0
            vy_0 = dy_0 / dt_0
            
            ax = (vx - vx_0) / dt    
            ay = (vy - vy_0) / dt          
            
            self._init_kalman_filter(
                current_time, 
                (current_x, current_y), 
                (vx, vy),
                (ax, ay),
                )
        

        self.kf.update(current_time, (current_x, current_y))

        detect_frame = draw_region(frame, ball)
        detect_frame = draw_trajectory(detect_frame, self.trajectory, self.pred_points)

        if len(self.trajectory) >= 7:  # adjust this
            # predict trajectory and grab point
            # Todo: make this function in class

            predicted_points = [self.kf.predict(self.kf.dt * i) for i in range(1, 5)]
            self.pred_points.extend(predicted_points)

            grab_x, grab_y = self.kf.predict(self.t_pred) # adjust this
            self.grab_point = (int(grab_x), int(grab_y))

            cv2.circle(
                detect_frame, 
                self.grab_point,
                5, (0, 255, 255), -1
                )

        cv2.imshow("detected", detect_frame)
        cv2.waitKey(1)
        
        # self.write_frame(detect_frame)

    # def write_frame(self, frame):
    #     if self.video_writer is not None and self.is_recording:
    #         # resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
    #         self.video_writer.write(resized_frame)

    # def release_resources(self):
    #     if self.video_writer is not None:
    #         self.video_writer.release()
    #         rospy.loginfo("Sucessfully save!")
    #     cv2.destroyAllWindows()

    def run(self):
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            if self.grab_point is not None:
                self.goal_pick.pos_x = self.k1 * self.grab_point[1] + self.b1
                self.goal_pick.pos_y = self.k2 * self.grab_point[0] + self.b2
                self.client.send_goal_and_wait(self.goal_pick, rospy.Duration.from_sec(30))

                ret = self.client.get_result()
                if ret.result == SGRCtrlResult.PLAN_NOT_FOUND:
                    rospy.logwarn("no plan return. pass")
                elif ret.result == SGRCtrlResult.GRASP_FAILD:
                    rospy.logwarn("grasp failed. pass")
                else:
                    rospy.loginfo("grasp success")
                    
                self.client.send_goal_and_wait(self.goal_search, rospy.Duration.from_sec(30))    
                self.grab_point = None               
            rate.sleep()
            
        # self.release_resources()


if __name__ == '__main__':
    handler = GrabPointHandler(3)
    handler.run()
