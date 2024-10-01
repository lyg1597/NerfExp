import numpy as np
from math import cos, sin, atan2, sqrt, pi, asin
import rospy
import time
from scipy.integrate import odeint
import rospy 
import rospkg 
import cv2
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage

from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
# from rosplane_msgs.msg import State
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from sensor_msgs.msg import Image

import torch
import torch.nn.functional as F
from unet import UNet
from utils.data_loading import BasicDataset
import os

import pickle
import matplotlib.pyplot as plt 
from std_msgs.msg import ColorRGBA

import copy 
import squaternion 

body_height = 0.77
pitch_offset = 0

def create_state_msd(x, y, z, roll, pitch, yaw):
    state_msg = ModelState()
    state_msg.model_name = 'fixedwing'
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.position.z = z

    q = squaternion.Quaternion.from_euler(roll, pitch, yaw)
    
    state_msg.pose.orientation.x = q.x
    state_msg.pose.orientation.y = q.y
    state_msg.pose.orientation.z = q.z
    state_msg.pose.orientation.w = q.w

    return state_msg

class Perception:
    def __init__(self, net, device):
        self.net = net 
        self.device = device
        self.image_updated = False
        self.image = None
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        rospy.Subscriber("/fixedwing/chase/camera/rgb", Image, self.image_callback)
        self.bridge = CvBridge()

        # Camera matrix (intrinsic matrix)
        self.K = np.array([[1253.2215566867008, 0.0, 320.5], [0.0, 1253.2215566867008, 240.5], [0.0, 0.0, 1.0]])

        # The set of key points used for state estimation.
        self.keypoints = [[-1221.370483, 16.052534, 5.0],
                        [-1279.224854, 16.947235, 5.0],
                        [-1279.349731, 8.911615, 5.0],
                        [-1221.505737, 8.033512, 5.0],
                        [-1221.438110, -8.496282, 5.0],
                        [-1279.302002, -8.493725, 5.0],
                        [-1279.315796, -16.504263, 5.0],
                        [-1221.462402, -16.498976, 5.0],
                        [-1520.81, 26.125700, 5.0],
                        [-1559.122925, 26.101082, 5.0],
                        [-1559.157471, -30.753305, 5.0],
                        [-1520.886353,  -30.761044, 5.0],
                        [-1561.039063, 31.522200, 5.0],
                        [-1561.039795, -33.577713, 5.0]]
    
        self.state = None
        self.error_idx = []
        self.idx = None

    def state_callback(self, msg):
        pos = msg.pose[1].position
        ori = msg.pose[1].orientation
        self.state = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
 
    def image_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'passthrough')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.image = copy.deepcopy(cv_image)
            self.image_updated = True
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def predict_img(self, full_img, scale_factor=1.0, out_threshold=0.5):
        '''
        Unet.
        '''
        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img).cpu()

            return output 

    def pose_estimation(self, image_points, object_points, camera_matrix, dist_coeffs=np.zeros((4, 1))):
        '''
        Pose estimation via solvePnP.
        '''
        return cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    def show_image(self, cv_image, keypoints):
        kp_img = cv_image
        for i in range(len(keypoints)):
           kp_img = cv2.circle(kp_img, (int(keypoints[i][0]), int(keypoints[i][1])), radius=2, color=(0, 0, 255))

        # kp_img = self.add_noise_to_image(kp_img)
  
        cv2.imshow("Image Window", kp_img)
        cv2.waitKey(3)

    def vision_estimation(self, cv_image):
        # Try to convert the ROS image to a CV2 image
        img = PILImage.fromarray(cv_image, mode='RGB')
        
        # Get probabilistic heat maps corresponding to the key points.
        output = self.predict_img(img)

        # plt.figure(0)
        # plt.imshow(img)
        # plt.figure(1)
        # plt.imshow(np.sum(output[0].detach().numpy(),axis=0))
        # plt.show()

        # Key points detection using trained nn. Extract pixels with highest probability.
        keypoints = []
        for i in range(14):
            p = (((output[0,i,:,:]==torch.max(output[0,i,:,:])).nonzero())/1.0).tolist()
            p[0].reverse()
            keypoints.append(p[0])

        # Pose estimation via PnP.
        success, rotation_vector, translation_vector = self.pose_estimation(np.array(keypoints), np.array(self.keypoints), self.K)
        if success:
            # TODO: THIS PART NEEDS TO BE REVISED.
            Rot = cv2.Rodrigues(rotation_vector)[0]
            RotT = np.matrix(Rot).T
            camera_position = -RotT*np.matrix(translation_vector)

            R = Rot
            sin_x = sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
            singular  = sin_x < 1e-6
            if not singular:
                z1 = atan2(R[2,0], R[2,1])     # around z1-axis
                x = atan2(sin_x,  R[2,2])     # around x-axis
                z2 = atan2(R[0,2], -R[1,2])    # around z2-axis
            else:
                z1 = 0                                         # around z1-axis
                x = atan2(sin_x,  R[2,2])     # around x-axis
                z2 = 0                                         # around z2-axis

            angles = np.array([z1, x, z2])
            yawpitchroll_angles = -angles
            yawpitchroll_angles[0] = (yawpitchroll_angles[0] + (5/2)*pi)%(2*pi) # change rotation sense if needed, comment this line otherwise
            yawpitchroll_angles[1] = -(yawpitchroll_angles[1]+pi/2)
            if yawpitchroll_angles[0] > pi:
                yawpitchroll_angles[0] -= 2*pi

            self.estimated_state = [camera_position[0].item(), camera_position[1].item(), camera_position[2].item() - body_height, yawpitchroll_angles[2], yawpitchroll_angles[1] - pitch_offset, yawpitchroll_angles[0]]
        else:
            print("Pose Estimation Failed.")

        self.show_image(cv_image, keypoints)

    def wait_img_update(self):
        while not self.image_updated:
            time.sleep(0.1)

    def set_percept(self, point: np.ndarray) -> np.ndarray:
        # Set aircraft to pos
        self.set_pos(point)

        self.wait_img_update()

        img = self.image
        # img = set_rain_properties(img, e[0])
        self.vision_estimation(img)

        estimated_state = np.array([
            self.estimated_state[0],
            self.estimated_state[1],
            self.estimated_state[2],
            self.estimated_state[5],
            self.estimated_state[4],
            point[5]
        ])

        # if np.linalg.norm(estimated_state - point) > 50:
        #     print(">>>>>> Estimated Corrupted ", estimated_state)
        #     estimated_state = point 
        #     self.error_idx.append(self.idx)
        

        return estimated_state

    def set_pos(self, point: np.ndarray) -> np.ndarray:
        # Set aircraft to given pose
        init_msg = create_state_msd(point[0], point[1], point[2], 0, point[4], point[3])
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            # Set initial state.
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(init_msg)
        except rospy.ServiceException:
            print("Service call failed")
 
