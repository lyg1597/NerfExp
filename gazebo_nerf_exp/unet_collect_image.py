'''
Collect additional training images for fine tuning the detector on nerfenv
'''

import numpy as np
from math import cos, sin, atan2, sqrt, pi, asin
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import squaternion 
from scipy.spatial.transform import Rotation 
from ns_dp_info import dpDict

from gazebo_msgs.msg import ModelState 
# from gazebo_msgs.srv import SetModelState
# from rosplane_msgs.msg import State
# from gazebo_msgs.msg import ModelStates
# from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
# from sensor_msgs.msg import Image

# from aircraft_model import *
# from aircraft_mpc import *
# from aircraft_simulator import *

import os
import random
from ns_renderer import SplatRenderer
import matplotlib.pyplot as plt 

script_dir = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(script_dir, './data_unet/data_08-19-21-18')
if not os.path.exists(data_path):
    os.mkdir(data_path)
img_path = os.path.join(script_dir,'./data_unet/data_08-19-21-18/imgs')
if not os.path.exists(img_path):
    os.mkdir(img_path)
    
keypoints = [[-1221.370483, 16.052534, 5.0],
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

def sample_pose():
    x = random.uniform(-2000, -3200)
    y = random.uniform(-45, 45)
    z = random.uniform(20, 150)
    roll = random.uniform(-0.05, 0.05)
    pitch = random.uniform(-0.1, 0.1)
    yaw = random.uniform(-0.1, 0.1)

    return [x, y, z, roll, pitch, yaw]

def convert_to_image(world_pos, ego_pos, ego_ori, cameraMatrix):
    objectPoints = np.array(world_pos) 
    R = Rotation.from_quat(ego_ori)
    R2 = Rotation.from_euler('xyz',[-np.pi/2, -np.pi/2, 0])
    R_roted = R2*R.inv()

    #TODO: The way of converting rvec is wrong
    rvec = R_roted.as_rotvec()
    tvec = -R_roted.apply(np.array(ego_pos))
    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    pnt,_ = cv2.projectPoints(objectPoints,rvec,tvec,cameraMatrix,distCoeffs)
    return pnt  

def pose_estimation(image_points, object_points, camera_matrix, dist_coeffs=np.zeros((4, 1))):
        res = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        return res 

def check_valid_img(success, rotation_vector, translation_vector, state_ground_truth, tolr=1):
    if success:
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
        angles = np.array([[z1], [x], [z2]])
        yawpitchroll_angles = -angles
        yawpitchroll_angles[0,0] = (yawpitchroll_angles[0,0] + (5/2)*pi)%(2*pi) # change rotation sign if needed, comment this line otherwise
        yawpitchroll_angles[1,0] = -(yawpitchroll_angles[1,0]+pi/2)

        # yawpitchroll_angles[0,0] = yawpitchroll_angles[0,0]%pi
        # if yawpitchroll_angles[0,0] > pi:
        #     yawpitchroll_angles[0,0] = yawpitchroll_angles[0,0]%(pi*2)-2*pi
        # yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]%pi
        # if yawpitchroll_angles[1,0] > pi:
        #     yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]%(pi*2)-2*pi
        # yawpitchroll_angles[2,0] = yawpitchroll_angles[2,0]%pi
        # if yawpitchroll_angles[2,0] > pi:
        #     yawpitchroll_angles[2,0] = yawpitchroll_angles[2,0]%(pi*2)-2*pi

        estimated_state = [camera_position[0].item(), camera_position[1].item(), camera_position[2].item(), yawpitchroll_angles[2,0]-pi, yawpitchroll_angles[1,0], yawpitchroll_angles[0,0]]
        estimation_error = np.linalg.norm(np.array(estimated_state) - np.array(state_ground_truth))
        print("State estimation: ", estimated_state)
        print("State ground truth: ", state_ground_truth)
        if 0.5*(abs(estimated_state[0] - state_ground_truth[0]) + abs(estimated_state[1] - state_ground_truth[1]) + abs(estimated_state[2] - state_ground_truth[2])) + (abs(estimated_state[3] - state_ground_truth[3]) + abs(estimated_state[4] - state_ground_truth[4]) + abs(estimated_state[5] - state_ground_truth[5])) > 1:
            return False, estimation_error, estimated_state
        return True, estimation_error, estimated_state
    else:
        return False, np.inf, []

def get_position():
    while True:
        x = np.random.uniform(-3500, 1500)
        y = np.random.uniform(-2000, 2000)
        z = np.random.uniform(50, 750) 
        if not (z<300 and (y<-50 or y>50)):
            return [x,y,z] 

# Function to calculate the quaternion for the camera to face the target
def calculate_orientation(position, target_position=(-1500, 0, 0)):
    direction = np.array([target_position[0] - position[0],
                          target_position[1] - position[1],
                          target_position[2] - position[2]])
    direction = direction / np.linalg.norm(direction)
    yaw = np.arctan2(direction[1], direction[0])
    pitch = -np.arcsin(direction[2])
    return [0,pitch,yaw]

def sample_pose2():
    position = get_position()
    orientation = calculate_orientation(position)
    return position+orientation

def update_aircraft_position(imWidth, imHeight, fx, fy):
    initial_state =  [-3000.0, 0.0, 100.0, 0, 0, 0]
    
    cur_state = initial_state

    data_fn = os.path.join(data_path, f"data_08-19-21-18.txt")
    with open(data_fn,'w+') as f:
        pass    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_fn = os.path.normpath(os.path.join(
        script_dir,
        "../outputs/gazebo4_transformed/splatfacto/2024-08-05_204928/config.yml"
    ))
    renderer = SplatRenderer(
        config_fn, 
        imWidth, 
        imHeight, 
        fx, 
        fy
    )

    cx = imWidth/2
    cy = imHeight/2

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


    idx = 0
    while True:
        # Sampled camera pose in World (Gazebo) frame
        if np.random.uniform()>0.3:
            state_rand = sample_pose()
        else:
            state_rand = sample_pose2()

        # state_rand = [-3158.875853940029, 32.827547811424, 41.31578260068058, -0.04476461460798924, 0.0002528190803569863, -0.08145549616417824]
        # state_rand = [-643.5870811335876,-1315.487922957998,986.3821236027896,-3.27806592e-17,  5.61033194e-01,  2.14789038e+00]
        camera_pose = np.zeros((4,4))
        camera_pose[3,3] = 1
        camera_pose[:3,:3] = Rotation.from_euler('xyz',[state_rand[3],state_rand[4],state_rand[5]]).as_matrix()
        camera_pose[:3,3] = state_rand[:3]

        # Convert camera pose to what's stated in transforms_orig.json
        tmp = Rotation.from_euler('zyx',[-np.pi/2,np.pi/2,0]).as_matrix()
        mat = camera_pose[:3,:3]@tmp 
        camera_pose[:3,:3] = mat 
        
        # Convert camera pose to Colmap frame in transforms.json
        camera_pose[0:3,1:3] *= -1
        camera_pose = camera_pose[np.array([0,2,1,3]),:]
        camera_pose[2,:] *= -1 

        # Convert colmap pose to nerfstudio pose 
        # Only apply to outputs/gazebo4_transformed/splatfacto/2024-08-05_204928
        dp_trans_info = dpDict[config_fn]
        transform = np.array(dp_trans_info['transform'])
        scale_factor = dp_trans_info['scale']
        camera_pose = transform@camera_pose 
        camera_pose[:3,3] *= scale_factor
        camera_pose = camera_pose[:3,:]

        cv_img = renderer.render(camera_pose)

        # time.sleep(0.01)
        q = squaternion.Quaternion.from_euler(state_rand[3], state_rand[4], state_rand[5])
        # offset_vec = np.array([-1.1*0.20,0,0.8*0.77])
        aircraft_pos = np.array([state_rand[0], state_rand[1], state_rand[2]]) # x,y,z
        aircraft_ori = [q.x, q.y, q.z, q.w] # x,y,z,w
        # R = Rotation.from_quat(aircraft_ori)
        # aircraft_pos += offset_vec

        keypoint_position_in_image = []
        # plt.imshow(cv_img)
        # plt.show()
        for i in range(len(keypoints)):
            image_coordinate = convert_to_image(keypoints[i], aircraft_pos, aircraft_ori, K).flatten()
            if image_coordinate[0] > 640 or image_coordinate[0] < 0 or image_coordinate[1] > 480 or image_coordinate[1] < 0:
                break
            keypoint_position_in_image.append(image_coordinate)
        if len(np.array(keypoint_position_in_image)) < 14:
            continue
        # print(keypoint_position_in_image)
        # print(np.array(keypoints).shape)
        success, rotation_vector, translation_vector = pose_estimation(np.array(keypoint_position_in_image), np.array(keypoints), K)
        valid_img, error, estimated_state = check_valid_img(success, rotation_vector, translation_vector, state_rand)
        q_estimated = squaternion.Quaternion.from_euler(estimated_state[3], estimated_state[4], estimated_state[5])
        if not valid_img:
            continue
        with open(data_fn,'a+') as f:
            # f.write(f"\n{idx},{state_rand[0]},{state_rand[1]},{state_rand[2]},{q.x},{q.y},{q.z},{q.w},{error},{estimated_state[0]},{estimated_state[1]},{estimated_state[2]},{q_estimated.x},{q_estimated.y},{q_estimated.z},{q_estimated.w}")
            f.write(f"\n{idx},{state_rand[0]},{state_rand[1]},{state_rand[2]},{q.x},{q.y},{q.z},{q.w}")
        # cv2.imshow('camera', cv_img)
        # cv2.waitKey(1)
        

        path = os.path.join(img_path, f"img_{idx}.png")
        if cv_img is not None:
            # img = set_rain_properties(cv_img, np.random.uniform(0,0.3))
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            print(path)
            cv2.imwrite(path, cv_img)
        idx += 1
        if idx > 30000:
            break


if __name__ == "__main__":
    # rospy.init_node('update_poses', anonymous=True)
    
    update_aircraft_position(640,480,1253.2215566867008,1253.2215566867008)
    # update_aircraft_position(2560,1440,2343.0242837919386,2343.0242837919386)
    # try:
    #     update_aircraft_position()
    # except rospy.exceptions.ROSInterruptException:
    #     rospy.loginfo("Stop updating aircraft positions.")
        
