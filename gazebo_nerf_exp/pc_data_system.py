import numpy as np 
from typing import Tuple 
import cv2
from cv_bridge import CvBridge, CvBridgeError

import torch 
import copy 
from unet import UNet
from utils.data_loading import BasicDataset
from PIL import Image as PILImage
import time 
# import squaternion 
import os 
import matplotlib.pyplot as plt 
import cv2 
from scipy.spatial.transform import Rotation 
from ns_renderer import SplatRenderer
from ns_dp_info import dpDict
import squaternion 

body_height = 0.77
pitch_offset = 0

# def set_rain_properties(img: np.ndarray, rain_value: float) -> None:
#     density = rain_value*0.5
#     aug = iaa.Rain(drop_size=(0.02, 0.02), speed=(0.1,0.1), nb_iterations=(2,2), density=(density,density))
#     img_aug = aug(image = img)
#     return img_aug

# def set_snow_properties(img: np.ndarray, snow_value: float) -> None:
#     density = snow_value*0.5
#     aug = iaa.Snowflakes(density = (density, density))
#     img_aug = aug(image = img)
#     return img_aug

e_param_list = []
# e_img_updater_list = []

class Perception:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = UNet(n_channels=3, n_classes=14, bilinear=False)
        self.net.to(device=self.device)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # model_path = os.path.join(script_dir, './checkpoints/model_08-06-23-00.pth')
        model_path = os.path.join(script_dir, './checkpoints/checkpoints_08-19-21-18/checkpoint_epoch30.pth')
        state_dict = torch.load(model_path, map_location = self.device)
        self.net.load_state_dict(state_dict)
        self.image_updated = False
        self.image = None
        config_path = os.path.join(script_dir,'../outputs/gazebo4_transformed/splatfacto/2024-08-05_204928/config.yml')
        # config_path = os.path.join(script_dir,'../outputs/gazebo4_resampled4_dataset/splatfacto/2024-09-02_213741/config.yml')
        config_path = os.path.normpath(config_path)
        self.dp_trans_info = dpDict[config_path]
        self.renderer = SplatRenderer(
            config_path, 
            640,
            480,
            1253.2215566867008,
            1253.2215566867008
        )

        # rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        # rospy.Subscriber("/fixedwing/chase/camera/rgb", Image, self.image_callback)
        # self.bridge = CvBridge()

        # Camera matrix (intrinsic matrix)
        self.K = np.array([[1253.2215566867008, 0.0, 320], [0.0, 1253.2215566867008, 240], [0.0, 0.0, 1.0]])

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
        self.estimated_state = None # x, y, z, yaw, pitch, roll

    # def state_callback(self, msg):
    #     pos = msg.pose[1].position
    #     ori = msg.pose[1].orientation
    #     self.state = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
 
    # def image_callback(self, img_msg):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'passthrough')
    #         cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #         self.image = copy.deepcopy(cv_image)
    #         del cv_image
    #         self.image_updated = True
    #     except CvBridgeError as e:
    #         rospy.logerr("CvBridge Error: {0}".format(e))

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
        return cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs)

    def show_image(self, cv_image, keypoints):
        kp_img = cv_image
        for i in range(len(keypoints)):
           kp_img = cv2.circle(kp_img, (int(keypoints[i][0]), int(keypoints[i][1])), radius=2, color=(0, 0, 255))

        # kp_img = self.add_noise_to_image(kp_img)
  
        cv2.imshow("Image Window", kp_img)
        cv2.waitKey(1)

    def vision_estimation(self, cv_image):
        # cv2.imwrite('tmp.png', cv_image)

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
        success, rotation_vector, translation_vector, _ = self.pose_estimation(np.array(keypoints), np.array(self.keypoints), self.K)
        # if success:
        # TODO: THIS PART NEEDS TO BE REVISED.
        Rot = cv2.Rodrigues(rotation_vector)[0]
        RotT = np.matrix(Rot).T
        camera_position = -RotT*np.matrix(translation_vector)

        R = Rot
        sin_x = np.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
        singular  = sin_x < 1e-6
        if not singular:
            z1 = np.arctan2(R[2,0], R[2,1])     # around z1-axis
            x = np.arctan2(sin_x,  R[2,2])     # around x-axis
            z2 = np.arctan2(R[0,2], -R[1,2])    # around z2-axis
        else:
            z1 = 0                                         # around z1-axis
            x = np.arctan2(sin_x,  R[2,2])     # around x-axis
            z2 = 0                                         # around z2-axis

        angles = np.array([z1, x, z2])
        yawpitchroll_angles = -angles
        yawpitchroll_angles[0] = (yawpitchroll_angles[0] + (5/2)*np.pi)%(2*np.pi) # change rotation sense if needed, comment this line otherwise
        yawpitchroll_angles[1] = -(yawpitchroll_angles[1]+np.pi/2)
        if yawpitchroll_angles[0] > np.pi:
            yawpitchroll_angles[0] -= 2*np.pi

        self.estimated_state = [camera_position[0].item(), camera_position[1].item(), camera_position[2].item(), yawpitchroll_angles[0], yawpitchroll_angles[1], yawpitchroll_angles[2]]
        if not success:
            print("Pose Estimation Failed.")

        # self.show_image(cv_image, keypoints)
        # del cv_image
        for i in range(len(keypoints)):
           cv_image = cv2.circle(cv_image, (int(keypoints[i][0]), int(keypoints[i][1])), radius=2, color=(0, 0, 255))

        del img
        return cv_image

    # def wait_img_update(self):
    #     while not self.image_updated:
    #         time.sleep(0.1)

    # def set_environment(self, e_param: np.ndarray) -> None:
    #     if len(e_param) != len(e_param_list)+len(e_img_updater_list):
    #         raise ValueError("Input environmental parameters doesn't match existing handlers")

    #     for i in range(len(e_param_list)):
    #         setter = e_param_list[i] 
    #         setter(e_param[i])

    # def apply_img_update(self, img, e_param):
    #     if len(e_param) != len(e_param_list)+len(e_img_updater_list):
    #         raise ValueError("Input environmental parameters doesn't match existing handlers")

    #     for i in range(len(e_img_updater_list)):
    #         updater = e_img_updater_list[i]
    #         e = e_param[i+len(e_param_list)]
    #         img = updater(img, e)

    #     return img

    def convert_to_image(self, world_pos, ego_pos, ego_ori, cameraMatrix):
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

    def  set_percept_validate(self, point, e_param, fn):
        camera_pose = np.zeros((4,4))
        camera_pose[3,3] = 1
        camera_pose[:3,:3] = Rotation.from_euler('xyz',[point[3],point[4],point[5]]).as_matrix()
        camera_pose[:3,3] = point[:3]

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
        dp_trans_info = self.dp_trans_info
        transform = np.array(dp_trans_info['transform'])
        scale_factor = dp_trans_info['scale']
        camera_pose = transform@camera_pose 
        camera_pose[:3,3] *= scale_factor
        camera_pose = camera_pose[:3,:]

        # time.sleep(0.01)
        q = squaternion.Quaternion.from_euler(point[3], point[4], point[5])
        # offset_vec = np.array([-1.1*0.20,0,0.8*0.77])
        aircraft_pos = np.array([point[0], point[1], point[2]]) # x,y,z
        aircraft_ori = [q.x, q.y, q.z, q.w] # x,y,z,w
        # R = Rotation.from_quat(aircraft_ori)
        # aircraft_pos += offset_vec

        keypoint_position_in_image = []
        # plt.imshow(cv_img)
        # plt.show()
        for i in range(len(self.keypoints)):
            image_coordinate = self.convert_to_image(self.keypoints[i], aircraft_pos, aircraft_ori, self.K).flatten()
            if image_coordinate[0] > 640 or image_coordinate[0] < 0 or image_coordinate[1] > 480 or image_coordinate[1] < 0:
                break
            keypoint_position_in_image.append(image_coordinate)
        if len(np.array(keypoint_position_in_image)) < 14:
            return False, None, None
        # print(keypoint_position_in_image)
        # print(np.array(keypoints).shape)
        print("render image")
        img = self.renderer.render(camera_pose)

        print("perform estimation")
        cv_img = self.vision_estimation(img)

        print("return value")
        return True, np.array([
            self.estimated_state[0],
            self.estimated_state[1],
            self.estimated_state[2],
            self.estimated_state[3],
            self.estimated_state[4],
            self.estimated_state[5],
        ]), cv_img

    # def pose_estimation(image_points, object_points, camera_matrix, dist_coeffs=np.zeros((4, 1))):
    #         res = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    #         return res 

    def check_valid_img(self,success, rotation_vector, translation_vector, state_ground_truth, tolr=1):
        if success:
            Rot = cv2.Rodrigues(rotation_vector)[0]
            RotT = np.matrix(Rot).T
            camera_position = -RotT*np.matrix(translation_vector)

            R = Rot
            sin_x = np.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
            singular  = sin_x < 1e-6
            if not singular:
                z1 = np.arctan2(R[2,0], R[2,1])     # around z1-axis
                x = np.arctan2(sin_x,  R[2,2])     # around x-axis
                z2 = np.arctan2(R[0,2], -R[1,2])    # around z2-axis
            else:
                z1 = 0                                         # around z1-axis
                x = np.arctan2(sin_x,  R[2,2])     # around x-axis
                z2 = 0                                         # around z2-axis
            angles = np.array([[z1], [x], [z2]])
            yawpitchroll_angles = -angles
            yawpitchroll_angles[0,0] = (yawpitchroll_angles[0,0] + (5/2)*np.pi)%(2*np.pi) # change rotation sign if needed, comment this line otherwise
            yawpitchroll_angles[1,0] = -(yawpitchroll_angles[1,0]+np.pi/2)

            # yawpitchroll_angles[0,0] = yawpitchroll_angles[0,0]%pi
            # if yawpitchroll_angles[0,0] > pi:
            #     yawpitchroll_angles[0,0] = yawpitchroll_angles[0,0]%(pi*2)-2*pi
            # yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]%pi
            # if yawpitchroll_angles[1,0] > pi:
            #     yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]%(pi*2)-2*pi
            # yawpitchroll_angles[2,0] = yawpitchroll_angles[2,0]%pi
            # if yawpitchroll_angles[2,0] > pi:
            #     yawpitchroll_angles[2,0] = yawpitchroll_angles[2,0]%(pi*2)-2*pi

            estimated_state = [camera_position[0].item(), camera_position[1].item(), camera_position[2].item(), yawpitchroll_angles[2,0]-np.pi, yawpitchroll_angles[1,0], yawpitchroll_angles[0,0]]
            estimation_error = np.linalg.norm(np.array(estimated_state) - np.array(state_ground_truth))
            print("State estimation: ", estimated_state)
            print("State ground truth: ", state_ground_truth)
            if 0.5*(abs(estimated_state[0] - state_ground_truth[0]) + abs(estimated_state[1] - state_ground_truth[1]) + abs(estimated_state[2] - state_ground_truth[2])) + (abs(estimated_state[3] - state_ground_truth[3]) + abs(estimated_state[4] - state_ground_truth[4]) + abs(estimated_state[5] - state_ground_truth[5])) > 1:
                return False, estimation_error, estimated_state
            return True, estimation_error, estimated_state
        else:
            return False, np.inf, []



    def set_percept(self, point: np.ndarray, e_param: np.ndarray, fn = None) -> np.ndarray:
        # Set aircraft to pos
        camera_pose = np.zeros((4,4))
        camera_pose[3,3] = 1
        camera_pose[:3,:3] = Rotation.from_euler('xyz',[point[5],point[4],point[3]]).as_matrix()
        camera_pose[:3,3] = point[:3]

        # Convert camera pose to what's stated in transforms_orig.json
        tmp = Rotation.from_euler('zyx',[-np.pi/2,np.pi/2,0]).as_matrix()
        mat = camera_pose[:3,:3]@tmp 
        camera_pose[:3,:3] = mat 
        
        # Convert camera pose to Colmap frame in transforms.json
        camera_pose[0:3,1:3] *= -1
        camera_pose = camera_pose[np.array([0,2,1,3]),:]
        camera_pose[2,:] *= -1 

        transform = np.array(self.dp_trans_info['transform'])
        scale_factor = self.dp_trans_info['scale']
        camera_pose = transform@camera_pose 
        camera_pose[:3,3] *= scale_factor
        camera_pose = camera_pose[:3,:]

        img = self.renderer.render(camera_pose)

        # img = self.image
        # img = self.apply_img_update(img, e_param)
        cv_img = self.vision_estimation(img)
        if fn is not None:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(fn, cv_img)

        # x, y, z, yaw, pitch, roll
        estimated_state = np.array([
            self.estimated_state[0],
            self.estimated_state[1],
            self.estimated_state[2],
            self.estimated_state[3],
            self.estimated_state[4],
            self.estimated_state[5],
        ])

        # if np.linalg.norm(estimated_state - point) > 50:
        #     print(">>>>>> Estimated Corrupted ", estimated_state)
        #     estimated_state = point 
        #     self.error_idx.append(self.idx)

        del img
        
        return estimated_state

    def set_percept_output(self, point: np.ndarray, e_param: np.ndarray, fn = None) -> np.ndarray:
        # Set aircraft to pos
        camera_pose = np.zeros((4,4))
        camera_pose[3,3] = 1
        camera_pose[:3,:3] = Rotation.from_euler('xyz',[point[5],point[4],point[3]]).as_matrix()
        camera_pose[:3,3] = point[:3]

        # Convert camera pose to what's stated in transforms_orig.json
        tmp = Rotation.from_euler('zyx',[-np.pi/2,np.pi/2,0]).as_matrix()
        mat = camera_pose[:3,:3]@tmp 
        camera_pose[:3,:3] = mat 
        
        # Convert camera pose to Colmap frame in transforms.json
        camera_pose[0:3,1:3] *= -1
        camera_pose = camera_pose[np.array([0,2,1,3]),:]
        camera_pose[2,:] *= -1 

        transform = np.array(self.dp_trans_info['transform'])
        scale_factor = self.dp_trans_info['scale']
        camera_pose = transform@camera_pose 
        camera_pose[:3,3] *= scale_factor
        camera_pose = camera_pose[:3,:]

        img = self.renderer.render(camera_pose)

        # img = self.image
        # img = self.apply_img_update(img, e_param)
        cv_img = self.vision_estimation(copy.deepcopy(img))
        if fn is not None:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(fn, cv_img)

        # x, y, z, yaw, pitch, roll
        estimated_state = np.array([
            self.estimated_state[0],
            self.estimated_state[1],
            self.estimated_state[2],
            self.estimated_state[3],
            self.estimated_state[4],
            self.estimated_state[5],
        ])

        # if np.linalg.norm(estimated_state - point) > 50:
        #     print(">>>>>> Estimated Corrupted ", estimated_state)
        #     estimated_state = point 
        #     self.error_idx.append(self.idx)

        # del img
        
        return estimated_state, img, cv_img

    def set_pos(self, point: np.ndarray) -> np.ndarray:
        # Set aircraft to given pose
        init_msg = create_state_msd(point[0], point[1], point[2], point[5], -point[4], point[3])
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            # Set initial state.
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(init_msg)
        except rospy.ServiceException:
            print("Service call failed")

        self.image_updated = False

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

def sample_box(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    res = np.random.uniform(lb, ub)
    return res 

def sample_pose() -> np.ndarray:
    x_idx = np.random.uniform(0,1)
    x_idx = np.sqrt(x_idx)
    x = -(x_idx *1000+2000) 
    y_r_max = -0.06*(x-(-3000))+100
    y = np.random.uniform(-y_r_max, y_r_max)
    z_center = -0.07*(x+3000)+120
    z_r = np.random.uniform(-10, 10)
    z = z_center + z_r
    roll = 0 
    pitch_r = -0.003*(x-(-3000))+6
    pitch = np.deg2rad(-3 + np.random.uniform(-pitch_r, pitch_r))
    yaw_r = -0.003*(x-(-3000))+10
    yaw = np.deg2rad(np.random.uniform(-yaw_r, yaw_r))
    return [x,y,z,yaw, pitch,roll]

def sample_2X0():
    lb, ub, Elb, Eub, Ermax = (
        [-3000,-20,110, 0.0012853, 0.0396328, -0.0834173],
        [-2500,20,130, 0.0012853, 0.0396328, -0.0834173],
        [0.0, 0.0],
        [0.05, 0.05],
        [0.01, 0.01],
    )

    x = sample_pose()
    Ec = sample_box(Elb, Eub)
    Er1 = np.random.uniform(0, Ermax)
    for i in range(len(Er1)):
        if Ec[i]-Er1[i]<Elb[i]:
            Er1[i] = Ec[i]-Elb[i]
        elif Ec[i]+Er1[i]>Eub[i]:
            Er1[i] = Eub[i]-Ec[i]

    Er2 = np.random.uniform(0, Ermax)
    for i in range(len(Er2)):
        if Ec[i]-Er2[i]<Elb[i]:
            Er2[i] = Ec[i]-Elb[i] 
        elif Ec[i]+Er2[i]>Eub[i]:
            Er2[i] = Eub[i]-Ec[i]
    return (x, Ec, Er1), (x, Ec, Er2)

def sample_X0() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lb, ub, Elb, Eub, Ermax = (
        [-3000,-20,110, 0.0012853, 0.0396328, -0.0834173],
        [-2500,20,130, 0.0012853, 0.0396328, -0.0834173],
        [0.0, 0.0],
        [1.0, 0.03],
        [0.0, 0.0],
    )

    # x = sample_box(lb, ub)
    x = sample_pose()
    Ec = sample_box(Elb, Eub)
    Er1 = np.random.uniform(0, Ermax)
    for i in range(len(Er1)):
        if Ec[i]-Er1[i]<Elb[i]:
            Er1[i] = Ec[i]-Elb[i]
        elif Ec[i]+Er1[i]>Eub[i]:
            Er1[i] = Eub[i]-Ec[i]
    return (x, Ec, Er1)

def sample_x0(X0: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    x, Ec, Er = X0
    e = sample_box(Ec-Er, Ec+Er)
    return (x, e)

def simulate(x0: Tuple[np.ndarray, np.ndarray], perception: Perception):
    x, e = x0 
    res = perception.set_percept(x, e)
    return res 

def simulate2(x0: Tuple[np.ndarray, np.ndarray], perception: Perception):
    x, e = x0 
    res = perception.set_percept(x, e)
    return res 

def get_init_center(X0: Tuple[np.ndarray,np.ndarray,np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    x, Ec, Er = X0 
    return (x, Ec)

if __name__ == "__main__":
    perception = Perception()

    x0 = np.array([-2.95301775e+03,  9.25878974e+01,  1.16544436e+02,  7.67133510e-02,
 -1.34227392e-01,  0.00000000e+00])
    e = np.array([0,0])
    res = simulate((x0, e), perception)

    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # config_path = os.path.join(script_dir,'../outputs/gazebo4_resampled3_dataset/splatfacto/2024-08-19_160517/config.yml')
    # perception.renderer = SplatRenderer(
    #         config_path,
    #         2560,
    #         1440,
    #         2342.8638660735796,
    #         2342.8638660735796
    #     )

    # x,y,z,roll,pitch,yaw = [-643.5870811335876,-1315.487922957998,986.3821236027896,-3.27806592e-17,  5.61033194e-01,  2.14789038e+00]

    # x = [x,y,z,yaw,pitch,roll]
    # e = [0.5, 0.03]
    # x0 = (x,e)

    # res = perception.set_percept(x, e, fn='res.png')
 