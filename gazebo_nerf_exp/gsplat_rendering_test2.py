import torch 
from nerfstudio.models.splatfacto import SplatfactoModel
# from nerfstudio.utils import load_config, load_model
# from nerfstudio
from scipy.spatial.transform import Rotation as R 
import cv2 
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps
import numpy as np 
import os 
from pathlib import Path
import matplotlib.pyplot as plt 
from ns_renderer import SplatRenderer

if __name__ == "__main__":
    # Sampled camera pose in World (Gazebo) frame
    state_rand = np.array([-643.5870811335876, -1315.487922957998, 986.3821236027896, 0.45802836116014156, -0.24337730962498508, 0.13196425582817917, 0.8447265478936802])

    renderer = SplatRenderer(
        '../outputs/gazebo4_transformed/splatfacto/2024-08-05_204928/config.yml', 
        2560, 
        1440, 
        2343.0242837919386, 
        2343.0242837919386
    )

    camera_pose = np.zeros((4,4))
    camera_pose[3,3] = 1
    camera_pose[:3,:3] = R.from_quat(state_rand[[4,5,6,3]]).as_matrix()
    camera_pose[:3,3] = state_rand[:3]

    # Convert camera pose to what's stated in transforms_orig.json
    tmp = R.from_euler('zyx',[-np.pi/2,np.pi/2,0]).as_matrix()
    mat = camera_pose[:3,:3]@tmp 
    camera_pose[:3,:3] = mat 
    
    # Convert camera pose to Colmap frame in transforms.json
    camera_pose[0:3,1:3] *= -1
    camera_pose = camera_pose[np.array([0,2,1,3]),:]
    camera_pose[2,:] *= -1 

    # Convert colmap pose to nerfstudio pose 
    # Only apply to outputs/gazebo4_transformed/splatfacto/2024-08-05_204928
    dp_trans_info = {
    "transform": [
        [
            0.9947746992111206,
            -0.07210789620876312,
            -0.07227572053670883,
            1278.867919921875
        ],
        [
            -0.07210789620876312,
            0.004931271076202393,
            -0.9973846673965454,
            -75.30574035644531
        ],
        [
            0.07227572053670883,
            0.9973846673965454,
            -0.00029408931732177734,
            -566.6797485351562
        ]
    ],
    "scale": 0.00042486766191778165
    }
    transform = np.array(dp_trans_info['transform'])
    scale_factor = dp_trans_info['scale']
    camera_pose = transform@camera_pose 
    camera_pose[:3,3] *= scale_factor
    camera_pose = camera_pose[:3,:]

    # rospy.wait_for_service('/gazebo/set_model_state')
    # try:
    #     set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    #     _ = set_state( state_msg )
    # except rospy.ServiceException:
    #     print("Service call failed")

    cv_img = renderer.render(camera_pose)
    plt.imshow(cv_img)
    plt.show()