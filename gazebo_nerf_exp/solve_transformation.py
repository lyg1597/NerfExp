# Solve the transformation from ground truth frame to colmap frame 

import numpy as np 
from scipy.optimize import minimize  
import pandas as pd
import os 
import json 
from typing import List 
import re
from scipy.spatial.transform import Rotation as R

script_dir = os.path.dirname(os.path.realpath(__file__))

def norm_path(path):
    return os.path.normpath(os.path.abspath(path))

def find_path_index(paths: List[str], target_path:str) -> int:
    normalized_target = norm_path(target_path)

    try:
        return paths.index(normalized_target)
    except ValueError:
        return None

def objective_function(x, res_list: List):
    # tf@gazebo_pose = colmap_pose
    tf = np.zeros((4,4))
    tf[3,3] = 1
    tf[:3,:] = np.reshape(x, (3,4))
    res = 0
    for elem in res_list:
        gazebo_pose = elem['gazebo_pose']
        colmap_pose = elem['colmap_pose']

        res += np.linalg.norm(colmap_pose - tf@gazebo_pose)
    return res 

if __name__ == "__main__":
    image_matching_fn = os.path.join(script_dir, 'matches.csv') 
    image_matching = pd.read_csv(image_matching_fn) 
    input_fns = [norm_path(elem) for elem in image_matching['input_fns']]
    output_fns = [norm_path(elem) for elem in image_matching['output_fns']]

    gazebo_pose_fn = os.path.join(script_dir, '../gazebo3/pose.csv')
    gazebo_pose = pd.read_csv(gazebo_pose_fn)

    colmap_dir = os.path.join(script_dir, '../data/gazebo3')
    colmap_pose_fn = os.path.join(colmap_dir, 'transforms.json')
    with open(colmap_pose_fn, 'r') as f:
        colmap_pose = json.load(f)

    # Match pairs of colmap poses and gazebo poses 
    matched_list = []
    for i in range(len(colmap_pose['frames'])):
        res_dict = {}
        frame = colmap_pose['frames'][i]
        output_pose = np.array(frame['transform_matrix'])
        colmap_image_fn = frame['file_path']
        image_path = os.path.join(colmap_dir, colmap_image_fn)
        idx = find_path_index(output_fns, image_path)
        input_fn = input_fns[idx]
        match = re.search(r'image_(\d+)\.png', input_fn)
        input_idx = int(match.group(1))
        input_tx = gazebo_pose['x'][input_idx]
        input_ty = gazebo_pose['y'][input_idx]
        input_tz = gazebo_pose['z'][input_idx]
        input_rx = gazebo_pose['qx'][input_idx]
        input_ry = gazebo_pose['qy'][input_idx]
        input_rz = gazebo_pose['qz'][input_idx]
        input_rw = gazebo_pose['qw'][input_idx]
        input_pose = np.zeros((4,4))
        input_pose[3,3] = 1 
        input_pose[:3,3] = np.array([input_tx, input_ty, input_tz])
        input_pose[:3,:3] = R.from_quat((input_rx, input_ry, input_rz, input_rw)).as_matrix()
        res_dict['gazebo_fn'] = input_fn 
        res_dict['colmap_fn'] = image_path
        res_dict['gazebo_pose'] = input_pose 
        res_dict['colmap_pose'] = output_pose
        matched_list.append(res_dict)

    tmp = np.eye(4)
    x0 = tmp[:3, :].flatten()
    res = minimize(
        objective_function, 
        x0,
        (matched_list, )
    )

    if res.success:
        print(res.x)