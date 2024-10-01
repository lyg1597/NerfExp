import json 
import csv 
import os 
import numpy as np 
from scipy.spatial.transform import Rotation
from PIL import Image
import shutil 

def scale_down_image(img, output_path, factor = 2):
    # with Image.open(image_path) as img:
    # Get current size
    width, height = img.size
    # Scale down by half
    new_width, new_height = width // factor, height // factor
    # Resize the image
    img_resized = img.resize((new_width, new_height))
    # Save the resized image
    img_resized.save(output_path)

script_dir = os.path.dirname(os.path.realpath(__file__))

output_dir = os.path.join(script_dir, 'gazebo4_resampled4_dataset')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_json_fn = os.path.join(output_dir, 'transforms_orig.json')

output_img_dir = os.path.join(output_dir, './images')
if not os.path.exists(output_img_dir):
    os.mkdir(output_img_dir)
output_img2_dir = os.path.join(output_dir, './images_2')
if not os.path.exists(output_img2_dir):
    os.mkdir(output_img2_dir)
output_img4_dir = os.path.join(output_dir, './images_4')
if not os.path.exists(output_img4_dir):
    os.mkdir(output_img4_dir)
output_img8_dir = os.path.join(output_dir, './images_8')
if not os.path.exists(output_img8_dir):
    os.mkdir(output_img8_dir)

input_dir = os.path.join(script_dir, 'gazebo')
input_img_dir = os.path.join(input_dir, 'images')
input_csv = os.path.join(input_dir, 'pose.csv')

res_dict = {
    "w": 640,
    "h": 480,
    "fl_x": 1253.2215566867008,
    "fl_y": 1253.2215566867008,
    "cx": 320.5,
    "cy": 240.5,
    "k1": 0,
    "k2": 0,
    "p1": 0,
    "p2": 0,
    "applied_transform": [
        [
            1,
            0,
            0,
            0
        ],
        [
            0,
            1,
            0,
            0
        ],
        [
            0,
            0,
            1,
            0
        ]
    ],
    "ply_file_path": "sparse_pc.ply",
    "camera_model": "OPENCV",
    "frames": []
}

data = np.genfromtxt(input_csv, dtype=float, delimiter=',', names = True)

frames = []
for i in range(data.shape[0]):
    print(i)
    x,y,z,qw,qx,qy,qz = data[i]
    input_img_fn = os.path.join(input_img_dir, f'image_{i:05d}.png')
    pos_vector = np.array([x,y,z])
    rot_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    R = Rotation.from_euler('yzx',[np.pi/2, 0, -np.pi/2]).as_matrix()
    rot_matrix = rot_matrix@R
    transform_matrix = np.zeros((4,4))
    transform_matrix[:3,:3] = rot_matrix
    transform_matrix[:3,3] = pos_vector 
    transform_matrix[3,3] = 1
    frame = {}
    frame['file_path'] = f'images/frames_{i+1:05d}.png'
    frame['colmap_im_id'] = i+1
    frame['original_fn'] = os.path.normpath(input_img_fn)
    frame['transform_matrix'] = transform_matrix.tolist()
    frames.append(frame)

    output_img_fn = os.path.join(output_img_dir, f'frames_{i+1:05d}.png')
    shutil.copyfile(input_img_fn, output_img_fn)
    img = Image.open(input_img_fn)
    output_img2_fn = os.path.join(output_img2_dir, f'frames_{i+1:05d}.png')
    scale_down_image(img, output_img2_fn, 2)
    output_img4_fn = os.path.join(output_img4_dir, f'frames_{i+1:05d}.png')
    scale_down_image(img, output_img4_fn, 4)
    output_img8_fn = os.path.join(output_img8_dir, f'frames_{i+1:05d}.png')
    scale_down_image(img, output_img8_fn, 8)

res_dict['frames'] = frames 
with open(output_json_fn, 'w+') as f:
    json.dump(res_dict, f, indent=4)
