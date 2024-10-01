import json 
import os 
import shutil 
import copy 
import numpy as np 

script_dir = os.path.dirname(os.path.realpath(__file__))
gazebo_dir = os.path.join(script_dir, './gazebo6_dataset')
boeing_dir = os.path.join(script_dir, './data/boeing_airport4')
output_dir = os.path.join(script_dir, './mixed_dataset')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
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

if __name__ == "__main__":
    output_json = {
        "camera_model": "OPENCV",
        "w": 4096,
        "h": 3000,
        "fl_x": 7371.907949216963,
        "fl_y": 7369.5606333180995,
        "cx": 2010.4485880343118,
        "cy": 1414.4028715939817,
        "k1": -0.005917495376564054,
        "k2": -0.053551366492332504,
        "p1": -0.002083526399758475,
        "p2": -0.002349387620412934,
        "applied_transform": [
            [
                0,
                1,
                0,
                0
            ],
            [
                1,
                0,
                0,
                0
            ],
            [
                0,
                0,
                -1,
                0
            ]
        ],
    }

    count = 0
    all_frames = []

    gazebo_json_fn = os.path.join(gazebo_dir, 'transforms.json')
    with open(gazebo_json_fn, 'r') as f:
        gazebo_json = json.load(f)

    frames = gazebo_json['frames']
    for i in range(len(frames)):
        print(count)
        frame = frames[i]
        img_path:str = frame['file_path']
        img_fn = img_path.split('/')[1]
        
        new_frame = copy.deepcopy(frame)
        new_frame['file_path'] = f'images/frame_{count:06d}.png'
        new_frame['colmap_im_id'] = count
        
        # transform_matrix = np.array(new_frame['transform_matrix'])
        # transform_matrix[0:3, 1:3] *= -1
        # transform_matrix = transform_matrix[np.array([0, 2, 1, 3]), :]
        # transform_matrix[2, :] *= -1
        # new_frame['transform_matrix'] = transform_matrix.tolist()

        all_frames.append(new_frame)
        
        # Copy images
        src = os.path.join(gazebo_dir, f'./images/{img_fn}')
        dest = os.path.join(output_dir, f'./images/frame_{count:06d}.png')
        shutil.copy(src, dest)
        src = os.path.join(gazebo_dir, f'./images_2/{img_fn}')
        dest = os.path.join(output_dir, f'./images_2/frame_{count:06d}.png')
        shutil.copy(src, dest)
        src = os.path.join(gazebo_dir, f'./images_4/{img_fn}')
        dest = os.path.join(output_dir, f'./images_4/frame_{count:06d}.png')
        shutil.copy(src, dest)
        src = os.path.join(gazebo_dir, f'./images_8/{img_fn}')
        dest = os.path.join(output_dir, f'./images_8/frame_{count:06d}.png')
        shutil.copy(src, dest)

        count += 1

    boeing_json_fn = os.path.join(boeing_dir, 'transforms_orig.json')
    with open(boeing_json_fn, 'r') as f:
        boeing_json = json.load(f)

    frames = boeing_json['frames']
    for i in range(len(frames)):
        print(count)
        frame = frames[i]
        img_path:str = frame['file_path']
        img_fn = img_path.split('/')[1]
        
        new_frame = copy.deepcopy(frame)
        new_frame['file_path'] = f'images/frame_{count:06d}.png'
        new_frame['colmap_im_id'] = count
        
        # transform_matrix = np.array(new_frame['transform_matrix'])
        # transform_matrix[0:3, 1:3] *= -1
        # transform_matrix = transform_matrix[np.array([0, 2, 1, 3]), :]
        # transform_matrix[2, :] *= -1
        # new_frame['transform_matrix'] = transform_matrix.tolist()

        all_frames.append(new_frame)
        
        # Copy images
        src = os.path.join(boeing_dir, f'./images/{img_fn}')
        dest = os.path.join(output_dir, f'./images/frame_{count:06d}.png')
        shutil.copy(src, dest)
        src = os.path.join(boeing_dir, f'./images_2/{img_fn}')
        dest = os.path.join(output_dir, f'./images_2/frame_{count:06d}.png')
        shutil.copy(src, dest)
        src = os.path.join(boeing_dir, f'./images_4/{img_fn}')
        dest = os.path.join(output_dir, f'./images_4/frame_{count:06d}.png')
        shutil.copy(src, dest)
        src = os.path.join(boeing_dir, f'./images_8/{img_fn}')
        dest = os.path.join(output_dir, f'./images_8/frame_{count:06d}.png')
        shutil.copy(src, dest)

        count += 1

output_json['frames'] = all_frames
output_json_fn = os.path.join(output_dir, 'transforms_orig.json')
with open(output_json_fn, 'w+') as f:
    json.dump(output_json, f, indent=4)