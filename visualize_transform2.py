import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from PIL import Image
import shutil 
import json 

import numpy as np
from scipy.spatial.transform import Rotation


def odom_to_colmap(pos_xyz, quat_c2w, world2colmap=None):
    """
    Convert a camera-to-world pose into COLMAP's world-to-camera convention.

    Parameters
    ----------
    pos_xyz : (3,) array-like
        Camera position C = (x, y, z) in world coordinates.
    quat_c2w : (4,) array-like
        Quaternion (qx, qy, qz, qw) rotating *camera*-frame into *world*-frame.
        Hamilton convention (right-handed, scalar last).
    world2colmap : (3,3) array-like, optional
        If your odometry world frame differs from COLMAP's (+X right, +Y down,
        +Z forward), pass a transform that converts odom-world axes into
        COLMAP-world axes (e.g. np.diag([1, -1, -1]) for ROS → COLMAP).

    Returns
    -------
    quat_w2c : np.ndarray, shape (4,)
        Quaternion (qw, qx, qy, qz) that rotates *world*-frame into *camera*-frame.
    trans_w2c : np.ndarray, shape (3,)
        Translation T = (Tx, Ty, Tz) s.t. X_cam = R_wc · X_world + T.
    """
    pos_xyz = np.asarray(pos_xyz, dtype=float)
    qx, qy, qz, qw = quat_c2w
    R_c2w = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    # Optional axis conversion
    if world2colmap is not None:
        R_c2w = world2colmap @ R_c2w
        pos_xyz = world2colmap @ pos_xyz

    # Invert the pose ---------------------------------------------------------
    R_w2c = R_c2w.T                      # rotation inverse
    trans_w2c = -R_w2c @ pos_xyz         # translation inverse

    # Quaternion for R_w2c, reordered to (w, x, y, z) as COLMAP wants
    quat_xyzw = Rotation.from_matrix(R_w2c).as_quat()
    quat_wxyz = np.roll(quat_xyzw, 1)    # (x,y,z,w) → (w,x,y,z)

    return quat_wxyz, trans_w2c


def make_pose_prior_line(image_name, pos_xyz, quat_c2w, world2colmap=None):
    """Return a ready-to-write line for pose_priors.txt."""
    q, t = odom_to_colmap(pos_xyz, quat_c2w, world2colmap)
    return f"{image_name} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} " \
           f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}"


# # ---------------------------------------------------------------------------
# # Example
# if __name__ == "__main__":
#     # identity pose, camera at (1,2,0.5) in metres
#     pos = (1.0, 2.0, 0.5)
#     quat_c2w = (0.0, 0.0, 0.0, 1.0)  # (qx,qy,qz,qw)
#     print(make_pose_prior_line("frame_000000.png", pos, quat_c2w))


# ────────────────────────────── helpers ──────────────────────────────
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

def match_rgb_odom(folder_path):
    pat = re.compile(r'^(rgb|odom)_(\d+\.\d+)\.(jpg|txt)$')
    mapping = {}
    for fname in os.listdir(folder_path):
        m = pat.match(fname)
        if not m:
            continue
        typ, idx, _ = m.groups()
        mapping.setdefault(idx, {})[typ] = os.path.join(folder_path, fname)

    pairs = [
        (d['rgb'], d['odom']) for d in mapping.values()
        if 'rgb' in d and 'odom' in d
    ]
    pairs.sort(key=lambda p: float(os.path.basename(p[0]).split('_')[1].split('.')[0]))
    return pairs

def quaternion_to_matrix(q):
    """qx,qy,qz,qw  →  3×3 rotation matrix"""
    return Rotation.from_quat(q).as_matrix()

def euler_offset_matrix(roll_deg=0.0, pitch_deg=-5.0, yaw_deg=0.0):
    """Intrinsic XYZ (roll→pitch→yaw) offset in degrees → 3×3 matrix."""
    return Rotation.from_euler('xyz',
                               [np.radians(roll_deg),
                                np.radians(pitch_deg),
                                np.radians(yaw_deg)],
                               degrees=False).as_matrix()

R_OFF = euler_offset_matrix(0.0, -19.0, 0.0)   # global –19° pitch

# ─────────────────────── pyramid utilities (unchanged) ───────────────────────
def create_pyramid(position, rotation_matrix):
    base_size = 0.1
    height = 0.15
    base_vertices = np.array([
        [-base_size*2, -base_size, 0],
        [ base_size*2, -base_size, 0],
        [ base_size*2,  base_size, 0],
        [-base_size*2,  base_size, 0]
    ])
    apex = np.array([0, 0, -height])

    # rotate and translate
    rotated_base  = base_vertices @ rotation_matrix.T
    rotated_apex  = apex          @ rotation_matrix.T
    return rotated_base + position, rotated_apex + position

def plot_pyramid(ax, base_vertices, apex):
    base = Poly3DCollection([base_vertices], color='cyan', alpha=0.6)
    ax.add_collection3d(base)
    # four sides
    for i in range(4):
        face = [[apex,
                 base_vertices[i],
                 base_vertices[(i + 1) % 4]]]
        side = Poly3DCollection(face, color='blue', alpha=0.6)
        ax.add_collection3d(side)
    ax.scatter(*apex, color='red')

# ─────────────────────────────── main ────────────────────────────────
def visualize(folder, stride=40):
    odom_files = match_rgb_odom(folder)
    if not odom_files:
        raise RuntimeError("No odom_*.txt files found in", folder)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    poses = []
    fns = []
    pos_lines = []
    for idx, (img_path, odom_path) in enumerate(odom_files):
        if idx % stride:        # skip to lighten the plot
            continue

        vals      = np.loadtxt(odom_path)
        position  = vals[:3]
        quat      = vals[3:]                    # qx qy qz qw

        if position.size!=3 or quat.size!=4:
            continue 
        R_cam     = quaternion_to_matrix(quat)  # camera→world
        R_final   = R_cam @ R_OFF               # apply global –5° pitch

        R = Rotation.from_euler('zyx',[-np.pi/2, np.pi/2, 0]).as_matrix()
        R_final = R_final@R

        base, apex = create_pyramid(position, R_final)
        plot_pyramid(ax, base, apex)

        poses.append((position, R_final))
        fns.append((img_path, odom_path))
        pos_line = make_pose_prior_line("", position, quat, np.diag([1, -1, -1]))
        pos_lines.append(pos_line)

    # axis labels & limits (adjust to your data scale)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim([-5,  5])
    ax.set_ylim([-5,  5])
    ax.set_zlim([-5, 5])
    plt.tight_layout()
    plt.show()
    return poses, fns, pos_lines

def create_nerfstudio_dataset(output_dir, poses, fns, pose_lines):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images/')
    output_img2_dir = os.path.join(output_dir, 'images_2/')
    output_img4_dir = os.path.join(output_dir, 'images_4/')
    output_img8_dir = os.path.join(output_dir, 'images_8/')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)
    if not os.path.exists(output_img2_dir):
        os.mkdir(output_img2_dir)
    if not os.path.exists(output_img4_dir):
        os.mkdir(output_img4_dir)
    if not os.path.exists(output_img8_dir):
        os.mkdir(output_img8_dir)

    f_pose = open(os.path.join(output_dir, 'init_poses.txt'),'w+')

    output_json_fn = os.path.join(output_dir, 'transforms_orig.json')
    res_dict = {
        "w": 1296,
        "h": 972,
        "fl_x": 624.7025861923801,
        "fl_y": 622.1024899860136,
        "cx": 655.0802437532551,
        "cy": 494.9500170238977,
        "k1": -0.1633556496674091,
        "k2": 0.019164905599367132,
        "p1": 7.588347139885713e-05,
        "p2": -0.00010445711172148531,
        "applied_transform": [
            [
                1,
                0,
                0,
                0
            ],
            [
                0,
                0,
                1,
                0
            ],
            [
                0,
                -1,
                0,
                0
            ]
        ],
        "ply_file_path": "sparse_pc.ply",
        "camera_model": "OPENCV",
        "frames": []
    }

    frames = []
    for i in range(len(poses)):
        
        pos_vector, rot_matrix = poses[i]
        input_img_fn = fns[i][0]
        transform_matrix = np.zeros((4,4))
        transform_matrix[:3,:3] = rot_matrix
        transform_matrix[:3,3] = pos_vector 
        transform_matrix[3,3] = 1
        R2 = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        transform_matrix = R2@transform_matrix
        tmp = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
        transform_matrix = tmp@transform_matrix

        f_pose.write(f"frames_{i+1:05d}.png {pose_lines[i]}\n")

        frame = {}
        frame['file_path'] = f'images/frames_{i+1:05d}.png'
        frame['colmap_im_id'] = i+1
        frame['original_fn'] = os.path.normpath(input_img_fn)
        frame['transform_matrix'] = transform_matrix.tolist()
        # frame['env_params'] = [env1, env2]
        frames.append(frame)

        img = Image.open(input_img_fn)
        output_img_fn = os.path.join(output_img_dir, f'frames_{i+1:05d}.png')
        scale_down_image(img, output_img_fn, 1)
        output_img2_fn = os.path.join(output_img2_dir, f'frames_{i+1:05d}.png')
        scale_down_image(img, output_img2_fn, 2)
        output_img4_fn = os.path.join(output_img4_dir, f'frames_{i+1:05d}.png')
        scale_down_image(img, output_img4_fn, 4)
        output_img8_fn = os.path.join(output_img8_dir, f'frames_{i+1:05d}.png')
        scale_down_image(img, output_img8_fn, 8)

    res_dict['frames'] = frames 
    with open(output_json_fn, 'w+') as f:
        json.dump(res_dict, f, indent=4)
    f_pose.close()

# ─────────────────────────── run it! ────────────────────────────
if __name__ == "__main__":
    output_dir = './big_room_dataset'
    poses, fns, pose_lines = visualize("big_room", stride=6)   # ← change to your odom directory
    create_nerfstudio_dataset(output_dir, poses, fns, pose_lines)
