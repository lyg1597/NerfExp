import os
import re
import math
import numpy as np
import pyvista as pv


# ───────────────────────── file-pairing helper ─────────────────────────
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


# ───────────────────────── orientation helpers ─────────────────────────
def load_odom(path):
    vals = np.loadtxt(path)
    return vals[:3], vals[3:7]                       # position, quaternion


def quat_to_R(q):
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])


# --- global roll / pitch / yaw offsets (degrees) -----------------------
ROLL_OFF_DEG  = 0.0
PITCH_OFF_DEG = -5.0          # tilt down 5°
YAW_OFF_DEG   = 0.0

def euler_offset_R(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r),  math.cos(r)]])
    Ry = np.array([[ math.cos(p), 0, math.sin(p)],
                   [0,           1, 0],
                   [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0],
                   [math.sin(y),  math.cos(y), 0],
                   [0,           0,           1]])
    # intrinsic XYZ (roll→pitch→yaw)
    return Rz @ Ry @ Rx

R_OFF = euler_offset_R(ROLL_OFF_DEG, PITCH_OFF_DEG, YAW_OFF_DEG)
# -----------------------------------------------------------------------


# ─────────────────────────── visualisation ─────────────────────────────
def visualize(folder,
              axis_len_world=0.2,     # length of global reference axes
              cam_axis_len=0.05,      # length of each local camera axis
              point_radius=0.01):     # radius of the red camera-origin sphere

    pairs = match_rgb_odom(folder)
    if not pairs:
        raise RuntimeError("No rgb/odom matches found.")

    p = pv.Plotter()
    p.background_color = "white"

    # Global XYZ reference axes (world frame)
    p.add_mesh(pv.Line((0, 0, 0), (axis_len_world, 0, 0)), color="red",   line_width=4)
    p.add_mesh(pv.Line((0, 0, 0), (0, axis_len_world, 0)), color="green", line_width=4)
    p.add_mesh(pv.Line((0, 0, 0), (0, 0, axis_len_world)), color="blue",  line_width=4)

    for i, (_, odom_path) in enumerate(pairs):           # rgb_path unused here
        pos, quat = load_odom(odom_path)

        if i%20!=0 or pos.size!=3 or quat.size!=4:
            continue 

        # Full camera rotation with –5° pitch offset applied
        R_cam = quat_to_R(quat)
        R_final = R_cam @ R_OFF

        # Local axes endpoints
        x_end = pos + R_final @ np.array([cam_axis_len, 0, 0])
        y_end = pos + R_final @ np.array([0, cam_axis_len, 0])
        z_end = pos + R_final @ np.array([0, 0, cam_axis_len])

        # Draw camera-frame axes (short lines)
        p.add_mesh(pv.Line(pos, x_end), color="red",   line_width=2)
        p.add_mesh(pv.Line(pos, y_end), color="green", line_width=2)
        p.add_mesh(pv.Line(pos, z_end), color="blue",  line_width=2)

        # Red point marking the camera position
        p.add_mesh(pv.Sphere(radius=point_radius, center=pos), color="red")

    p.show(cpos="xy")                     # top-down view


if __name__ == "__main__":
    visualize("gsplat_recording_long_2")     # ← set your data folder
