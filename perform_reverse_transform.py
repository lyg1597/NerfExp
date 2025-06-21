import numpy as np 
from scipy.spatial.transform import Rotation
import json 
import os 
import pyvista as pv
from scipy.optimize import least_squares
script_dir = os.path.dirname(os.path.realpath(__file__))

def add_camera_frames(
    plotter: pv.Plotter,
    poses: np.ndarray,
    color_x="r",
    color_y="g",
    color_z="b",
    axis_length: float = 0.05,
    line_width: int | float = 2,
) -> None:
    """
    Draw a small coordinate frame (x-, y-, z-axes) for each camera pose.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The figure to which frames will be added.
    poses : ndarray, shape (N, 4, 4)
        Homogeneous transforms, each mapping camera → world coordinates.
    color_x, color_y, color_z : str | tuple, optional
        Colours for the x, y, z axes (any PyVista/Matplotlib colour spec).
    axis_length : float, optional
        Length of each axis arrow/line, in world units.
    line_width : int | float, optional
        Thickness of the axis lines.
    """
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("`poses` must have shape (N, 4, 4)")

    # Pre-compute 3-D unit vectors for each axis (columns 0–2 of R)
    for T in poses:
        origin = T[:3, 3]
        x_tip = origin + axis_length * T[:3, 0]
        y_tip = origin + axis_length * T[:3, 1]
        z_tip = origin + axis_length * T[:3, 2]

        # Draw the three coloured axes
        plotter.add_mesh(pv.Line(origin, x_tip), color=color_x, line_width=line_width)
        plotter.add_mesh(pv.Line(origin, y_tip), color=color_y, line_width=line_width)
        plotter.add_mesh(pv.Line(origin, z_tip), color=color_z, line_width=line_width)

def perform_inverse_transform(transform_matrix:np.ndarray):
    transform_matrix[2,:]*=-1
    transform_matrix = transform_matrix[np.array([0,2,1,3]), :]
    transform_matrix[0:3, 1:3] *= -1
    R = Rotation.from_euler('zyx',[-np.pi/2, np.pi/2, 0]).as_matrix()
    # transform_matrix[:3,:3] = transform_matrix[:3,:3]@np.linalg.inv(R)
    return transform_matrix

def solve_T_s(a: np.ndarray, b: np.ndarray,
              init_s: float | None = None) -> tuple[np.ndarray, float]:
    """
    Estimate T (4×4) and scalar s such that T @ (a with scaled translation) ≈ b.

    Parameters
    ----------
    a, b : ndarray, shape (N, 4, 4)
        Homogeneous transforms for two coordinate systems.
    init_s : float, optional
        Initial guess for the scale (defaults to 1.0 or the ratio of median
        translation norms).

    Returns
    -------
    T_hat : ndarray, shape (4, 4)
        Estimated similarity transform.
    s_hat : float
        Estimated translation scale factor.
    """
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.shape[1:] == (4, 4)
    N = a.shape[0]

    # Extract rotations and translations
    Ra = a[:, :3, :3]
    ta = a[:, :3, 3]
    Rb = b[:, :3, :3]
    tb = b[:, :3, 3]

    # ---------- initial guesses ------------------------------------------------
    # Rotation: average of R_b R_aᵀ (projected back to SO(3))
    R_init = Rotation.from_matrix(Rb @ np.transpose(Ra, (0, 2, 1))).mean()
    r_init = R_init.as_rotvec()
    # Translation: rough by aligning centroids (with s ≈ 1)
    if init_s is None:
        init_s = np.median(np.linalg.norm(tb, axis=1) /
                           np.maximum(np.linalg.norm(ta, axis=1), 1e-9))
    t_init = tb.mean(axis=0) - R_init.apply(init_s * ta.mean(axis=0))

    # Parameter vector p = [rx, ry, rz, tx, ty, tz, s]
    p0 = np.hstack([r_init, t_init, init_s])

    # ---------- residual --------------------------------------------------------
    def residual(p):
        r_vec, t_vec, s = p[:3], p[3:6], p[6]
        R = Rotation.from_rotvec(r_vec).as_matrix()
        # orientation residual: flatten matrices
        R_pred = R @ Ra                        # (N, 3, 3)
        orient_res = (R_pred - Rb).reshape(N, -1)
        # translation residual
        trans_pred = t_vec + (R @ (ta.T * s)).T
        trans_res = trans_pred - tb
        return np.concatenate([orient_res, trans_res], axis=1).ravel()

    # ---------- solve -----------------------------------------------------------
    res = least_squares(residual, p0, method="lm")   # Levenberg–Marquardt

    r_opt, t_opt, s_opt = res.x[:3], res.x[3:6], res.x[6]
    R_opt = Rotation.from_rotvec(r_opt).as_matrix()

    T_opt = np.eye(4)
    T_opt[:3, :3] = R_opt
    T_opt[:3, 3] = t_opt
    return T_opt, s_opt

def a_to_b(a, T, s):
    """
    Apply T and scale s to take poses in frame A → frame B.

    b_pred_i = T @ [ R_a_i | s * t_a_i ]
    """
    a_scaled = a.copy()
    a_scaled[:, :3, 3] *= s               # scale the translation only
    return (T @ a_scaled)                 # broadcasted matrix product

def b_to_a(b, T, s):
    """
    Inverse mapping: recover A-frame poses from B-frame ones.

    a_pred_i = inv(T) @ b_i;  then un-scale the translation.
    """
    T_inv = np.linalg.inv(T)
    a_pred = T_inv @ b
    a_pred[:, :3, 3] /= s                 # undo the scale on translation
    return a_pred

if __name__ == "__main__":
    transform_gt_fn = os.path.join(script_dir, 'data/kaliber9/transforms.json')

    transform_colmap_fn = os.path.join(script_dir, 'data/kaliber10/transforms.json')

    with open(transform_gt_fn, 'r') as f:
        transform_gt_json = json.load(f)

    with open(transform_colmap_fn, 'r') as f:
        transform_colmap_json = json.load(f)

    transform_gt = transform_gt_json['frames']
    transform_colmap = transform_colmap_json['frames']
    recovered_transform_gt_list = []
    recovered_transform_colmap_list = []
    for i in range(len(transform_colmap)):
        transform_matrix_gt = np.array(transform_gt[i]['transform_matrix'])
        transform_matrix_colmap = np.array(transform_colmap[i]['transform_matrix'])
        
        applied_transform_colmap = np.linalg.inv(np.array([
            [1,0,0,0],
            [0,0,1,0],
            [0,-1,0,0],
            [0,0,0,1]
        ]))

        transform_matrix_colmap = applied_transform_colmap@transform_matrix_colmap

        recovered_transform_gt = perform_inverse_transform(transform_matrix_gt)
        recovered_transform_colmap = perform_inverse_transform(transform_matrix_colmap)

        recovered_transform_gt_list.append(recovered_transform_gt)
        recovered_transform_colmap_list.append(recovered_transform_colmap)

    recovered_transform_gt_array = np.array(recovered_transform_gt_list)
    recovered_transform_colmap_array = np.array(recovered_transform_colmap_list)

    T_hat, s_hat = solve_T_s(recovered_transform_gt_array, recovered_transform_colmap_array)
    print(T_hat, s_hat)

    recovered_transform_gt_array_transformed = a_to_b(recovered_transform_gt_array, T_hat, s_hat)
    
    plotter = pv.Plotter(shape=(1, 3))
    plotter.subplot(0,0)
    plotter.add_text('GT', font_size=30)
    add_camera_frames(plotter, recovered_transform_gt_array, axis_length=0.02)
    plotter.add_mesh(pv.Line([0,0,0], [1,0,0]), color='r')
    plotter.add_mesh(pv.Line([0,0,0], [0,1,0]), color='g')
    plotter.add_mesh(pv.Line([0,0,0], [0,0,1]), color='b')

    plotter.subplot(0,1)
    plotter.add_text('GT transformed', font_size=30)
    add_camera_frames(plotter, recovered_transform_gt_array_transformed)
    plotter.add_mesh(pv.Line([0,0,0], [1,0,0]), color='r')
    plotter.add_mesh(pv.Line([0,0,0], [0,1,0]), color='g')
    plotter.add_mesh(pv.Line([0,0,0], [0,0,1]), color='b')
        
    plotter.subplot(0,2)
    plotter.add_text('COLMAP', font_size=30)
    add_camera_frames(plotter, recovered_transform_colmap_array)
    plotter.add_mesh(pv.Line([0,0,0], [1,0,0]), color='r')
    plotter.add_mesh(pv.Line([0,0,0], [0,1,0]), color='g')
    plotter.add_mesh(pv.Line([0,0,0], [0,0,1]), color='b')
    
    plotter.show()