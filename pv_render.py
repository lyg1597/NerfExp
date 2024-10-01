import pyvista as pv
import numpy as np 
from verse.plotter.plotter3D import plot_line_3d
# Load the ply file
import pickle 
import os 
import polytope as pc 

def plot_polytope_3d(A, b, ax=None, color="red", trans=0.2, edge=True):
    if ax is None:
        ax = pv.Plotter()

    poly = pc.Polytope(A=A, b=b)
    vertices = pc.extreme(poly)
    cloud = pv.PolyData(vertices)
    volume = cloud.delaunay_3d()
    shell = volume.extract_geometry()
    if len(shell.points) <= 0:
        return ax
    ax.add_mesh(shell, opacity=trans, color=color)
    if edge:
        edges = shell.extract_feature_edges(20)
        ax.add_mesh(edges, color="r", line_width=1, opacity=1)
    return ax

def plot3dReachtubeSingle(tube, x_dim, y_dim, z_dim, ax, color, edge=True):
    for i in range(0,len(tube),2):
        box = tube[i:i+2,(x_dim, y_dim, z_dim)]
        # box = [[lb[x_dim], lb[y_dim], lb[z_dim]], [ub[x_dim], ub[y_dim], ub[z_dim]]]
        poly = pc.box2poly(np.array(box).T)
        ax = plot_polytope_3d(poly.A, poly.b, ax=ax, color=color, edge=edge, trans=0.1)
    return ax

def plot_trajectory(traj, x_dim, y_dim, z_dim, ax, color, line_width):
    for i in range(len(traj)-1):
        start = traj[i][(x_dim, y_dim, z_dim),]
        end = traj[i+1][(x_dim, y_dim, z_dim),]
        ax = plot_line_3d(start, end, ax, color, line_width)
    return ax 

point_cloud = pv.read('./exports/pcd/point_cloud.ply')

# Extract RGB values and normalize to [0, 1]
rgb = point_cloud.point_data['RGB'] / 255.0

# Convert to RGBA by adding an alpha channel
rgba = np.c_[rgb, np.ones(rgb.shape[0])*200]

# Set the RGBA colors to the mesh
point_cloud.point_data['RGBA'] = rgba


# Visualize the point cloud
fig = pv.Plotter()


with open('./NeRF_UAV_simulation/verse/vcs_sim_exp1_unsafe.pickle','rb') as f:
    state_list = pickle.load(f)
with open('./NeRF_UAV_simulation/verse/vcs_estimate_exp1_unsafe.pickle','rb') as f:
    est_list = pickle.load(f)
with open('./NeRF_UAV_simulation/verse/vcs_init_exp1_unsafe.pickle','rb') as f:
    e_list = pickle.load(f)


for i in range(18,len(state_list)):
        fig = plot_trajectory(state_list[i], 0, 1, 2, fig, 'blue', 5)


with open('./NeRF_UAV_simulation/verse/exp2_safe.pickle', 'rb') as f: 
    M, E, C_list, reachtube = pickle.load(f)

fig = plot3dReachtubeSingle(reachtube,1,5,9,fig,'red')

fig.add_mesh(point_cloud, point_size=5, render_points_as_spheres=True, scalars='RGBA', rgb=True, ambient=0.5)
fig.show()
