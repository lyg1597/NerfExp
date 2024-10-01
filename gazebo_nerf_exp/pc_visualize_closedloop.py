import pickle 
import numpy as np 
import torch
from enum import Enum, auto
from typing import Tuple
import os 
import json 

from verse.plotter.plotter2D import *

import mpl_toolkits.mplot3d as a3
import polytope as pc
import pypoman as ppm
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt 
import copy

from pc_closedloop_dynamics import FixedWingAgent3
from verse import Scenario, ScenarioConfig

import os 
from pc_Rrect import R1
from pc_simple_models import pre_process_data, get_vision_estimation_batch

model_radius_decay = lambda r, r_max: (1/np.sqrt(r_max))*np.sqrt(r) # Input to this function is the radius of environmental parameters

class Faces():
    def __init__(self,tri, sig_dig=12, method="convexhull"):
        self.method=method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms,return_inverse=True, axis=0)

    def norm(self,sq):
        cr = np.cross(sq[2]-sq[0],sq[1]-sq[0])
        return np.abs(cr/np.linalg.norm(cr))

    def isneighbor(self, tr1,tr2):
        a = np.concatenate((tr1,tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0))+2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n,v[1]-v[0])
        y = y/np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1]-v[0],y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c,axis=0)
            d = c-mean
            s = np.arctan2(d[:,0], d[:,1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j,tri2 in enumerate(self.tri):
                if j > i:
                    if self.isneighbor(tri1,tri2) and \
                       self.inv[i]==self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups

def plot_polytope_3d(A, b, ax = None, edgecolor = 'k', color = 'red', trans = 0.2):
    verts = np.array(ppm.compute_polytope_vertices(A, b))
    # compute the triangles that make up the convex hull of the data points
    hull = ConvexHull(verts)
    triangles = [verts[s] for s in hull.simplices]
    # combine co-planar triangles into a single face
    faces = Faces(triangles, sig_dig=1).simplify()
    # plot
    if ax == None:
        ax = a3.Axes3D(plt.figure())

    pc = a3.art3d.Poly3DCollection(faces,
                                    facecolor=color,
                                    edgecolor=edgecolor, alpha=trans)
    ax.add_collection3d(pc)
    # define view
    yllim, ytlim = ax.get_ylim()
    xllim, xtlim = ax.get_xlim()
    zllim, ztlim = ax.get_zlim()
    x = verts[:,0]
    x = np.append(x, [xllim+1, xtlim-1])
    y = verts[:,1]
    y = np.append(y, [yllim+1, ytlim-1])
    z = verts[:,2]
    z = np.append(z, [zllim+1, ztlim-1])
    # print(np.min(x)-1, np.max(x)+1, np.min(y)-1, np.max(y)+1, np.min(z)-1, np.max(z)+1)
    ax.set_xlim(np.min(x)-1, np.max(x)+1)
    ax.set_ylim(np.min(y)-1, np.max(y)+1)
    ax.set_zlim(np.min(z)-1, np.max(z)+1)

script_dir = os.path.realpath(os.path.dirname(__file__))

model_x_name = './model_pc/model_x2.json'
model_y_name = './model_pc/model_y2.json'
model_z_name = './model_pc/model_z2.json'
model_yaw_name = './model_pc/model_yaw2.json'
model_pitch_name = './model_pc/model_pitch2.json'

with open(os.path.join(script_dir, model_x_name), 'r') as f:
    model_x = json.load(f)
with open(os.path.join(script_dir, model_y_name), 'r') as f:
    model_y = json.load(f)
with open(os.path.join(script_dir, model_z_name), 'r') as f:
    model_z = json.load(f)
with open(os.path.join(script_dir, model_pitch_name), 'r') as f:
    model_pitch = json.load(f)
with open(os.path.join(script_dir, model_yaw_name), 'r') as f:
    model_yaw = json.load(f)

def run_ref(ref_state, time_step, approaching_angle=3):
    k = np.tan(approaching_angle*(np.pi/180))
    delta_x = ref_state[-1]*time_step
    delta_z = k*delta_x # *time_step
    return np.array([ref_state[0]+delta_x, 0, ref_state[2]-delta_z, ref_state[3], ref_state[4], ref_state[5]])

class FixedWingMode(Enum):
    # NOTE: Any model should have at least one mode
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)

def apply_model(model, point):
    dim = model['dim']
    cc = model['coef_center']
    cr = model['coef_radius']

    if dim == 'x':
        point = point[0]
    elif dim == 'y':
        point = point[(0,1),]
    elif dim == 'z':
        point = point[(0,2),]
    elif dim == 'yaw':
        point = point[(0,3),]
    elif dim == 'pitch':
        point = point[(0,4),]

    if dim == 'x':
        x = point
        center_center = cc[0] + cc[1]*x
        radius = cr[0] + x*cr[1] + x**2*cr[2]
        return center_center, radius 
    if dim == 'pitch' or dim == 'z':
        x = point[0]
        y = point[1]
        center_center = cc[0] + cc[1]*x + cc[2]*y
        radius = cr[0] + x*cr[1] + y*cr[2]
        return center_center, abs(radius)
    else:
        x = point[0]
        y = point[1]
        center_center = cc[0] + cc[1]*x + cc[2]*y
        radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
        return center_center, abs(radius)
        
def get_vision_estimation(point: np.ndarray, models) -> Tuple[np.ndarray, np.ndarray]:
    x_c, x_r = apply_model(models[0], point)
    y_c, y_r = apply_model(models[1], point)
    z_c, z_r = apply_model(models[2], point)
    yaw_c, yaw_r = apply_model(models[3], point)
    pitch_c, pitch_r = apply_model(models[4], point)


    low = np.array([x_c-x_r, y_c-y_r, z_c-z_r, yaw_c-yaw_r, pitch_c-pitch_r, point[5]])
    high = np.array([x_c+x_r, y_c+y_r, z_c+z_r, yaw_c+yaw_r, pitch_c+pitch_r, point[5]])

    return low, high

def sample_point(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.random.uniform(low, high) 

def run_vision_sim(scenario, init_point, init_ref, pc, time_horizon, computation_step, time_step):
    time_points = np.arange(0, time_horizon+computation_step/2, computation_step)

    traj = [np.insert(init_point, 0, 0)]
    point = init_point 
    ref = init_ref
    for t in time_points[1:]:
        estimate_lower, estimate_upper = get_vision_estimation(point, pc)
        estimate_point = sample_point(estimate_lower, estimate_upper)
        init = np.concatenate((point, estimate_point, ref))
        scenario.set_init(
            [[init]],
            [(FixedWingMode.Normal,)]
        )
        res = scenario.simulate(computation_step, time_step)
        trace = res.nodes[0].trace['a1']
        point = trace[-1,1:7]
        traj.append(np.insert(point, 0, t))
        ref = run_ref(ref, computation_step)
    return traj

script_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    import sys 
    arg = sys.argv[1]
    # arg = 'small'
    fig = plt.figure(0)
    ax = plt.axes(projection='3d')
    ax.set_xlim(-3050, -3010)
    ax.set_ylim(-20, 20)
    ax.set_zlim(110, 130)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    fn = os.path.join(script_dir, f'./res/exp_08-13-13-00/pc_closedloop_{arg}.pickle')
    with open(fn, 'rb') as f:
        M, E, C_list = pickle.load(f)
    C_list_truncate = C_list[:12]
    for i in range(len(C_list_truncate)):
        rect = C_list[i]

        pos_rect = rect[:,1:4]
        poly = pc.box2poly(pos_rect.T)
        # plot_polytope_3d(poly.A, poly.b, ax, trans=0.1, edgecolor='k')

    for i in range(len(R1)):
        rect =  R1[i]

        pos_rect = rect[:, 0:3]
        poly = pc.box2poly(pos_rect.T)
        plot_polytope_3d(poly.A, poly.b, ax, trans=0.1, edgecolor='k', color='b')

    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'pc_closedloop_dl.py')
    aircraft = FixedWingAgent3("a1")
    fixed_wing_scenario.add_agent(aircraft)
    

    state = np.array([
        [-3020.0, -5, 118.0, 0-0.0001, -np.deg2rad(3)-0.0001, 10-0.0001], 
        [-3010.0, 5, 122.0, 0+0.0001, -np.deg2rad(3)+0.0001, 10+0.0001]
    ])
    ref = np.array([-3000.0, 0, 120.0, 0, -np.deg2rad(3), 10])
    time_horizon = 0.1*((len(C_list_truncate)-1)*80+1)

    for i in range(10):
        init_point = sample_point(state[0,:], state[1,:])
        init_ref = copy.deepcopy(ref)
        trace = run_vision_sim(fixed_wing_scenario, init_point, init_ref, M, time_horizon, 0.1, 0.01)
        trace = np.array(trace)
        # ax.plot(trace[:,1], trace[:,2], trace[:,3], linewidth=1, color='g')
        # ax.scatter(trace[::80,1], trace[::80,2], trace[::80,3], marker='x', color='m', s=30)
    
    # plt.show()

    frame_path = os.path.join(script_dir, 'res_frames')
    if not os.path.exists(frame_path):
        os.mkdir(frame_path)
    
    i = 0
    fn_list = []
    for angle in range(0,360, 2):
        ax.view_init(azim=angle)
        fn = os.path.join(frame_path, f'./frame_{i:04d}.png')
        plt.savefig(fn)
        fn_list.append(fn)
        i += 1

    import imageio 
    frames = []
    for fn in fn_list:
        frames.append(imageio.imread(fn))
    
    print("read image complete!")
    imageio.mimsave('rotation.gif', frames, fps=20)