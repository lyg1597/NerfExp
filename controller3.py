import dynamics3
from dynamics3 import g, A, B, kT
import numpy as np
import scipy
from scipy.integrate import odeint
import copy 
import matplotlib.pyplot as plt 
import json 
import os 

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)

####################### solve LQR #######################
n = A.shape[0]
m = B.shape[1]
Q = np.eye(n)
Q[0, 0] = 10.
Q[1, 1] = 10.
Q[2, 2] = 10.
# Q[11,11] = 0.01
R = np.diag([1., 1., 1.])
K, _, _ = lqr(A, B, Q, R)

####################### The controller ######################
def u(x, goal):
    yaw = x[10]
    err = [goal[0],0,0,0, goal[1],0,0,0, goal[2],0] - x[:10]
    err_pos = err[[0,4,8]]

    err_pos = np.linalg.inv(np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0,0,1]
    ]))@err_pos

    err[[0,4,8]] = err_pos
    u_pos = K.dot(err) + [0, 0, g / kT]
    u_ori = (goal[3]-yaw)*0.1+(0-x[11])*1.0

    return np.concatenate((u_pos, [u_ori]))

######################## The closed_loop system #######################
def cl_nonlinear(x, t, goal):
    x = np.array(x)
    dot_x = dynamics3.f(x, u(x, goal))
    return dot_x

# simulate
def simulate(x, goal, dt):
    curr_position = np.array(x)[[0, 4, 8]]
    goal_pos = goal[:3]
    error = goal_pos - curr_position
    distance = np.sqrt((error**2).sum())
    if distance > 1:
        goal[:3] = curr_position + error / distance
    return odeint(cl_nonlinear, x, [0, dt], args=(goal,))[-1]

if __name__ == "__main__":
    # x0 = np.zeros(12)
    # x0[10] = np.pi/3
    # dt = 0.01 
    # goal = np.array([5.,5.,5.,np.pi/4])
    # x_list = [copy.deepcopy(x0)]
    # for i in range(4000):
    #     if i==1500:
    #         print('a')
    #     res = simulate(x0, copy.deepcopy(goal), dt)
    #     x_list.append(copy.deepcopy(res))
    #     x0 = res
    # x_list = np.array(x_list)
    # print(x_list.shape)
    # plt.figure(0)
    # plt.plot(x_list[:,0])
    # plt.figure(1)
    # plt.plot(x_list[:,4])
    # plt.figure(2)
    # plt.plot(x_list[:,8])
    # plt.figure(3)
    # plt.plot(x_list[:,10])
    # plt.show()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir, 'camera_path.json'), 'r') as f:
        camera_path = json.load(f)

    x0 = [-0.8441804331384228,0,0,0, -0.05182509322668888,0,0,0, -0.3332843105241055,0,0,0]
    x_list = [copy.deepcopy(x0)]
    path = camera_path['camera_path']
    dt = 0.1
    for elem in path:
        pose = np.array(elem['camera_to_world']).reshape((4,4))
        camera_pos = pose[:3, 3]
        goal = np.array([camera_pos[0], camera_pos[1], camera_pos[2], 0])
        for i in range(10):
            res = simulate(x0, copy.deepcopy(goal), dt)
            x_list.append(copy.deepcopy(res))
            x0 = res
    x_list = np.array(x_list)
    # plt.figure(0)
    # plt.plot(x_list[:,0])
    # plt.figure(1)
    # plt.plot(x_list[:,4])
    # plt.figure(2)
    # plt.plot(x_list[:,8])
    # plt.figure(3)
    # plt.plot(x_list[:,10])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_list[:,0], x_list[:,4], x_list[:,8], 'gray')

    plt.show()

    