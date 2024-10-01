from model_pc.model_x2 import compute_model_x
from model_pc.model_y2 import compute_model_y
from model_pc.model_z2 import compute_model_z
from model_pc.model_yaw2 import compute_model_yaw
from model_pc.model_pitch2 import compute_model_pitch
import os 
import pickle 
import numpy as np
import json 
from typing import Tuple

def get_all_models(data, pr = [0.9, 0.9, 0.8, 0.9, 0.9]):
    state_array, trace_array, E_array = data
    pr_x, pr_y, pr_z, pr_yaw, pr_pitch = pr
    model_x = compute_model_x(state_array, trace_array, E_array, 0.5, pr_x)
    model_y = compute_model_y(state_array, trace_array, E_array, 0.5, pr_y)
    model_z = compute_model_z(state_array, trace_array, E_array, 0.5, pr_z)
    model_yaw = compute_model_yaw(state_array, trace_array, E_array, 0.5, pr_yaw)
    model_pitch = compute_model_pitch(state_array, trace_array, E_array, 0.5, pr_pitch)

    return model_x, model_y, model_z, model_yaw, model_pitch

def pre_process_data(data):

    state_list = []
    trace_list = []
    E_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        init = data[i][2]
        for tmp in init:
            E_list.append(tmp[1])
    # Getting Model for center center model
    state_list = np.array(state_list)
    E_list = np.array(E_list)

    # Flatten Lists
    state_array = np.zeros((E_list.shape[0],state_list.shape[1]))
    trace_array = np.zeros(state_array.shape)
    E_array = E_list

    num = trace_list[0].shape[0]
    for i in range(state_list.shape[0]):
        state_array[i*num:(i+1)*num,:] = state_list[i,:]
        trace_array[i*num:(i+1)*num,:] = trace_list[i] 
    return state_array, trace_array, E_array 

def pre_process_data2(data):

    state_list = []
    trace_list = []
    E_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        init = data[i][2]
        # for tmp in init:
        #     E_list.append(tmp[1])
        E_list.append(init[1])
    # Getting Model for center center model
    state_array = np.array(state_list)
    trace_array = np.array(trace_list).squeeze()
    E_array = np.array(E_list)

    return state_array, trace_array, E_array 

def strip_data(data):
    state_array, trace_array, E_array = data 

    state_max = np.max(state_array, axis=0)
    state_min = np.min(state_array, axis=0)
    state_diff = (state_max - state_min)[:3]

    est_diff = np.abs(state_array[:,:3]-trace_array[:,:3])

    comparison = est_diff <= state_diff 
    mask = np.all(comparison, axis = 1)
    stripped_state = state_array[mask]
    stripped_trace = trace_array[mask]
    stripped_E = E_array[mask]
    return (stripped_state, stripped_trace, stripped_E)

def strip_data_more(data):
    state_array, trace_array, E_array = data 

    state_max = np.max(state_array, axis=0)
    state_min = np.min(state_array, axis=0)
    state_diff = (state_max - state_min)[:3]

    est_diff = np.abs(state_array[:,:3]-trace_array[:,:3])

    comparison = est_diff <= state_diff 
    mask = np.all(comparison, axis = 1)
    stripped_state = state_array[mask]
    stripped_trace = trace_array[mask]
    stripped_E = E_array[mask]
    removed_state = state_array[~mask]
    removed_trace = trace_array[~mask]
    removed_E = E_array[~mask]
    return (stripped_state, stripped_trace, stripped_E), (removed_state, removed_trace, removed_E)

def apply_model_batch(model, point):
    dim = model['dim']
    cc = model['coef_center']
    cr = model['coef_radius']

    if dim == 'x':
        point = point[:,0]
    elif dim == 'y':
        point = point[:,(0,1)]
    elif dim == 'z':
        point = point[:,(0,2)]
    elif dim == 'yaw':
        point = point[:,(0,3)]
    elif dim == 'pitch':
        point = point[:,(0,4)]

    if dim == 'x':
        x = point 
        center = cc[0]+cc[1]*x
        radius = cr[0]+x*cr[1]+x**2*cr[2]
        # radius = 0
        return center, np.abs(radius)
    elif dim == 'pitch' or dim == 'z':
        x = point[:,0]
        y = point[:,1]
        center = cc[0] + cc[1]*x +cc[2]*y
        radius = cr[0] + x*cr[1] + y*cr[2]
        # radius = 0
        return center, np.abs(radius)
    else:
        x = point[:,0]
        y = point[:,1]
        center = cc[0]+cc[1]*x+cc[2]*y
        radius = cr[0]+x*cr[1]+y*cr[2]+x*y*cr[3]+x**2*cr[4]+y**2*cr[5]
        # radius = 0
        return center, np.abs(radius)

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
        # radius = 0
        return center_center, abs(radius) 
    elif dim == 'pitch' or dim == 'z':
        x = point[0]
        y = point[1]
        center = cc[0] + cc[1]*x +cc[2]*y
        radius = cr[0] + x*cr[1] + y*cr[2]
        # radius = 0
        return center, abs(radius)
    else:
        x = point[0]
        y = point[1]
        center_center = cc[0] + cc[1]*x + cc[2]*y
        radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
        # radius = 0
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

def get_vision_estimation_batch(point: np.ndarray, models) -> Tuple[np.ndarray, np.ndarray]:
    x_c, x_r = apply_model_batch(models[0], point)
    y_c, y_r = apply_model_batch(models[1], point)
    z_c, z_r = apply_model_batch(models[2], point)
    yaw_c, yaw_r = apply_model_batch(models[3], point)
    pitch_c, pitch_r = apply_model_batch(models[4], point)


    low = np.array([x_c-x_r, y_c-y_r, z_c-z_r, yaw_c-yaw_r, pitch_c-pitch_r, point[:,5]]).T
    high = np.array([x_c+x_r, y_c+y_r, z_c+z_r, yaw_c+yaw_r, pitch_c+pitch_r, point[:,5]]).T

    return low, high

def realize_data(data):
    state_array, trace_array, E_array = data 

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, './data_pc/data_08-07-16-39.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)
    data = pre_process_data(data)
    state_array, trace_array, E_array = data 

    data = strip_data(data)
    data = realize_data(data)

    pr = [0.95,0.95,0.95,0.95,0.95]
    model_x, model_y, model_z, model_yaw, model_pitch = get_all_models(data)

    res_x = apply_model_batch(model_x, state_array)
    res_y = apply_model_batch(model_y, state_array)
    res_z = apply_model_batch(model_z, state_array)
    res_yaw = apply_model_batch(model_yaw, state_array)
    res_pitch = apply_model_batch(model_pitch, state_array)

    with open(os.path.join(script_dir, './model_pc/model_x2.json'), 'w+') as f:
        json.dump(model_x, f)

    with open(os.path.join(script_dir, './model_pc/model_y2.json'), 'w+') as f:
        json.dump(model_y, f)

    with open(os.path.join(script_dir, './model_pc/model_z2.json'), 'w+') as f:
        json.dump(model_z, f)

    with open(os.path.join(script_dir, './model_pc/model_yaw2.json'), 'w+') as f:
        json.dump(model_yaw, f)

    with open(os.path.join(script_dir, './model_pc/model_pitch2.json'), 'w+') as f:
        json.dump(model_pitch, f)
