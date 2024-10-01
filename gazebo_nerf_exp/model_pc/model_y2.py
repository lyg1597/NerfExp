import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import statsmodels.api as sm 
import json 

dim = 1

def compute_model_y(state_array, trace_array, E_array, pc=0.9, pr=0.95):
    X = state_array[:, (0,dim)]
    Y = trace_array[:,dim]
    model_center = sm.QuantReg(Y, sm.add_constant(X))
    result = model_center.fit(q=0.5)
    cc = result.params
    # -------------------------------------

    # Getting Model for Radius
    X = state_array[:, (0,dim)]
    X1 = state_array[:,0]
    X2 = state_array[:, dim]
    center_center = cc[0] + cc[1]*X1 + cc[2]*X2 
    trace_array_radius = trace_array[:,(0, dim)]

    # tmp = np.array(tmp)
    Y_radius = np.abs(trace_array_radius[:,1]-center_center)
    # Y_radius = np.abs(trace_list_radius[:,0]-X_radius)
    quantile = pr
    X_radius = np.hstack((
        X1.reshape((-1,1)),
        X2.reshape((-1,1)),
        (X1*X2).reshape((-1,1)),
        (X1**2).reshape((-1,1)),
        (X2**2).reshape((-1,1))
    ))
    model_radius = sm.QuantReg(Y_radius, sm.add_constant(X_radius))
    result = model_radius.fit(q=quantile)
    cr = result.params

    # cc = mcc.coef_.tolist()+[mcc.intercept_]
    min_cc = 0
    min_r = 0
    for i in range(state_array.shape[0]):
        x = state_array[i,0]
        y = state_array[i, dim]
        center_center = cc[0] + cc[1]*x + cc[2]*y
        if center_center < min_cc:
            min_cc = center_center
        radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
        if radius < min_r:
            min_r = radius
    cr[0] += (-min_r)
    res = {
        'dim': 'y',
        'coef_center':cc.tolist(),
        'coef_radius': cr.tolist()
    }
    return res
# -------------------------------------

# Testing the obtained models
# The center of perception contract. 
# mcc # Input to this function is the ground truth state and center of range of environmental parameter

# # The radius of possible center 
# model_center_radius 
# ccr # Input to this function is the ground truth state and center of range of environmental parameter

# # The radius of perception contract 
# model_radius 
# cr # Input to this function is the ground truth state

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '../data_pc/data.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

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

    res = compute_model_y(state_array, trace_array, E_array, pc = 0.8, pr=0.999)
    cc = res['coef_center']
    cr = res['coef_radius']
    with open(os.path.join(script_dir,'model_y2.json'),'w+') as f:
        json.dump(res, f, indent=4)

    sample_contained = 0
    total_sample = 0
    for i in range(state_list.shape[0]):
        x = state_list[i,0]
        y = state_list[i, dim]
        center_center = cc[0] + cc[1]*x + cc[2]*y
        radius = cr[0] + x*cr[1] + y*cr[2] + x*y*cr[3] + x**2*cr[4] + y**2*cr[5]
        traces = trace_list[i]
        for j in range(trace_list[i].shape[0]):
            x_est = trace_list[i][j,dim]
            if x_est<center_center+radius and \
                x_est>center_center-radius:
                sample_contained += 1
                total_sample += 1 
            else:
                total_sample += 1

    print(sample_contained/total_sample)
