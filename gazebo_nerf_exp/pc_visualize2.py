import matplotlib.pyplot as plt 
import numpy as np 
import pickle 
import json 
import os 
from pc_simple_models import get_all_models, get_vision_estimation_batch, strip_data

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
        # for tmp in init:
        #     E_list.append(tmp[1])
        E_list.append(init[1])
    # Getting Model for center center model
    state_array = np.array(state_list)
    trace_array = np.array(trace_list).squeeze()
    E_array = np.array(E_list)

    return state_array, trace_array, E_array 

script_dir = os.path.dirname(os.path.realpath(__file__))
# data_fn = os.path.join(script_dir, './data_pc/data_08-07-16-39.pickle')
data_fn = os.path.join(script_dir, './data_pc/data_09-02-17-23.pickle')
# data_fn = os.path.join(script_dir, './data_pc/data_09-03-09-52.pickle')
# data_fn = os.path.join(script_dir, './data_pc/data_09-16-20-58.pickle')

with open(data_fn, 'rb') as f:
    data = pickle.load(f)

state_array, trace_array, E_array = pre_process_data(data)

state_array, trace_array, E_array = strip_data((state_array, trace_array, E_array))

pr = [0.95, 0.95, 0.95, 0.95, 0.95] # Mid
# pr = [0.9, 0.9, 0.8, 0.9, 0.9] # Mid
# pr = [0.3, 0.3, 0.3, 0.3, 0.3] # Mid
M = get_all_models((state_array, trace_array, E_array), pr)

lb, ub = get_vision_estimation_batch(state_array, M)

plt.figure(0)
plt.plot(state_array[:,0], trace_array[:,0], 'b*')
plt.plot(state_array[:,0], lb[:,0], 'r*')
plt.plot(state_array[:,0], ub[:,0], 'r*')
plt.xlabel('gt x')
plt.ylabel('est x')

plt.figure(1)
plt.plot(state_array[:,1], trace_array[:,1], 'b*')
plt.plot(state_array[:,1], lb[:,1], 'r*')
plt.plot(state_array[:,1], ub[:,1], 'r*')
plt.xlabel('gt y')
plt.ylabel('est y')

plt.figure(2)
plt.plot(state_array[:,2], trace_array[:,2], 'b*')
plt.plot(state_array[:,2], lb[:,2], 'r*')
plt.plot(state_array[:,2], ub[:,2], 'r*')
plt.xlabel('gt z')
plt.ylabel('est z')

plt.figure(3)
plt.plot(state_array[:,3], trace_array[:,3], 'b*')
plt.plot(state_array[:,3], lb[:,3], 'r*')
plt.plot(state_array[:,3], ub[:,3], 'r*')
plt.xlabel('gt yaw')
plt.ylabel('est yaw')

plt.figure(4)
plt.plot(state_array[:,4], trace_array[:,4], 'b*')
plt.plot(state_array[:,4], lb[:,4], 'r*')
plt.plot(state_array[:,4], ub[:,4], 'r*')
plt.xlabel('gt pitch')
plt.ylabel('est pitch')

plt.show()