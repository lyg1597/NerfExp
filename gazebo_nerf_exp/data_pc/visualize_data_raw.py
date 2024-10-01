import pickle 
import numpy as np 
import os 
import matplotlib.pyplot as plt 

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

script_dir = os.path.dirname(os.path.realpath(__file__))
data_fn = os.path.join(script_dir, './data_08-07-16-39.pickle')
with open(data_fn, 'rb') as f:
    data = pickle.load(f)


state_array, trace_array, E_array = pre_process_data(data)

plt.figure(0)
plt.plot(state_array[:,0], trace_array[:,0],'b*')
plt.figure(1)
plt.plot(state_array[:,1], trace_array[:,1],'b*')
plt.figure(2)
plt.plot(state_array[:,2], trace_array[:,2],'b*')
plt.figure(3)
plt.plot(state_array[:,3], trace_array[:,3],'b*')
plt.figure(4)
plt.plot(state_array[:,4], trace_array[:,4],'b*')
plt.show()
