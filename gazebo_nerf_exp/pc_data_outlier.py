from ns_renderer import GazeboSplatRenderer
import os 
from pc_simple_models import pre_process_data, strip_data, get_all_models, apply_model_batch, strip_data_more
import pickle 
from pc_data_system import Perception
import numpy as np 
import cv2 
import pandas as pd 
import copy 

if __name__ == "__main__":
    perception = Perception()

    pr = 0.9 

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, './data_pc/data_08-07-16-39.pickle')
    with open(data_file_path,'rb') as f:
        data = pickle.load(f)
    data = pre_process_data(data)
    data_orig = copy.deepcopy(data)
    data, data_removed = strip_data_more(data)

    # for i in range(12):
    #     E = refineEnv(E, None, data, i)
    # M_out = computeContract(data, E)
    M_out = get_all_models(data)

    state_array, trace_array, E_array = data

    cx, rx = apply_model_batch(M_out[0], state_array)
    cy, ry = apply_model_batch(M_out[1], state_array)
    cz, rz = apply_model_batch(M_out[2], state_array)

    inx = ((cx-rx) <= trace_array[:,0]) & (trace_array[:,0] <= (cx+rx))
    iny = ((cy-ry) <= trace_array[:,1]) & (trace_array[:,1] <= (cy+ry))
    inz = ((cz-rz) <= trace_array[:,2]) & (trace_array[:,2] <= (cz+rz))

    notinxyz = ~(inx & iny & inz)

    outliers = state_array[notinxyz,:]
    outliers_est = trace_array[notinxyz,:]

    outliers_dir = os.path.join(script_dir, 'data_pc_outliers')
    if not os.path.exists(outliers_dir):
        os.mkdir(outliers_dir)
    outliers_list = []
    for i in range(outliers.shape[0]):
        estimated_state,img, kp_img = perception.set_percept_output(
            outliers[i,:], None, None
        )
        print(estimated_state)
        print(outliers_est[i,:])
        if np.linalg.norm(estimated_state-outliers_est[i,:])<10:
            data_dict = {}
            outlier_fn = os.path.join(outliers_dir, f'frame_{i:05d}.png')
            outlier_kp_fn = os.path.join(outliers_dir, f'frame_{i:05d}_kp.png')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            kp_img = cv2.cvtColor(kp_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(outlier_fn, img)
            cv2.imwrite(outlier_kp_fn, kp_img)
            data_dict['gt'] = outliers[i,:]
            data_dict['est'] = estimated_state 
            data_dict['diff'] = np.linalg.norm(estimated_state[:3] - outliers[i,:3])
            data_dict['img_fn'] = os.path.normpath(outlier_fn)
            data_dict['kp_fn'] = os.path.normpath(outlier_kp_fn)
            data_dict['removed'] = False
            outliers_list.append(data_dict)

            df = pd.DataFrame(outliers_list)
            csv_file = os.path.join(outliers_dir, 'outlier.csv')
            df.to_csv(csv_file, index=False)

    outliers = data_removed[0]
    outliers_est = data_removed[1]
    for i in range(outliers.shape[0]):
        estimated_state,img, kp_img = perception.set_percept_output(
            outliers[i,:], None, None
        )
        print(estimated_state)
        print(outliers_est[i,:])
        if np.linalg.norm(estimated_state-outliers_est[i,:])<10:
            if i == 126:
                print("stop")
            data_dict = {}
            outlier_fn = os.path.join(outliers_dir, f'frame_removed_{i:05d}.png')
            outlier_kp_fn = os.path.join(outliers_dir, f'frame_removed_{i:05d}_kp.png')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            kp_img = cv2.cvtColor(kp_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(outlier_fn, img)
            cv2.imwrite(outlier_kp_fn, kp_img)
            data_dict['gt'] = outliers[i,:]
            data_dict['est'] = estimated_state 
            data_dict['diff'] = np.linalg.norm(estimated_state[:3] - outliers[i,:3])
            data_dict['img_fn'] = os.path.normpath(outlier_fn)
            data_dict['kp_fn'] = os.path.normpath(outlier_kp_fn)
            data_dict['removed'] = True
            outliers_list.append(data_dict)

            df = pd.DataFrame(outliers_list)
            csv_file = os.path.join(outliers_dir, 'outlier.csv')
            df.to_csv(csv_file, index=False)