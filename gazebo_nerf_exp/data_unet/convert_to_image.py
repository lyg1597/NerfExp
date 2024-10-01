import numpy as np 
import cv2
from scipy.spatial.transform import Rotation 
import os 

script_dir = os.path.dirname(os.path.realpath(__file__))

def gaussian(xL, yL, sigma, H, W):

    grid = np.meshgrid(list(range(W)), list(range(H)))
    channel = np.exp(-((grid[0] - xL) ** 2 + (grid[1] - yL) ** 2) / (2 * sigma ** 2))

    return channel

def convertToHM(H, W, keypoints, sigma=5):
    nKeypoints = len(keypoints)

    img_hm = np.zeros(shape=(H, W, nKeypoints // 2), dtype=np.float32)

    for i in range(0, nKeypoints // 2):
        x = keypoints[i * 2]
        y = keypoints[1 + 2 * i]

        channel_hm = gaussian(x, y, sigma, H, W)

        img_hm[:, :, i] = channel_hm
    
    img_hm = img_hm.transpose((2,0,1))
    return img_hm



def convert_to_image(world_pos, ego_pos, ego_ori, K):
    objectPoints = np.array(world_pos) 
    R = Rotation.from_quat(ego_ori)
    # R_euler = R.as_euler('xyz')
    # # R_euler[1] = -R_euler[1]
    # R = Rotation.from_euler('xyz', R_euler)
    R2 = Rotation.from_euler('xyz',[-np.pi/2, -np.pi/2, 0])
    R_roted = R2*R.inv()

    #TODO: The way of converting rvec is wrong
    rvec = R_roted.as_rotvec()
    tvec = -R_roted.apply(np.array(ego_pos))
    cameraMatrix = K
    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    pnt,_ = cv2.projectPoints(objectPoints,rvec,tvec,cameraMatrix,distCoeffs)
    return pnt  

if __name__ == "__main__":
    imWidth = 640
    imHeight = 480
    fx = 1253.2215566867008
    fy = 1253.2215566867008
    K = np.array([[fx,0,imWidth/2],[0,fy,imHeight/2],[0,0,1]])
    for i in range(0, 100000):
        idx = i
        print(idx)
        # if idx == 6910:
        #     print("stop")

        with open(os.path.join(script_dir, './data_08-19-21-18/data.txt'),'r') as f:
            data = f.read()
            data = data.strip('\n').split('\n')
            pose = data[idx]
            pose = pose.split(',')
            pose = [float(elem) for elem in pose]
    
        kp1 = [-1221.370483, 16.052534, 5.0] # (290, 327)
        kp2 = [-1279.224854, 16.947235, 5.0] # (285, 346)
        kp3 = [-1279.349731, 8.911615, 5.0] # (299, 346)
        kp4 = [-1221.505737, 8.033512, 5.0] # (304, 327)

        kp5 = [-1221.438110, -8.496282, 5.0] # (329, 327)
        kp6 = [-1279.302002, -8.493725, 5.0] # (333, 346)
        kp7 = [-1279.315796, -16.504263, 5.0] # (348, 346)
        kp8 = [-1221.462402, -16.498976, 5.0] # (340, 327)

        kp9 = [-1520.81, 26.125700, 5.0] # (290, 327)
        kp10 = [-1559.122925, 26.101082,  5.0] # (285, 346)
        kp11 = [-1559.157471, -30.753305,  5.0] # (299, 346)
        kp12 = [-1520.886353,  -30.761044, 5.0] # (304, 327)

        kp13 = [-1561.039063, 31.522200, 5.0] # (329, 327)
        kp14 = [-1561.039795, -33.577713, 5.0] # (333, 346)
        # kp15 = [-600.0, 31.5, 5.0] # (348, 346)
        # kp16 = [-600.0, -23.5, 5.0] # (340, 327)

        keypoint_list = [kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12, kp13, kp14]
        num_keypoint = len(keypoint_list)

        offset_vec = np.array([-1.1*0.20,0,0.8*0.77])
        drone_pos = np.array([pose[1], pose[2], pose[3]]) # x,y,z
        drone_ori = [pose[4], pose[5], pose[6], pose[7]] # x,y,z,w
        R = Rotation.from_quat(drone_ori)
        drone_pos += offset_vec

        position_in_image = []
        for i in range(num_keypoint):
            position_in_image.append(convert_to_image(keypoint_list[i], drone_pos, drone_ori, K))

        u_vectors = []
        v_vectors = []
        kp_vectors = []
        for i in range(num_keypoint):
            u = imWidth-int(position_in_image[i][0][0][0])
            v = imHeight-int(position_in_image[i][0][0][1])

            u_vectors.append(u)
            v_vectors.append(v)

            kp_vectors.append(u)
            kp_vectors.append(v)

        img_fn = os.path.join(script_dir, f'./data_08-19-21-18/imgs/img_{idx}.png')
        img = cv2.imread(img_fn)

        
        cv2.line(img, (u_vectors[0], v_vectors[0]), (u_vectors[1], v_vectors[1]),(0,0,255))
        cv2.line(img, (u_vectors[1], v_vectors[1]), (u_vectors[2], v_vectors[2]),(0,0,255))
        cv2.line(img, (u_vectors[2], v_vectors[2]), (u_vectors[3], v_vectors[3]),(0,0,255))
        cv2.line(img, (u_vectors[3], v_vectors[3]), (u_vectors[0], v_vectors[0]),(0,0,255))

        cv2.line(img, (u_vectors[4], v_vectors[4]), (u_vectors[5], v_vectors[5]),(0,0,255))
        cv2.line(img, (u_vectors[5], v_vectors[5]), (u_vectors[6], v_vectors[6]),(0,0,255))
        cv2.line(img, (u_vectors[6], v_vectors[6]), (u_vectors[7], v_vectors[7]),(0,0,255))
        cv2.line(img, (u_vectors[7], v_vectors[7]), (u_vectors[4], v_vectors[4]),(0,0,255))

        cv2.line(img, (u_vectors[8], v_vectors[8]), (u_vectors[9], v_vectors[9]),(0,0,255))
        cv2.line(img, (u_vectors[9], v_vectors[9]), (u_vectors[10], v_vectors[10]),(0,0,255))
        cv2.line(img, (u_vectors[10], v_vectors[10]), (u_vectors[11], v_vectors[11]),(0,0,255))
        cv2.line(img, (u_vectors[11], v_vectors[11]), (u_vectors[8], v_vectors[8]),(0,0,255))

        cv2.line(img, (u_vectors[12], v_vectors[12]), (u_vectors[13], v_vectors[13]),(0,0,255))
        # cv2.line(img, (u_vectors[13], v_vectors[13]), (u_vectors[14], v_vectors[14]),(0,0,255))
        # cv2.line(img, (u_vectors[14], v_vectors[14]), (u_vectors[15], v_vectors[15]),(0,0,255))
        # cv2.line(img, (u_vectors[15], v_vectors[15]), (u_vectors[12], v_vectors[12]),(0,0,255))

        # cv2.imshow('keypoints', img)
        # cv2.waitKey(3)
        hms = convertToHM(imHeight, imWidth, kp_vectors, sigma=2)
        # hm_img = np.vstack((np.hstack(hms[0:4,:,:]), np.hstack(hms[4:,:,:])))
        hm_img = np.sum(hms, axis=0)
        # cv2.imshow('heat map', hm_img)
        # cv2.waitKey(1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not os.path.exists(os.path.join(script_dir,'./data_08-19-21-18/label_kp')):
            os.mkdir(os.path.join(script_dir,'./data_08-19-21-18/label_kp'))
        # cv2.imwrite(os.path.join(script_dir, f"./label_kp/img_marker_{idx}.png"), img)
        # # print("HM shape: ", hms.shape[0])
        # for j in range(hms.shape[0]):
        #     cv2.imwrite(os.path.join(script_dir, f"./label_kp/img_{idx}_{j}.png"), hms[j,:,:]*255)

        # cv2.imwrite(os.path.join(script_dir, f"./label_kp/img_{idx}_hm.png"), hm_img*255)

        with open(os.path.join(script_dir, f"./data_08-19-21-18/label_kp/img_{idx}.txt"), "w+") as f:
            for i in range(num_keypoint):
                f.write(f"{u_vectors[i]}, {v_vectors[i]}\n")

            # f.write(f"{u_vectors[0]}, {v_vectors[0]}, {u_vectors[1]}, {v_vectors[1]}, {u_vectors[2]}, {v_vectors[2]}, \
            #         {u_vectors[3]}, {v_vectors[3]}, {u_veimg_rgbctors[4]}, {v_vectors[4]}, {u_vectors[5]}, {v_vectors[5]}, \
            #         {u_vectors[6]}, {v_vectors[6]}, {u_vectors[7]}, {v_vectors[7]}, {u_vectors[8]}, {v_vectors[8]}, \
            #         {u_vectors[9]}, {v_vectors[9]}, {u_vectors[10]}, {v_vectors[10]}, {u_vectors[11]}, {v_vectors[11]},\
            #         {u_vectors[12]}, {v_vectors[12]}, {u_vectors[13]}, {v_vectors[7]}, {u_vectors[8]}, {v_vectors[8]},\
            #         ")
        # import matplotlib.pyplot as plt 
        # plt.imshow(img_rgb)
        # plt.show()
