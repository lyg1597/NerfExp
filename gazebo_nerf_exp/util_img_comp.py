import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

if __name__ == "__main__":
    sd = os.path.dirname(os.path.realpath(__file__))
    im_rendered = cv2.imread(os.path.join(sd,'rendered.png')).astype(np.float64)
    im_gazebo = cv2.imread(os.path.join(sd,'frames_000001.png')).astype(np.float64)

    diff = np.linalg.norm(im_rendered-im_gazebo, axis = 2) 
    diff = diff/np.linalg.norm([255,255,255])
    # plt.imshow(diff)
    # plt.show()
    print("average diff", np.average(diff))
    cv2.imwrite(os.path.join(sd, 'diff.png'), diff*255)