import cv2 
import os 
import matplotlib.pyplot as plt 
import random
import re 
import numpy as np 

def mark_keypoints(image, keypoints):
    # Iterate through each keypoint and mark it on the image
    for point in keypoints:
        x, y = point
        # Draw a small circle at each keypoint location
        cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    return image

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))

    img_dir = os.path.join(script_dir, './data_unet/data_08-07-13-11/imgs')
    label_dir = os.path.join(script_dir, './data_unet/data_08-07-13-11/label_kp')

    all_fns = list(os.listdir(img_dir))

    img_fn = random.choice(all_fns)

    img = cv2.imread(os.path.join(img_dir, img_fn))

    match = re.search(r'\d+', img_fn)
    idx = int(match.group())

    label_fn = f"img_{idx}.txt"
    
    kps = np.loadtxt(os.path.join(label_dir, label_fn), delimiter = ',')

    marked_image = mark_keypoints(img, kps)

    cv2.imshow('keypoints', marked_image)

    key = cv2.waitKey()
    if key==27:
        cv2.destroyAllWindows()