import cv2
import os 

if not os.path.exists('./frames'):
    os.mkdir('./frames')

vidcap = cv2.VideoCapture('./renders/2024-09-02-21-37-44.mp4')
i = 0
while True:
    success, image = vidcap.read()
    if not success:
        break 
    cv2.imwrite(f'./frames/frame_{i:05d}.png', image)
    i+=1
