import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import os 

def adjust_color_temperature(image, temperature=0):
    """
    Adjusts the color temperature of an image.
    :param image_path: Path to the image.
    :param temperature: Positive values make the image warmer, negative make it cooler.
    :return: Adjusted image.
    """

    # Define temperature scaling
    if temperature > 0:
        # Increase Red and Green channels
        image[:, :, 2] += temperature
        image[:, :, 1] += temperature / 2
    elif temperature < 0:
        # Increase Blue channel
        image[:, :, 0] -= temperature
        image[:, :, 1] -= temperature / 2

    # Clip the values to [0,255]
    image = np.clip(image, 0, 255)

    plt.imshow(image)
    plt.show()

    return image

def adjust_saturation(image, saturation_scale=1.5):    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust the saturation
    hsv[:, :, 1] *= (saturation_scale+1)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Convert back to BGR
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_image

def adjust_hue(image, hue_shift=10):    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust the hue
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 360  # OpenCV hue range is [0,179]
    
    # Convert back to BGR
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_image

script_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":

    data = np.load(os.path.join(script_dir,"tiny_nerf_data.npz"))
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    print(type(images), type(poses), type(focal))
    print(images.shape, poses.shape, focal.shape)
    print(focal)

    data_new = {}
    data_new['images'] = np.zeros((images.shape[0]*10, images.shape[1], images.shape[2], images.shape[3]))
    data_new['poses'] = np.zeros((poses.shape[0]*10, poses.shape[1], poses.shape[2]))
    data_new['focal'] = focal 
    data_new['env'] = np.zeros((images.shape[0]*10, 2))

    hue_range = [-30, 30]
    saturation_range = [-0.5, 0.5]

    for i in range(images.shape[0]):
        image = images[i,:,:,:]
        pose = poses[i,:,:]

        data_new['images'][i*10+0,:,:,:] = image
        data_new['poses'][i*10+0,:,:] = pose
        data_new['env'][i*10+0,:] = np.array([0,0])
        # print(i*10+0)
        cv2img = cv2.cvtColor(image*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'./adjusted/frame_{i:03d}_{0:02d}.png', cv2img)

        # idx 1-4
        for j in range(len(hue_range)):
            for k in range(len(saturation_range)):
                hue = hue_range[j]
                satur = saturation_range[k]
                idx = j*len(saturation_range)+k 

                cv2img = cv2.cvtColor(image*255, cv2.COLOR_RGB2BGR)
                cv2img = adjust_hue(cv2img, hue)
                # cv2img = adjust_saturation(cv2img, satur)
                image_modified = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f'./adjusted/frame_{i:03d}_{idx+1:02d}.png', cv2img)
                data_new['images'][i*10+idx+1,:,:,:] = image_modified/255 
                data_new['poses'][i*10+idx+1,:,:] = pose 
                data_new['env'][i*10+idx+1,:] = np.array([hue, satur])
                # print(i*10+idx+1)

        # for idx 5-9
        for j in range(5,10):
            hue = np.random.uniform(*hue_range)
            satur = np.random.uniform(*saturation_range)

            cv2img = cv2.cvtColor(image*255, cv2.COLOR_RGB2BGR)
            cv2img = adjust_hue(cv2img, hue)
            cv2img = adjust_saturation(cv2img, satur)
            cv2.imwrite(f'./adjusted/frame_{i:03d}_{j:02d}.png', cv2img)
            image_modified = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
            data_new['images'][i*10+j,:,:,:] = image_modified/255
            data_new['poses'][i*10+j,:,:] = pose 
            data_new['env'][i*10+j,:] = np.array([hue, satur])
            # print(i*10+j)
        # print("aa")

    np.savez('tiny_nerf_data_new.npz', images=data_new['images'], poses=data_new['poses'], focal=data_new['focal'], env=data_new['env'])
    # testimg_idx = 101 
    # testimg, testpose = images[testimg_idx], poses[testimg_idx]
    # cv2img = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)

    # temp_range = [-30, 30]

    # temp = np.random.uniform(*temp_range)
    # hue = np.random.uniform(*hue_range)
    # satur = 1+np.random.uniform(*saturation_range)

    # # cv2img = adjust_color_temperature(cv2img, temp)
    # cv2img = adjust_hue(cv2img, hue)
    # cv2img = adjust_saturation(cv2img, satur)

    # test_img_modified = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)

    # print(f"hue: {hue}; satur: {satur}")

    # plt.figure(0)
    # plt.imshow(testimg)
    # plt.title('Original Image')
    # plt.figure(1)
    # plt.imshow(test_img_modified)
    # plt.title('Modified Image')
    # plt.show()
