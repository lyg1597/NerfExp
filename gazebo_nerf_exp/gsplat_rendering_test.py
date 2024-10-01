import torch 
from nerfstudio.models.splatfacto import SplatfactoModel
# from nerfstudio.utils import load_config, load_model
# from nerfstudio
from scipy.spatial.transform import Rotation as R 
import cv2 
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps
import numpy as np 
import os 
from pathlib import Path
import matplotlib.pyplot as plt 
from ns_renderer import SplatRenderer

# class SplatRenderer:
#     def __init__(self,
#             config_path: str,    
#             width: int,
#             height: int,
#             fov: float,
#             camera_type = CameraType.PERSPECTIVE
#         ):
#         self._script_dir = os.path.dirname(os.path.realpath(__file__))
#         self.config_path = Path(os.path.join(self._script_dir, config_path))


#         self.fx = (width/2)/(np.tan(np.deg2rad(fov)/2))
#         self.fy = (height/2)/(np.tan(np.deg2rad(fov)/2))
#         self.cx = width/2
#         self.cy = height/2
#         self.nerfW = width
#         self.nerfH = height
#         self.camera_type  = camera_type

#         self.focal = self.fx

        
#         _, pipeline, _, step = eval_setup(
#             self.config_path,
#             eval_num_rays_per_chunk=None,
#             test_mode='inference'
#         )
#         self.model = pipeline.model 

#     def render(self, cam_state):
#         # rpy = R.from_matrix(cam_state[0, :3,:3])
        
#         camera_to_world = torch.FloatTensor( cam_state )

#         camera = Cameras(camera_to_worlds = camera_to_world, fx = self.fx, fy = self.fy, cx = self.cx, cy = self.cy, width=self.nerfW, height=self.nerfH, camera_type=self.camera_type)
#         camera = camera.to('cuda')
#         ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)

#         with torch.no_grad():
#             # tmp = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
#             tmp = self.model.get_outputs_for_camera(camera)

#         img = tmp['rgb']
#         img =(colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions())).cpu().numpy()
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = (img * 255).astype(np.uint8)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


#         # image1 = self.set_dark_properties(self.set_fog_properties(img,fog_num), dark_num)
#         # image1 = image1/255.

#         # if save:
#         #     output_dir = f"NeRF_UAV_simulation/images/Iteration_{iter}/{save_name}{particle_number}.jpg"
#         #     cv2.imwrite(output_dir, img)

#         return img
    
if __name__ == "__main__":
    renderer = SplatRenderer(
        '../outputs/gazebo4_transformed/splatfacto/2024-08-05_204928/config.yml',
        width = 1920,
        height = 1080,
        fov = 50
    )

    camera_pose = np.array([[
        [-2.6127e-02,  2.3050e-01, -9.7272e-01, -9.9032e-01],
        [-9.9966e-01, -6.0243e-03,  2.5423e-02,  8.8765e-02],
        [ 5.5511e-17,  9.7305e-01,  2.3058e-01, -2.1638e-01]
    ]])

    img = renderer.render(camera_pose)

    plt.imshow(img)
    plt.show()