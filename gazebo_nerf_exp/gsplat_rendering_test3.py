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
import torch
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
        '../outputs/gazebo5_transformed_env-1/splatfacto-env/2024-10-15_005454/config.yml',
        2560, 
        1440, 
        2343.0242837919386, 
        2343.0242837919386,
        metadata={"env_params":torch.tensor([0.5,0.5])}
    )

    camera_pose = np.array([
                [
                    0.07585322596629693,
                    0.2607750091672529,
                    -0.9624150262253418,
                    -3050.83116632528
                ],
                [
                    -2.7412222166117605e-16,
                    0.9651957610452591,
                    0.26152847428198583,
                    407.8386916124485
                ],
                [
                    0.9971189939573442,
                    -0.019837778456332084,
                    0.07321321216427805,
                    114.17181528627293
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
    ])

    transform = np.array([
        [
            0.9996061325073242,
            -0.01975083164870739,
            -0.01993674598634243,
            997.8822021484375
        ],
        [
            -0.01975083164870739,
            0.009563744068145752,
            -0.9997591972351074,
            -42.26317596435547
        ],
        [
            0.01993674598634243,
            0.9997591972351074,
            0.00916987657546997,
            -242.0419158935547
        ]
    ])

    scale_factor = 0.0003946526873285077

    camera_pose = transform@camera_pose
    camera_pose[:3,3] *= scale_factor
    camera_pose = camera_pose[:3,:]

    img = renderer.render(camera_pose)

    plt.imshow(img)
    plt.show()