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
from ns_dp_info import dpDict
from scipy.spatial.transform import Rotation

class NerfRenderer:
    def __init__(
        self,
        config_path: str, 
        width: int,
        height: int, 
        fx: float, 
        fy: float, 
        distortion_params: np.ndarray = None,
        camera_type = CameraType.PERSPECTIVE,
        metadata = None 
    ):
        self._script_dir = os.path.dirname(os.path.realpath(__file__))
        self.config_path = Path(os.path.join(self._script_dir, config_path))

        self.fx = fx
        self.fy = fy
        self.cx = width/2
        self.cy = height/2
        self.nerfW = width
        self.nerfH = height
        self.distortion_params = distortion_params
        self.camera_type  = camera_type

        self.focal = self.fx

        self.metadata = metadata

        _, pipeline, _, step = eval_setup(
            self.config_path,
            eval_num_rays_per_chunk=None,
            test_mode='inference'
        )
        self.model = pipeline.model 

    def render(self, cam_state:np.ndarray):
        # rpy = R.from_matrix(cam_state[0, :3,:3])
        if cam_state.ndim == 2:
            cam_state = np.expand_dims(cam_state, axis=0)
        
        camera_to_world = torch.FloatTensor( cam_state )

        camera = Cameras(camera_to_worlds = camera_to_world, fx = self.fx, fy = self.fy, cx = self.cx, cy = self.cy, width=self.nerfW, height=self.nerfH, distortion_params=self.distortion_params, camera_type=self.camera_type, metadata=self.metadata)
        camera = camera.to('cuda')
        ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)
        with torch.no_grad():
            tmp = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)

        img = tmp['rgb']
        img =(colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions())).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

class SplatRenderer:
    def __init__(self,
            config_path: str,    
            width: int,
            height: int,
            # fov: float,
            fx: float, 
            fy: float, 
            distortion_params: np.ndarray = None,
            camera_type = CameraType.PERSPECTIVE,
            metadata = None
        ):
        self._script_dir = os.path.dirname(os.path.realpath(__file__))
        self.config_path = Path(os.path.join(self._script_dir, config_path))


        self.fx = fx
        self.fy = fy
        self.cx = width/2
        self.cy = height/2
        self.nerfW = width
        self.nerfH = height
        self.distortion_params = distortion_params
        self.camera_type  = camera_type

        self.focal = self.fx

        self.metadata = metadata

        _, pipeline, _, step = eval_setup(
            self.config_path,
            eval_num_rays_per_chunk=None,
            test_mode='inference'
        )
        self.model = pipeline.model 

    def render(self, cam_state:np.ndarray):
        # rpy = R.from_matrix(cam_state[0, :3,:3])
        if cam_state.ndim == 2:
            cam_state = np.expand_dims(cam_state, axis=0)
        
        camera_to_world = torch.FloatTensor( cam_state )

        camera = Cameras(camera_to_worlds = camera_to_world, fx = self.fx, fy = self.fy, cx = self.cx, cy = self.cy, width=self.nerfW, height=self.nerfH, distortion_params=self.distortion_params, camera_type=self.camera_type, metadata=self.metadata)
        camera = camera.to('cuda')
        ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)

        with torch.no_grad():
            # tmp = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
            tmp = self.model.get_outputs_for_camera(camera)

        img = tmp['rgb']
        img =(colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions())).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        # image1 = self.set_dark_properties(self.set_fog_properties(img,fog_num), dark_num)
        # image1 = image1/255.

        # if save:
        #     output_dir = f"NeRF_UAV_simulation/images/Iteration_{iter}/{save_name}{particle_number}.jpg"
        #     cv2.imwrite(output_dir, img)

        return img
    
class GazeboSplatRenderer(SplatRenderer):
    def __init__(
        self, 
        config_path: str, 
        width: int, 
        height: int, 
        fx: float, 
        fy: float, 
        distortion_params: np.ndarray = None,
        camera_type = CameraType.PERSPECTIVE
    ):
        super().__init__(
            config_path,
            width,
            height,
            fx,
            fy,
            distortion_params,
            camera_type
        )

        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir,'../outputs/gazebo4_resampled3_dataset/splatfacto/2024-08-19_160517/config.yml')
        config_path = os.path.normpath(config_path)
        
        self.dp_trans_info = dpDict[config_path]

    def render(self, point: np.ndarray):
        # Set aircraft to pos
        camera_pose = np.zeros((4,4))
        camera_pose[3,3] = 1
        camera_pose[:3,:3] = Rotation.from_euler('xyz',[point[5],point[4],point[3]]).as_matrix()
        camera_pose[:3,3] = point[:3]

        # Convert camera pose to what's stated in transforms_orig.json
        tmp = Rotation.from_euler('zyx',[-np.pi/2,np.pi/2,0]).as_matrix()
        mat = camera_pose[:3,:3]@tmp 
        camera_pose[:3,:3] = mat 
        
        # Convert camera pose to Colmap frame in transforms.json
        camera_pose[0:3,1:3] *= -1
        camera_pose = camera_pose[np.array([0,2,1,3]),:]
        camera_pose[2,:] *= -1 

        transform = np.array(self.dp_trans_info['transform'])
        scale_factor = self.dp_trans_info['scale']
        camera_pose = transform@camera_pose 
        camera_pose[:3,3] *= scale_factor
        camera_pose = camera_pose[:3,:]

        image = super().render(camera_pose) 
        
        return img 


if __name__ == "__main__":
    fx = (1920/2)/(np.tan(np.deg2rad(50)/2))
    fy = (1080/2)/(np.tan(np.deg2rad(50)/2))

    renderer = SplatRenderer(
        '../outputs/gazebo4_transformed/splatfacto/2024-08-05_204928/config.yml',
        width = 2560,
        height = 1440,
        fx = 2343.0242837919386, 
        fy = 2343.0242837919386 
    )

    camera_pose = np.array([
[
                    0.8978860619290513,
                    -0.2417503202706265,
                    0.3679095030609964,
                    -614.2608252935024
                ],
                [
                    -2.133238163238067e-16,
                    0.8357250597051481,
                    0.5491480898453773,
                    575.7609662327382
                ],
                [
                    -0.4402279180078654,
                    -0.49307241580712685,
                    0.7503858827140766,
                    786.7511676140264
                ]])

    img = renderer.render(camera_pose)

    plt.imshow(img)
    plt.show()