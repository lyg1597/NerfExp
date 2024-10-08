import nerfstudio
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
import torch 
import numpy as np 
import json 
import os 
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import yaml
import matplotlib.pyplot as plt 
from nerfstudio.utils import colormaps

if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fn = os.path.join(script_dir, '../data/IRL2/transforms.json')
    with open(fn,'r') as f:
        data = json.load(f)
    transforms = []
    frames = data['frames']
    for frame in frames:
        transform_matrix = frame['transform_matrix']
        tmp = np.array(transform_matrix)[:3,:]
        transforms.append(tmp)
    transforms = np.array(transforms)
    print(transforms.shape)
    transforms_tensor = torch.FloatTensor(transforms)
    scene_box = SceneBox.from_camera_poses(transforms_tensor, 1)
    
    config_fn = os.path.join(script_dir, '../outputs/IRL2/nerfacto/2023-09-14_230330/config.yml')
    config_path = Path(config_fn)
    _, pipeline, _, step = eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode='inference'
    )
    # config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    # pipeline = config.pipeline.setup(device='cuda', test_mode='test')
    model = pipeline.model
    # scene_box = SceneBox(aabb=torch.FloatTensor([[-1.,-1.,-1.],[1.,1.,1.]]))
    # model = config.pipeline.model.setup(
    #     scene_box = scene_box,
    #     num_train_data = 0,
    # ).to('cuda')
    camera_to_world = torch.FloatTensor([
       [ 6.6384e-01, -1.2349e-01,  7.3761e-01,  6.1554e-01],
        [ 7.4787e-01,  1.0962e-01, -6.5473e-01, -5.5424e-01],
        [ 6.9389e-17,  9.8627e-01,  1.6512e-01, -9.1273e-03]
    ])
    # camera_to_world = torch.FloatTensor([
    #     [
    #         0.012817036360502243,
    #         0.19525104761123657,
    #         -0.9806695580482483,
    #         -0.07062844187021255
    #     ],
    #     [
    #         0.9997958540916443,
    #         0.012817036360502243,
    #         0.015618880279362202,
    #         -0.027591701596975327
    #     ],
    #     [
    #         0.015618880279362202,
    #         -0.9806695580482483,
    #         -0.19504690170288086,
    #         -0.001556280069053173
    #     ]
    # ])
    fx = fy = (320.0/2)/(np.tan(np.deg2rad(50)/2))
    cx = cy = 160.0
    width = height = 320
    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    camera_type  = CameraType.PERSPECTIVE
    camera = Cameras(camera_to_worlds = camera_to_world, fx = fx, fy = fy, cx = cx, cy = cy, width=width, height=height, camera_type=camera_type)
    camera = camera.to('cuda')
    ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)
    print(len(ray_bundle))


    # model = Model(config, scene_box, num_train_data=1)


    # loaded_state = torch.load(model_fn)
    # model.load_state_dict(loaded_state)
    with torch.no_grad():
        tmp = model.get_outputs_for_camera_ray_bundle(ray_bundle)

    # img = model.get_rgba_image(res)
    img = tmp['rgb']
    img = (colormaps.apply_colormap(image=img, colormap_options=colormaps.ColormapOptions())).cpu().numpy()
    plt.imshow(img)
    plt.show()

    print("stop")
