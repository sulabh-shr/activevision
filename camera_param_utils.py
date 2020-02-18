import os
from .defaults import camera


def load_camera_params(dataset_root, scene, return_default=False):
    CAMERA_PARAMS_FOLDER = 'camera_params'
    CAMERA_PARAMS_FILENAME = 'cameras.txt'

    camera_params_path = os.path.join(dataset_root, CAMERA_PARAMS_FOLDER)

    if os.path.isdir(camera_params_path):
        scene_params_path = os.path.join(camera_params_path, scene)
    else:
        scene_params_path = os.path.join(dataset_root, scene)

    scene_params_path = os.path.join(scene_params_path, CAMERA_PARAMS_FILENAME)

    if os.path.isfile(scene_params_path):
        with open(scene_params_path, 'r') as f:
            file_contents = f.read()
        params_line = file_contents.split('\n')[3]
        params_split = params_line.split(' ')

        return {
            'fx': float(params_split[4]),
            'fy': float(params_split[5]),
            'cx': float(params_split[6]),
            'cy': float(params_split[7])
        }
    elif return_default:
        print(f'Camera parameters not found for **{scene}** in \n\t {scene_params_path}.'
              '\nReturning default values.')
        return camera['intrinsics']
    else:
        raise FileNotFoundError(f'Path {scene_params_path} does not exists!')
