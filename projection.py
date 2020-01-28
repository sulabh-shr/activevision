import numpy as np

from active_vision_utils.parameters import *
from active_vision_utils.coordinate_utils import generate_flat_xyz
from active_vision_utils.tR_utils import camera_to_world_tR, inter_camera_tR


def project_xyz_to_camera(x_flat, y_flat, z_flat, center_x=cx,
                          center_y=cy, focal_x=fx, focal_y=fy,
                          filter_depth=False, rgb=None):
    """ Project xyz in image coordinate to camera coordinate system.
        This function assumes that xyz do not need to be checked for unusual
        depth value because it's either already checked or not required.

    Parameters
    ----------
    x_flat:
    y_flat:
    z_flat
    center_x
    center_y
    focal_x
    focal_y
    filter_depth:
    rgb : None or numpy array, optional (default=None)
        Numpy array of shape (3, num_pts) containing the rgb values of each
        pixel in 2d image.

    Returns
    -------

    """
    # Center align so that (0, 0) corresponds to origin of camera coordinate
    x_flat = x_flat - center_x
    y_flat = y_flat - center_y

    # The 2d to 3d transformation --> [(x1, ...), (y1, ...), (z1, ...)]
    pcloud = np.array((x_flat * z_flat / focal_x, y_flat * z_flat / focal_y,
                       z_flat))  # (3, num_pts)

    if filter_depth:
        print(f'Filtering z-values while projecting xyz to camera.')
        # Find valid indices i.e. those that have non-zero depth values
        valid_idx = z_flat != 0
        pcloud = pcloud[:, valid_idx]  # (3, valid_idx)

        if rgb is not None:
            rgb = rgb[:, valid_idx]

    return pcloud, rgb


def project_img_to_camera(image, depth, center_x=cx, center_y=cy, focal_x=fx,
                          focal_y=fy, filter_depth=False, project_rgb=True):
    """ Project 2d image to Camera coordinate.

    Parameters
    ----------
    image: PIL Image
        The 2d image of shape (width, height, 3) in which the third dimension
        denotes the rgb values.
    depth: numpy array
        Depth image of shape (width, height) with each pixel denoting its depth.
    center_x: int or float, optional
        Value by which x axis needs to be translated so that x = 0 is center
        aligned with origin of camera coordinate system. If not specified,
        default value from
    center_y: int or float
        Value by which y axis needs to be translated so that y=0 is center
        aligned with origin of camera coordinate system.
    focal_x: int or float
        Focal length in x direction.
    focal_y: int or float
        Focal length in y direction.
    filter_depth: bool, optional (default=False)
        Flag for filtering the pixels that have zero z-values.
    project_rgb: bool, optional (default=True)
        Flag for getting respective rgb values in camera coordinate.

    Returns
    -------
    pcloud: numpy array
        3d point cloud in Camera coordinate system of shape (3, valid_points)
        where values in first dimension are xyz.
    rgb: numpy array or None
        None if project_rgb is False. Else, rgb values of each 3d coordinate of
        shape (3, valid_points) where values in first dimension are rgb.
    """

    # Converting PIL Image to np array SWAPS the dimension
    depth = np.array(depth, dtype=np.float64)  # (imh, imw)

    if project_rgb:
        rgb = np.array(image).T.reshape(3, -1)  # (3, num_pts)
    else:
        rgb = None

    # X and Y coordinate of each pixel in separate matrix
    # X increases going left to right and remains constant for single column
    # Y increases going top to bottom and remains constant for single row
    x_flat, y_flat, z_flat = generate_flat_xyz(depth)

    pcloud, rgb = project_xyz_to_camera(x_flat, y_flat, z_flat,
                                        center_x=center_x, center_y=center_y,
                                        focal_x=focal_x, focal_y=focal_y,
                                        filter_depth=filter_depth, rgb=rgb)

    return pcloud, rgb


def project_camera_to_2d(pcl, center_x=cx, center_y=cy,
                         focal_x=fx, focal_y=fy):
    z = pcl[2, :]
    x = pcl[0, :] * focal_x / z + center_x
    y = pcl[1, :] * focal_y / z + center_y

    return np.array([x, y])


if __name__ == '__main__':
    import os
    from PIL import Image
    from matlab_utils import get_tR, load_image_struct
    from plot_utils import plot_in_image

    # IMPORT IMAGE

    # img1_name = '000110000010101.jpg'
    # img2_name = '000110000860101.jpg'
    img1_name = '000110012940101.jpg'
    img2_name = '000110012950101.jpg'
    depth1_name = img1_name.split('.')[0][:-1] + '3.png'
    depth2_name = img2_name.split('.')[0][:-1] + '3.png'

    img1 = Image.open(os.path.join(root_path, 'jpg_rgb', img1_name))
    img2 = Image.open(os.path.join(root_path, 'jpg_rgb', img2_name))
    depth1 = Image.open(os.path.join(root_path, 'high_res_depth', depth1_name))
    depth2 = Image.open(os.path.join(root_path, 'high_res_depth', depth2_name))

    # PROJECT 2D IMAGE TO 3D CAMERA COORDINATE
    pcl_cam1, rgb_cam1 = project_img_to_camera(img1, depth1, filter_depth=False,
                                               project_rgb=True)

    print(f'The point cloud is of shape: {pcl_cam1.shape}')

    # pcl1, rgb3d1 = project_img_to_camera(img1, depth1)
    image_struct, scale = load_image_struct(root_path)

    # LOAD CAMERA TRANSFORMATION PARAMETERS
    tc1w, Rc1w = get_tR(img1_name, image_struct)
    tc2w, Rc2w = get_tR(img2_name, image_struct)
    tc1w = tc1w * scale
    tc2w = tc2w * scale
    twc1, Rwc1 = camera_to_world_tR(tc1w, Rc1w)

    # FIND CAMERA1 TO CAMERA2 TRANSFORMATION PARAMETERS
    tc2c1, Rc2c1 = inter_camera_tR(twc1, Rwc1, tc2w, Rc2w)

    # TRANSFORM CAMERA1 TO CAMERA2 COORDINATE SYSTEM
    pcl_cam21 = np.matmul(Rc2c1, pcl_cam1) + tc2c1

    print(f'The transformed cloud is of shape: {pcl_cam21.shape}')

    # PROJECT CAMERA2 TO IMAGE2
    proj_img2 = project_camera_to_2d(pcl_cam21)

    print(proj_img2.shape)
    print(proj_img2)
    print(proj_img2.T)

