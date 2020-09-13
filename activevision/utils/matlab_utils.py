import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from activevision.defaults import IMG_STRUCT_FNAME, IMG_STRUCT_FIELDS


def load_mat(path, file=None):
    if file:
        path = os.path.join(path, file)
    return loadmat(path)


def load_image_struct(path, file=None):
    """

    Parameters
    ----------
    path
    file

    Returns
    -------

    """
    if file is None and not path.endswith('.mat'):
        file = IMG_STRUCT_FNAME

    mat_file = load_mat(path, file)
    image_structs = mat_file['image_structs'][0]
    scale = mat_file['scale'][0][0]

    return image_structs, scale


def mat_to_dict(mat_file):
    return


def get_tR(image_name, image_struct):
    for idx, row in enumerate(image_struct):
        row_img_name = row[0][0]

        if row_img_name == image_name:
            t = row[1].copy()
            R = row[2].copy()

            return t, R

    return None


def get_world_pos(image_name, image_struct):
    for idx, row in enumerate(image_struct):
        row_img_name = row[0][0]

        if row_img_name == image_name:
            return row[3][:, 0].copy()

    return None


def get_all_world_pos(image_struct):
    world_pos_list = []

    for row in image_struct:
        world_pos_list.append(row[3][:, 0].copy())

    return np.unique(world_pos_list, axis=0)


def get_image_from_nodes(nodes, image_struct):
    images = []
    nodes = nodes.tolist()

    for row in image_struct:
        world_pos = list(row[3][:, 0][[0, 2]])

        if world_pos in nodes:
            row_img_name = row[0][0]
            images.append(row_img_name)

    return images


if __name__ == '__main__':
    # data_root = os.environ['ACTIVEVISION_DATASET']
    data_root = '/mnt/sda2/workspace/DATASETS/ActiveVision'
    load_image_struct(os.path.join(data_root, 'Home_003_1'))
