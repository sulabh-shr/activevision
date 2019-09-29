import os
from scipy.io import loadmat


def load_mat(path, file=None):
    if file:
        path = os.path.join(path, file)
    return loadmat(path)


def load_image_struct(path, file=None):
    if file is None and not path.endswith('.mat'):
        file = 'image_structs.mat'

    mat_file = load_mat(path, file)

    image_structs = mat_file['image_structs'][0]
    scale = mat_file['scale'][0][0]

    return image_structs, scale


def get_tR(image_name, image_struct):
    
    for idx, row in enumerate(image_struct):
        row_img_name = row[0][0]

        if row_img_name == image_name:
            t = row[1]
            R = row[2]
            
            return t, R

    return None 
