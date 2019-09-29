import numpy as np


def generate_flat_xyz(depth_image):
    """ Generate flattened xyz coordinates using depth image.
        The function assumes the input is either a PIL image with default
        dimension or a numpy array with shape (width, height). The function
        works even if the dimensions are switched but this has to be uniform
        throughout other calculations.

    Parameters
    ----------
    depth_image: PIL Image or numpy array
        The PIL image with default dimension or a numpy array with shape
        (width, height).

    Returns
    -------
    x_flat, y_flat, z_flat: numpy array
        Numpy arrays with flattened x, y and z coordinates.
    """
    depth = np.array(depth_image, dtype=np.float64)  # imh, imw
    imh, imw = depth.shape

    x_grid, y_grid = np.meshgrid(np.arange(imw), np.arange(imh))
    print(f'Each of the coordinate axes is of shape {x_grid.shape}')

    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)
    z_flat = depth.reshape(-1)

    return x_flat, y_flat, z_flat


def xy_to_index(xy, x_flat, y_flat, mode='n2'):
    """ Get indices in flattened axes that corresponds to input coordinates.

    Parameters
    ----------
    xy: list of numpy array
        List of coordinates of shape (num_pts, 2) or (2, num_pts) corresponding
        to the mode argument.
    x_flat: numpy array
        Flattened x-axis coordinates.
    y_flat: numpy array
        Flattened y-axis coordinates.
    mode: {'2n', 'n2'}, optional
        Determines the dimension of the input coordinates. If 2n then the shape
        must be (2, num_pts) and vice versa.

    Returns
    -------
        Indices for input coordinates that correspond simultaneously to X and Y
        coordinates in the flattened axes.
    """
    assert mode in ['2n', 'n2'], f'Invalid mode {mode}!!!'

    xy = np.array(xy)

    if mode == 'n2':
        assert xy.shape[1] == 2, \
            f'In mode={mode} second dimension length must be 2!!!'
    else:
        assert xy.shape[0] == 2, \
            f'In mode={mode} first dimension length must be 2!!!'
        xy = xy.T

    indices = []

    for a, b in xy:
        matched_index = np.where(np.logical_and(x_flat == a, y_flat == b))
        print(a, b, matched_index)
        indices.append(matched_index[0][0])

    return indices


def bbox_pixel_indices(bounding_box, x_flat, y_flat, z_flat,
                       filter_depth=False):
    """ Get indices of pixels inside the bounding box.
        Using boundaries of the bounding box, find the indices that correspond
        to pixels inside these boundaries in the flattened axes. Only find
        indices to non-zero depth values if `filter_depth` is set.

    Parameters
    ----------
    bounding_box: list or numpy array
        List containing the [xmin, ymin, xmax, ymax] of the bounding box.
    x_flat: numpy array
        Flattened x-axis coordinates.
    y_flat: numpy array
        Flattened y-axis coordinates.
    z_flat: numpy array
        Flattened y-axis coordinates.
    filter_depth: bool, optional (default=False)
        Flag for whether to filter the indices that have zero z-values.

    Returns
    -------
    matched_indices: list
        Indices of the flattened axes that correspond to the pixels inside the
        bounding box that have valid depth values.
    """
    xmin = bounding_box[0]
    ymin = bounding_box[1]
    xmax = bounding_box[2]
    ymax = bounding_box[3]

    valid_x = np.logical_and((x_flat >= xmin), (x_flat <= xmax))
    valid_y = np.logical_and((y_flat >= ymin), (y_flat <= ymax))

    if filter_depth:
        valid_z = z_flat != 0
        # all_and = np.logical_and(np.logical_and(valid_x, valid_y), valid_z)
        all_and = np.logical_and.reduce(np.array((valid_x, valid_y, valid_z)))
    else:
        all_and = np.logical_and(valid_x, valid_y)

    matched_indices = all_and.nonzero()[0]

    return matched_indices


def bbox_pixel_indices_list(bounding_boxes, x_flat, y_flat, z_flat,
                            filter_depth=False, coordinates=False):
    """ Get either indices or coordinates of pixels for each bounding box.
        Depending of the flags, the function returns a list of list of either
        the indices or the coordinates inside the bounding boxes.

    Parameters
    ----------
    bounding_boxes: list or numpy array
        List containing the [[xmin1, ymin1, xmax1, ymax1],
        [xmin2, ymin2, xmax2, yman2], ...] of each bounding box.
    x_flat: numpy array
        Flattened x-axis coordinates.
    y_flat: numpy array
        Flattened y-axis coordinates.
    z_flat: numpy array
        Flattened y-axis coordinates.
    filter_depth: bool, optional (default=False)
        Flag for whether to filter the indices that have zero z-values.
    coordinates: bool, optional (default=False)
        Flag for whether to return coordinates instead of indices.

    Returns
    -------
    matched_indices: numpy array
        List of indices of the flattened axes that correspond to the pixels
        inside each bounding box.
    """

    bboxes_xyz_list = []

    for bbox in bounding_boxes:
        matched_idx = bbox_pixel_indices(bbox, x_flat=x_flat, y_flat=y_flat,
                                         z_flat=z_flat,
                                         filter_depth=filter_depth)
        if coordinates:
            bboxes_xyz_list.append(np.array((x_flat[matched_idx],
                                            y_flat[matched_idx],
                                            z_flat[matched_idx])))
        else:
            bboxes_xyz_list.append(matched_idx)

    return bboxes_xyz_list


if __name__ == '__main__':
    w = 5
    h = 4

    rand_x, rand_y = np.meshgrid(np.arange(w), np.arange(h))
    rand_x = rand_x.reshape(-1)
    rand_y = rand_y.reshape(-1)
    coordinates_to_find = [[np.random.randint(w), np.random.randint(h)],
                           [np.random.randint(w), np.random.randint(h)]]
    found_indices = xy_to_index(coordinates_to_find, rand_x, rand_y)

    print(rand_x)
    print(rand_y)
    print(f'Coordinates to find = {coordinates_to_find}')
    print(f'Found indices = {found_indices}')
    for i in found_indices:
        print(rand_x[i], rand_y[i])
    # print(generate_flat_xyz(np.random.randint(0, 5, (8, 3))))
