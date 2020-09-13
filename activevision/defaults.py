camera = {
    # Use Scene 1 camera intrinsics as default
    'intrinsics': {'fx': 1049.51,
                   'fy': 1092.28,
                   'cx': 927.269,
                   'cy': 545.76}
}
ALL_SCENES = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1', 'Home_003_2', 'Home_004_1',
              'Home_004_2', 'Home_005_1', 'Home_005_2', 'Home_006_1', 'Home_007_1', 'Home_008_1',
              'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1', 'Home_014_2', 'Home_015_1',
              'Home_016_1', 'Office_001_1']
IMG_STRUCT_FIELDS = ['image_name', 't', 'R', 'world_pos', 'direction', 'quat',
                       'scaled_world_pos', 'image_id', 'camera_id', 'cluster_id',
                       'rotate_cw', 'rotate_ccw', 'translate_forward',
                       'translate_backward', 'translate_right', 'translate_left']
IMG_STRUCT_FNAME = 'image_structs.mat'
