import os
from PIL import Image

from parameters import *
from projection import *
from coordinate_utils import *
from plot_utils import *
from matlab_utils import *

img1_name = '000110000860101.jpg'
depth1_name = img1_name.split('.')[0][:-1] + '3.png'
img2_name = '000110000010101.jpg'
depth2_name = img2_name.split('.')[0][:-1] + '3.png'

img1 = Image.open(os.path.join(root_path, 'jpg_rgb', img1_name))
depth1 = Image.open(os.path.join(root_path, 'high_res_depth', depth1_name))
img2 = Image.open(os.path.join(root_path, 'jpg_rgb', img2_name))
depth2 = Image.open(os.path.join(root_path, 'high_res_depth', depth2_name))

image_struct, scale = load_image_struct(root_path)

tc1w, Rc1w = get_tR(img1_name, image_struct)
tc2w, Rc2w = get_tR(img2_name, image_struct)
tc1w = tc1w * scale
tc2w = tc2w * scale
twc1, Rwc1 = camera_to_world_tR(tc1w, Rc1w)

tc2c1, Rc2c1 = inter_camera_tR(twc1, Rwc1, tc2w, Rc2w)
# --------------------------------------------------------------

# plot_in_image(img1, [[1030, 630], [1135, 820], [1453, 522], [1590, 820],
#                     [595, 550], [632, 659]], mode='n2', numbering=False)

bboxes = [[1030, 630, 1135, 820], [1453, 522, 1590, 820],
          [595, 550, 632, 659]]


# GENERATE XYZ
x, y, z = generate_flat_xyz(depth1)

# GET INDICES OF EACH BOUNDING BOX'S PIXELS
bbox_px_idx = bbox_pixel_indices_list(np.array(bboxes), x_flat=x, y_flat=y,
                                      z_flat=z, filter_depth=True,
                                      coordinates=False)
for i in bbox_px_idx:
    print(min(z[i]), max(z[i]), sum(z[i])/len(z[i]))
    plot_in_image(img1, coordinates=(x[i], y[i]))

pcl_cam1, _ = project_xyz_to_camera(x_flat=x, y_flat=y, z_flat=z)
pcl_cam21 = np.matmul(Rc2c1, pcl_cam1) + tc2c1
proj21 = project_camera_to_2d(pcl_cam21)

for i in bbox_px_idx:
    plot_in_image(img2, coordinates=(proj21[0][i], proj21[1][i]))

