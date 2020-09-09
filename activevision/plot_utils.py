import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D

# import plotly.express as px
# import plotly.graph_objects as go
#
#
# def plot3D(pts, color=None):
#     n = 600000
#     pts = pts[:, :n]
#     print(f'Plotting points of shape {pts.shape}')
#
#     fig = go.Figure(data=[
#         go.Scatter3d(x=pts[0, :], y=pts[1, :], z=pts[2, :], mode='markers',
#                      marker=dict(color=color, size=12
#                                  # color=[f'rgb({i[0]}, {i[1]}, {i[2]})' for i in color[:, :n]]))],
#                                  ))])
#
#     fig.show()


# n = 100000
# def plot3D(pts, color=None):
#   ax = plt.axes(projection='3d')
#   print(pts.shape)
#   # raise Exception()
#   ax.scatter3D(pts[0, :], pts[1, :], pts[2, :])
#   plt.show()

def scatterplot_in_img(img, coordinates, mode='2n', numbering=False, s=60,
                  fontsize=20, edgecolor=None, fig=None, ax=None, return_fig=False):

    assert mode in ['2n', 'n2'], f'Invalid mode {mode}!!!'

    coordinates = np.array(coordinates)

    if mode == '2n':
        assert coordinates.shape[0] == 2, \
            f'In mode={mode} first dimension length must be 2!!!'
        coordinates = coordinates.T
    else:
        assert coordinates.shape[1] == 2, \
            f'In mode={mode} second dimension length must be 2!!!'

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(25, 14))
        ax.imshow(img)

    if numbering:
        count = 0
        for x, y in coordinates:
            ax.scatter(x, y, s=s, color='r', edgecolors=edgecolor)
            ax.text(x, y, str(count), fontsize=fontsize, color='black')
            count += 1
    else:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=s, color='r',
                   edgecolors=edgecolor, alpha=0.4)

    if return_fig:
        return fig, ax

    plt.show()


def bboxplot_in_img(img, bboxes, mode='xyxy', s=60, fontsize=20, edgecolor='b',
                    linewidth=1, fig=None, ax=None, return_fig=False, numbering=True):

    assert mode in ['xyxy', 'xywh'], f'Invalid mode {mode}!!!'

    bboxes = np.array(bboxes, dtype=float)

    if mode == 'xyxy':
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(25, 14))
        ax.imshow(img)
    count = 0

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
        if numbering:
            ax.text(bbox[0], bbox[1], str(count), fontsize=fontsize, color='black')
        count += 1

    if return_fig:
        return fig, ax

    plt.show(fig)


if __name__ == '__main__':
    import os
    from PIL import Image
    from parameters import root_path

    img1_name = '000110000010101.jpg'
    img2_name = '000110000860101.jpg'
    img1 = Image.open(os.path.join(root_path, 'jpg_rgb', img2_name))

    scatterplot_in_img(img1, [[780, 600], [969, 526], [595, 693], [153, 847]])
