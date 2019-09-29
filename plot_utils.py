import numpy as np
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.mplot3d import Axes3D

# import plotly.express as px
import plotly.graph_objects as go


def plot3D(pts, color=None):
    n = 600000
    pts = pts[:, :n]
    print(f'Plotting points of shape {pts.shape}')

    fig = go.Figure(data=[
        go.Scatter3d(x=pts[0, :], y=pts[1, :], z=pts[2, :], mode='markers',
                     marker=dict(color=color, size=12
                                 # color=[f'rgb({i[0]}, {i[1]}, {i[2]})' for i in color[:, :n]]))],
                                 ))])

    fig.show()


# n = 100000
# def plot3D(pts, color=None):
#   ax = plt.axes(projection='3d')
#   print(pts.shape)
#   # raise Exception()
#   ax.scatter3D(pts[0, :], pts[1, :], pts[2, :])
#   plt.show()

def plot_in_image(img, coordinates, mode='2n', numbering=False, s=60,
                  fontsize=20, edgecolor=None):

    assert mode in ['2n', 'n2'], f'Invalid mode {mode}!!!'

    coordinates = np.array(coordinates)

    if mode == '2n':
        assert coordinates.shape[0] == 2, \
            f'In mode={mode} first dimension length must be 2!!!'
        coordinates = coordinates.T
    else:
        assert coordinates.shape[1] == 2, \
            f'In mode={mode} second dimension length must be 2!!!'

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
                   edgecolors=edgecolor)

    plt.show()


if __name__ == '__main__':
    import os
    from PIL import Image
    from parameters import root_path

    img1_name = '000110000010101.jpg'
    img2_name = '000110000860101.jpg'
    img1 = Image.open(os.path.join(root_path, 'jpg_rgb', img2_name))

    plot_in_image(img1, [[780, 600], [969, 526], [595, 693], [153, 847]])
