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


def bboxplot_in_img(img, bboxes, labels=None, mode='xyxy', box_attribs=None, text_attribs=None,
                    fig=None, ax=None, return_fig=False, label_number=True):
    """

    Parameters
    ----------
    img: PIL Image
        Image to plot the boxes on. If 'fig' and 'ax' both are provided, this can be None.
    bboxes: list or numpy array
        Bounding boxes to plot with values in format corresponding to 'mode'. The length of each
        box can be greater than 4 (for example when they have label index or difficulty), but only
        the first 4 elements of each box are used.
    labels: list or None
        List of labels corresponding to bounding boxes for add text above boxes. Length must be same
        as the number of boxes. Can be None for skipping plotting labels.
    mode: 'xyxy' or 'xywh'
        Mode in which the bounding boxes values are input.
    box_attribs: dict or None
        Bounding box attributes.
    text_attribs: dict or None
        Text box attributes. The bbox attribute surrounding the text can also be set in 'box' key.
    fig: matplotlib.fig
    ax: matplotlib.axes
    return_fig: bool
        Flag to return fig. If true, fig and axes are returned without being shown.
    label_number: bool
        Flag to add index of box when text labels are absent. Does not work when labels are
        provided.

    Returns
    -------

    """
    assert mode in ['xyxy', 'xywh'], f'Invalid mode {mode}!!!'
    assert labels is None or len(bboxes) == len(labels), \
        f'Number of bounding boxes and labels mismatch: {len(bboxes)} vs {len(labels)}'

    # Default values for text and box attributes
    box_p = {'linewidth': 2,
             'colors': ['r', 'g', 'b', 'y', 'brown', 'orange']}
    text_p = {'fontsize': 8, 'color': 'black'}
    text_box = {'facecolor': 'wheat', 'alpha': 0.5, 'pad': 0.1, 'boxstyle': 'round'}

    # Update default text and box attributes with specified values
    if box_attribs is not None:
        for k, v in box_attribs.items():
            assert k in box_p, f'Invalid key for box attribute: {k}'
            box_p[k] = v
    if text_attribs is not None:
        for k, v in text_attribs.items():
            if k != 'box':
                assert k in text_p, f'Invalid key for text attribute: {k}'
                text_p[k] = v
            else:
                text_box.update(v)

    # Create a new plot if not provided
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(25, 14))

    ax.imshow(img)

    if len(bboxes) > 0:
        # Convert to xywh
        bboxes = np.array(bboxes, dtype=float)
        if mode == 'xyxy':
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

        # Plot boxes
        for idx, bbox in enumerate(bboxes):
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=box_p['linewidth'],
                                     edgecolor=box_p['colors'][idx % len(box_p['colors'])],
                                     facecolor='none')
            ax.add_patch(rect)
            if labels:
                ax.text(bbox[0], max(bbox[1] - 5, 0), labels[idx], fontdict=text_p, bbox=text_box)
            elif label_number:
                ax.text(bbox[0], max(bbox[1] - 5, 0), str(idx), fontdict=text_p, bbox=text_box)

    if return_fig:
        return fig, ax

    plt.pause(0.01)


if __name__ == '__main__':
    pass
