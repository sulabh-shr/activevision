import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from active_vision_utils.matlab_utils import load_image_struct, get_all_world_pos


# direction_list = ['left', 'rotate_ccw', 'forward', 'rotate_cw', 'right',
#                   'backward']
direction_list = ['rotate_ccw', 'forward', 'rotate_cw']
root_path = '/mnt/sda2/workspace/DATASETS/ActiveVision'


def get_neighbors(img_name, root_path=None, folder=None, ann=None):
    neighbors = []

    if folder is not None:
        assert root_path is not None, 'Root path must be provided with folder!!!'
        if ann is not None:
            warnings.warn('Annotation parameter is not being used!')
        ann_path = os.path.join(root_path, folder, 'annotations.json')
        with open(ann_path, 'r') as f:
            ann = json.load(f)
    elif ann is None:
        raise ValueError(
            'Both folder and annotation parameters cannot be None!!!')

    current_img_ann = ann[img_name]
    for direction in direction_list:
        neighbor_img = current_img_ann[direction]
        # Remove empty and self-referring neighbor
        if neighbor_img != '' and neighbor_img != img_name:
            neighbors.append(neighbor_img)

    return neighbors


def get_bfs_nodes(root_path=None, folder=None, image_struct=None,
                  near_threshold=25, visualize_nodes=False, test_split=0.3):

    if folder is not None:
        assert root_path is not None, 'Root path must be provided with folder!!!'
        if image_struct is not None:
            warnings.warn('Image_struct parameter is not being used!')
        image_struct, scale = load_image_struct(os.path.join(root_path, folder))
    elif image_struct is None:
        raise ValueError(
            'Both folder and image_struct parameters cannot be None!!!')

    all_nodes = get_all_world_pos(image_struct=image_struct)
    all_nodes_xy = all_nodes[:, [0, 2]].copy()
    all_nodes_xy = all_nodes_xy*scale

    clustering = DBSCAN(eps=near_threshold, min_samples=1)
    labels = clustering.fit_predict(all_nodes_xy)

    # Get unique sorted labels
    labels_sorted = np.unique(labels)
    labels_sorted.sort()
    print(f'Clusters = {len(labels_sorted)}')
    num_train_nodes = round(len(labels_sorted)*(1-test_split))
    print(f'Num train nodes = {num_train_nodes}')
    nearest_cluster_centers = []

    # Find node in each cluster nearest to cluster mean
    for label in labels_sorted:
        cluster_points = all_nodes_xy[labels == label]
        cluster_mean = np.mean(cluster_points, axis=0)
        distance_from_cluster_mean = np.linalg.norm(cluster_points-cluster_mean,
                                                    axis=1)
        nearest_cluster_centers.append(cluster_points[
                                           distance_from_cluster_mean.argmin()])
    nearest_cluster_centers = np.array(nearest_cluster_centers)

    if visualize_nodes:
        plt.scatter(nearest_cluster_centers[:, 0], nearest_cluster_centers[:, 1])
        plt.axis('equal')
        plt.show()

    # Populate queue with first node
    remaining_center_nodes = nearest_cluster_centers.copy().tolist()
    start_idx = np.random.randint(len(labels_sorted))
    queue = [remaining_center_nodes[start_idx]]
    processed_idx = [start_idx]
    bfs_nodes = []

    while len(queue) != 0 and len(bfs_nodes) < num_train_nodes:

        current_node = queue.pop(0)
        bfs_nodes.append(current_node)
        # remaining_center_nodes.remove(current_node)
        print(f'Calculating for {current_node}')

        difference = np.array(remaining_center_nodes) - current_node
        distance = np.linalg.norm(difference, axis=1)
        # Normalize be nearest neighbor node distance (1 because 0 is self)
        normalized_distance = distance/sorted(distance)[1]

        # TODO: Make this a parameter
        next_hierarchy_indices = np.where(normalized_distance <= 1.3)[0]
        plt.scatter(nearest_cluster_centers[:, 0],
                    nearest_cluster_centers[:, 1])

        plt.scatter(current_node[0], current_node[1], edgecolors='black',
                    c='y', label='current')

        print(f'Neighbors are:')
        for idx in next_hierarchy_indices:
            next = remaining_center_nodes[idx]
            print(next)
            if idx not in processed_idx:
                plt.scatter(next[0], next[1], edgecolors='pink', c='brown',
                            label='neighbor')
                if remaining_center_nodes[idx] not in queue:
                    queue.append(remaining_center_nodes[idx])
                processed_idx.append(idx)

        # Last one is current node
        for coord in bfs_nodes[1:-1]:
            plt.scatter(coord[0], coord[1], edgecolors='blue', c='white')

        if len(bfs_nodes)>1:
            plt.scatter(bfs_nodes[0][0], bfs_nodes[0][1], edgecolors='r', c='r')
        # Current node

        plt.axis('equal')
        plt.legend()
        plt.show()


def distance_sort_nodes(root_path=None, folder=None, image_struct=None,
                        scale=None, near_threshold=25, visualize_nodes=False):

    if folder is not None:
        assert root_path is not None, 'Root path must be provided with folder!!!'
        if image_struct is not None:
            warnings.warn('Image_struct parameter is not being used!')
        image_struct, scale = load_image_struct(os.path.join(root_path, folder))
    elif image_struct is None or scale is None:
        raise ValueError(
            'Both folder and image_struct/scale parameters cannot be None!!!')

    all_nodes = get_all_world_pos(image_struct=image_struct)
    all_nodes_xy = all_nodes[:, [0, 2]].copy()
    all_nodes_xy = all_nodes_xy*scale

    clustering = DBSCAN(eps=near_threshold, min_samples=1)
    labels = clustering.fit_predict(all_nodes_xy)

    # Get unique sorted labels
    labels_sorted = np.unique(labels)
    labels_sorted.sort()
    print(f'Clusters = {len(labels_sorted)}')

    nearest_cluster_centers = []

    # Find node in each cluster nearest to cluster mean
    for label in labels_sorted:
        cluster_points = all_nodes_xy[labels == label]
        cluster_mean = np.mean(cluster_points, axis=0)
        distance_from_cluster_mean = np.linalg.norm(cluster_points-cluster_mean,
                                                    axis=1)
        nearest_cluster_centers.append(cluster_points[
                                           distance_from_cluster_mean.argmin()])
    nearest_cluster_centers = np.array(nearest_cluster_centers)

    start_idx = np.random.randint(len(nearest_cluster_centers))
    start_node = nearest_cluster_centers[start_idx]
    distance_from_start_node = np.linalg.norm(nearest_cluster_centers-start_node,
                                              axis=1)
    sorted_idx = distance_from_start_node.argsort()
    sorted_cluster_centers = nearest_cluster_centers[sorted_idx]
    sorted_clusters = []

    for idx in sorted_idx:
        cluster_label = labels_sorted[idx]
        cluster_node_indices = np.where(labels==cluster_label)
        cluster_nodes = all_nodes_xy[cluster_node_indices]
        sorted_clusters.append(cluster_nodes)

    if visualize_nodes:
        plt.scatter(nearest_cluster_centers[:, 0], nearest_cluster_centers[:, 1])

        plt.plot(start_node[0], start_node[1], 'ro', label='Starting Node')

        for i in range(0):
            plt.scatter(sorted_cluster_centers[i][0], sorted_cluster_centers[i][1], edgecolors='b',
                        c='white')
        plt.title('Center nodes of the cluster')
        plt.axis('equal')
        plt.legend()
        plt.show()

    return sorted_cluster_centers/scale, np.array(sorted_clusters)/scale


if __name__ == '__main__':
    d = get_hierarchy_neighbors(img_name='000110000010101.jpg',
                                root_path='/mnt/sda2/workspace/DATASETS/ActiveVision',
                                folder='Home_001_1')

    for k, v in d.items():
        print(k)
        print(v)
